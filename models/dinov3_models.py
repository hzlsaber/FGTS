# models/dinov3_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Alias mapping: your alias in VALID_NAMES -> actual timm model name
# Convention:
#   * *_14 series: maps to DINOv2 ViT patch14 weights (common and stable)
#   * *_16 series: maps to DINOv3 ViT patch16 weights (requires newer timm)
_ALIAS_TO_TIMM = {
    # DINOv2 (patch14)
    'dinov2_vits14': 'vit_small_patch14_dinov2',
    'dinov2_vitb14': 'vit_base_patch14_dinov2',
    'dinov2_vitl14': 'vit_large_patch14_dinov2',
    'dinov2_vith14': 'vit_giant_patch14_dinov2',

    # DINOv3 (patch16) - Note: timm doesn't have .lvd1689m suffix, huge_plus replaces giant
    'dinov3_vits16': 'vit_small_patch16_dinov3',
    'dinov3_vitb16': 'vit_base_patch16_dinov3',
    'dinov3_vitl16': 'vit_large_patch16_dinov3',
    'dinov3_vith16': 'vit_huge_plus_patch16_dinov3', 
    'dinov3_vit_7b': 'vit_7b_patch16_dinov3',  # timm doesn't have giant, use huge_plus
}

class DinoV3Model(nn.Module):
    def __init__(self, model_name: str = 'dinov3_vitl16', img_size: int = None, pool_type: str = 'patch_avg', custom_token_indices: list = None):
        """
        model_name: The value from name.split(':',1)[1] in __init__.py,
                    e.g., 'dinov3_vitl16' or 'dinov3_vitl14'
        img_size: Optional input image size. If specified, timm will automatically interpolate
                  position embeddings to adapt to this size.
                  Example: img_size=256 adapts DINOv2 from 518x518 to 256x256
        pool_type: Token pooling method
                  - 'patch_avg': Average PATCH tokens (default, original behavior)
                  - 'cls': Use CLS token
                  - 'reg_avg': Average REG tokens
                  - 'cls+reg': Concatenate CLS and REG average
                  - 'cls+patch': Concatenate CLS and PATCH average (without registers)
                  - 'custom_tokens': Average tokens specified by custom_token_indices
        custom_token_indices: Custom token index list, only used when pool_type='custom_tokens'
                             Example: [1, 0, 4, 3, 2, 5, 18, 187, 200, 6]
        """
        super().__init__()
        key = model_name.lower()
        timm_name = _ALIAS_TO_TIMM.get(key)
        if timm_name is None:
            raise ValueError(
                f"Unknown DINO alias '{model_name}'. "
                f"Valid: {', '.join(sorted(_ALIAS_TO_TIMM.keys()))}"
            )

        # Create backbone; requires timm version that supports the entry name
        # If img_size is specified, timm will automatically interpolate position embeddings after loading pretrained weights
        if img_size is not None:
            self.backbone = timm.create_model(timm_name, pretrained=True, img_size=img_size)
            print(f"[DinoV3Model] Created {timm_name} with custom img_size={img_size}")
        else:
            self.backbone = timm.create_model(timm_name, pretrained=True)
            print(f"[DinoV3Model] Created {timm_name} with default img_size")

        # Remove classifier head, keep features only
        if hasattr(self.backbone, 'reset_classifier'):
            self.backbone.reset_classifier(0)

        # Save pool_type configuration
        self.pool_type = pool_type
        self.custom_token_indices = custom_token_indices

        # Get feature dimension (if unavailable, probe with one forward pass)
        feat_dim = getattr(self.backbone, 'num_features', None)
        if feat_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feats = self._extract_feats(self.backbone, dummy, pool_type=pool_type, custom_token_indices=custom_token_indices)
                feat_dim = feats.shape[-1]
        else:
            # If cls+reg or cls+patch, feature dimension doubles
            if pool_type in ['cls+reg', 'cls+patch']:
                feat_dim = feat_dim * 2

        self.feat_dim = feat_dim

        # Binary classification head (for BCEWithLogitsLoss)
        self.fc = nn.Linear(self.feat_dim, 1)

        if pool_type == 'custom_tokens':
            print(f"[DinoV3Model] Using pool_type='custom_tokens', indices={custom_token_indices}, feat_dim={feat_dim}")
        else:
            print(f"[DinoV3Model] Using pool_type='{pool_type}', feat_dim={feat_dim}")

    @staticmethod
    def _extract_feats(backbone, x: torch.Tensor, pool_type: str = 'patch_avg', custom_token_indices: list = None) -> torch.Tensor:
        """
        Safely extract pre-logits features from timm model.
        Supports multiple token pooling methods.

        Args:
            backbone: DINOv3 backbone model
            x: Input images [B, 3, H, W]
            pool_type: Pooling method
                - 'patch_avg': Average PATCH tokens (original timm behavior)
                - 'cls': CLS token
                - 'reg_avg': Average REG tokens
                - 'cls+reg': Concatenate CLS and REG average
                - 'cls+patch': Concatenate CLS and PATCH average (without registers)
                - 'custom_tokens': Average tokens specified by custom_token_indices
            custom_token_indices: Custom token index list, only used when pool_type='custom_tokens'

        Returns:
            feats: [B, D] or [B, 2D] (when pool_type='cls+reg' or 'cls+patch')
        """
        feats = backbone.forward_features(x)  # [B, num_tokens, D]

        # Handle dict return case
        if isinstance(feats, dict):
            # Try to get CLS token directly first
            if pool_type == 'cls' and 'x_norm_clstoken' in feats:
                return feats['x_norm_clstoken']

            # Otherwise extract all tokens
            for k in ('x', 'x_norm_clspool', 'pool'):
                if k in feats and feats[k] is not None:
                    feats = feats[k]
                    break
            else:
                # Fallback: use forward_head
                feats = backbone.forward_head(feats, pre_logits=True)
                return feats

        # Now feats should be [B, num_tokens, D] Tensor
        # Extract features based on pool_type
        if pool_type == 'cls':
            # Extract CLS token (first token)
            return feats[:, 0, :]  # [B, D]

        elif pool_type == 'reg_avg':
            # Extract REG tokens (tokens 1-4) and average
            num_prefix = getattr(backbone, 'num_prefix_tokens', 5)
            num_reg = num_prefix - 1  # Subtract CLS
            if num_reg > 0:
                reg_tokens = feats[:, 1:1+num_reg, :]  # [B, num_reg, D]
                return reg_tokens.mean(dim=1)  # [B, D]
            else:
                # If no REG tokens, fallback to CLS
                print("[Warning] No REG tokens found, using CLS instead")
                return feats[:, 0, :]

        elif pool_type == 'cls+reg':
            # Concatenate CLS and REG average
            cls_feat = feats[:, 0, :]  # [B, D]

            num_prefix = getattr(backbone, 'num_prefix_tokens', 5)
            num_reg = num_prefix - 1
            if num_reg > 0:
                reg_tokens = feats[:, 1:1+num_reg, :]  # [B, num_reg, D]
                reg_feat = reg_tokens.mean(dim=1)  # [B, D]
            else:
                # If no REG tokens, duplicate CLS
                reg_feat = cls_feat

            return torch.cat([cls_feat, reg_feat], dim=-1)  # [B, 2D]

        elif pool_type == 'cls+patch':
            # Concatenate CLS and PATCH average (without register tokens)
            cls_feat = feats[:, 0, :]  # [B, D]

            # Get patch tokens starting position (skip CLS and register tokens)
            num_prefix = getattr(backbone, 'num_prefix_tokens', 5)
            patch_tokens = feats[:, num_prefix:, :]  # [B, num_patches, D]
            patch_feat = patch_tokens.mean(dim=1)  # [B, D]

            return torch.cat([cls_feat, patch_feat], dim=-1)  # [B, 2D]

        elif pool_type == 'patch_avg':
            # Use original forward_head (PATCH average)
            return backbone.forward_head(feats, pre_logits=True)

        elif pool_type == 'custom_tokens':
            # Average custom token indices
            if custom_token_indices is None or len(custom_token_indices) == 0:
                raise ValueError("custom_token_indices must be provided when pool_type='custom_tokens'")

            # Extract tokens at specified indices
            try:
                selected_tokens = feats[:, custom_token_indices, :]  # [B, num_selected, D]
                return selected_tokens.mean(dim=1)  # [B, D]
            except IndexError as e:
                raise ValueError(f"Token indices {custom_token_indices} out of range. "
                               f"Total tokens: {feats.shape[1]}. Error: {e}")

        else:
            raise ValueError(f"Unknown pool_type: {pool_type}. "
                           f"Valid options: 'patch_avg', 'cls', 'reg_avg', 'cls+reg', 'cls+patch', 'custom_tokens'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._extract_feats(self.backbone, x, pool_type=self.pool_type, custom_token_indices=self.custom_token_indices)   # [B, D] or [B, 2D]
        feats = F.normalize(feats, dim=-1)
        logits = self.fc(feats)                         # [B, 1]
        return logits


# ==================== Multi-layer Feature Version ====================

class DinoV3MultiLayerModel(nn.Module):
    """
    Multi-layer feature version of DINOv3 model, extracts and fuses shallow and deep features

    Supported strategies:
    - 'shallow_only': Use only shallow features (test low-level dependency)
    - 'deep_only': Use only deep features (original version, default)
    - 'concat': Concatenate shallow + deep then pass through FC
    - 'weighted': Weighted sum of shallow + deep
    - 'mean_pool': Mean pooling over multi-layer features
    - 'attention': Fuse multi-layer features using attention mechanism
    """

    def __init__(
        self,
        model_name: str = 'dinov3_vitl16',
        img_size: int = None,
        layer_strategy: str = 'deep_only',
        shallow_layers: list = None,  # e.g., [2, 5, 8] represents layers 3,6,9
        deep_layers: list = None,     # e.g., [-1] represents last layer
        token_pool: str = 'auto',     # 'auto', 'cls', 'patch_mean', 'patch_max'
    ):
        """
        Args:
            model_name: DINO model name
            img_size: Input image size
            layer_strategy: Feature fusion strategy
            shallow_layers: Shallow layer index list (None uses default [2, 5, 8])
            deep_layers: Deep layer index list (None uses default [-1])
            token_pool: Token pooling method
                - 'auto': shallow_only uses patch_mean, others use cls
                - 'cls': Use CLS token (global semantics)
                - 'patch_mean': Average patch tokens (local texture)
                - 'patch_max': Max pool patch tokens
        """
        super().__init__()

        # Default layer configuration
        if shallow_layers is None:
            shallow_layers = [2, 5, 8]  # Layers 3,6,9 (0-indexed)
            #shallow_layers = [8, 11, 14]  # Layers 9,12,15 (0-indexed)
        if deep_layers is None:
            deep_layers = [-1]  # Last layer

        self.layer_strategy = layer_strategy
        self.shallow_layers = shallow_layers
        self.deep_layers = deep_layers

        # Auto-select token pooling strategy
        if token_pool == 'auto':
            # shallow_only uses patch features, others use CLS
            if layer_strategy == 'shallow_only':
                self.token_pool = 'patch_mean'
            else:
                self.token_pool = 'cls'
        else:
            self.token_pool = token_pool

        # Create backbone
        key = model_name.lower()
        timm_name = _ALIAS_TO_TIMM.get(key)
        if timm_name is None:
            raise ValueError(
                f"Unknown DINO alias '{model_name}'. "
                f"Valid: {', '.join(sorted(_ALIAS_TO_TIMM.keys()))}"
            )

        if img_size is not None:
            self.backbone = timm.create_model(timm_name, pretrained=True, img_size=img_size)
            print(f"[DinoV3MultiLayerModel] Created {timm_name} with img_size={img_size}")
        else:
            self.backbone = timm.create_model(timm_name, pretrained=True)
            print(f"[DinoV3MultiLayerModel] Created {timm_name} with default img_size")

        if hasattr(self.backbone, 'reset_classifier'):
            self.backbone.reset_classifier(0)

        # Get base feature dimension
        self.base_feat_dim = getattr(self.backbone, 'num_features', 1024)

        # Calculate final feature dimension and create FC layer based on strategy
        self._setup_classifier()

        print(f"[DinoV3MultiLayerModel] Strategy: {layer_strategy}")
        print(f"  Shallow layers: {shallow_layers}")
        print(f"  Deep layers: {deep_layers}")
        print(f"  Token pooling: {self.token_pool}")
        print(f"  Final feature dim: {self.feat_dim}")

    def _setup_classifier(self):
        """Setup classifier based on fusion strategy"""
        if self.layer_strategy == 'shallow_only':
            # Use shallow layers only
            self.feat_dim = self.base_feat_dim * len(self.shallow_layers)
            self.fc = nn.Linear(self.feat_dim, 1)

        elif self.layer_strategy == 'deep_only':
            # Use deep layers only (original approach)
            self.feat_dim = self.base_feat_dim
            self.fc = nn.Linear(self.feat_dim, 1)

        elif self.layer_strategy == 'concat':
            # Concatenate shallow and deep layers
            self.feat_dim = self.base_feat_dim * (len(self.shallow_layers) + len(self.deep_layers))
            self.fc = nn.Linear(self.feat_dim, 1)

        elif self.layer_strategy == 'weighted':
            # Weighted sum (learnable weights)
            self.feat_dim = self.base_feat_dim
            num_layers = len(self.shallow_layers) + len(self.deep_layers)
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            self.fc = nn.Linear(self.feat_dim, 1)

        elif self.layer_strategy == 'mean_pool':
            # Mean pooling
            self.feat_dim = self.base_feat_dim
            self.fc = nn.Linear(self.feat_dim, 1)

        elif self.layer_strategy == 'attention':
            # Attention fusion
            self.feat_dim = self.base_feat_dim
            num_layers = len(self.shallow_layers) + len(self.deep_layers)
            self.attention = nn.MultiheadAttention(
                embed_dim=self.base_feat_dim,
                num_heads=8,
                batch_first=True
            )
            self.fc = nn.Linear(self.feat_dim, 1)
        else:
            raise ValueError(f"Unknown layer_strategy: {self.layer_strategy}")

    def _forward_and_collect(self, x, indices, norm=True):
        """
        Manually forward propagate and collect intermediate layer outputs (fallback for forward_intermediates)

        Args:
            x: [B, 3, H, W] Input images
            indices: List of layer indices to collect
            norm: Whether to apply LayerNorm to outputs

        Returns:
            final_x: [B, 1+N, D] Last layer output
            intermediates: list of [B, 1+N, D] Intermediate layer outputs
        """
        bb = self.backbone

        # Patch embedding
        x = bb.patch_embed(x)

        # Position embedding (handle CLS/register/pos)
        if hasattr(bb, '_pos_embed'):
            x = bb._pos_embed(x)
            # _pos_embed may return tuple, need to unpack
            if isinstance(x, tuple):
                x = x[0]
        elif hasattr(bb, 'pos_embed'):
            # Fallback approach
            cls_token = bb.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + bb.pos_embed

        # Position dropout
        if hasattr(bb, 'pos_drop'):
            x = bb.pos_drop(x)

        # Forward propagate and collect intermediate layers
        intermediates = []
        num_blocks = len(bb.blocks)

        # Convert negative indices to positive
        positive_indices = []
        for idx in indices:
            if idx < 0:
                positive_indices.append(num_blocks + idx)
            else:
                positive_indices.append(idx)

        # Find max index for early break optimization
        max_index = max(positive_indices)

        for i, blk in enumerate(bb.blocks):
            x = blk(x)
            if i in positive_indices:
                # If norm is needed and model has norm layer
                if norm and hasattr(bb, 'norm'):
                    y = bb.norm(x)
                else:
                    y = x
                intermediates.append(y)

            # Early break: stop forward propagation if all needed layers are collected
            if i >= max_index:
                break

        # Final output (if needed)
        if hasattr(bb, 'norm'):
            x = bb.norm(x)

        return x, intermediates

    def _pool_layer_tokens(self, x_layer, pool='patch_mean'):
        """
        Pool token features from layer output

        Args:
            x_layer: [B, 1+N, D] Layer output containing CLS and patch tokens
                    (DINOv3 may also have register tokens: [CLS, REG*, PATCH*])
                    May also be in tuple/dict format
            pool: Pooling method
                - 'cls': Take CLS token (global semantic features)
                - 'patch_mean': Average patch tokens (local/texture features)
                - 'patch_max': Max pool patch tokens

        Returns:
            [B, D] Pooled features
        """
        # Handle possible tuple/dict wrapping
        if isinstance(x_layer, tuple):
            x_layer = x_layer[0]  # Take first element
        elif isinstance(x_layer, dict):
            # Try common keys
            for key in ['x', 'x_norm_patchtokens', 'x_norm_clstoken']:
                if key in x_layer:
                    x_layer = x_layer[key]
                    break

        # If already 2D, return directly
        if x_layer.dim() == 2:
            return x_layer

        # Handle 4D spatial feature map [B, D, H, W] (timm forward_intermediates return format)
        if x_layer.dim() == 4:
            # [B, D, H, W] -> [B, D]
            if pool in ['patch_mean', 'cls']:  # Approximate CLS with GAP
                return x_layer.mean(dim=[2, 3])  # Global Average Pooling
            elif pool == 'patch_max':
                return x_layer.amax(dim=[2, 3])  # Global Max Pooling
            else:
                raise ValueError(f"Unknown pool method: {pool}")

        # Handle 3D token sequence [B, 1+N, D]
        elif x_layer.dim() == 3:
            if pool == 'cls':
                # Take first token (CLS)
                return x_layer[:, 0, :]  # [B, D]

            elif pool in ['patch_mean', 'patch_max']:
                # Properly handle register tokens
                # DINOv3 format: [CLS, REG_0, ..., REG_k, PATCH_0, ..., PATCH_n]
                # Need to find starting position of patch tokens

                # Try to get num_prefix_tokens from backbone
                num_prefix = getattr(self.backbone, 'num_prefix_tokens', None)
                if num_prefix is None:
                    # Fallback: infer from other attributes
                    num_register_tokens = getattr(self.backbone, 'num_register_tokens', 0)
                    num_prefix = 1 + num_register_tokens  # CLS + register tokens

                # Extract pure patch tokens (skip CLS and register tokens)
                patch_tokens = x_layer[:, num_prefix:, :]  # [B, N_patches, D]

                if pool == 'patch_mean':
                    return patch_tokens.mean(dim=1)  # [B, D]
                else:  # patch_max
                    return patch_tokens.amax(dim=1)  # [B, D]

            else:
                raise ValueError(f"Unknown pool method: {pool}")

        else:
            raise ValueError(f"Unexpected x_layer shape: {x_layer.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation, extract and fuse multi-layer features based on strategy
        """
        # Decide which layers to extract
        if self.layer_strategy == 'shallow_only':
            indices = self.shallow_layers
        elif self.layer_strategy == 'deep_only':
            indices = self.deep_layers
        else:
            # Need both shallow and deep layers
            indices = self.shallow_layers + self.deep_layers

        # Use forward_intermediates to extract intermediate layer features
        # Returns: (final_output, intermediates)
        # intermediates is a list, each element corresponds to one layer's output
        try:
            # Try to use timm's forward_intermediates API
            final_output, intermediates = self.backbone.forward_intermediates(
                x,
                indices=indices,
                return_prefix_tokens=True,  # Include CLS token
                norm=True  # Apply LayerNorm to outputs
            )
        except (TypeError, AttributeError):
            # Fallback: manual forward propagation
            try:
                final_output, intermediates = self.backbone.forward_intermediates(x, indices=indices)
            except (TypeError, AttributeError):
                # Complete fallback: manual implementation
                final_output, intermediates = self._forward_and_collect(x, indices, norm=True)

        # Extract features using configured pooling method
        features = []
        for inter in intermediates:
            feat = self._pool_layer_tokens(inter, pool=self.token_pool)
            features.append(feat)

        # Fuse features based on strategy
        if self.layer_strategy == 'concat':
            # Concatenate all layer features
            fused = torch.cat(features, dim=-1)  # [B, D * num_layers]

        elif self.layer_strategy == 'weighted':
            # Weighted sum
            # features: list of [B, D]
            stacked = torch.stack(features, dim=1)  # [B, num_layers, D]
            weights = F.softmax(self.layer_weights, dim=0)  # [num_layers]
            weights = weights.view(1, -1, 1)  # [1, num_layers, 1]
            fused = (stacked * weights).sum(dim=1)  # [B, D]

        elif self.layer_strategy == 'mean_pool':
            # Mean pooling
            stacked = torch.stack(features, dim=1)  # [B, num_layers, D]
            fused = stacked.mean(dim=1)  # [B, D]

        elif self.layer_strategy == 'attention':
            # Attention fusion
            stacked = torch.stack(features, dim=1)  # [B, num_layers, D]
            # Use last layer as query, all layers as key and value
            query = features[-1].unsqueeze(1)  # [B, 1, D]
            attended, _ = self.attention(query, stacked, stacked)  # [B, 1, D]
            fused = attended.squeeze(1)  # [B, D]

        elif self.layer_strategy in ['shallow_only', 'deep_only']:
            # Use a single group of layers
            if len(features) == 1:
                fused = features[0]
            else:
                # Concatenate multiple layers (consistent with _setup_classifier dimension)
                fused = torch.cat(features, dim=-1)
        else:
            raise ValueError(f"Unknown strategy: {self.layer_strategy}")

        # Normalize and classify
        fused = F.normalize(fused, dim=-1)
        logits = self.fc(fused)  # [B, 1]

        return logits
