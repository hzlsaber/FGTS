"""
Unified model loading interface

Supported models:
- DINOv2 (timm)
- DINOv3 (timm)
- CLIP (OpenAI)
- HuggingFace dinov2-with-registers
"""

from pathlib import Path
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


def load_model(model_name: str,
               checkpoint: Optional[str] = None,
               img_size: int = 224,
               device: str = 'cuda') -> Tuple[nn.Module, transforms.Compose, str]:
    """
    Unified model loading function

    Args:
        model_name: Model name
            - DINOv3: 'dinov3_vitl16', 'dinov3_vitb16', 'dinov3_vit_small', etc.
            - CLIP: 'clip_ViT-L/14', 'clip_ViT-B/32', etc.
            - HuggingFace: 'hf_registers' (requires additional hf_model_path)
        checkpoint: Custom checkpoint path (optional)
        img_size: Input image size
        device: Device ('cuda' or 'cpu')

    Returns:
        model: Loaded model (parameters frozen)
        transform: Corresponding data preprocessing
        model_type: Model type ('timm', 'clip', 'hf')
    """
    is_clip = model_name.startswith('clip_')
    is_hf = model_name == 'hf_registers'

    if is_clip:
        return _load_clip_model(model_name, device)
    elif is_hf:
        raise ValueError("HF models need to be loaded via load_hf_model() with hf_model_path")
    else:
        return _load_timm_model(model_name, checkpoint, img_size, device)


def load_hf_model(hf_model_path: str,
                  device: str = 'cuda') -> Tuple[nn.Module, transforms.Compose, str, object]:
    """
    Load HuggingFace model (e.g., dinov2-with-registers)

    Args:
        hf_model_path: HuggingFace model path or ID
        device: Device

    Returns:
        model: Loaded model
        transform: Data preprocessing
        model_type: 'hf'
        config: HuggingFace config object
    """
    try:
        from transformers import AutoModel, AutoConfig
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")

    print(f"[HF] Loading model from {hf_model_path}")

    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_model_path, trust_remote_code=True)

    model = model.to(device)
    model.eval()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # HF model image size
    hf_img_size = getattr(config, 'image_size', 518)

    transform = transforms.Compose([
        transforms.Resize((hf_img_size, hf_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    print(f"[HF] Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"[HF] Image size: {hf_img_size}x{hf_img_size}")
    print(f"[HF] Register tokens: {getattr(config, 'num_register_tokens', 0)}")

    return model, transform, 'hf', config


def _load_timm_model(model_name: str,
                     checkpoint: Optional[str],
                     img_size: int,
                     device: str) -> Tuple[nn.Module, transforms.Compose, str]:
    """Load timm model (DINOv2/DINOv3)"""
    import timm

    # Import model alias mapping
    try:
        from models.dinov3_models import _ALIAS_TO_TIMM
        timm_name = _ALIAS_TO_TIMM.get(model_name, model_name)
    except ImportError:
        # If no alias mapping, use model name directly
        timm_name = model_name

    print(f"[timm] Loading model: {timm_name}")

    # Create model
    model = timm.create_model(
        timm_name,
        pretrained=(checkpoint is None),
        img_size=img_size
    )

    # Load custom checkpoint
    if checkpoint is not None:
        print(f"[timm] Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Remove 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Load weights
        msg = model.load_state_dict(new_state_dict, strict=False)
        if msg.missing_keys:
            print(f"[timm] Missing keys: {len(msg.missing_keys)}")
        if msg.unexpected_keys:
            print(f"[timm] Unexpected keys: {len(msg.unexpected_keys)}")

    # Remove classification head
    if hasattr(model, 'reset_classifier'):
        model.reset_classifier(0)

    model = model.to(device)
    model.eval()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    print(f"[timm] Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"[timm] Image size: {img_size}x{img_size}")

    return model, transform, 'timm'


def _load_clip_model(model_name: str,
                     device: str) -> Tuple[nn.Module, transforms.Compose, str]:
    """Load CLIP model"""
    try:
        from models.clip import clip
    except ImportError:
        raise ImportError("CLIP model not found. Please check models/clip.py")

    clip_name = model_name.replace('clip_', '')
    print(f"[CLIP] Loading model: {clip_name}")

    model, preprocess = clip.load(clip_name, device=device)
    model.eval()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # CLIP transform
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    print(f"[CLIP] Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    return model, transform, 'clip'


def get_token_layout(model, num_tokens: int, model_config=None) -> Dict[str, List[int]]:
    """
    Get token layout (CLS/REG/PATCH indices)

    Args:
        model: Model instance
        num_tokens: Total number of tokens
        model_config: HuggingFace model config (optional)

    Returns:
        Dictionary containing:
        - cls_idx: CLS token index list
        - reg_idx: Register token index list
        - patch_idx: Patch token index list
    """
    # Detect number of register tokens
    num_register_tokens = 0

    # Get from config first (HF model)
    if model_config is not None:
        num_register_tokens = getattr(model_config, 'num_register_tokens', 0)
    else:
        # Get from model itself (timm model)
        num_register_tokens = getattr(model, 'num_register_tokens', 0)
        num_prefix = getattr(model, 'num_prefix_tokens', None)
        if num_prefix is not None and num_prefix > 1:
            num_register_tokens = num_prefix - 1  # Subtract CLS

    # Build token indices
    cls_idx = [0]

    if num_register_tokens > 0:
        reg_idx = list(range(1, 1 + num_register_tokens))
        patch_start = 1 + num_register_tokens
        patch_idx = list(range(patch_start, num_tokens))
    else:
        reg_idx = []
        patch_idx = list(range(1, num_tokens))

    print(f"[Token Layout] CLS={cls_idx}, REG={len(reg_idx)} tokens, PATCH={len(patch_idx)} tokens")

    return {
        "cls_idx": cls_idx,
        "reg_idx": reg_idx,
        "patch_idx": patch_idx,
    }


def freeze_model(model):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
