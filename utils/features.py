"""
Feature extraction utilities

Unified feature extraction interface supporting different model types
"""

from typing import Tuple, List, Optional
import torch
import numpy as np
from tqdm import tqdm


def extract_features(model, dataloader, model_type: str, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified feature extraction function

    Args:
        model: Model instance
        dataloader: Data loader
        model_type: Model type ('timm', 'clip', 'hf')
        device: Device

    Returns:
        features: [N, num_tokens, D] for timm/hf, [N, 1, D] for CLIP
        labels: [N]
    """
    if model_type == 'timm':
        return _extract_features_timm(model, dataloader, device)
    elif model_type == 'clip':
        return _extract_features_clip(model, dataloader, device)
    elif model_type == 'hf':
        return _extract_features_hf(model, dataloader, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _extract_features_timm(model, dataloader, device='cuda'):
    """Extract features using timm model"""
    all_feats = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for imgs, labs in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)

            # Extract token features
            out = model.forward_features(imgs)
            if isinstance(out, dict):
                # Get token-level output uniformly
                for k in ["x", "tokens", "last_hidden_state"]:
                    if k in out and out[k] is not None:
                        out = out[k]
                        break

            all_feats.append(out.cpu().numpy())
            all_labels.append(labs.numpy())

    features = np.concatenate(all_feats, axis=0)  # [N, T, D]
    labels = np.concatenate(all_labels, axis=0)    # [N]

    print(f"[extract_features] shape={features.shape}")
    return features, labels


def _extract_features_clip(model, dataloader, device='cuda'):
    """Extract features using CLIP model"""
    all_feats = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for imgs, labs in tqdm(dataloader, desc="Extracting features (CLIP)"):
            imgs = imgs.to(device)

            # CLIP extracts global features
            features = model.encode_image(imgs)  # [B, D]

            all_feats.append(features.cpu().numpy())
            all_labels.append(labs.numpy())

    features = np.concatenate(all_feats, axis=0)  # [N, D]
    labels = np.concatenate(all_labels, axis=0)    # [N]

    # Expand dimension to match [N, num_tokens, D] format
    features = features[:, np.newaxis, :]  # [N, 1, D]

    print(f"[extract_features_clip] shape={features.shape}")
    return features, labels


def _extract_features_hf(model, dataloader, device='cuda'):
    """Extract features using HuggingFace model"""
    all_feats = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for imgs, labs in tqdm(dataloader, desc="Extracting features (HF)"):
            imgs = imgs.to(device)

            # HF model extracts token features
            out = model(pixel_values=imgs, output_hidden_states=True)
            tokens = out.last_hidden_state  # [B, num_tokens, D]

            all_feats.append(tokens.cpu().numpy())
            all_labels.append(labs.numpy())

    features = np.concatenate(all_feats, axis=0)  # [N, T, D]
    labels = np.concatenate(all_labels, axis=0)    # [N]

    print(f"[extract_features_hf] shape={features.shape}")
    return features, labels


def select_tokens(features: np.ndarray, token_indices: List[int]) -> np.ndarray:
    """
    Select specified tokens

    Args:
        features: [N, num_tokens, D]
        token_indices: List of token indices to select

    Returns:
        selected_features: [N, len(token_indices), D]
    """
    return features[:, token_indices, :]


def filter_special_tokens(indices: np.ndarray, num_register_tokens: int) -> np.ndarray:
    """
    Filter out special tokens (CLS and register tokens) from indices

    Args:
        indices: [K] - Token indices (e.g., top-k Fisher scores)
        num_register_tokens: Number of register tokens
            - DINOv2: 0 (only CLS at index 0)
            - DINOv3: 4 (CLS at 0, registers at 1-4)

    Returns:
        filtered_indices: Filtered indices with special tokens removed

    Examples:
        DINOv2 (no registers):
            Input:  [0, 6, 7, 22, 1, 65]
            Output: [6, 7, 22, 65]  # removed 0(CLS)

        DINOv3 (4 registers):
            Input:  [0, 6, 7, 22, 65, 75, 1, 2, 3, 4, 88]
            Output: [6, 7, 22, 65, 75, 88]  # removed 0,1,2,3,4
    """
    # Special tokens to remove: CLS (0) + register tokens (1 to num_register_tokens)
    special_token_indices = set(range(num_register_tokens + 1))

    # Filter out special tokens
    filtered = indices[~np.isin(indices, list(special_token_indices))]

    return filtered


def select_top_k_tokens(fisher_scores: np.ndarray,
                        top_k: int,
                        num_register_tokens: int,
                        auto_extend: bool = True) -> np.ndarray:
    """
    Select top-k tokens based on Fisher scores, excluding special tokens

    Args:
        fisher_scores: [num_tokens] - Fisher score for each token
        top_k: Number of top tokens to select
        num_register_tokens: Number of register tokens (0 for DINOv2, 4 for DINOv3)
        auto_extend: If True, automatically extend selection to get exactly top_k patch tokens

    Returns:
        top_indices: [top_k] - Selected token indices (only patch tokens)

    Example:
        Fisher scores ranking: [0, 6, 7, 22, 65, 75, 1, 2, 3, 4, 88, 99, 105, 110, 120]
        top_k = 10, num_register_tokens = 4

        Step 1: Get top-15 to ensure we have enough after filtering
        Step 2: Filter out 0,1,2,3,4 -> [6, 7, 22, 65, 75, 88, 99, 105, 110, 120]
        Step 3: Return first 10 -> [6, 7, 22, 65, 75, 88, 99, 105, 110, 120]
    """
    num_special = num_register_tokens + 1  # CLS + register tokens

    # If auto_extend, fetch more candidates to ensure we have enough after filtering
    if auto_extend:
        # Estimate: fetch top_k + num_special tokens to account for filtered ones
        fetch_k = min(top_k + num_special + 10, len(fisher_scores))
    else:
        fetch_k = min(top_k, len(fisher_scores))

    # Get top-K indices (descending order by Fisher score)
    all_top_indices = np.argsort(fisher_scores)[-fetch_k:][::-1]

    # Filter out special tokens
    filtered_indices = filter_special_tokens(all_top_indices, num_register_tokens)

    # Take exactly top_k tokens
    if len(filtered_indices) < top_k:
        print(f"[Warning] Only {len(filtered_indices)} patch tokens available, requested {top_k}")
        return filtered_indices
    else:
        return filtered_indices[:top_k]


def compute_fisher_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute Fisher discrimination score for each token

    Fisher ratio = (inter-class distance) / (intra-class distance)

    Args:
        features: [N, num_tokens, D]
        labels: [N] - 0=real, 1=fake

    Returns:
        fisher_scores: [num_tokens] - Discrimination score for each token
    """
    N, num_tokens, D = features.shape

    # Separate real/fake samples
    real_mask = labels == 0
    fake_mask = labels == 1

    real_feats = features[real_mask]  # [N_real, T, D]
    fake_feats = features[fake_mask]  # [N_fake, T, D]

    fisher_scores = np.zeros(num_tokens)

    for t in range(num_tokens):
        # Extract all sample features for this token
        real_t = real_feats[:, t, :]  # [N_real, D]
        fake_t = fake_feats[:, t, :]  # [N_fake, D]

        # Compute centers
        real_center = real_t.mean(axis=0)  # [D]
        fake_center = fake_t.mean(axis=0)  # [D]

        # Inter-class distance
        inter_dist = np.linalg.norm(real_center - fake_center)

        # Intra-class distance
        real_intra = np.mean([np.linalg.norm(f - real_center) for f in real_t])
        fake_intra = np.mean([np.linalg.norm(f - fake_center) for f in fake_t])
        intra_dist = (real_intra + fake_intra) / 2

        # Fisher discrimination ratio
        fisher_scores[t] = inter_dist / (intra_dist + 1e-8)

    print(f"[Fisher scores] shape={fisher_scores.shape}")
    print(f"[Fisher scores] Top-5 scores: {np.sort(fisher_scores)[-5:][::-1]}")

    return fisher_scores


def pool_features(features: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Pool token features

    Args:
        features: [N, num_tokens, D]
        method: Pooling method ('mean', 'max', 'cls')

    Returns:
        pooled: [N, D]
    """
    if features.ndim == 2:
        # Already pooled features
        return features

    if method == 'mean':
        return features.mean(axis=1)
    elif method == 'max':
        return features.max(axis=1)
    elif method == 'cls':
        return features[:, 0, :]  # Take only CLS token
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def compute_register_subspace(features: np.ndarray, reg_idx: List[int]) -> np.ndarray:
    """
    Compute subspace basis matrix U using register tokens (for zero-shot experiments)

    Args:
        features: [N, num_tokens, D]
        reg_idx: List of register token indices

    Returns:
        U: [D, k] - Subspace basis matrix
    """
    if len(reg_idx) == 0:
        raise ValueError("No register tokens available")

    # Extract register tokens
    reg_tokens = features[:, reg_idx, :]  # [N, num_reg, D]
    N, num_reg, D = reg_tokens.shape

    # Flatten
    reg_tokens_flat = reg_tokens.reshape(N * num_reg, D)

    # Center
    mean_reg = reg_tokens_flat.mean(axis=0)
    reg_centered = reg_tokens_flat - mean_reg[np.newaxis, :]

    # SVD decomposition
    U, S, Vt = np.linalg.svd(reg_centered.T @ reg_centered, full_matrices=False)

    # Take first num_reg principal components
    k = min(num_reg, D, len(S))
    U_subspace = U[:, :k]  # [D, k]

    print(f"[Register subspace] dim={k}, top singular values: {S[:5]}")

    return U_subspace


def decompose_tokens(features: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose features into parallel and orthogonal components (for zero-shot experiments)

    Args:
        features: [N, num_tokens, D]
        U: [D, k] - Subspace basis matrix

    Returns:
        parallel: [N, num_tokens, D] - Parallel component
        orthogonal: [N, num_tokens, D] - Orthogonal component
    """
    N, T, D = features.shape

    # Reshape
    features_flat = features.reshape(N * T, D)

    # Projection matrix
    P = U @ U.T  # [D, D]

    # Parallel component
    parallel_flat = features_flat @ P.T
    parallel = parallel_flat.reshape(N, T, D)

    # Orthogonal component
    orthogonal = features - parallel

    return parallel, orthogonal
