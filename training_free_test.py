"""
Zero-Shot Deepfake Detection

Completely training-free approach using frozen foundation models.
Uses feature centroids from a reference dataset to classify test images.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Dict
import torch
import numpy as np

from utils import (
    build_dataloader,
    get_all_test_datasets,
    load_model,
    load_hf_model,
    get_token_layout,
    extract_features,
    select_tokens,
    compute_metrics,
    save_json,
    generate_report,
    load_fisher_scores,
    compute_fisher_scores,
    select_top_k_tokens,
    save_fisher_scores,
    compute_register_subspace,
    decompose_tokens
)


class ZeroShotClassifier:
    """
    Zero-shot classifier based on feature centroid distance

    Principle:
    1. Compute real/fake centroids from reference dataset
    2. For test images: classify based on nearest centroid
    3. No training required
    """
    def __init__(self, token_indices: List[int]):
        self.token_indices = token_indices
        self.real_center = None
        self.fake_center = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Compute real/fake centroids

        Args:
            features: [N, num_tokens, D]
            labels: [N] - 0=real, 1=fake
        """
        # Select and pool tokens
        selected = features[:, self.token_indices, :]  # [N, K, D]
        pooled = selected.mean(axis=1)                  # [N, D]

        # Separate real/fake
        real_mask = labels == 0
        fake_mask = labels == 1

        real_feats = pooled[real_mask]
        fake_feats = pooled[fake_mask]

        # Compute centroids
        self.real_center = real_feats.mean(axis=0)  # [D]
        self.fake_center = fake_feats.mean(axis=0)  # [D]

        # Statistics
        inter_dist = np.linalg.norm(self.real_center - self.fake_center)
        real_intra = np.mean([np.linalg.norm(f - self.real_center) for f in real_feats])
        fake_intra = np.mean([np.linalg.norm(f - self.fake_center) for f in fake_feats])
        intra_dist = (real_intra + fake_intra) / 2
        fisher_ratio = inter_dist / (intra_dist + 1e-8)

        print(f"\n[ZeroShot Classifier]")
        print(f"  Real samples: {real_mask.sum()}")
        print(f"  Fake samples: {fake_mask.sum()}")
        print(f"  Inter-class distance: {inter_dist:.4f}")
        print(f"  Intra-class distance: {intra_dist:.4f}")
        print(f"  Fisher ratio: {fisher_ratio:.4f}")

    def predict(self, features: np.ndarray):
        """
        Predict labels for test images

        Args:
            features: [N, num_tokens, D]

        Returns:
            predictions: [N]
            probs: [N] - probability of being fake
        """
        # Select and pool tokens
        selected = features[:, self.token_indices, :]
        pooled = selected.mean(axis=1)  # [N, D]

        # Compute distances to centroids
        dist_to_real = np.linalg.norm(pooled - self.real_center[np.newaxis, :], axis=1)
        dist_to_fake = np.linalg.norm(pooled - self.fake_center[np.newaxis, :], axis=1)

        # Classify based on nearest centroid
        predictions = (dist_to_fake < dist_to_real).astype(int)

        # Convert distances to probabilities
        total_dist = dist_to_real + dist_to_fake + 1e-8
        probs_fake = dist_to_real / total_dist

        return predictions, probs_fake


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Deepfake Detection')

    # Reference dataset
    parser.add_argument('--reference_dataset', type=str, required=True,
                       help='Reference dataset path for computing centroids')
    parser.add_argument('--reference_category', type=str, default='car',
                       help='Reference category')
    parser.add_argument('--reference_fake_type', type=str, default='1_fake_ldm',
                       help='Reference fake directory name')
    parser.add_argument('--max_reference', type=int, default=1000,
                       help='Max reference samples per class')

    # Test datasets
    parser.add_argument('--test_base_dir', type=str, required=True,
                       help='Test datasets base directory')
    parser.add_argument('--test_category', type=str, default='car',
                       help='Test category')
    parser.add_argument('--test_mode', type=str, default='so-fake-ood',
                       choices=['so-fake-ood', 'GenImage', 'AIGCDetectionBenchmark'],
                       help='Test mode - affects directory structure expectations')
    parser.add_argument('--max_test', type=int, default=500,
                       help='Max test samples per class')

    # Model
    parser.add_argument('--model', type=str, default='dinov3_vitl16',
                       help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Custom checkpoint path')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--use_hf', action='store_true',
                       help='Use HuggingFace model')
    parser.add_argument('--hf_model_path', type=str, default=None,
                       help='HuggingFace model path')

    # Token strategy
    parser.add_argument('--token_strategy', type=str, default='all',
                       choices=['all', 'cls', 'reg', 'patch', 'cls+reg', 'top_fisher', 'auto_fisher', 'custom_indices'],
                       help='Token selection strategy')
    parser.add_argument('--fisher_scores_path', type=str, default=None,
                       help='Fisher scores .npy file path (for top_fisher strategy)')
    parser.add_argument('--top_k', type=int, default=20,
                       help='Top-K tokens (for top_fisher/auto_fisher strategy)')
    parser.add_argument('--custom_token_indices', type=str, default=None,
                       help='Comma-separated list of token indices to use (for custom_indices strategy), e.g., "187,200,18,5"')

    # Output
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./results/zero_shot',
                       help='Output directory')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Zero-Shot Deepfake Detection")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Reference: {args.reference_dataset}/{args.reference_category}")
    print(f"  Token strategy: {args.token_strategy}")
    print(f"  Device: {device}")

    # Load model
    print(f"\n{'='*80}")
    print("Loading model")
    print("="*80)

    if args.use_hf:
        if args.hf_model_path is None:
            raise ValueError("--hf_model_path required when --use_hf is set")
        model, transform, model_type, config = load_hf_model(args.hf_model_path, device)
    else:
        model, transform, model_type = load_model(
            args.model,
            checkpoint=args.checkpoint,
            img_size=args.img_size,
            device=device
        )
        config = None

    # Load reference dataset
    print(f"\n{'='*80}")
    print("Loading reference dataset")
    print("="*80)

    ref_path = Path(args.reference_dataset)
    ref_loader = build_dataloader(
        real_dir=ref_path / args.reference_category / '0_real',
        fake_dir=ref_path / args.reference_category / args.reference_fake_type,
        transform=transform,
        batch_size=args.batch_size,
        max_samples=args.max_reference
    )

    if ref_loader is None:
        raise ValueError("Reference dataset is empty!")

    # Extract reference features
    print("\nExtracting reference features...")
    ref_features, ref_labels = extract_features(model, ref_loader, model_type, device)

    # Determine token selection strategy
    print(f"\n{'='*80}")
    print("Token selection")
    print("="*80)

    layout = get_token_layout(model, ref_features.shape[1], config)
    num_register_tokens = len(layout['reg_idx'])

    if args.token_strategy == 'all':
        token_indices = layout['cls_idx'] + layout['reg_idx'] + layout['patch_idx']
    elif args.token_strategy == 'cls':
        token_indices = layout['cls_idx']
    elif args.token_strategy == 'reg':
        if len(layout['reg_idx']) == 0:
            print("[Warning] No register tokens, using CLS instead")
            token_indices = layout['cls_idx']
        else:
            token_indices = layout['reg_idx']
    elif args.token_strategy == 'patch':
        token_indices = layout['patch_idx']
    elif args.token_strategy == 'cls+reg':
        token_indices = layout['cls_idx'] + layout['reg_idx']
    elif args.token_strategy == 'top_fisher':
        if args.fisher_scores_path is None:
            raise ValueError("--fisher_scores_path required for top_fisher strategy")

        fisher_scores = load_fisher_scores(args.fisher_scores_path)
        top_k = min(args.top_k, len(fisher_scores))
        top_indices = np.argsort(fisher_scores)[-top_k:][::-1]
        token_indices = top_indices.tolist()

        print(f"\nUsing Top-{top_k} Fisher tokens:")
        for i, idx in enumerate(top_indices[:10]):
            token_type = "CLS" if idx == 0 else ("REG" if idx in layout['reg_idx'] else "PATCH")
            print(f"  {i+1}. Token {idx:3d} ({token_type:5s}): Fisher={fisher_scores[idx]:.4f}")
    elif args.token_strategy == 'auto_fisher':
        # Auto Fisher mode: compute Fisher scores on reference data
        print("\n[Auto Fisher Mode] Computing Fisher scores on reference data...")

        fisher_scores = compute_fisher_scores(ref_features, ref_labels)
        save_fisher_scores(fisher_scores, output_dir)
        print(f"Fisher scores saved to: {output_dir / 'fisher_scores.npy'}")

        # Select top-k tokens (automatically filtering special tokens)
        top_indices = select_top_k_tokens(
            fisher_scores,
            top_k=args.top_k,
            num_register_tokens=num_register_tokens
        )
        token_indices = top_indices.tolist()

        print(f"\nUsing Top-{args.top_k} Fisher tokens (patch tokens only):")
        for i, idx in enumerate(top_indices[:10]):
            token_type = "CLS" if idx == 0 else ("REG" if idx in layout['reg_idx'] else "PATCH")
            print(f"  {i+1}. Token {idx:3d} ({token_type:5s}): Fisher={fisher_scores[idx]:.4f}")
    elif args.token_strategy == 'custom_indices':
        if args.custom_token_indices is None:
            raise ValueError("--custom_token_indices required for custom_indices strategy")

        token_indices = [int(idx.strip()) for idx in args.custom_token_indices.split(',')]

        print(f"Using {len(token_indices)} custom token indices: {token_indices}")

        max_tokens = ref_features.shape[1]
        if any(idx >= max_tokens or idx < 0 for idx in token_indices):
            raise ValueError(f"Token indices must be in range [0, {max_tokens-1}]")

        print("\nToken details:")
        for i, idx in enumerate(token_indices[:20]):  # Show first 20
            token_type = "CLS" if idx == 0 else ("REG" if idx in layout['reg_idx'] else "PATCH")
            print(f"  {i+1}. Token {idx:3d} ({token_type:5s})")
        if len(token_indices) > 20:
            print(f"  ... and {len(token_indices)-20} more tokens")
    else:
        raise ValueError(f"Unknown token strategy: {args.token_strategy}")

    print(f"\nUsing {len(token_indices)} tokens")

    # Create zero-shot classifier
    print(f"\n{'='*80}")
    print("Creating zero-shot classifier")
    print("="*80)

    classifier = ZeroShotClassifier(token_indices)
    classifier.fit(ref_features, ref_labels)

    # Evaluate on test datasets
    print(f"\n{'='*80}")
    print("Evaluating on test datasets")
    print("="*80)

    test_datasets = get_all_test_datasets(args.test_base_dir, test_mode=args.test_mode)
    all_results = []

    for ds_info in test_datasets:
        ds_name = ds_info['name']
        has_cat = ds_info['has_categories']

        # Skip if category not found
        if has_cat and args.test_category not in ds_info['categories']:
            print(f"\n[Skip] {ds_name} - no category '{args.test_category}'")
            continue

        print(f"\n[Evaluating] {ds_name}/{args.test_category if has_cat else 'N/A'}")

        # Build test loader using resolve_dataset_paths
        from utils import resolve_dataset_paths
        real_dir, fake_dir = resolve_dataset_paths(
            ds_info['path'],
            category=args.test_category if has_cat else None,
            test_mode=args.test_mode
        )

        test_loader = build_dataloader(
            real_dir=real_dir,
            fake_dir=fake_dir,
            transform=transform,
            batch_size=args.batch_size,
            max_samples=args.max_test
        )

        if test_loader is None:
            print("  [Skip] Empty dataset")
            continue

        # Extract features and predict
        test_features, test_labels = extract_features(model, test_loader, model_type, device)
        predictions, probs = classifier.predict(test_features)

        # Compute metrics
        metrics = compute_metrics(predictions, test_labels, probs)

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC:      {metrics['auc']:.4f}")
        print(f"  AP:       {metrics['ap']:.4f}")

        all_results.append({
            'dataset': ds_name,
            'category': args.test_category if has_cat else 'N/A',
            **metrics
        })

    # Save results
    print(f"\n{'='*80}")
    print("Saving results")
    print("="*80)

    save_json(all_results, output_dir / 'zero_shot_results.json')

    generate_report(
        all_results,
        config=vars(args),
        output_path=output_dir / 'report.txt',
        title=f"Zero-Shot Detection ({args.token_strategy})"
    )

    # Summary
    if all_results:
        accs = [r['accuracy'] for r in all_results]
        aucs = [r['auc'] for r in all_results]
        print(f"\nSummary:")
        print(f"  Evaluated {len(all_results)} datasets")
        print(f"  Average Accuracy: {np.mean(accs):.4f}")
        print(f"  Average AUC: {np.mean(aucs):.4f}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
