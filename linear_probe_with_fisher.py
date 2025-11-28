"""
Linear Probe Training with Fisher-Guided Token Selection

Train a linear classifier on top of frozen foundation model features,
using Fisher discriminability scores to select the most informative tokens.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

from utils import (
    build_dataloader,
    get_all_test_datasets,
    load_model,
    load_hf_model,
    get_token_layout,
    extract_features,
    select_tokens,
    compute_fisher_scores,
    select_top_k_tokens,
    evaluate_model,
    save_json,
    save_numpy,
    generate_report,
    save_fisher_scores,
    print_metrics
)


class LinearProbe(nn.Module):
    """Simple linear classifier"""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Pool if needed
        if x.dim() == 3:  # [B, T, D]
            x = x.mean(dim=1)  # [B, D]
        return self.fc(x)


def train_linear_probe(train_features, train_labels, val_features, val_labels,
                      num_epochs=50, lr=0.01, device='cuda'):
    """
    Train linear probe on selected tokens

    Args:
        train_features: [N, num_selected_tokens, D] or [N, D]
        train_labels: [N]
        val_features: [N, num_selected_tokens, D] or [N, D]
        val_labels: [N]
        num_epochs: number of training epochs
        lr: learning rate
        device: device

    Returns:
        Trained model
    """
    # Pool features if needed
    if train_features.ndim == 3:
        train_features = train_features.mean(axis=1)
        val_features = val_features.mean(axis=1)

    # Convert to tensors
    X_train = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(train_labels).to(device)
    X_val = torch.FloatTensor(val_features).to(device)
    y_val = torch.LongTensor(val_labels).to(device)

    # Create model
    model = LinearProbe(train_features.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_acc = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # clone tensor values to avoid being overwritten in later epochs
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: val_acc={val_acc:.4f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  Best validation accuracy: {best_val_acc:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Linear Probe with Fisher Token Selection')

    # Data
    parser.add_argument('--train_dataset', type=str, required=True,
                       help='Training dataset path')
    parser.add_argument('--train_category', type=str, default='car',
                       help='Training category')
    parser.add_argument('--train_fake_type', type=str, default='1_fake_ldm',
                       help='Fake image directory name')
    parser.add_argument('--val_dataset', type=str, default=None,
                       help='Validation dataset path (if None, will split from training data)')
    parser.add_argument('--val_category', type=str, default=None,
                       help='Validation category (if None, use train_category)')
    parser.add_argument('--val_fake_type', type=str, default='1_fake',
                       help='Validation fake image directory name')
    parser.add_argument('--no_validation', action='store_true',
                       help='Skip validation, use all training data for training')
    parser.add_argument('--test_base_dir', type=str, required=True,
                       help='Test datasets base directory')
    parser.add_argument('--test_category', type=str, default='car',
                       help='Test category')
    parser.add_argument('--test_mode', type=str, default='so-fake-ood',
                       choices=['so-fake-ood', 'GenImage', 'AIGCDetectionBenchmark'],
                       help='Test mode - affects directory structure expectations')
    parser.add_argument('--max_train_samples', type=int, default=1000,
                       help='Max training samples per class')
    parser.add_argument('--max_val_samples', type=int, default=500,
                       help='Max validation samples per class (only used with --val_dataset)')
    parser.add_argument('--max_test_samples', type=int, default=500,
                       help='Max test samples per class')

    # Model
    parser.add_argument('--model', type=str, default='dinov3_vitl16',
                       help='Model name (dinov3_vitl16, dinov3_vitb16, etc.)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Custom checkpoint path')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--use_hf', action='store_true',
                       help='Use HuggingFace model')
    parser.add_argument('--hf_model_path', type=str, default=None,
                       help='HuggingFace model path (if use_hf=True)')

    # Token Selection Strategy
    parser.add_argument('--token_strategy', type=str, default='auto_fisher',
                       choices=['all', 'cls', 'reg', 'patch', 'cls+reg', 'top_fisher', 'auto_fisher', 'custom_indices'],
                       help='Token selection strategy')
    parser.add_argument('--top_k', type=int, default=20,
                       help='Number of top Fisher tokens to select (for top_fisher/auto_fisher)')
    parser.add_argument('--fisher_scores_path', type=str, default=None,
                       help='Path to pre-computed Fisher scores (for top_fisher strategy)')
    parser.add_argument('--custom_token_indices', type=str, default=None,
                       help='Comma-separated list of token indices to use (for custom_indices strategy), e.g., "187,200,18,5"')
    parser.add_argument('--filter_special_tokens', action='store_true',
                       help='[DEPRECATED] Use --token_strategy auto_fisher instead')

    # Training
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results/linear_probe',
                       help='Output directory')
    parser.add_argument('--probe_checkpoint', type=str, default=None,
                       help='Path to pre-trained linear probe (skip training, eval only)')

    args = parser.parse_args()
    eval_only = args.probe_checkpoint is not None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle deprecated flag
    if args.filter_special_tokens:
        print("[Warning] --filter_special_tokens is deprecated. Using --token_strategy auto_fisher instead.")
        args.token_strategy = 'auto_fisher'

    print("="*80)
    print("Linear Probe Training with Fisher Token Selection")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Training dataset: {args.train_dataset}")
    print(f"  Category: {args.train_category}")
    print(f"  Token strategy: {args.token_strategy}")
    if args.token_strategy in ['top_fisher', 'auto_fisher']:
        print(f"  Top-K tokens: {args.top_k}")
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

    if eval_only:
        # Load pre-trained probe and token indices
        ckpt = torch.load(args.probe_checkpoint, map_location=device)
        if 'probe_state_dict' in ckpt:
            probe_state = ckpt['probe_state_dict']
        elif 'model_state_dict' in ckpt:
            probe_state = ckpt['model_state_dict']
        else:
            raise ValueError("Invalid probe checkpoint: missing probe_state_dict/model_state_dict")

        weight = probe_state.get('fc.weight', None)
        if weight is None:
            raise ValueError("Invalid probe checkpoint: missing fc.weight")
        in_dim = weight.shape[1]
        probe_model = LinearProbe(in_dim).to(device)
        probe_model.load_state_dict(probe_state)

        token_indices = None
        if 'token_indices' in ckpt:
            token_indices = ckpt['token_indices']
        elif 'config' in ckpt and ckpt['config'].get('custom_tokens'):
            token_indices = ckpt['config']['custom_tokens']
        if token_indices is None and args.custom_token_indices:
            token_indices = [int(idx.strip()) for idx in args.custom_token_indices.split(',')]
        if token_indices is None:
            raise ValueError("Probe checkpoint missing token indices; specify --custom_token_indices")
        token_indices = [int(t) for t in token_indices]

        print(f"\nLoaded probe checkpoint: {args.probe_checkpoint}")
        print(f"  in_dim={in_dim}, tokens={len(token_indices)}")
    else:
        # Load training data
        print(f"\n{'='*80}")
        print("Loading training data")
        print("="*80)

        train_path = Path(args.train_dataset)
        train_loader = build_dataloader(
            real_dir=train_path / args.train_category / '0_real',
            fake_dir=train_path / args.train_category / args.train_fake_type,
            transform=transform,
            batch_size=args.batch_size,
            max_samples=args.max_train_samples,
            shuffle=False
        )

        if train_loader is None:
            raise ValueError("Training dataset is empty!")

        # Extract training features
        print("\nExtracting training features...")
        train_features, train_labels = extract_features(model, train_loader, model_type, device)

        # Load validation data (independent or split from training)
        if args.no_validation:
            print("\nNo validation mode: using all training data for training")
            train_feats = train_features
            train_labs = train_labels
            val_feats = train_features
            val_labs = train_labels
            print(f"Train: {len(train_labs)} samples, Val: {len(val_labs)} samples (same as training)")
        elif args.val_dataset is not None:
            print(f"\n{'='*80}")
            print("Loading independent validation data")
            print("="*80)

            val_path = Path(args.val_dataset)
            val_category = args.val_category if args.val_category else args.train_category

            val_loader = build_dataloader(
                real_dir=val_path / val_category / '0_real',
                fake_dir=val_path / val_category / args.val_fake_type,
                transform=transform,
                batch_size=args.batch_size,
                max_samples=args.max_val_samples,
                shuffle=False
            )

            if val_loader is None:
                raise ValueError("Validation dataset is empty!")

            print("\nExtracting validation features...")
            val_features, val_labels = extract_features(model, val_loader, model_type, device)

            train_feats = train_features
            train_labs = train_labels
            val_feats = val_features
            val_labs = val_labels

            print(f"Train: {len(train_labs)} samples, Val: {len(val_labs)} samples (independent)")
        else:
            print("\nSplitting training data into train/val (80/20)...")
            n_train = int(0.8 * len(train_labels))
            train_feats = train_features[:n_train]
            train_labs = train_labels[:n_train]
            val_feats = train_features[n_train:]
            val_labs = train_labels[n_train:]

            print(f"Train: {len(train_labs)} samples, Val: {len(val_labs)} samples (split from training)")

        layout = get_token_layout(model, train_feats.shape[1], config)
        num_register_tokens = len(layout['reg_idx'])

    if not eval_only:
        print(f"\n{'='*80}")
        print(f"Token Selection Strategy: {args.token_strategy}")
        print("="*80)

        if args.token_strategy == 'all':
            token_indices = layout['cls_idx'] + layout['reg_idx'] + layout['patch_idx']
            print(f"Using all {len(token_indices)} tokens (CLS + Register + Patch)")
        elif args.token_strategy == 'cls':
            token_indices = layout['cls_idx']
            print(f"Using {len(token_indices)} CLS token(s)")
        elif args.token_strategy == 'reg':
            if len(layout['reg_idx']) == 0:
                print("[Warning] No register tokens, using CLS instead")
                token_indices = layout['cls_idx']
            else:
                token_indices = layout['reg_idx']
            print(f"Using {len(token_indices)} register token(s)")
        elif args.token_strategy == 'patch':
            token_indices = layout['patch_idx']
            print(f"Using {len(token_indices)} patch tokens")
        elif args.token_strategy == 'cls+reg':
            token_indices = layout['cls_idx'] + layout['reg_idx']
            print(f"Using {len(token_indices)} tokens (CLS + Register)")
        elif args.token_strategy == 'top_fisher':
            if args.fisher_scores_path is None:
                raise ValueError("--fisher_scores_path required for top_fisher strategy")

            print(f"Loading Fisher scores from: {args.fisher_scores_path}")
            fisher_scores = np.load(args.fisher_scores_path)

            print(f"Selecting top-{args.top_k} patch tokens based on Fisher scores...")
            token_indices = select_top_k_tokens(
                fisher_scores,
                top_k=args.top_k,
                num_register_tokens=num_register_tokens
            )

            print(f"\nTop-{args.top_k} patch token indices: {token_indices.tolist()[:10]}...")
            for i, idx in enumerate(token_indices[:10]):
                token_type = "CLS" if idx == 0 else ("REG" if idx in layout['reg_idx'] else "PATCH")
                print(f"  {i+1}. Token {idx:3d} ({token_type:5s}): Fisher={fisher_scores[idx]:.4f}")
        elif args.token_strategy == 'auto_fisher':
            print("Computing Fisher scores on training data...")
            fisher_scores = compute_fisher_scores(train_feats, train_labs)
            save_fisher_scores(fisher_scores, output_dir)

            print(f"Selecting top-{args.top_k} patch tokens based on Fisher scores...")
            token_indices = select_top_k_tokens(
                fisher_scores,
                top_k=args.top_k,
                num_register_tokens=num_register_tokens
            )

            print(f"\nTop-{args.top_k} patch token indices: {token_indices.tolist()[:10]}...")
            for i, idx in enumerate(token_indices[:10]):
                token_type = "CLS" if idx == 0 else ("REG" if idx in layout['reg_idx'] else "PATCH")
                print(f"  {i+1}. Token {idx:3d} ({token_type:5s}): Fisher={fisher_scores[idx]:.4f}")
        elif args.token_strategy == 'custom_indices':
            if args.custom_token_indices is None:
                raise ValueError("--custom_token_indices required for custom_indices strategy")

            token_indices = [int(idx.strip()) for idx in args.custom_token_indices.split(',')]
            token_indices = np.array(token_indices)

            print(f"Using {len(token_indices)} custom token indices: {token_indices.tolist()}")

            max_tokens = train_feats.shape[1]
            if np.any(token_indices >= max_tokens) or np.any(token_indices < 0):
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

        # Select features
        train_selected = select_tokens(train_feats, token_indices)
        val_selected = select_tokens(val_feats, token_indices)

        # Train linear probe
        print(f"\n{'='*80}")
        print(f"Training linear probe ({len(token_indices)} tokens)")
        print("="*80)

        probe_model = train_linear_probe(
            train_selected, train_labs,
            val_selected, val_labs,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=device
        )

        # Save checkpoint
        checkpoint_path = output_dir / 'linear_probe.pth'
        checkpoint_data = {
            'model_state_dict': probe_model.state_dict(),
            'token_indices': token_indices if isinstance(token_indices, list) else token_indices.tolist() if hasattr(token_indices, 'tolist') else list(token_indices),
            'config': vars(args)
        }
        if args.token_strategy in ['auto_fisher', 'top_fisher']:
            checkpoint_data['fisher_scores'] = fisher_scores.tolist()
        torch.save(checkpoint_data, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")

    # Evaluate on test sets
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
            max_samples=args.max_test_samples
        )

        if test_loader is None:
            print("  [Skip] Empty dataset")
            continue

        # Extract and select features
        test_features, test_labels = extract_features(model, test_loader, model_type, device)
        test_selected = select_tokens(test_features, token_indices)

        # Evaluate
        metrics = evaluate_model(probe_model, test_selected, test_labels, device)

        print_metrics(metrics, prefix="  Results")

        all_results.append({
            'dataset': ds_name,
            'category': args.test_category if has_cat else 'N/A',
            **metrics
        })

    # Save results
    print(f"\n{'='*80}")
    print("Saving results")
    print("="*80)

    save_json(all_results, output_dir / 'test_results.json')

    generate_report(
        all_results,
        config=vars(args),
        output_path=output_dir / 'report.txt',
        title=f"Linear Probe Evaluation ({args.token_strategy})"
    )

    # Summary
    if all_results:
        accs = [r['accuracy'] for r in all_results]
        print(f"\nSummary:")
        print(f"  Evaluated {len(all_results)} datasets")
        print(f"  Average Accuracy: {np.mean(accs):.4f}")
        print(f"  Std Dev: {np.std(accs):.4f}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
