"""
Input/output utilities

For saving and loading results, checkpoints, etc.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any


def save_json(data: Any, filepath: str):
    """
    Save data in JSON format

    Args:
        data: Data to save
        filepath: Save path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[Saved] {filepath}")


def load_json(filepath: str) -> Any:
    """
    Load JSON file

    Args:
        filepath: File path

    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"[Loaded] {filepath}")
    return data


def save_numpy(array: np.ndarray, filepath: str):
    """
    Save numpy array

    Args:
        array: Numpy array
        filepath: Save path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    np.save(filepath, array)
    print(f"[Saved] {filepath}")


def load_numpy(filepath: str) -> np.ndarray:
    """
    Load numpy array

    Args:
        filepath: File path

    Returns:
        Numpy array
    """
    array = np.load(filepath)
    print(f"[Loaded] {filepath} with shape {array.shape}")
    return array


def save_checkpoint(model, optimizer, epoch, metrics, filepath: str):
    """
    Save training checkpoint

    Args:
        model: Model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Evaluation metrics
        filepath: Save path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    torch.save(checkpoint, filepath)
    print(f"[Saved checkpoint] {filepath}")


def load_checkpoint(filepath: str, model, optimizer=None):
    """
    Load training checkpoint

    Args:
        filepath: Checkpoint path
        model: Model instance
        optimizer: Optimizer instance (optional)

    Returns:
        epoch, metrics
    """
    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    print(f"[Loaded checkpoint] {filepath} (epoch {epoch})")

    return epoch, metrics


def generate_report(results: List[Dict],
                   config: Dict,
                   output_path: str,
                   title: str = "Evaluation Report"):
    """
    Generate text report

    Args:
        results: Results list
        config: Experiment configuration
        output_path: Output path
        title: Report title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Title
        f.write("=" * 80 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 80 + "\n\n")

        # Configuration
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        # Results table
        f.write("=" * 80 + "\n")
        f.write(f"Results ({len(results)} datasets)\n")
        f.write("=" * 80 + "\n\n")

        # Table header
        f.write(f"{'Dataset':<25} {'Category':<15} {'Accuracy':<12} {'AUC':<12} {'AP':<12}\n")
        f.write("-" * 80 + "\n")

        # Data rows
        for r in results:
            dataset = r.get('dataset', 'N/A')
            category = r.get('category', 'N/A')
            accuracy = r.get('accuracy', 0)
            auc = r.get('auc', 0)
            ap = r.get('ap', 0)

            f.write(f"{dataset:<25} {category:<15} {accuracy:<12.4f} {auc:<12.4f} {ap:<12.4f}\n")

        # Statistics
        if len(results) > 1:
            accs = [r['accuracy'] for r in results if 'accuracy' in r]
            aucs = [r['auc'] for r in results if 'auc' in r]
            aps = [r['ap'] for r in results if 'ap' in r]

            f.write("-" * 80 + "\n")
            f.write(f"{'Average':<25} {'':<15} {np.mean(accs):<12.4f} "
                   f"{np.mean(aucs):<12.4f} {np.mean(aps):<12.4f}\n")
            f.write(f"{'Std Dev':<25} {'':<15} {np.std(accs):<12.4f} "
                   f"{np.std(aucs):<12.4f} {np.std(aps):<12.4f}\n")

            # Best/Worst
            best_idx = int(np.argmax(accs))
            worst_idx = int(np.argmin(accs))

            f.write("\n")
            f.write(f"Best:  {results[best_idx]['dataset']} (Acc={accs[best_idx]:.4f})\n")
            f.write(f"Worst: {results[worst_idx]['dataset']} (Acc={accs[worst_idx]:.4f})\n")

    print(f"[Report saved] {output_path}")


def save_fisher_scores(fisher_scores: np.ndarray, output_dir: str, prefix: str = ""):
    """
    Save Fisher discrimination scores

    Args:
        fisher_scores: Fisher score array
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{prefix}fisher_scores.npy" if prefix else "fisher_scores.npy"
    save_numpy(fisher_scores, output_dir / filename)


def load_fisher_scores(filepath: str) -> np.ndarray:
    """
    Load Fisher discrimination scores

    Args:
        filepath: Fisher scores file path

    Returns:
        Fisher score array
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Fisher scores not found: {filepath}")

    return load_numpy(filepath)
