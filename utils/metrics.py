"""
Evaluation metrics computation and result saving
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


def compute_metrics(predictions: np.ndarray,
                   labels: np.ndarray,
                   probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute classification metrics

    Args:
        predictions: [N] - Predicted labels (0 or 1)
        labels: [N] - True labels (0 or 1)
        probs: [N] - Prediction probabilities (optional, for computing AUC and AP)

    Returns:
        Dictionary containing:
        - accuracy: Accuracy
        - auc: AUC-ROC (if probs provided)
        - ap: Average Precision (if probs provided)
    """
    metrics = {
        'accuracy': accuracy_score(labels, predictions)
    }

    if probs is not None:
        try:
            metrics['auc'] = roc_auc_score(labels, probs)
            metrics['ap'] = average_precision_score(labels, probs)
        except ValueError:
            # If only one class, skip AUC/AP computation
            metrics['auc'] = 0.0
            metrics['ap'] = 0.0

    return metrics


def evaluate_model(model, features: np.ndarray, labels: np.ndarray, device='cuda') -> Dict[str, float]:
    """
    Evaluate linear probe model

    Args:
        model: Linear classifier
        features: [N, D] or [N, T, D] - Features
                  If 3D, the model's forward() method should handle pooling
        labels: [N] - Labels
        device: Device

    Returns:
        Evaluation metrics dictionary
    """
    import torch

    # DO NOT pool here! Let the model's forward() method handle it
    # This ensures consistency between training and evaluation
    # The LinearProbe model has built-in pooling in its forward() method

    # Convert to tensors
    X = torch.FloatTensor(features).to(device)
    y = torch.LongTensor(labels).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(X)  # [N, 2] - model.forward() handles pooling if needed
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Get fake class probability
        preds = outputs.argmax(dim=1)

    # Compute metrics
    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()
    labels_np = labels

    return compute_metrics(preds_np, labels_np, probs_np)


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print evaluation metrics

    Args:
        metrics: Metrics dictionary
        prefix: Prefix string
    """
    if prefix:
        print(f"{prefix}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    if 'auc' in metrics:
        print(f"  AUC:      {metrics['auc']:.4f}")
    if 'ap' in metrics:
        print(f"  AP:       {metrics['ap']:.4f}")


def aggregate_results(results: List[Dict]) -> Dict[str, float]:
    """
    Aggregate mean and standard deviation of multiple results

    Args:
        results: List of result dictionaries

    Returns:
        Aggregated statistics dictionary
    """
    if not results:
        return {}

    # Extract all metric values
    metrics_values = {}
    for key in results[0].keys():
        if key in ['dataset', 'category', 'name']:
            continue
        values = [r[key] for r in results if key in r]
        if values:
            metrics_values[key] = values

    # Compute statistics
    stats = {}
    for metric_name, values in metrics_values.items():
        stats[f'{metric_name}_mean'] = np.mean(values)
        stats[f'{metric_name}_std'] = np.std(values)
        stats[f'{metric_name}_min'] = np.min(values)
        stats[f'{metric_name}_max'] = np.max(values)

    return stats


def format_result_table(results: List[Dict],
                       columns: List[str] = ['dataset', 'accuracy', 'auc', 'ap'],
                       include_stats: bool = True) -> str:
    """
    Format results as table string

    Args:
        results: List of result dictionaries
        columns: Columns to display
        include_stats: Whether to include statistics rows

    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display"

    # Build table header
    col_widths = {col: max(len(col), 12) for col in columns}

    # Update column widths to fit data
    for result in results:
        for col in columns:
            if col in result:
                value_str = str(result[col])
                if isinstance(result[col], float):
                    value_str = f"{result[col]:.4f}"
                col_widths[col] = max(col_widths[col], len(value_str))

    # Build separator
    separator = "-" * (sum(col_widths.values()) + len(columns) * 3 + 1)

    # Build table
    lines = []
    lines.append(separator)

    # Table header
    header = " | ".join([col.ljust(col_widths[col]) for col in columns])
    lines.append(header)
    lines.append(separator)

    # Data rows
    for result in results:
        row_values = []
        for col in columns:
            value = result.get(col, 'N/A')
            if isinstance(value, float):
                value_str = f"{value:.4f}".ljust(col_widths[col])
            else:
                value_str = str(value).ljust(col_widths[col])
            row_values.append(value_str)
        lines.append(" | ".join(row_values))

    # Statistics rows
    if include_stats and len(results) > 1:
        lines.append(separator)

        # Compute averages
        avg_row = ["Average".ljust(col_widths[columns[0]])]
        for col in columns[1:]:
            values = [r[col] for r in results if col in r and isinstance(r[col], (int, float))]
            if values:
                avg_value = np.mean(values)
                avg_row.append(f"{avg_value:.4f}".ljust(col_widths[col]))
            else:
                avg_row.append("N/A".ljust(col_widths[col]))
        lines.append(" | ".join(avg_row))

    lines.append(separator)

    return "\n".join(lines)
