"""
Evaluation Metrics for Hallucination Detection.

Computes precision, recall, F1, AUROC, AUPRC with bootstrap confidence
intervals. Also handles per-category and per-task-type breakdowns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from loguru import logger


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: Binary ground truth labels.
        y_probs: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Dict with precision, recall, f1, auroc, auprc.
    """
    y_pred = (y_probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(y_true, y_probs)),
        "auprc": float(average_precision_score(y_true, y_probs)),
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }

    return metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for all metrics.

    Args:
        y_true: True labels.
        y_probs: Predicted probabilities.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        random_seed: Random seed.

    Returns:
        Dict mapping metric_name -> {"mean", "lower", "upper", "std"}.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)

    bootstrap_metrics = {
        "auroc": [], "auprc": [], "f1": [], "precision": [], "recall": []
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_true = y_true[idx]
        boot_probs = y_probs[idx]

        # Skip if only one class in bootstrap sample
        if len(np.unique(boot_true)) < 2:
            continue

        boot_pred = (boot_probs >= 0.5).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            boot_true, boot_pred, average="binary", zero_division=0
        )

        bootstrap_metrics["auroc"].append(roc_auc_score(boot_true, boot_probs))
        bootstrap_metrics["auprc"].append(average_precision_score(boot_true, boot_probs))
        bootstrap_metrics["f1"].append(f)
        bootstrap_metrics["precision"].append(p)
        bootstrap_metrics["recall"].append(r)

    # Compute CIs
    alpha = (1 - confidence_level) / 2
    results = {}
    for metric_name, values in bootstrap_metrics.items():
        values = np.array(values)
        results[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "lower": float(np.percentile(values, 100 * alpha)),
            "upper": float(np.percentile(values, 100 * (1 - alpha))),
        }

    return results


def compute_cost_comparison(
    method_results: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Compare methods by both accuracy and computational cost.

    Args:
        method_results: Dict mapping method_name -> {metrics + cost info}.

    Returns:
        Formatted comparison dict.
    """
    comparison = {}
    for method, metrics in method_results.items():
        comparison[method] = {
            "auroc": metrics.get("auroc", 0),
            "f1": metrics.get("f1", 0),
            "forward_passes": metrics.get("forward_passes", 1),
            "wall_time_s": metrics.get("wall_time_s", 0),
            "gpu_memory_mb": metrics.get("gpu_memory_mb", 0),
        }

    return comparison


def format_results_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
) -> str:
    """Format results as a markdown table.

    Args:
        results: Dict mapping method_name -> metric_dict.
        metrics: Which metrics to include.

    Returns:
        Markdown-formatted table string.
    """
    if metrics is None:
        metrics = ["auroc", "auprc", "f1", "precision", "recall"]

    # Header
    header = "| Method | " + " | ".join(m.upper() for m in metrics) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"

    rows = [header, separator]
    for method, metric_dict in results.items():
        values = [f"{metric_dict.get(m, 0):.4f}" for m in metrics]
        row = f"| {method} | " + " | ".join(values) + " |"
        rows.append(row)

    return "\n".join(rows)
