"""Evaluation metrics for clustering: ARI, NMI, Pairwise Precision, Recall, and F1."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class ClusteringEvaluationResult:
    """Encapsulates clustering evaluation metrics."""

    status: str  # 'COMPLETED' or 'PENDING_ANNOTATIONS'
    total_pairs: int = 0
    labeled_pairs: int = 0
    pairwise_precision: Optional[float] = None
    pairwise_recall: Optional[float] = None
    pairwise_f1: Optional[float] = None
    adjusted_rand_index: Optional[float] = None
    normalized_mutual_info: Optional[float] = None
    cluster_purity: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


def compute_pairwise_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """Compute pairwise Precision, Recall, and F1 over binary pair comparisons.

    Args:
        y_true: Ground truth binary flags (1 for SAME_EVENT, 0 for DIFFERENT_EVENT).
        y_pred: Predicted binary flags (1 for assigned to same cluster, 0 otherwise).

    Returns:
        Dict containing 'precision', 'recall', and 'f1'.
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_cluster_purity(labels_true: List[int], labels_pred: List[int]) -> float:
    """Compute cluster purity metric."""
    if not labels_true or len(labels_true) != len(labels_pred):
        return 0.0

    from collections import Counter
    clusters = {}
    for t, p in zip(labels_true, labels_pred):
        clusters.setdefault(p, []).append(t)

    total_correct = 0
    for cluster_labels in clusters.values():
        most_common_count = Counter(cluster_labels).most_common(1)[0][1]
        total_correct += most_common_count

    return total_correct / len(labels_true)


def compute_clustering_metrics(
    labels_true: Optional[List[int]], labels_pred: List[int]
) -> ClusteringEvaluationResult:
    """Compute comprehensive clustering metrics if true reference labels are present."""
    if labels_true is None or len(labels_true) == 0:
        return ClusteringEvaluationResult(
            status="PENDING_ANNOTATIONS",
            total_pairs=len(labels_pred),
            labeled_pairs=0,
            details={"message": "Reference labels not provided or pending human annotation."},
        )

    ari = float(adjusted_rand_score(labels_true, labels_pred))
    nmi = float(normalized_mutual_info_score(labels_true, labels_pred))
    purity = compute_cluster_purity(labels_true, labels_pred)

    return ClusteringEvaluationResult(
        status="COMPLETED",
        total_pairs=len(labels_true),
        labeled_pairs=len(labels_true),
        adjusted_rand_index=ari,
        normalized_mutual_info=nmi,
        cluster_purity=purity,
    )
