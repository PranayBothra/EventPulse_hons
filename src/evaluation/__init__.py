"""Evaluation and validation modules for EventPulse."""

from src.evaluation.metrics import (
    ClusteringEvaluationResult,
    compute_clustering_metrics,
    compute_pairwise_metrics,
)
from src.evaluation.pair_generator import generate_validation_pairs_csv
from src.evaluation.sensitivity import run_threshold_sensitivity

__all__ = [
    "compute_pairwise_metrics",
    "compute_clustering_metrics",
    "ClusteringEvaluationResult",
    "generate_validation_pairs_csv",
    "run_threshold_sensitivity",
]
