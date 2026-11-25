"""Comprehensive evaluation metrics for recommendation quality, ranking, novelty, and diversity."""

import math
from typing import Any, Dict, List, Set, Union
import numpy as np


def precision_at_k(actual_clicks: Set[Union[str, int]], ranked_items: List[Union[str, int]], k: int = 5) -> float:
    """Compute Precision@K: proportion of recommended top-K items that were clicked."""
    if not ranked_items or k <= 0 or not actual_clicks:
        return 0.0
    top_k = ranked_items[:k]
    hits = sum(1 for item in top_k if item in actual_clicks)
    return hits / float(k)


def recall_at_k(actual_clicks: Set[Union[str, int]], ranked_items: List[Union[str, int]], k: int = 5) -> float:
    """Compute Recall@K: proportion of actual clicked items retrieved in top-K."""
    if not actual_clicks or not ranked_items or k <= 0:
        return 0.0
    top_k = ranked_items[:k]
    hits = sum(1 for item in top_k if item in actual_clicks)
    return hits / float(len(actual_clicks))


def mrr(actual_clicks: Set[Union[str, int]], ranked_items: List[Union[str, int]]) -> float:
    """Compute Mean Reciprocal Rank (MRR): reciprocal rank of the first clicked item."""
    if not actual_clicks or not ranked_items:
        return 0.0
    for rank, item in enumerate(ranked_items, start=1):
        if item in actual_clicks:
            return 1.0 / float(rank)
    return 0.0


def dcg_at_k(actual_clicks: Set[Union[str, int]], ranked_items: List[Union[str, int]], k: int = 5) -> float:
    """Compute Discounted Cumulative Gain at K with binary relevance."""
    if not actual_clicks or not ranked_items or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        rel = 1.0 if item in actual_clicks else 0.0
        dcg += rel / math.log2(rank + 1.0)
    return dcg


def ndcg_at_k(actual_clicks: Set[Union[str, int]], ranked_items: List[Union[str, int]], k: int = 5) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K)."""
    if not actual_clicks or not ranked_items or k <= 0:
        return 0.0
    dcg = dcg_at_k(actual_clicks, ranked_items, k)
    # Ideal DCG: all hits ranked at top
    num_hits = min(k, len(actual_clicks))
    if num_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, num_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def auc_score(labels: List[int], scores: List[float]) -> float:
    """Compute Area Under ROC Curve (AUC) using rank-sum formula."""
    if not labels or len(set(labels)) < 2:
        return 0.5  # Degenerate case when all positive or all negative

    pos_indices = [i for i, y in enumerate(labels) if y == 1]
    neg_indices = [i for i, y in enumerate(labels) if y == 0]
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Pairwise comparison count
    wins = 0.0
    for pos in pos_indices:
        for neg in neg_indices:
            if scores[pos] > scores[neg]:
                wins += 1.0
            elif scores[pos] == scores[neg]:
                wins += 0.5
    return wins / (n_pos * n_neg)


def intra_list_diversity(embeddings: np.ndarray) -> float:
    """Compute Intra-List Diversity (ILD): mean pairwise cosine distance 1 - cos(vi, vj)."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms

    sim_matrix = np.dot(normed, normed.T)
    # Extract strictly upper triangle
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]
    pairwise_dist = 1.0 - pairwise_sims
    return float(np.mean(pairwise_dist))


def novelty_at_k(novelty_scores: List[float], k: int = 5) -> float:
    """Compute mean novelty score among top-K recommendations."""
    if not novelty_scores or k <= 0:
        return 0.0
    return float(np.mean(novelty_scores[:k]))


def redundancy_at_k(redundancy_scores: List[float], k: int = 5) -> float:
    """Compute mean redundancy penalty among top-K recommendations."""
    if not redundancy_scores or k <= 0:
        return 0.0
    return float(np.mean(redundancy_scores[:k]))
