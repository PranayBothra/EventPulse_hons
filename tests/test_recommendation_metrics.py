"""Unit tests for recommendation metrics: Precision, Recall, NDCG, MRR, AUC, ILD."""

import math
import pytest
import numpy as np

from src.evaluation.recommendation_metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mrr,
    auc_score,
    intra_list_diversity,
    novelty_at_k,
    redundancy_at_k,
)


class TestRecommendationMetrics:
    """Test mathematical accuracy of IR and recommendation metrics."""

    def test_precision_and_recall_at_k(self):
        clicks = {"item_1", "item_3"}
        ranked = ["item_1", "item_2", "item_3", "item_4", "item_5"]

        # K = 1: item_1 is a hit -> P@1 = 1/1 = 1.0, R@1 = 1/2 = 0.5
        assert precision_at_k(clicks, ranked, k=1) == 1.0
        assert recall_at_k(clicks, ranked, k=1) == 0.5

        # K = 2: 1 hit out of 2 -> P@2 = 0.5, R@2 = 0.5
        assert precision_at_k(clicks, ranked, k=2) == 0.5
        assert recall_at_k(clicks, ranked, k=2) == 0.5

        # K = 3: 2 hits out of 3 -> P@3 = 2/3, R@3 = 2/2 = 1.0
        assert round(precision_at_k(clicks, ranked, k=3), 4) == round(2 / 3, 4)
        assert recall_at_k(clicks, ranked, k=3) == 1.0

    def test_ndcg_at_k_perfect_vs_reversed(self):
        clicks = {"item_1", "item_2"}
        perfect_ranked = ["item_1", "item_2", "item_3", "item_4"]
        poor_ranked = ["item_3", "item_4", "item_2", "item_1"]

        ndcg_perfect = ndcg_at_k(clicks, perfect_ranked, k=4)
        ndcg_poor = ndcg_at_k(clicks, poor_ranked, k=4)

        assert round(ndcg_perfect, 4) == 1.0
        assert ndcg_poor < ndcg_perfect
        assert ndcg_poor > 0.0

    def test_mrr_first_hit(self):
        clicks = {"item_target"}
        assert mrr(clicks, ["item_target", "other"]) == 1.0
        assert mrr(clicks, ["other", "item_target"]) == 0.5
        assert mrr(clicks, ["other1", "other2", "item_target"]) == 1.0 / 3.0
        assert mrr(clicks, ["other1", "other2"]) == 0.0

    def test_auc_score(self):
        labels = [1, 0, 1, 0]
        # Perfect ranking: positive items have strictly higher scores
        perfect_scores = [0.9, 0.2, 0.8, 0.1]
        assert auc_score(labels, perfect_scores) == 1.0

        # Completely inverted ranking
        inverted_scores = [0.1, 0.8, 0.2, 0.9]
        assert auc_score(labels, inverted_scores) == 0.0

        # Random / tie ranking
        tied_scores = [0.5, 0.5, 0.5, 0.5]
        assert auc_score(labels, tied_scores) == 0.5

    def test_intra_list_diversity(self):
        # 3 mutually orthogonal unit vectors: pairwise distance is 1.0
        orthogonal_vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        ild_orth = intra_list_diversity(orthogonal_vecs)
        assert round(ild_orth, 4) == 1.0

        # 3 identical collinear vectors: pairwise distance is 0.0
        collinear_vecs = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        ild_collinear = intra_list_diversity(collinear_vecs)
        assert round(ild_collinear, 4) == 0.0
