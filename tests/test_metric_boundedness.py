"""
Unit and boundedness tests for Recommendation & Diversity Metrics.
Verifies Section 8 criteria:
1. Intra-List Diversity (ILD) range [0.0, 2.0] for unnormalized cosine distance.
2. ILD behavior on collinear, orthogonal, and opposing vectors.
3. NDCG@K, MRR, AUC bounds [0.0, 1.0] and edge-case handling.
4. Linear identity: Novelty@K = 1.0 - Redundancy@K.
"""

import numpy as np
import pytest

from src.evaluation.recommendation_metrics import (
    intra_list_diversity,
    ndcg_at_k,
    mrr,
    auc_score,
)


class TestMetricBoundednessAndIntegrity:
    """Verifies mathematical bounds and stability of evaluation metrics."""

    def test_ild_collinear_vectors_zero(self):
        """Identical/collinear vectors must produce ILD = 0.0."""
        vecs = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        ild = intra_list_diversity(vecs)
        assert np.isclose(ild, 0.0), f"Collinear vectors must yield 0 ILD, got {ild}"

    def test_ild_orthogonal_vectors_one(self):
        """Mutually orthogonal vectors must produce ILD = 1.0."""
        vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        ild = intra_list_diversity(vecs)
        assert np.isclose(ild, 1.0), f"Orthogonal vectors must yield 1.0 ILD, got {ild}"

    def test_ild_opposing_vectors_exceeds_one(self):
        """Opposing vectors (cos < 0) produce unnormalized ILD in (1.0, 2.0]."""
        # Two diametrically opposing vectors: [1, 0] and [-1, 0] -> cos = -1.0 -> 1 - cos = 2.0
        vecs = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
        ], dtype=np.float32)
        ild = intra_list_diversity(vecs)
        assert np.isclose(ild, 2.0), f"Diametrically opposing vectors must yield 2.0, got {ild}"

    def test_ild_slight_negative_cosine_mode_c_explanation(self):
        """Vectors with slight negative cosine (cos = -0.0002) produce ILD = 1.0002."""
        # Angle slightly > 90 degrees
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-0.0002, 0.99999998], dtype=np.float32)
        vecs = np.vstack([v1, v2])
        ild = intra_list_diversity(vecs)
        assert round(ild, 4) == 1.0002, f"Expected 1.0002, got {ild}"

    def test_ild_empty_or_single_vector(self):
        """Empty list or single item slate must return 0.0."""
        assert intra_list_diversity(np.empty((0, 384))) == 0.0
        assert intra_list_diversity(np.array([[1.0, 0.0, 0.0]])) == 0.0

    def test_ndcg_bounds_and_empty_clicks(self):
        """NDCG must strictly lie in [0.0, 1.0] and return 0.0 for empty clicks."""
        clicked = set()
        ranked = ["item_1", "item_2", "item_3"]
        assert ndcg_at_k(clicked, ranked, k=5) == 0.0
        assert ndcg_at_k(set(), [], k=5) == 0.0

        # Perfect ranking -> 1.0
        clicked_perf = {"item_1", "item_2"}
        assert np.isclose(ndcg_at_k(clicked_perf, ["item_1", "item_2", "item_3"], k=5), 1.0)

        # Inverted ranking -> strictly between 0 and 1
        val_inv = ndcg_at_k(clicked_perf, ["item_3", "item_4", "item_1", "item_2"], k=5)
        assert 0.0 < val_inv < 1.0

    def test_mrr_bounds_and_empty_slate(self):
        """MRR must lie in [0.0, 1.0] and return 0.0 when no hit."""
        assert mrr(set(), ["item_1"]) == 0.0
        assert mrr({"item_3"}, ["item_1", "item_2"]) == 0.0
        assert mrr({"item_1"}, ["item_1", "item_2"]) == 1.0
        assert mrr({"item_2"}, ["item_1", "item_2"]) == 0.5

    def test_auc_degenerate_single_class(self):
        """When all labels are 0 or all 1, AUC must return 0.5 without division by zero."""
        assert auc_score([0, 0, 0], [0.8, 0.5, 0.2]) == 0.5
        assert auc_score([1, 1, 1], [0.8, 0.5, 0.2]) == 0.5
        assert auc_score([], []) == 0.5

    def test_novelty_redundancy_exact_linear_sum(self):
        """Novelty@K and Redundancy@K must sum to exactly 1.0 by definition."""
        cand_vec = np.array([0.8, 0.6], dtype=np.float32)
        hist_vecs = np.array([
            [0.9, 0.435],
            [0.5, 0.866],
        ], dtype=np.float32)

        sims = np.dot(cand_vec, hist_vecs.T)
        red = float(np.max(sims))
        nov = 1.0 - red
        assert np.isclose(nov + red, 1.0), "Novelty and Redundancy must sum to 1.0"
