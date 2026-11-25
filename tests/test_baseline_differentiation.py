"""
Unit and regression tests for Baseline B vs Baseline C differentiation.
Ensures Baseline B (pure semantic content-based) and Baseline C (category/event proxy aware)
maintain strictly distinct scoring formulas, ranking behaviors, and test invariants.
"""

import numpy as np
import pytest

def compute_baseline_b_score(cand_vecs: np.ndarray, user_profile: np.ndarray) -> np.ndarray:
    """Pure classical content-based: semantic similarity only."""
    norm = np.linalg.norm(user_profile) or 1.0
    return np.dot(cand_vecs, user_profile / norm)

def compute_baseline_c_score(
    cand_vecs: np.ndarray,
    user_profile: np.ndarray,
    cand_categories: list,
    user_categories: set,
    w_long: float = 0.60,
    w_cat: float = 0.40,
) -> np.ndarray:
    """Event/Category-aware baseline: semantic similarity + category match proxy."""
    long_sim = compute_baseline_b_score(cand_vecs, user_profile)
    cat_match = np.array([1.0 if c in user_categories else 0.5 for c in cand_categories])
    return w_long * long_sim + w_cat * cat_match


class TestBaselineDifferentiation:
    """Test suite ensuring Baseline B and Baseline C never collapse to identity."""

    def test_baselines_diverge_on_category_discrepancy(self):
        """When candidate articles have identical semantic similarity but different categories,
        Baseline B must treat them equally, while Baseline C must distinguish them."""
        # Setup: 2 candidates with identical semantic vectors
        user_profile = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cand_vecs = np.array([
            [0.8, 0.1, 0.0],  # Candidate 1
            [0.8, 0.1, 0.0],  # Candidate 2 (same semantic vector)
        ], dtype=np.float32)

        cand_categories = ["sports", "cooking"]
        user_categories = {"sports"}

        b_scores = compute_baseline_b_score(cand_vecs, user_profile)
        c_scores = compute_baseline_c_score(cand_vecs, user_profile, cand_categories, user_categories)

        # Baseline B should give identical scores to both candidates
        assert np.isclose(b_scores[0], b_scores[1]), "Baseline B should not differentiate by category"

        # Baseline C must give higher score to matching category
        assert c_scores[0] > c_scores[1], "Baseline C must prioritize matching category"
        assert not np.allclose(b_scores, c_scores), "Baseline B and C must not produce identical scores"

    def test_ranking_inversion_between_baselines(self):
        """Verify that Baseline B and Baseline C can produce inverted ranking orders
        when semantic similarity and category match compete."""
        user_profile = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Cand 0: high semantic similarity (0.9), non-matching category
        # Cand 1: moderate semantic similarity (0.6), matching category
        cand_vecs = np.array([
            [0.9, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ], dtype=np.float32)

        cand_categories = ["finance", "sports"]
        user_categories = {"sports"}

        b_scores = compute_baseline_b_score(cand_vecs, user_profile)
        c_scores = compute_baseline_c_score(
            cand_vecs, user_profile, cand_categories, user_categories,
            w_long=0.40, w_cat=0.60
        )

        # Baseline B ranks Cand 0 > Cand 1
        assert b_scores[0] > b_scores[1]

        # Baseline C:
        # Cand 0: 0.40 * 0.9 + 0.60 * 0.5 = 0.36 + 0.30 = 0.66
        # Cand 1: 0.40 * 0.6 + 0.60 * 1.0 = 0.24 + 0.60 = 0.84
        # Baseline C ranks Cand 1 > Cand 0
        assert c_scores[1] > c_scores[0]

    def test_regression_stage3_2_equivalence_prevented(self):
        """Regression test: ensure Baseline B formula strictly ignores category metadata."""
        import inspect
        sig = inspect.signature(compute_baseline_b_score)
        params = list(sig.parameters.keys())
        assert "cand_categories" not in params, "Baseline B must not accept category parameters"
        assert "user_categories" not in params, "Baseline B must not accept user category set"
