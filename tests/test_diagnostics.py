import numpy as np
import pytest
from src.representation.diagnostics import (
    compute_mean_vector_norm,
    compute_pairwise_cosine_stats,
    compute_covariance_spectrum,
    compute_effective_rank,
    compute_full_representation_diagnostics,
)


def test_diagnostics_on_orthogonal_matrix():
    # Identity matrix: 10 orthogonal unit vectors in R^10
    X = np.eye(10, dtype=np.float32)
    
    # Mean vector norm: ||(1/10) * [1, 1, ..., 1]||_2 = sqrt(10 * (1/10)^2) = 1/sqrt(10) ~ 0.3162
    mean_norm = compute_mean_vector_norm(X)
    assert abs(mean_norm - 1.0 / np.sqrt(10)) < 1e-4
    
    # Random pairwise cosine for distinct orthogonal vectors is exactly 0.0
    pair_stats = compute_pairwise_cosine_stats(X, n_pairs=500)
    assert abs(pair_stats["mean_cosine"]) < 1e-5
    assert abs(pair_stats["max_cosine"]) < 1e-5
    
    # Covariance spectrum: all variance equally distributed
    eigs, var_stats = compute_covariance_spectrum(X)
    erank = compute_effective_rank(eigs)
    assert erank["effective_rank"] > 8.0  # Close to full rank 10


def test_diagnostics_on_collinear_matrix():
    # 10 identical vectors: maximum anisotropy / collapse
    v = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
    X = np.repeat(v, 10, axis=0)
    
    # Mean vector norm is exactly 1.0
    mean_norm = compute_mean_vector_norm(X)
    assert abs(mean_norm - 1.0) < 1e-5
    
    # Random pairwise cosine is exactly 1.0
    pair_stats = compute_pairwise_cosine_stats(X, n_pairs=100)
    assert abs(pair_stats["mean_cosine"] - 1.0) < 1e-5
    
    # Effective rank is 1.0
    eigs, var_stats = compute_covariance_spectrum(X)
    erank = compute_effective_rank(eigs)
    assert erank["effective_rank"] == 1.0


def test_full_pipeline_output_structure():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    res = compute_full_representation_diagnostics(X, "test_model")
    assert res["model_id"] == "test_model"
    assert res["sample_size"] == 20
    assert res["dimension"] == 16
    assert "mean_vector_norm" in res
    assert "pairwise_cosine" in res
    assert "variance_spectrum" in res
    assert "effective_rank" in res
