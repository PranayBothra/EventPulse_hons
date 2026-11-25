"""
Representation-Space Diagnostics for Embedding Anisotropy and Geometry Analysis.

Implements rigorous information-theoretic and geometric diagnostics:
1. Mean random-pair cosine similarity
2. Pairwise cosine similarity distribution (quantiles & variance)
3. Mean-vector norm (Centroid length ||mu||_2)
4. Covariance spectrum (Eigenvalues & variance explained by top-k components)
5. Effective Rank (Roy & Vetterli, 2007)
"""

import numpy as np
from typing import Dict, Any, Tuple


def compute_mean_vector_norm(embeddings: np.ndarray) -> float:
    """Compute the L2 norm of the unnormalized mean vector: ||mu||_2 = ||(1/N) * sum v_i||_2.
    
    For an isotropic uniform distribution on S^{d-1}, ||mu||_2 ~ 1 / sqrt(N).
    For a narrow cone (severe anisotropy), ||mu||_2 -> 1.0.
    """
    if len(embeddings) == 0:
        return 0.0
    mean_vec = np.mean(embeddings, axis=0)
    return float(np.linalg.norm(mean_vec))


def compute_pairwise_cosine_stats(
    embeddings: np.ndarray,
    n_pairs: int = 20000,
    seed: int = 42
) -> Dict[str, Any]:
    """Sample random pairs from the embedding matrix and compute cosine similarity distribution."""
    N = len(embeddings)
    if N < 2:
        return {}
    
    rng = np.random.default_rng(seed)
    # Sample random index pairs without replacement
    idx1 = rng.integers(0, N, size=n_pairs)
    idx2 = rng.integers(0, N, size=n_pairs)
    # Avoid self-pairs
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    
    v1 = embeddings[idx1]
    v2 = embeddings[idx2]
    
    # Cosine similarity (embeddings are already unit-normalized)
    sims = np.sum(v1 * v2, axis=1)
    
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q_vals = np.quantile(sims, quantiles)
    
    return {
        "n_pairs_sampled": int(len(sims)),
        "mean_cosine": float(np.mean(sims)),
        "std_cosine": float(np.std(sims)),
        "min_cosine": float(np.min(sims)),
        "max_cosine": float(np.max(sims)),
        "median_cosine": float(np.median(sims)),
        "p01": float(q_vals[0]),
        "p05": float(q_vals[1]),
        "p10": float(q_vals[2]),
        "p25": float(q_vals[3]),
        "p50": float(q_vals[4]),
        "p75": float(q_vals[5]),
        "p90": float(q_vals[6]),
        "p95": float(q_vals[7]),
        "p99": float(q_vals[8]),
    }


def compute_covariance_spectrum(embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute eigenvalues of the centered covariance matrix and variance explained.
    
    Returns:
        eigenvalues: Sorted array of eigenvalues (descending order).
        variance_stats: Dict of variance explained by top-1, top-5, top-10, top-20 components.
    """
    N, d = embeddings.shape
    if N < 2:
        return np.zeros(d), {}
    
    # Center the embeddings: X_tilde = X - mu
    mean_vec = np.mean(embeddings, axis=0, keepdims=True)
    centered = embeddings - mean_vec
    
    # Compute SVD of centered matrix: X_tilde = U * S * Vh
    # Covariance eigenvalues are S_i^2 / (N - 1)
    # SVD is more numerically stable than computing X^T X directly
    s = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = (s ** 2) / (N - 1)
    
    total_var = np.sum(eigenvalues)
    if total_var <= 1e-12:
        return eigenvalues, {"top1_ratio": 0.0, "top5_ratio": 0.0, "top10_ratio": 0.0}
    
    var_ratios = np.cumsum(eigenvalues) / total_var
    
    stats = {
        "total_variance": float(total_var),
        "top1_var_ratio": float(var_ratios[0]),
        "top5_var_ratio": float(var_ratios[min(4, len(var_ratios) - 1)]),
        "top10_var_ratio": float(var_ratios[min(9, len(var_ratios) - 1)]),
        "top20_var_ratio": float(var_ratios[min(19, len(var_ratios) - 1)]),
    }
    
    return eigenvalues, stats


def compute_effective_rank(eigenvalues: np.ndarray) -> Dict[str, float]:
    """Compute Roy & Vetterli (2007) Effective Rank:
    
    erank(Sigma) = exp( - sum p_i * ln(p_i) ) where p_i = lambda_i / sum(lambda_j).
    Normalized effective rank: erank / d in [1/d, 1.0].
    """
    d = len(eigenvalues)
    pos_eigs = eigenvalues[eigenvalues > 1e-12]
    
    if len(pos_eigs) == 0:
        return {"effective_rank": 1.0, "normalized_effective_rank": 1.0 / d, "dimension": d}
    
    total = np.sum(pos_eigs)
    p = pos_eigs / total
    
    # Shannon entropy
    entropy = -np.sum(p * np.log(p))
    erank = float(np.exp(entropy))
    norm_erank = float(erank / d)
    
    return {
        "dimension": d,
        "effective_rank": round(erank, 2),
        "normalized_effective_rank": round(norm_erank, 4),
        "entropy_nats": round(float(entropy), 4)
    }


def compute_full_representation_diagnostics(embeddings: np.ndarray, model_id: str) -> Dict[str, Any]:
    """Execute complete representation diagnostic pipeline on an embedding matrix."""
    N, d = embeddings.shape
    mean_norm = compute_mean_vector_norm(embeddings)
    pairwise_stats = compute_pairwise_cosine_stats(embeddings)
    eigenvalues, var_stats = compute_covariance_spectrum(embeddings)
    erank_stats = compute_effective_rank(eigenvalues)
    
    return {
        "model_id": model_id,
        "sample_size": N,
        "dimension": d,
        "mean_vector_norm": round(mean_norm, 5),
        "pairwise_cosine": pairwise_stats,
        "variance_spectrum": var_stats,
        "effective_rank": erank_stats
    }
