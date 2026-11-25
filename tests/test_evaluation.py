"""Unit tests for evaluation metrics, pair generator, and sensitivity analyzer."""

from pathlib import Path
import pytest

from src.evaluation.metrics import (
    compute_clustering_metrics,
    compute_cluster_purity,
    compute_pairwise_metrics,
)
from src.evaluation.pair_generator import generate_validation_pairs_csv
from src.evaluation.sensitivity import run_threshold_sensitivity
from src.representation.embedding import TfidfEmbeddingProvider
from src.storage.repository import SQLiteRepository


class TestEvaluationMetrics:
    def test_pairwise_metrics_perfect_agreement(self):
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 1, 0, 0, 1]
        res = compute_pairwise_metrics(y_true, y_pred)
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0
        assert res["f1"] == 1.0
        assert res["true_positives"] == 3
        assert res["false_positives"] == 0

    def test_pairwise_metrics_partial_overlap(self):
        y_true = [1, 1, 1, 0]
        y_pred = [1, 0, 1, 1]
        res = compute_pairwise_metrics(y_true, y_pred)
        # TP = 2, FP = 1, FN = 1, TN = 0
        assert res["true_positives"] == 2
        assert res["false_positives"] == 1
        assert res["false_negatives"] == 1
        assert res["true_negatives"] == 0
        assert pytest.approx(res["precision"]) == 2 / 3
        assert pytest.approx(res["recall"]) == 2 / 3
        assert pytest.approx(res["f1"]) == 2 / 3

    def test_pairwise_metrics_all_negative_predictions(self):
        y_true = [1, 1, 0]
        y_pred = [0, 0, 0]
        res = compute_pairwise_metrics(y_true, y_pred)
        assert res["precision"] == 0.0
        assert res["recall"] == 0.0
        assert res["f1"] == 0.0

    def test_cluster_purity(self):
        labels_true = [0, 0, 0, 1, 1, 1]
        labels_pred = [0, 0, 0, 1, 1, 1]
        assert compute_cluster_purity(labels_true, labels_pred) == 1.0

        # One mixed item
        labels_mixed = [0, 0, 1, 1, 1, 1]
        purity = compute_cluster_purity(labels_true, labels_mixed)
        assert 0.0 < purity < 1.0

        # Empty inputs
        assert compute_cluster_purity([], []) == 0.0

    def test_compute_clustering_metrics_unlabeled_status(self):
        pred = [0, 1, 0, 2]
        res = compute_clustering_metrics(None, pred)
        assert res.status == "PENDING_ANNOTATIONS"
        assert res.labeled_pairs == 0
        assert res.adjusted_rand_index is None

    def test_compute_clustering_metrics_labeled(self):
        true = [0, 0, 1, 1]
        pred = [0, 0, 1, 1]
        res = compute_clustering_metrics(true, pred)
        assert res.status == "COMPLETED"
        assert res.adjusted_rand_index == 1.0
        assert res.normalized_mutual_info == 1.0
        assert res.cluster_purity == 1.0


class TestSensitivityAndPairGenerator:
    def test_pair_generator_with_empty_db(self, temp_db: SQLiteRepository, tmp_path: Path):
        provider = TfidfEmbeddingProvider(dimension=32)
        out_csv = tmp_path / "test_pairs_blind.csv"
        blind_path, pred_path = generate_validation_pairs_csv(
            repository=temp_db,
            embedding_provider=provider,
            output_path=str(out_csv),
            sample_pairs_count=10,
        )
        assert blind_path.exists()
        assert pred_path.exists()
        blind_content = blind_path.read_text(encoding="utf-8")
        assert "Pair ID" in blind_content
        assert "Human label" in blind_content
        assert "Predicted similarity" not in blind_content

        pred_content = pred_path.read_text(encoding="utf-8")
        assert "Pair ID" in pred_content
        assert "Predicted similarity" in pred_content

    def test_sensitivity_analyzer_empty_db(self, temp_db: SQLiteRepository):
        provider = TfidfEmbeddingProvider(dimension=32)
        results = run_threshold_sensitivity(
            repository=temp_db,
            embedding_provider=provider,
            thresholds=[0.40, 0.50],
        )
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert "threshold" in r
            assert r["total_events"] == 0
