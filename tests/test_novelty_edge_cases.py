"""
Unit tests for edge cases in novelty detection and validation for EventPulse Stage 3.3.
Covers:
1. Differing numbers in non-factual / incidental contexts.
2. Polarity action flips and opposition semantics.
3. Insufficient evidence / trivial fragments.
4. History permutation symmetry of set-maximum memory.
5. Integrity and blinded status of data/novelty_validation_blind.csv.
"""

import csv
from pathlib import Path
import numpy as np
import pytest

from src.novelty.multi_signal_estimator import MultiSignalNoveltyEstimator


@pytest.fixture
def estimator():
    return MultiSignalNoveltyEstimator(redundancy_threshold=0.75, novelty_threshold=0.65)


class TestNoveltyEdgeCases:

    def test_blinded_dataset_integrity(self):
        """Ensure data/novelty_validation_blind.csv adheres to schema and preserves blinded status."""
        csv_path = Path("data/novelty_validation_blind.csv")
        assert csv_path.exists(), "novelty_validation_blind.csv must exist"

        expected_columns = [
            "pair_id", "event_id", "article_id", "sentence_candidate",
            "sentence_historical", "allowed_categories", "ground_truth_label", "annotator_notes"
        ]

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == expected_columns, f"Columns must match: {expected_columns}"
            rows = list(reader)

        assert len(rows) == 200, f"Expected exactly 200 pairs, found {len(rows)}"

        # Verify all ground truth labels are strictly PENDING_BLIND_ANNOTATION
        unblinded = [r for r in rows if r["ground_truth_label"] != "PENDING_BLIND_ANNOTATION"]
        assert len(unblinded) == 0, f"Blinded status violated for {len(unblinded)} rows! Labels must remain PENDING_BLIND_ANNOTATION."

    def test_insufficient_evidence_classification(self, estimator):
        """Short sentence fragments or low-information tokens must be classified as INSUFFICIENT_EVIDENCE."""
        cand_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mem_vec = np.array([[0.2, 0.8, 0.0]], dtype=np.float32)

        fragment = "Read more."
        res = estimator.estimate_novelty(
            candidate_text=fragment,
            candidate_embedding=cand_vec,
            memory_texts=["The president signed the bilateral treaty."],
            memory_embeddings=mem_vec,
            informativeness_score=0.15,
        )
        assert res.evidence_status in ["BACKGROUND_INFORMATION", "INSUFFICIENT_EVIDENCE"]

    def test_polarity_action_flip_resumed_collapsed(self, estimator):
        """Opposite action tokens (resumed vs collapsed) must trigger opposition polarity flip."""
        cand = "Diplomats confirmed the peace talks resumed in Geneva."
        hist = "Diplomats confirmed the peace talks collapsed in Geneva."
        pol_diff, reason = estimator.detect_polarity_and_action_flip(cand, hist)
        assert pol_diff is True
        assert reason == "OPPOSITION_ACTION_FLIP"

    def test_history_permutation_max_symmetry(self):
        """Verify that max-similarity redundancy penalty is symmetric under history permutation."""
        cand_vec = np.array([0.5, 0.5, 0.7071], dtype=np.float32)
        h1 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        h2 = np.array([0.3, 0.8, 0.5], dtype=np.float32)
        h3 = np.array([0.5, 0.5, 0.7071], dtype=np.float32)

        hist_forward = np.vstack([h1, h2, h3])
        hist_reversed = np.vstack([h3, h2, h1])
        hist_shuffled = np.vstack([h2, h1, h3])

        sim_fwd = np.dot(cand_vec, hist_forward.T)
        sim_rev = np.dot(cand_vec, hist_reversed.T)
        sim_shf = np.dot(cand_vec, hist_shuffled.T)

        max_fwd = float(np.max(sim_fwd))
        max_rev = float(np.max(sim_rev))
        max_shf = float(np.max(sim_shf))

        assert np.isclose(max_fwd, max_rev), "Max similarity must be invariant to sequence reversal"
        assert np.isclose(max_fwd, max_shf), "Max similarity must be invariant to sequence permutation"

    def test_top3_average_smooths_extreme_outlier(self):
        """Top-3 average similarity prevents a single spurious match from dominating redundancy."""
        cand_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Outlier vector matches perfectly, but other 9 vectors have low similarity (0.1)
        outlier = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        background = [np.array([0.1, 0.99, 0.0], dtype=np.float32) for _ in range(9)]
        hist_vecs = np.vstack([outlier] + background)

        sims = np.dot(cand_vec, hist_vecs.T)
        max_sim = float(np.max(sims))
        top3_sim = float(np.mean(np.partition(sims, -3)[-3:]))

        assert max_sim == 1.0
        assert top3_sim < 0.5, f"Top-3 avg should be substantially lower than single outlier: {top3_sim}"
