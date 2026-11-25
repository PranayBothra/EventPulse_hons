"""Unit tests for MultiSignalNoveltyEstimator addressing numerical and polarity updates."""

import numpy as np
import pytest

from src.novelty.multi_signal_estimator import MultiSignalNoveltyEstimator


@pytest.fixture
def estimator():
    return MultiSignalNoveltyEstimator(redundancy_threshold=0.75, novelty_threshold=0.65)


def test_numerical_change_detection(estimator):
    cand = "Authorities confirmed the confirmed death toll has climbed to 47."
    hist = "Authorities reported that the confirmed death toll was 38."
    num_diff, c_nums, h_nums = estimator.detect_numerical_difference(cand, hist)
    assert num_diff is True
    assert "47" in c_nums
    assert "38" in h_nums


def test_polarity_action_flip(estimator):
    cand = "The governor approved the new climate regulations."
    hist = "The governor rejected the new climate regulations."
    pol_diff, reason = estimator.detect_polarity_and_action_flip(cand, hist)
    assert pol_diff is True
    assert reason == "OPPOSITION_ACTION_FLIP"


def test_negation_flip(estimator):
    cand = "The jury found the defendant guilty on all counts."
    hist = "The jury found the defendant not guilty on all counts."
    pol_diff, reason = estimator.detect_polarity_and_action_flip(cand, hist)
    assert pol_diff is True
    assert "NEGATION_POLARITY_FLIP" in reason or "OPPOSITION_ACTION_FLIP" in reason


def test_promotional_boilerplate_detection(estimator):
    promo = "For more local weather updates, download the channel 4 news application now."
    assert estimator.is_promotional_or_boilerplate(promo) is True

    valid = "The municipal council voted unanimously to allocate funding for road repairs."
    assert estimator.is_promotional_or_boilerplate(valid) is False


def test_estimate_novelty_numerical_override(estimator):
    cand = "Officials said 47 people were injured in the train derailment."
    hist = "Officials said 38 people were injured in the train derailment."

    # Mock high cosine similarity embedding (0.95)
    cand_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mem_vec = np.array([[0.95, 0.312, 0.0]], dtype=np.float32)

    res = estimator.estimate_novelty(
        candidate_text=cand,
        candidate_embedding=cand_vec,
        memory_texts=[hist],
        memory_embeddings=mem_vec,
        informativeness_score=0.80,
    )

    # In single-signal bi-encoder, 0.95 similarity would be REPEATED_INFORMATION.
    # But multi-signal detects numerical change and overrides to POTENTIALLY_NEW!
    assert res.numerical_change_detected is True
    assert res.evidence_status == "POTENTIALLY_NEW"
    assert res.composite_novelty_score >= 0.80


def test_estimate_novelty_promotional_background(estimator):
    promo = "Download the mobile app today for breaking news alerts."
    cand_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mem_vec = np.array([[0.1, 0.9, 0.0]], dtype=np.float32)

    res = estimator.estimate_novelty(
        candidate_text=promo,
        candidate_embedding=cand_vec,
        memory_texts=["Rescue crews arrived on the scene."],
        memory_embeddings=mem_vec,
        informativeness_score=0.20,
    )

    assert res.evidence_status == "BACKGROUND_INFORMATION"
    assert res.composite_novelty_score == 0.0
