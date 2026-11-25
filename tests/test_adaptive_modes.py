"""Unit tests for adaptive recommendation modes in EventPulse."""

import pytest
from src.recommendation.ranking import (
    NoveltyAwareRecommender,
    RecommendationMode,
    RecommendationWeights,
)
from src.storage.repository import SQLiteRepository


def test_weights_for_mode_relevance_focused():
    w = RecommendationWeights.for_mode(RecommendationMode.RELEVANCE_FOCUSED)
    assert w.w_long == 0.40
    assert w.w_short == 0.35
    assert w.w_event_novelty == 0.05
    assert w.w_user_novelty == 0.05
    assert w.w_redundancy_penalty == 0.05
    assert w.mmr_lambda == 0.95


def test_weights_for_mode_balanced_intelligence():
    w = RecommendationWeights.for_mode(RecommendationMode.BALANCED_INTELLIGENCE)
    assert w.w_long == 0.25
    assert w.w_short == 0.25
    assert w.w_event_novelty == 0.15
    assert w.w_user_novelty == 0.10
    assert w.w_redundancy_penalty == 0.10
    assert w.mmr_lambda == 0.80


def test_weights_for_mode_discovery_focused():
    w = RecommendationWeights.for_mode(RecommendationMode.DISCOVERY_FOCUSED)
    assert w.w_long == 0.15
    assert w.w_short == 0.15
    assert w.w_event_novelty == 0.25
    assert w.w_user_novelty == 0.20
    assert w.w_redundancy_penalty == 0.20
    assert w.mmr_lambda == 0.65


def test_recommender_instantiation_with_modes(tmp_path):
    db_file = tmp_path / "test.db"
    repo = SQLiteRepository(str(db_file))

    rec_rel = NoveltyAwareRecommender(repo, mode=RecommendationMode.RELEVANCE_FOCUSED)
    assert rec_rel.weights.w_long == 0.40
    assert rec_rel.weights.mmr_lambda == 0.95

    rec_bal = NoveltyAwareRecommender(repo, mode=RecommendationMode.BALANCED_INTELLIGENCE)
    assert rec_bal.weights.w_long == 0.25
    assert rec_bal.weights.mmr_lambda == 0.80

    rec_disc = NoveltyAwareRecommender(repo, mode=RecommendationMode.DISCOVERY_FOCUSED)
    assert rec_disc.weights.w_long == 0.15
    assert rec_disc.weights.mmr_lambda == 0.65
