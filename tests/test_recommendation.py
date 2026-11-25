"""Unit tests for recommendation models, ranking algorithms, and baselines."""

import tempfile
import pytest
import numpy as np

from src.personalisation.user_profiles import UserProfileManager
from src.recommendation.baselines import (
    ChronologicalRecommender,
    ContentBasedRecommender,
    EventAwareWithoutNoveltyRecommender,
    ProposedEventPulseRecommender,
)
from src.recommendation.ranking import NoveltyAwareRecommender, RecommendationWeights
from src.storage.repository import SQLiteRepository


class TestRecommendationModels:
    """Test comparative behavior of all 4 recommendation models."""

    @pytest.fixture
    def test_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            repo = SQLiteRepository(tmp.name)
            repo.initialize_schema()
            return repo

    def test_chronological_recommender(self, test_db):
        test_db.upsert_feed("feed_1", "News", "https://example.com/rss")

        # Create two events with different timestamps
        test_db.create_event(
            event_id="ev_old",
            title="Old Event",
            summary="Happened yesterday",
            first_seen_at="2026-03-01T10:00:00Z",
            last_seen_at="2026-03-01T10:00:00Z",
            centroid_blob=np.array([1, 0, 0], dtype=np.float32).tobytes(),
            key_entities=["old"],
        )
        test_db.insert_article({
            "canonical_url": "https://example.com/art_old",
            "url": "https://example.com/art_old",
            "headline": "Old Event Headline",
            "headline_clean": "old event headline",
            "headline_hash": "hash_old",
            "published_at": "2026-03-01T10:00:00Z",
            "source_id": "feed_1",
            "is_sample": False,
        })
        art_old = test_db.get_article_by_canonical_url("https://example.com/art_old")
        test_db.assign_article_to_event("ev_old", art_old["id"], 0.9)

        test_db.create_event(
            event_id="ev_new",
            title="New Event",
            summary="Happened today",
            first_seen_at="2026-03-02T10:00:00Z",
            last_seen_at="2026-03-02T10:00:00Z",
            centroid_blob=np.array([0, 1, 0], dtype=np.float32).tobytes(),
            key_entities=["new"],
        )
        test_db.insert_article({
            "canonical_url": "https://example.com/art_new",
            "url": "https://example.com/art_new",
            "headline": "New Event Headline",
            "headline_clean": "new event headline",
            "headline_hash": "hash_new",
            "published_at": "2026-03-02T10:00:00Z",
            "source_id": "feed_1",
            "is_sample": False,
        })
        art_new = test_db.get_article_by_canonical_url("https://example.com/art_new")
        test_db.assign_article_to_event("ev_new", art_new["id"], 0.9)

        recommender = ChronologicalRecommender(test_db)
        recs = recommender.rank_events("any_user", top_k=2)

        assert len(recs) == 2
        assert recs[0].event_id == "ev_new"
        assert recs[1].event_id == "ev_old"

    def test_proposed_recommender_exposure_penalty(self, test_db):
        test_db.upsert_feed("feed_1", "News", "https://example.com/rss")
        profile_mgr = UserProfileManager(test_db, model_id="minilm")

        # Create two events
        centroid_a = np.array([1, 0, 0], dtype=np.float32)
        centroid_b = np.array([0, 1, 0], dtype=np.float32)

        test_db.create_event(
            event_id="ev_a",
            title="Event A",
            summary="Event A summary",
            first_seen_at="2026-03-02T10:00:00Z",
            last_seen_at="2026-03-02T10:00:00Z",
            centroid_blob=centroid_a.tobytes(),
            key_entities=["a"],
        )
        test_db.insert_article({
            "canonical_url": "https://example.com/art_a",
            "url": "https://example.com/art_a",
            "headline": "Headline A",
            "headline_clean": "headline a",
            "headline_hash": "hash_a",
            "published_at": "2026-03-02T10:00:00Z",
            "source_id": "feed_1",
            "is_sample": False,
        })
        art_a = test_db.get_article_by_canonical_url("https://example.com/art_a")
        test_db.assign_article_to_event("ev_a", art_a["id"], 0.9)

        test_db.create_event(
            event_id="ev_b",
            title="Event B",
            summary="Event B summary",
            first_seen_at="2026-03-02T10:00:00Z",
            last_seen_at="2026-03-02T10:00:00Z",
            centroid_blob=centroid_b.tobytes(),
            key_entities=["b"],
        )
        test_db.insert_article({
            "canonical_url": "https://example.com/art_b",
            "url": "https://example.com/art_b",
            "headline": "Headline B",
            "headline_clean": "headline b",
            "headline_hash": "hash_b",
            "published_at": "2026-03-02T10:00:00Z",
            "source_id": "feed_1",
            "is_sample": False,
        })
        art_b = test_db.get_article_by_canonical_url("https://example.com/art_b")
        test_db.assign_article_to_event("ev_b", art_b["id"], 0.9)

        user_id = "test_user_rec"

        # User reads Event A multiple times, making Event A saturated/exposed
        for i in range(4):
            profile_mgr.record_interaction(
                user_id=user_id,
                article_id=art_a["id"],
                event_id="ev_a",
                interaction_type="READ",
                interaction_epoch=100.0 + i,
            )

        recommender = ProposedEventPulseRecommender(test_db, profile_manager=profile_mgr, model_id="minilm")
        recs = recommender.rank_events(user_id, top_k=2)

        assert len(recs) == 2
        # Event A should have high exposure penalty recorded in breakdown
        rec_map = {r.event_id: r for r in recs}
        assert rec_map["ev_a"].score_breakdown["exposure_penalty"] > 0.80
        assert rec_map["ev_b"].score_breakdown["exposure_penalty"] == 0.0
        assert "breakdown" in str(rec_map["ev_a"].score_breakdown) or "raw_composite" in rec_map["ev_a"].score_breakdown
        assert rec_map["ev_a"].explanation is not None
