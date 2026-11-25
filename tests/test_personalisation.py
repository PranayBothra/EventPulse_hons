"""Unit tests for user profiling, short/long-term dynamics, and user-level novelty."""

import tempfile
import pytest
import numpy as np

from src.personalisation.user_profiles import UserProfile, UserProfileManager
from src.personalisation.user_novelty import UserNoveltyEngine
from src.novelty.sentence_extraction import SentencePipeline, SentenceExtractor
from src.representation.embedding import TfidfEmbeddingProvider
from src.storage.repository import SQLiteRepository


class TestUserProfileManager:
    """Test dynamic dual-profile interest modelling and temporal decay."""

    @pytest.fixture
    def test_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            repo = SQLiteRepository(tmp.name)
            repo.initialize_schema()
            return repo

    def test_cold_start_profile(self, test_db):
        manager = UserProfileManager(test_db)
        profile = manager.get_profile("new_user_1")
        assert profile.is_cold_start is True
        assert profile.interaction_count == 0
        assert profile.long_term_vector is None
        assert profile.short_term_vector is None

        # Relevance on cold start returns neutral prior 0.5
        dummy_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        lt, st, comb = manager.compute_relevance_scores(profile, dummy_vec)
        assert lt == 0.5
        assert st == 0.5
        assert comb == 0.5

    def test_dual_profile_temporal_decay(self, test_db):
        # 1-day half-life for clear test divergence
        manager = UserProfileManager(test_db, half_life_seconds=86400.0, model_id="minilm")

        # Create dummy articles in database
        test_db.upsert_feed("feed_1", "News", "https://example.com/rss")
        
        # Article A: Technology (vector [1, 0, 0])
        test_db.insert_article({
            "canonical_url": "https://example.com/art_a",
            "url": "https://example.com/art_a",
            "headline": "Technology Breakthrough in AI Chips",
            "headline_clean": "technology breakthrough in ai chips",
            "headline_hash": "hash_art_a",
            "description": "New AI chip released.",
            "source_id": "feed_1",
            "is_sample": False,
        })
        art_a = test_db.get_article_by_canonical_url("https://example.com/art_a")
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        test_db.batch_save_embeddings([(art_a["id"], "hash_art_a", "minilm", 3, vec_a.tobytes())])

        # Article B: Sports (vector [0, 1, 0])
        test_db.insert_article({
            "canonical_url": "https://example.com/art_b",
            "url": "https://example.com/art_b",
            "headline": "Championship Finals Conclude with Dramatic Win",
            "headline_clean": "championship finals conclude with dramatic win",
            "headline_hash": "hash_art_b",
            "description": "Sports tournament concludes.",
            "source_id": "feed_1",
            "is_sample": False,
        })
        art_b = test_db.get_article_by_canonical_url("https://example.com/art_b")
        vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        test_db.batch_save_embeddings([(art_b["id"], "hash_art_b", "minilm", 3, vec_b.tobytes())])

        user_id = "test_user_decay"

        # T = 0: Read Technology article A
        manager.record_interaction(
            user_id=user_id,
            article_id=art_a["id"],
            interaction_type="READ",
            interaction_epoch=0.0,
            recompute_profile=False,
        )

        # T = 172800 (2 days later = 2 half-lives): Read Sports article B twice
        manager.record_interaction(
            user_id=user_id,
            article_id=art_b["id"],
            interaction_type="READ",
            interaction_epoch=172800.0,
            recompute_profile=False,
        )
        manager.record_interaction(
            user_id=user_id,
            article_id=art_b["id"],
            interaction_type="LIKE",
            interaction_epoch=172800.0,
            recompute_profile=True,
        )

        profile = manager.get_profile(user_id)
        assert profile.is_cold_start is False
        assert profile.interaction_count == 3

        # In short-term profile, article A has decayed by 2 half-lives (factor of 0.25)
        # So short-term is overwhelmingly aligned with Sports (vec_b)
        st_sim_sports = float(np.dot(profile.short_term_vector, vec_b))
        st_sim_tech = float(np.dot(profile.short_term_vector, vec_a))
        assert st_sim_sports > 0.90
        assert st_sim_tech < 0.25

        # In long-term profile, article A is NOT decayed, so both Technology and Sports are present
        lt_sim_sports = float(np.dot(profile.long_term_vector, vec_b))
        lt_sim_tech = float(np.dot(profile.long_term_vector, vec_a))
        assert lt_sim_tech > 0.30
        assert lt_sim_sports > 0.70


class TestUserNoveltyEngine:
    """Test personalized sentence novelty and event exposure penalties."""

    @pytest.fixture
    def test_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            repo = SQLiteRepository(tmp.name)
            repo.initialize_schema()
            return repo

    def test_user_novelty_fresh_vs_consumed(self, test_db):
        provider = TfidfEmbeddingProvider(dimension=64)
        pipeline = SentencePipeline(test_db, embedding_provider=provider)
        novelty_engine = UserNoveltyEngine(test_db, model_id=provider.model_id)
        profile_manager = UserProfileManager(test_db, model_id=provider.model_id)

        test_db.upsert_feed("feed_1", "News", "https://example.com/rss")

        # Insert article 1
        test_db.insert_article({
            "canonical_url": "https://example.com/art_1",
            "url": "https://example.com/art_1",
            "headline": "Electric Vehicle Sales Surge Worldwide",
            "headline_clean": "electric vehicle sales surge worldwide",
            "headline_hash": "hash_art_1",
            "description": "Automakers report record deliveries of electric cars in Q1.",
            "source_id": "feed_1",
            "is_sample": False,
        })
        art1 = test_db.get_article_by_canonical_url("https://example.com/art_1")
        pipeline.process_and_save_article_sentences(
            article_id=art1["id"],
            headline="Electric Vehicle Sales Surge Worldwide",
            description="Automakers report record deliveries of electric cars in Q1."
        )

        user_id = "user_ev_reader"

        # 1. Fresh user: Article 1 is 100% novel
        score_fresh = novelty_engine.compute_user_novelty(user_id, art1["id"])
        assert score_fresh.user_novelty_score == 1.0
        assert score_fresh.redundancy_penalty == 0.0

        # 2. User reads Article 1
        profile_manager.record_interaction(
            user_id=user_id,
            article_id=art1["id"],
            interaction_type="READ",
            interaction_epoch=1000.0,
        )

        # 3. User is shown Article 1 again (or an identical duplicate article)
        score_after = novelty_engine.compute_user_novelty(user_id, art1["id"])
        assert score_after.user_novelty_score <= 0.10
        assert score_after.redundancy_penalty >= 0.90
        assert score_after.most_similar_consumed_similarity >= 0.95

    def test_event_exposure_penalty(self, test_db):
        novelty_engine = UserNoveltyEngine(test_db)
        profile_manager = UserProfileManager(test_db)

        test_db.upsert_feed("feed_1", "News", "https://example.com/rss")
        event_id = "event_penalty_01"
        test_db.create_event(
            event_id=event_id,
            title="Tech Summit",
            summary="Annual tech summit",
            first_seen_at="2026-03-01T10:00:00Z",
            last_seen_at="2026-03-01T10:00:00Z",
            centroid_blob=None,
            key_entities=["tech"],
        )

        test_db.insert_article({
            "canonical_url": "https://example.com/art_ev1",
            "url": "https://example.com/art_ev1",
            "headline": "Tech Summit Opening Keynote",
            "headline_clean": "tech summit opening keynote",
            "headline_hash": "hash_art_ev1",
            "source_id": "feed_1",
            "is_sample": False,
        })
        art = test_db.get_article_by_canonical_url("https://example.com/art_ev1")
        test_db.assign_article_to_event(event_id, art["id"], similarity_score=0.9)

        user_id = "user_penalty_test"
        # Zero interactions => zero penalty
        p0 = novelty_engine.compute_event_exposure_penalty(user_id, event_id)
        assert p0 == 0.0

        # After 1 interaction
        profile_manager.record_interaction(
            user_id=user_id, article_id=art["id"], interaction_type="CLICK", interaction_epoch=100.0
        )
        p1 = novelty_engine.compute_event_exposure_penalty(user_id, event_id)
        assert p1 > 0.0

        # After 3 interactions
        profile_manager.record_interaction(
            user_id=user_id, article_id=art["id"], interaction_type="READ", interaction_epoch=200.0
        )
        profile_manager.record_interaction(
            user_id=user_id, article_id=art["id"], interaction_type="SAVE", interaction_epoch=300.0
        )
        p3 = novelty_engine.compute_event_exposure_penalty(user_id, event_id)
        assert p3 > p1
