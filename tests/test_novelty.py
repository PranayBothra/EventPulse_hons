"""Unit tests for EventPulse information novelty detection and event memory."""

import tempfile
import pytest
import numpy as np

from src.novelty.event_memory import EventMemoryManager, MemorySentence
from src.novelty.novelty_scoring import SentenceNoveltyScorer, SentenceNoveltyResult
from src.novelty.novelty_aggregation import ArticleNoveltySummary, EventNoveltyEngine
from src.novelty.sentence_extraction import SentencePipeline, SentenceExtractor
from src.representation.embedding import TfidfEmbeddingProvider
from src.storage.repository import SQLiteRepository


class TestSentenceNoveltyScorer:
    """Test mathematical scoring of sentence novelty and evidence classification."""

    def test_score_against_empty_memory(self):
        scorer = SentenceNoveltyScorer()
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        res = scorer.score_sentence(
            sentence_id=1,
            article_id=10,
            sentence_text="Breaking news event occurs.",
            sentence_index=0,
            embedding=emb,
            informativeness_score=0.8,
            memory=[],
        )
        assert res.highest_similarity == 0.0
        assert res.most_similar_sentence_id is None
        assert res.novelty_score == 0.8
        assert res.evidence_status == "POTENTIALLY_NEW"

    def test_score_against_identical_memory(self):
        scorer = SentenceNoveltyScorer(redundancy_threshold=0.75)
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mem = [
            MemorySentence(
                sentence_id=99,
                article_id=9,
                published_at="2026-03-01T10:00:00Z",
                published_epoch=1000.0,
                sentence_text="Breaking news event occurs.",
                informativeness_score=0.8,
                embedding=emb,
                novelty_score=0.8,
                evidence_status="POTENTIALLY_NEW",
                named_entities=set(),
                has_numbers=False,
                has_dates=False,
            )
        ]
        res = scorer.score_sentence(
            sentence_id=100,
            article_id=10,
            sentence_text="Breaking news event occurs again.",
            sentence_index=0,
            embedding=emb,
            informativeness_score=0.8,
            memory=mem,
        )
        assert res.highest_similarity >= 0.99
        assert res.most_similar_sentence_id == 99
        assert res.novelty_score <= 0.05
        assert res.evidence_status == "REPEATED_INFORMATION"

    def test_score_orthogonal_development(self):
        scorer = SentenceNoveltyScorer(novelty_threshold=0.65)
        mem_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        new_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        mem = [
            MemorySentence(
                sentence_id=1,
                article_id=1,
                published_at="2026-03-01T10:00:00Z",
                published_epoch=1000.0,
                sentence_text="Old fact.",
                informativeness_score=0.7,
                embedding=mem_emb,
                novelty_score=0.7,
                evidence_status="POTENTIALLY_NEW",
                named_entities=set(),
                has_numbers=False,
                has_dates=False,
            )
        ]
        res = scorer.score_sentence(
            sentence_id=2,
            article_id=2,
            sentence_text="Entirely distinct substantive development.",
            sentence_index=1,
            embedding=new_emb,
            informativeness_score=0.75,
            memory=mem,
        )
        assert res.highest_similarity == 0.0
        assert res.novelty_score == 0.75
        assert res.evidence_status == "POTENTIALLY_NEW"

    def test_insufficient_evidence_for_low_informativeness(self):
        scorer = SentenceNoveltyScorer(min_informative_threshold=0.35)
        emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        res = scorer.score_sentence(
            sentence_id=3,
            article_id=3,
            sentence_text="Stay tuned for more.",
            sentence_index=2,
            embedding=emb,
            informativeness_score=0.20,  # Below threshold
            memory=[],
        )
        assert res.evidence_status == "INSUFFICIENT_EVIDENCE"


class TestEventNoveltyEngine:
    """Test full chronological novelty pipeline and multi-article event evolution."""

    @pytest.fixture
    def test_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            repo = SQLiteRepository(tmp.name)
            repo.initialize_schema()
            return repo

    def test_temporal_novelty_evolution(self, test_db):
        provider = TfidfEmbeddingProvider(dimension=64)
        pipeline = SentencePipeline(test_db, embedding_provider=provider)
        engine = EventNoveltyEngine(
            repository=test_db,
            pipeline=pipeline,
            model_id=provider.model_id,
        )

        test_db.upsert_feed("feed_1", "News", "https://example.com/rss")
        event_id = "event_test_001"
        test_db.create_event(
            event_id=event_id,
            title="Spacecraft Lands on Asteroid",
            summary="Spacecraft touches down on asteroid.",
            first_seen_at="2026-03-01T10:00:00Z",
            last_seen_at="2026-03-01T10:00:00Z",
            centroid_blob=None,
            key_entities=["asteroid", "spacecraft"],
        )

        # Article 1: Breaking news at T=1000
        test_db.insert_article({
            "canonical_url": "https://example.com/art_1",
            "url": "https://example.com/art_1",
            "headline": "Spacecraft Lands on Asteroid",
            "headline_clean": "spacecraft lands on asteroid",
            "headline_hash": "hash_art_1",
            "description": "The robotic probe touched down on the asteroid surface on Tuesday. Telemetry confirms all scientific instruments operational.",
            "content_clean": "The robotic probe touched down on the asteroid surface on Tuesday. Telemetry confirms all scientific instruments operational.",
            "source_id": "feed_1",
            "source_name": "SpaceDaily",
            "published_at": "2026-03-01T10:00:00Z",
            "is_sample": False,
        })
        art1 = test_db.get_article_by_canonical_url("https://example.com/art_1")

        summary1 = engine.evaluate_article_novelty(
            article_id=art1["id"],
            event_id=event_id,
            published_at="2026-03-01T10:00:00Z",
            published_epoch=1000.0,
            persist=True,
        )

        assert summary1.event_novelty_score > 0.40
        assert summary1.novel_sentence_count >= 1
        assert summary1.redundant_sentence_count == 0

        # Article 2: Repetitive coverage at T=2000 (repeats same facts)
        test_db.insert_article({
            "canonical_url": "https://example.com/art_2",
            "url": "https://example.com/art_2",
            "headline": "Spacecraft Lands on Asteroid",
            "headline_clean": "spacecraft lands on asteroid",
            "headline_hash": "hash_art_2",
            "description": "The robotic probe touched down on the asteroid surface on Tuesday. Telemetry confirms all scientific instruments operational.",
            "content_clean": "The robotic probe touched down on the asteroid surface on Tuesday. Telemetry confirms all scientific instruments operational.",
            "source_id": "feed_1",
            "source_name": "EchoNews",
            "published_at": "2026-03-01T11:00:00Z",
            "is_sample": False,
        })
        art2 = test_db.get_article_by_canonical_url("https://example.com/art_2")

        summary2 = engine.evaluate_article_novelty(
            article_id=art2["id"],
            event_id=event_id,
            published_at="2026-03-01T11:00:00Z",
            published_epoch=2000.0,
            persist=True,
        )

        # Article 2 must have much lower novelty and detected redundancy
        assert summary2.event_novelty_score < summary1.event_novelty_score
        assert summary2.redundant_sentence_count > 0

        # Verify temporal non-leakage:
        # Re-evaluating Article 1 with before_epoch=1000.0 must not see Article 2!
        mem_at_t1 = engine.memory_manager.get_event_memory(event_id, before_epoch=1000.0)
        assert len(mem_at_t1) == 0

        mem_at_t2 = engine.memory_manager.get_event_memory(event_id, before_epoch=2000.0)
        assert len(mem_at_t2) == len(summary1.sentence_results)

        # Verify database records
        db_nov1 = test_db.get_article_novelty(art1["id"], event_id)
        assert db_nov1 is not None
        assert db_nov1["event_novelty_score"] == summary1.event_novelty_score
