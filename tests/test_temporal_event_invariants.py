"""
Unit and invariant tests for Track B Chronological Event-Memory Invariants.
Verifies Section 2.1 criteria:
1. Strict temporal inequality: H(e, t) contains only propositions with published_at < t.
2. No future articles or self-matches enter memory.
3. Deterministic tie-breaking for concurrent / tied timestamps.
4. Physical insertion order invariance during chronological replay.
5. Invariant preservation under reordered ingestion.
"""

import numpy as np
import pytest

from src.storage.repository import SQLiteRepository
from src.novelty.event_memory import EventMemoryManager


def _insert_test_article(repo: SQLiteRepository, idx: int, pub_at: str) -> int:
    url = f"https://example.com/news/{idx}"
    article = {
        "canonical_url": url,
        "url": url,
        "headline": f"News Headline {idx}",
        "headline_clean": f"news headline {idx}",
        "headline_hash": f"hash_{idx}",
        "description": f"Description {idx}",
        "content_clean": f"Content {idx}",
        "source_id": "test_source",
        "source_name": "Test Source",
        "published_at": pub_at,
        "retrieved_at": pub_at,
        "is_sample": 0,
        "raw_metadata": "{}",
    }
    repo.insert_article(article)
    return repo.get_article_by_canonical_url(url)["id"]


def _insert_test_sentence(repo: SQLiteRepository, art_id: int, s_idx: int, text: str) -> int:
    sentences = [{
        "article_id": art_id,
        "sentence_index": s_idx,
        "sentence_hash": f"hash_{art_id}_{s_idx}",
        "sentence_text": text,
        "is_information_bearing": True,
        "informativeness_score": 0.85,
        "named_entities": "[]",
        "has_numbers": False,
        "has_dates": False,
        "numerical_tokens": "[]",
        "token_count": len(text.split()),
    }]
    return repo.save_article_sentences(sentences)[0]


class TestTemporalEventInvariants:
    """Rigorous verification of H(e, t) = {a_i | event(a_i) = e, published_at(a_i) < t}."""

    def test_strict_temporal_inequality_no_future_leakage(self, temp_db: SQLiteRepository):
        """Verify that sentences published at or after candidate time t are excluded from H(e, t)."""
        event_id = "evt_test_temporal_001"
        temp_db.create_event(
            event_id=event_id,
            title="Climate Summit Event",
            summary="Global summit in Geneva",
            first_seen_at="2026-11-20T10:00:00Z",
            last_seen_at="2026-11-20T10:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["Geneva"],
        )

        # Article 1: epoch 1000.0
        art1_id = _insert_test_article(temp_db, 1, "2026-11-20T10:00:00Z")
        s1_id = _insert_test_sentence(temp_db, art1_id, 0, "Delegates arrived at the venue.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s1_id, "article_id": art1_id,
            "published_at": "2026-11-20T10:00:00Z", "published_epoch": 1000.0,
            "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        # Article 2: epoch 2000.0
        art2_id = _insert_test_article(temp_db, 2, "2026-11-20T10:16:40Z")
        s2_id = _insert_test_sentence(temp_db, art2_id, 0, "Talks commenced behind closed doors.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s2_id, "article_id": art2_id,
            "published_at": "2026-11-20T10:16:40Z", "published_epoch": 2000.0,
            "novelty_score": 0.9, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": s1_id, "highest_similarity": 0.4,
        }])

        # Article 3: epoch 3000.0 (future relative to candidate at 2500.0)
        art3_id = _insert_test_article(temp_db, 3, "2026-11-20T10:33:20Z")
        s3_id = _insert_test_sentence(temp_db, art3_id, 0, "Joint communique released.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s3_id, "article_id": art3_id,
            "published_at": "2026-11-20T10:33:20Z", "published_epoch": 3000.0,
            "novelty_score": 0.8, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": s2_id, "highest_similarity": 0.5,
        }])

        mem_mgr = EventMemoryManager(repository=temp_db)

        # Case A: Query at candidate epoch 2000.0
        # Invariant: Must contain ONLY sentence 1 (epoch 1000.0 < 2000.0). Sentence 2 (epoch 2000.0) MUST NOT be included!
        mem_at_2000 = mem_mgr.get_event_memory(event_id, before_epoch=2000.0)
        assert len(mem_at_2000) == 1
        assert mem_at_2000[0].sentence_id == s1_id
        assert mem_at_2000[0].published_epoch < 2000.0

        # Case B: Query at candidate epoch 2500.0
        # Must contain sentence 1 and sentence 2, but NOT sentence 3 (epoch 3000.0)
        mem_at_2500 = mem_mgr.get_event_memory(event_id, before_epoch=2500.0)
        assert len(mem_at_2500) == 2
        retrieved_ids = [m.sentence_id for m in mem_at_2500]
        assert s1_id in retrieved_ids
        assert s2_id in retrieved_ids
        assert s3_id not in retrieved_ids

    def test_tied_timestamps_deterministic_ordering(self, temp_db: SQLiteRepository):
        """When multiple articles share identical timestamps (e.g. Day-level diurnal offset),
        memory must order deterministically by secondary key (sentence_id ASC)."""
        event_id = "evt_test_tied_002"
        temp_db.create_event(
            event_id=event_id,
            title="Tied Batch Event",
            summary="Batch processing event",
            first_seen_at="2026-11-20T12:00:00Z",
            last_seen_at="2026-11-20T12:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["Batch"],
        )

        art_id = _insert_test_article(temp_db, 201, "2026-11-20T12:00:00Z")
        s1 = _insert_test_sentence(temp_db, art_id, 0, "First sentence text")
        s2 = _insert_test_sentence(temp_db, art_id, 1, "Second sentence text")
        s3 = _insert_test_sentence(temp_db, art_id, 2, "Third sentence text")

        records = [
            {"event_id": event_id, "sentence_id": sid, "article_id": art_id,
             "published_at": "2026-11-20T12:00:00Z", "published_epoch": 5000.0,
             "novelty_score": 0.5, "evidence_status": "POTENTIALLY_NEW",
             "most_similar_sentence_id": None, "highest_similarity": 0.0}
            for sid in [s1, s2, s3]
        ]
        temp_db.save_event_memory_sentences(records)

        mem_mgr = EventMemoryManager(repository=temp_db)
        mem = mem_mgr.get_event_memory(event_id, before_epoch=6000.0)

        # Invariant: Count must be 3 and ordering must be strictly ascending by sentence_id
        assert len(mem) == 3
        ids = [m.sentence_id for m in mem]
        assert ids == sorted(ids), "Tied timestamps must be deterministically ordered by sentence_id"

    def test_reordered_ingestion_invariance(self, temp_db: SQLiteRepository):
        """Reversing or shuffling the order in which sentences are inserted must NOT
        change the chronological retrieval sequence."""
        event_id = "evt_test_reorder_003"
        temp_db.create_event(
            event_id=event_id,
            title="Reorder Test Event",
            summary="Reordering ingestion event",
            first_seen_at="2026-11-20T00:00:00Z",
            last_seen_at="2026-11-20T00:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["AP"],
        )
        art_id = _insert_test_article(temp_db, 301, "2026-11-20T00:00:00Z")

        s1 = _insert_test_sentence(temp_db, art_id, 0, "Sentence 1 at epoch 100")
        s2 = _insert_test_sentence(temp_db, art_id, 1, "Sentence 2 at epoch 200")
        s3 = _insert_test_sentence(temp_db, art_id, 2, "Sentence 3 at epoch 300")

        # Insert in REVERSE chronological order: s3 (300.0), s2 (200.0), s1 (100.0)
        temp_db.save_event_memory_sentences([
            {"event_id": event_id, "sentence_id": s3, "article_id": art_id,
             "published_at": "2026-11-20T03:00:00Z", "published_epoch": 300.0,
             "novelty_score": 0.8, "evidence_status": "POTENTIALLY_NEW",
             "most_similar_sentence_id": None, "highest_similarity": 0.0},
            {"event_id": event_id, "sentence_id": s2, "article_id": art_id,
             "published_at": "2026-11-20T02:00:00Z", "published_epoch": 200.0,
             "novelty_score": 0.8, "evidence_status": "POTENTIALLY_NEW",
             "most_similar_sentence_id": None, "highest_similarity": 0.0},
            {"event_id": event_id, "sentence_id": s1, "article_id": art_id,
             "published_at": "2026-11-20T01:00:00Z", "published_epoch": 100.0,
             "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
             "most_similar_sentence_id": None, "highest_similarity": 0.0},
        ])

        mem_mgr = EventMemoryManager(repository=temp_db)
        mem = mem_mgr.get_event_memory(event_id, before_epoch=400.0)

        # Invariant: Retrieved memory must be strictly chronologically sorted (epoch 100, then 200, then 300)
        retrieved_epochs = [m.published_epoch for m in mem]
        assert retrieved_epochs == [100.0, 200.0, 300.0]
        assert [m.sentence_id for m in mem] == [s1, s2, s3]
