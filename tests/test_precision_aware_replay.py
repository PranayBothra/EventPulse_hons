"""
Unit and invariant tests for Precision-Aware Temporal Replay (Stage 3.5).
Verifies:
1. Same-day records do not enter each other's memory when precision is DAY.
2. Earlier calendar dates are allowed.
3. Genuine earlier second-level records are allowed.
4. Equal timestamps do not establish precedence.
5. Artificial standardized diurnal offsets do not affect chronology.
6. Missing/unknown timestamps are excluded.
7. Reordering database ingestion does not change chronological replay.
"""

from datetime import datetime, timezone
import numpy as np
import pytest

from src.storage.repository import SQLiteRepository
from src.novelty.event_memory import EventMemoryManager


def _insert_test_article(
    repo: SQLiteRepository,
    idx: int,
    pub_at: str,
    precision: str = "DAY",
    is_inferred: bool = True,
    source_id: str = "test_source",
) -> int:
    url = f"https://example.com/news/precision/{idx}"
    article = {
        "canonical_url": url,
        "url": url,
        "headline": f"Precision News Headline {idx}",
        "headline_clean": f"precision news headline {idx}",
        "headline_hash": f"hash_prec_{idx}",
        "description": f"Description {idx}",
        "content_clean": f"Content {idx}",
        "source_id": source_id,
        "source_name": "Test Source",
        "published_at": pub_at,
        "retrieved_at": pub_at,
        "is_sample": 0,
        "raw_metadata": "{}",
        "timestamp_precision": precision,
        "timestamp_source": "test_provenance",
        "timestamp_timezone": "UTC",
        "timestamp_is_inferred": 1 if is_inferred else 0,
        "published_at_original": pub_at[:10] if precision == "DAY" else pub_at,
    }
    repo.insert_article(article)
    return repo.get_article_by_canonical_url(url)["id"]


def _insert_test_sentence(repo: SQLiteRepository, art_id: int, s_idx: int, text: str) -> int:
    sentences = [{
        "article_id": art_id,
        "sentence_index": s_idx,
        "sentence_hash": f"hash_s_{art_id}_{s_idx}",
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


class TestPrecisionAwareReplay:
    """Rigorous verification of Stage 3.5 precision-aware temporal invariants."""

    def test_same_day_records_day_precision_mutually_excluded(self, temp_db: SQLiteRepository):
        """Invariant 1: Two articles published on the same calendar day with precision DAY
        must be treated as simultaneous/co-temporal and mutually excluded from memory."""
        event_id = "evt_same_day_001"
        temp_db.create_event(
            event_id=event_id,
            title="Same Day Event",
            summary="Testing same-day mutual exclusion",
            first_seen_at="2020-04-13T12:00:00Z",
            last_seen_at="2020-04-13T12:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["Event"],
        )

        art1_id = _insert_test_article(temp_db, 101, "2020-04-13T12:00:00Z", precision="DAY", is_inferred=True)
        s1 = _insert_test_sentence(temp_db, art1_id, 0, "Morning briefing occurred.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s1, "article_id": art1_id,
            "published_at": "2020-04-13T12:00:00Z", "published_epoch": 1586779200.0,
            "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        art2_id = _insert_test_article(temp_db, 102, "2020-04-13T12:00:00Z", precision="DAY", is_inferred=True)
        s2 = _insert_test_sentence(temp_db, art2_id, 0, "Afternoon remarks delivered.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s2, "article_id": art2_id,
            "published_at": "2020-04-13T12:00:00Z", "published_epoch": 1586779200.0,
            "novelty_score": 0.9, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": s1, "highest_similarity": 0.3,
        }])

        mem_mgr = EventMemoryManager(repository=temp_db)

        # Evaluating Article 2: Candidate is on 2020-04-13 with precision DAY
        # Invariant: Article 1 (also on 2020-04-13) MUST NOT enter Article 2's memory
        mem_for_art2 = mem_mgr.get_event_memory(
            event_id=event_id,
            candidate_published_at="2020-04-13T12:00:00Z",
            candidate_precision="DAY",
            candidate_is_inferred=True,
            candidate_article_id=art2_id,
            mode="conservative",
        )
        assert len(mem_for_art2) == 0, "Same-day records with precision DAY must not enter memory"

    def test_earlier_calendar_date_allowed(self, temp_db: SQLiteRepository):
        """Invariant 2: When candidate is on date D_c, records from strictly earlier
        calendar dates (D_m < D_c) must enter memory."""
        event_id = "evt_earlier_date_002"
        temp_db.create_event(
            event_id=event_id,
            title="Multi-Day Story",
            summary="Story spanning consecutive days",
            first_seen_at="2020-04-12T12:00:00Z",
            last_seen_at="2020-04-13T12:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["Story"],
        )

        art1_id = _insert_test_article(temp_db, 201, "2020-04-12T12:00:00Z", precision="DAY", is_inferred=True)
        s1 = _insert_test_sentence(temp_db, art1_id, 0, "Day 1 development revealed.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s1, "article_id": art1_id,
            "published_at": "2020-04-12T12:00:00Z", "published_epoch": 1586692800.0,
            "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        art2_id = _insert_test_article(temp_db, 202, "2020-04-13T12:00:00Z", precision="DAY", is_inferred=True)

        mem_mgr = EventMemoryManager(repository=temp_db)

        # Evaluating Article 2 on 2020-04-13
        mem = mem_mgr.get_event_memory(
            event_id=event_id,
            candidate_published_at="2020-04-13T12:00:00Z",
            candidate_precision="DAY",
            candidate_is_inferred=True,
            candidate_article_id=art2_id,
            mode="conservative",
        )
        assert len(mem) == 1
        assert mem[0].sentence_id == s1
        assert mem[0].published_at[:10] == "2020-04-12"

    def test_genuine_earlier_second_level_records_allowed(self, temp_db: SQLiteRepository):
        """Invariant 3: Genuine second-level timestamps (Live RSS) with verified authentic
        seconds (is_inferred=False) admit earlier seconds."""
        event_id = "evt_second_level_003"
        temp_db.create_event(
            event_id=event_id,
            title="Live Breaking News",
            summary="Sub-hourly updates from live RSS",
            first_seen_at="2026-11-20T10:00:00Z",
            last_seen_at="2026-11-20T10:15:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["Live"],
        )

        art1_id = _insert_test_article(temp_db, 301, "2026-11-20T10:00:00Z", precision="SECOND", is_inferred=False)
        s1 = _insert_test_sentence(temp_db, art1_id, 0, "Explosion reported at the docks.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s1, "article_id": art1_id,
            "published_at": "2026-11-20T10:00:00Z", "published_epoch": 1795168800.0,
            "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        art2_id = _insert_test_article(temp_db, 302, "2026-11-20T10:15:00Z", precision="SECOND", is_inferred=False)

        mem_mgr = EventMemoryManager(repository=temp_db)

        # Evaluating Article 2 at 10:15:00Z
        mem = mem_mgr.get_event_memory(
            event_id=event_id,
            before_epoch=1795169700.0,
            candidate_published_at="2026-11-20T10:15:00Z",
            candidate_precision="SECOND",
            candidate_is_inferred=False,
            candidate_article_id=art2_id,
            mode="second_only",
        )
        assert len(mem) == 1
        assert mem[0].sentence_id == s1

    def test_equal_timestamps_do_not_establish_precedence(self, temp_db: SQLiteRepository):
        """Invariant 4: Simultaneous / equal timestamps must not establish chronological precedence."""
        event_id = "evt_equal_time_004"
        temp_db.create_event(
            event_id=event_id,
            title="Simultaneous Event",
            summary="Equal timestamps",
            first_seen_at="2026-11-20T10:00:00Z",
            last_seen_at="2026-11-20T10:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["Equal"],
        )

        art1_id = _insert_test_article(temp_db, 401, "2026-11-20T10:00:00Z", precision="SECOND", is_inferred=False)
        s1 = _insert_test_sentence(temp_db, art1_id, 0, "Wire feed announcement A.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s1, "article_id": art1_id,
            "published_at": "2026-11-20T10:00:00Z", "published_epoch": 1795168800.0,
            "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        art2_id = _insert_test_article(temp_db, 402, "2026-11-20T10:00:00Z", precision="SECOND", is_inferred=False)

        mem_mgr = EventMemoryManager(repository=temp_db)

        # Evaluating Article 2 at identical epoch 1795168800.0
        mem = mem_mgr.get_event_memory(
            event_id=event_id,
            before_epoch=1795168800.0,
            candidate_published_at="2026-11-20T10:00:00Z",
            candidate_precision="SECOND",
            candidate_is_inferred=False,
            candidate_article_id=art2_id,
            mode="conservative",
        )
        assert len(mem) == 0, "Equal timestamps must never establish precedence"

    def test_artificial_standardized_diurnal_offsets_do_not_affect_chronology(self, temp_db: SQLiteRepository):
        """Invariant 5: Artificial publisher offsets (e.g. ABC 08:00 vs HuffPost 12:00)
        on the exact same calendar date must NOT create false chronological precedence."""
        event_id = "evt_diurnal_offset_005"
        temp_db.create_event(
            event_id=event_id,
            title="Cross-Publisher Story",
            summary="ABC vs HuffPost on same date",
            first_seen_at="2019-11-18T08:00:00Z",
            last_seen_at="2019-11-18T12:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["CrossPublisher"],
        )

        # Article 1 from ABC News Australia: assigned standardized T08:00:00Z
        art1_id = _insert_test_article(temp_db, 501, "2019-11-18T08:00:00Z", precision="DAY", is_inferred=True, source_id="abc_news_australia")
        s1 = _insert_test_sentence(temp_db, art1_id, 0, "Bushfires intensify across NSW.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s1, "article_id": art1_id,
            "published_at": "2019-11-18T08:00:00Z", "published_epoch": 1574064000.0,
            "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        # Article 2 from HuffPost: assigned standardized T12:00:00Z on the same date (2019-11-18)
        art2_id = _insert_test_article(temp_db, 502, "2019-11-18T12:00:00Z", precision="DAY", is_inferred=True, source_id="www_huffpost_com")

        mem_mgr = EventMemoryManager(repository=temp_db)

        # Under naive replay, 08:00 < 12:00 would have admitted s1.
        # Under precision-aware replay, both are DAY precision on 2019-11-18, so s1 MUST NOT be admitted!
        mem = mem_mgr.get_event_memory(
            event_id=event_id,
            candidate_published_at="2019-11-18T12:00:00Z",
            candidate_precision="DAY",
            candidate_is_inferred=True,
            candidate_article_id=art2_id,
            mode="conservative",
        )
        assert len(mem) == 0, "Artificial diurnal offset (08:00 vs 12:00) must not establish chronological precedence"

    def test_missing_unknown_timestamps_excluded(self, temp_db: SQLiteRepository):
        """Invariant 6: Articles with UNKNOWN or missing precision are excluded from chronological replay."""
        event_id = "evt_unknown_time_006"
        temp_db.create_event(
            event_id=event_id,
            title="Unknown Timestamp Event",
            summary="Testing unknown handling",
            first_seen_at="2020-01-01T00:00:00Z",
            last_seen_at="2020-01-01T00:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["Unknown"],
        )

        art1_id = _insert_test_article(temp_db, 601, "2020-01-01T00:00:00Z", precision="UNKNOWN", is_inferred=False)
        s1 = _insert_test_sentence(temp_db, art1_id, 0, "Undated news snippet.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s1, "article_id": art1_id,
            "published_at": "2020-01-01T00:00:00Z", "published_epoch": 1577836800.0,
            "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        art2_id = _insert_test_article(temp_db, 602, "2020-01-02T12:00:00Z", precision="DAY", is_inferred=True)

        mem_mgr = EventMemoryManager(repository=temp_db)
        mem = mem_mgr.get_event_memory(
            event_id=event_id,
            candidate_published_at="2020-01-02T12:00:00Z",
            candidate_precision="DAY",
            candidate_is_inferred=True,
            candidate_article_id=art2_id,
            mode="conservative",
        )
        assert len(mem) == 0, "Propositions with precision UNKNOWN must be excluded from memory"

    def test_reordered_ingestion_invariance_precision_aware(self, temp_db: SQLiteRepository):
        """Invariant 7: Inverting or shuffling the physical row insertion order does not
        alter precision-aware chronological retrieval."""
        event_id = "evt_reorder_prec_007"
        temp_db.create_event(
            event_id=event_id,
            title="Reordered Precision Event",
            summary="Testing order invariance",
            first_seen_at="2020-04-10T12:00:00Z",
            last_seen_at="2020-04-14T12:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["Order"],
        )

        art1_id = _insert_test_article(temp_db, 701, "2020-04-10T12:00:00Z", precision="DAY", is_inferred=True)
        art2_id = _insert_test_article(temp_db, 702, "2020-04-12T12:00:00Z", precision="DAY", is_inferred=True)

        s1 = _insert_test_sentence(temp_db, art1_id, 0, "Sentence from April 10")
        s2 = _insert_test_sentence(temp_db, art2_id, 0, "Sentence from April 12")

        # Save in reverse physical order: s2 first, then s1
        temp_db.save_event_memory_sentences([
            {"event_id": event_id, "sentence_id": s2, "article_id": art2_id,
             "published_at": "2020-04-12T12:00:00Z", "published_epoch": 1586692800.0,
             "novelty_score": 0.8, "evidence_status": "POTENTIALLY_NEW",
             "most_similar_sentence_id": None, "highest_similarity": 0.0},
            {"event_id": event_id, "sentence_id": s1, "article_id": art1_id,
             "published_at": "2020-04-10T12:00:00Z", "published_epoch": 1586520000.0,
             "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
             "most_similar_sentence_id": None, "highest_similarity": 0.0},
        ])

        mem_mgr = EventMemoryManager(repository=temp_db)
        mem = mem_mgr.get_event_memory(
            event_id=event_id,
            candidate_published_at="2020-04-14T12:00:00Z",
            candidate_precision="DAY",
            candidate_is_inferred=True,
            candidate_article_id=999,
            mode="conservative",
        )
        assert len(mem) == 2
        assert [m.sentence_id for m in mem] == [s1, s2], "Retrieval order must be strictly chronological by date and epoch regardless of insertion order"

    def test_day_vs_second_cross_date_and_same_date_pairwise(self, temp_db: SQLiteRepository):
        """Invariant 8: Rigorous verification of partial temporal order for mixed DAY and SECOND pairs.
        - Cross-date (mem date < cand date) is ADMITTED.
        - Same-date (mem date == cand date) is EXCLUDED due to unknown sub-day ordering."""
        event_id = "evt_mixed_pairwise_008"
        temp_db.create_event(
            event_id=event_id,
            title="Mixed Precision Event",
            summary="Testing mixed DAY and SECOND interactions",
            first_seen_at="2026-11-20T00:00:00Z",
            last_seen_at="2026-11-21T18:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["MixedPrecision"],
        )

        # Article A: DAY precision on 2026-11-20
        art_a = _insert_test_article(temp_db, 801, "2026-11-20T12:00:00Z", precision="DAY", is_inferred=True)
        s_a = _insert_test_sentence(temp_db, art_a, 0, "Day-level news on Nov 20.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s_a, "article_id": art_a,
            "published_at": "2026-11-20T12:00:00Z", "published_epoch": 1795176000.0,
            "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        # Article B: SECOND precision on 2026-11-20T18:00:00Z (same day as A)
        art_b = _insert_test_article(temp_db, 802, "2026-11-20T18:00:00Z", precision="SECOND", is_inferred=False)
        s_b = _insert_test_sentence(temp_db, art_b, 0, "Second-level evening flash on Nov 20.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s_b, "article_id": art_b,
            "published_at": "2026-11-20T18:00:00Z", "published_epoch": 1795197600.0,
            "novelty_score": 0.9, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        # Article C: SECOND precision on 2026-11-21T09:00:00Z (following day)
        art_c = _insert_test_article(temp_db, 803, "2026-11-21T09:00:00Z", precision="SECOND", is_inferred=False)
        s_c = _insert_test_sentence(temp_db, art_c, 0, "Second-level morning flash on Nov 21.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s_c, "article_id": art_c,
            "published_at": "2026-11-21T09:00:00Z", "published_epoch": 1795251600.0,
            "novelty_score": 0.85, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        # Article D: DAY precision on 2026-11-22
        art_d = _insert_test_article(temp_db, 804, "2026-11-22T12:00:00Z", precision="DAY", is_inferred=True)

        mem_mgr = EventMemoryManager(repository=temp_db)

        # Case 1: Candidate is B (SECOND, Nov 20). Memory has A (DAY, Nov 20).
        # Same calendar date + DAY precision -> A MUST NOT enter B's memory
        mem_b = mem_mgr.get_event_memory(
            event_id=event_id,
            candidate_published_at="2026-11-20T18:00:00Z",
            candidate_precision="SECOND",
            candidate_is_inferred=False,
            candidate_article_id=art_b,
            mode="conservative",
        )
        assert len(mem_b) == 0, "Same-day DAY record must be excluded from candidate SECOND memory"

        # Case 2: Candidate is C (SECOND, Nov 21).
        # Nov 20 is strictly earlier date (< Nov 21), so BOTH A (DAY) and B (SECOND) MUST enter memory
        mem_c = mem_mgr.get_event_memory(
            event_id=event_id,
            candidate_published_at="2026-11-21T09:00:00Z",
            candidate_precision="SECOND",
            candidate_is_inferred=False,
            candidate_article_id=art_c,
            mode="conservative",
        )
        assert len(mem_c) == 2, "Cross-date earlier records (DAY and SECOND) must enter memory"
        assert {m.sentence_id for m in mem_c} == {s_a, s_b}

        # Case 3: Candidate is D (DAY, Nov 22).
        # Both Nov 20 (A, B) and Nov 21 (C) are strictly earlier dates -> All 3 enter memory
        mem_d = mem_mgr.get_event_memory(
            event_id=event_id,
            candidate_published_at="2026-11-22T12:00:00Z",
            candidate_precision="DAY",
            candidate_is_inferred=True,
            candidate_article_id=art_d,
            mode="conservative",
        )
        assert len(mem_d) == 3, "Candidate DAY on later date must admit earlier DAY and SECOND records"
        assert {m.sentence_id for m in mem_d} == {s_a, s_b, s_c}

    def test_unknown_precision_mutual_isolation(self, temp_db: SQLiteRepository):
        """Invariant 9: UNKNOWN precision candidate gets zero memory, and UNKNOWN propositions
        never enter any candidate's memory."""
        event_id = "evt_unknown_isolation_009"
        temp_db.create_event(
            event_id=event_id,
            title="Unknown Precision Event",
            summary="Testing unknown isolation",
            first_seen_at="2026-01-01T00:00:00Z",
            last_seen_at="2026-01-02T00:00:00Z",
            centroid_blob=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            key_entities=["UnknownIsolation"],
        )

        art_known = _insert_test_article(temp_db, 901, "2026-01-01T12:00:00Z", precision="DAY", is_inferred=True)
        s_known = _insert_test_sentence(temp_db, art_known, 0, "Known date article sentence.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s_known, "article_id": art_known,
            "published_at": "2026-01-01T12:00:00Z", "published_epoch": 1767268800.0,
            "novelty_score": 1.0, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        art_unk = _insert_test_article(temp_db, 902, "2026-01-01T12:00:00Z", precision="UNKNOWN", is_inferred=False)
        s_unk = _insert_test_sentence(temp_db, art_unk, 0, "Undated sentence.")
        temp_db.save_event_memory_sentences([{
            "event_id": event_id, "sentence_id": s_unk, "article_id": art_unk,
            "published_at": "2026-01-01T12:00:00Z", "published_epoch": 1767268800.0,
            "novelty_score": 0.5, "evidence_status": "POTENTIALLY_NEW",
            "most_similar_sentence_id": None, "highest_similarity": 0.0,
        }])

        mem_mgr = EventMemoryManager(repository=temp_db)

        # Candidate with UNKNOWN precision must receive empty list
        mem_for_unk = mem_mgr.get_event_memory(
            event_id=event_id,
            candidate_published_at="2026-01-02T12:00:00Z",
            candidate_precision="UNKNOWN",
            candidate_is_inferred=False,
            candidate_article_id=art_unk,
            mode="conservative",
        )
        assert len(mem_for_unk) == 0

        # Known candidate on Jan 2 must receive s_known, but NOT s_unk
        art_cand = _insert_test_article(temp_db, 903, "2026-01-02T12:00:00Z", precision="DAY", is_inferred=True)
        mem_for_known = mem_mgr.get_event_memory(
            event_id=event_id,
            candidate_published_at="2026-01-02T12:00:00Z",
            candidate_precision="DAY",
            candidate_is_inferred=True,
            candidate_article_id=art_cand,
            mode="conservative",
        )
        assert len(mem_for_known) == 1
        assert mem_for_known[0].sentence_id == s_known
