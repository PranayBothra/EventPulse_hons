"""Chronological event memory management for EventPulse.

Maintains the temporal sequence of information-bearing propositions associated
with an event prior to any given timestamp, preventing lookahead leakage.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone
import numpy as np

from src.storage.repository import SQLiteRepository


@dataclass
class MemorySentence:
    """Represents a historical sentence in an event's memory."""
    sentence_id: int
    article_id: int
    published_at: str
    published_epoch: float
    sentence_text: str
    informativeness_score: float
    embedding: np.ndarray
    novelty_score: float
    evidence_status: str
    named_entities: Set[str]
    has_numbers: bool
    has_dates: bool
    timestamp_precision: str = "UNKNOWN"
    timestamp_source: Optional[str] = None
    timestamp_is_inferred: bool = False
    published_at_original: Optional[str] = None


class EventMemoryManager:
    """Manages chronological event memory H(e, t) backed by SQLite."""

    def __init__(self, repository: SQLiteRepository, model_id: str = "all-MiniLM-L6-v2"):
        self.repo = repository
        self.model_id = model_id
        # In-memory LRU-like cache of event embeddings: event_id -> List[MemorySentence]
        self._memory_cache: Dict[str, List[MemorySentence]] = {}

    def get_event_memory(
        self,
        event_id: str,
        before_epoch: Optional[float] = None,
        candidate_published_at: Optional[str] = None,
        candidate_precision: Optional[str] = None,
        candidate_is_inferred: bool = False,
        candidate_article_id: Optional[int] = None,
        mode: str = "conservative",  # "conservative", "second_only", "day_date_only", "legacy"
    ) -> List[MemorySentence]:
        """Retrieve chronological sentences for an event published before a given epoch.

        Strict Temporal Invariant:
        Only returns sentences with published_epoch < before_epoch.
        When candidate metadata is provided, enforces precision-aware chronological ordering:
        - Mode 'second_only': Evaluates only verified second-level records (epoch_m < epoch_c).
        - Mode 'day_date_only': Evaluates day-level records using date-only ordering (date_m < date_c).
        - Mode 'conservative': Unified precision-aware policy:
            - If both candidate and memory proposition have verified second-level precision: epoch_m < epoch_c.
            - If either candidate or memory has day-level precision or inferred offset: date_m < date_c.
            - Same-day records with date-level precision are co-temporal and strictly excluded.
            - Unknown timestamps are strictly excluded.
        - Mode 'legacy': Standard backward-compatible before_epoch filter.
        """
        raw_records = self.repo.get_event_memory_sentences(event_id, before_epoch=None if mode != "legacy" else before_epoch)
        if not raw_records:
            return []

        # Collect sentence IDs to fetch embeddings
        sentence_ids = [r["sentence_id"] for r in raw_records]
        embedding_blobs = self.repo.get_sentence_embeddings(sentence_ids, model_id=self.model_id)

        all_sentences: List[MemorySentence] = []
        for r in raw_records:
            sid = r["sentence_id"]
            blob = embedding_blobs.get(sid)
            if blob is not None:
                vec = np.frombuffer(blob, dtype=np.float32)
                # Ensure unit normalization
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            else:
                # Fallback zero vector if embedding missing
                vec = np.zeros(384, dtype=np.float32)

            all_sentences.append(
                MemorySentence(
                    sentence_id=sid,
                    article_id=r["article_id"],
                    published_at=r["published_at"],
                    published_epoch=r["published_epoch"],
                    sentence_text=r["sentence_text"],
                    informativeness_score=r["informativeness_score"],
                    embedding=vec,
                    novelty_score=r["novelty_score"],
                    evidence_status=r["evidence_status"],
                    named_entities=r["named_entities"],
                    has_numbers=r["has_numbers"],
                    has_dates=r["has_dates"],
                    timestamp_precision=r.get("timestamp_precision", "UNKNOWN"),
                    timestamp_source=r.get("timestamp_source"),
                    timestamp_is_inferred=bool(r.get("timestamp_is_inferred", False)),
                    published_at_original=r.get("published_at_original"),
                )
            )

        # Sort strictly chronologically by (epoch, sentence_id) for deterministic sequencing
        all_sentences.sort(key=lambda s: (s.published_epoch, s.sentence_id))

        if mode == "legacy" or (candidate_published_at is None and candidate_precision is None):
            # Backward compatibility mode
            if before_epoch is None:
                return all_sentences
            return [
                s for s in all_sentences
                if s.published_epoch < before_epoch and (candidate_article_id is None or s.article_id != candidate_article_id)
            ]

        # Precision-Aware Temporal Replay Filtering
        cand_epoch = before_epoch
        if cand_epoch is None and candidate_published_at:
            try:
                dt = datetime.fromisoformat(candidate_published_at.replace("Z", "+00:00"))
                cand_epoch = dt.timestamp()
            except Exception:
                cand_epoch = None

        cand_date = candidate_published_at[:10] if candidate_published_at else ""
        cand_prec = candidate_precision or "UNKNOWN"

        if cand_prec == "UNKNOWN":
            return []

        filtered_sentences: List[MemorySentence] = []
        for s in all_sentences:
            # 1. Exclude self-matches
            if candidate_article_id is not None and s.article_id == candidate_article_id:
                continue

            # 2. Exclude unknown timestamps
            if s.timestamp_precision == "UNKNOWN":
                continue

            # 3. Apply mode-specific rules
            if mode == "second_only":
                if cand_prec != "SECOND" or s.timestamp_precision != "SECOND" or candidate_is_inferred or s.timestamp_is_inferred:
                    continue
                if cand_epoch is not None and s.published_epoch < cand_epoch:
                    filtered_sentences.append(s)

            elif mode == "day_date_only":
                if cand_prec != "DAY" or s.timestamp_precision != "DAY":
                    continue
                mem_date = s.published_at[:10]
                if mem_date < cand_date:
                    filtered_sentences.append(s)

            elif mode == "conservative":
                # Combined precision-aware policy
                if cand_prec == "SECOND" and s.timestamp_precision == "SECOND" and not candidate_is_inferred and not s.timestamp_is_inferred:
                    # Verified second-level precision on both sides
                    if cand_epoch is not None and s.published_epoch < cand_epoch:
                        filtered_sentences.append(s)
                else:
                    # Day-level precision or inferred diurnal offset on either side
                    mem_date = s.published_at[:10]
                    # Strictly earlier calendar date required
                    if mem_date < cand_date:
                        filtered_sentences.append(s)

        return filtered_sentences

    def commit_sentences_to_memory(
        self,
        event_id: str,
        memory_records: List[Dict[str, Any]],
    ) -> int:
        """Persist evaluated article sentences to the event's permanent memory."""
        if not memory_records:
            return 0
        return self.repo.save_event_memory_sentences(memory_records)
