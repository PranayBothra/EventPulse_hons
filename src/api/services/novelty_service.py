"""Incremental sentence-level novelty inspection service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.novelty.event_memory import EventMemoryManager
from src.novelty.novelty_aggregation import EventNoveltyEngine
from src.storage.repository import SQLiteRepository


class NoveltyService:
    """Wraps EventNoveltyEngine and EventMemoryManager for API inspection."""

    def __init__(
        self,
        repository: SQLiteRepository,
        novelty_engine: EventNoveltyEngine,
        memory_manager: Optional[EventMemoryManager] = None,
    ):
        self.repo = repository
        self.novelty_engine = novelty_engine
        self.memory_manager = memory_manager or novelty_engine.memory_manager

    def list_evolving_events(self, min_articles: int = 2, limit: int = 30) -> List[Dict[str, Any]]:
        return self.repo.get_evolving_events(min_articles=min_articles, limit=limit)

    def inspect_article_novelty(
        self,
        event_id: str,
        article_id: Optional[int] = None,
        persist: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate sentence novelty of a target (or latest) article against chronological memory H(e,t)."""
        event_row = self.repo.get_event_by_id(event_id)
        if not event_row:
            return None

        articles_in_event = self.repo.get_event_articles(event_id)
        if not articles_in_event:
            return None

        target_art = None
        if article_id is not None:
            target_art = next((a for a in articles_in_event if a["id"] == article_id), None)
            if target_art is None:
                return None
        else:
            target_art = articles_in_event[-1]

        pub_at = target_art.get("published_at") or ""
        try:
            pub_epoch = datetime.fromisoformat(pub_at.replace("Z", "+00:00")).timestamp()
        except Exception:
            pub_epoch = 0.0

        prior_mem = self.memory_manager.get_event_memory(event_id, before_epoch=pub_epoch)
        summary = self.novelty_engine.evaluate_article_novelty(
            article_id=target_art["id"],
            event_id=event_id,
            published_at=pub_at,
            published_epoch=pub_epoch,
            persist=persist,
        )

        return {
            "article_id": summary.article_id,
            "event_id": summary.event_id,
            "article_headline": target_art["headline"],
            "source_name": target_art.get("source_name") or "",
            "published_at": pub_at,
            "event_novelty_score": summary.event_novelty_score,
            "novel_sentence_count": summary.novel_sentence_count,
            "redundant_sentence_count": summary.redundant_sentence_count,
            "insufficient_evidence_count": summary.insufficient_evidence_count,
            "total_sentence_count": summary.total_sentence_count,
            "novelty_ratio": summary.novelty_ratio,
            "prior_memory_sentence_count": len(prior_mem),
            "sentence_results": [
                {
                    "sentence_id": sr.sentence_id,
                    "article_id": sr.article_id,
                    "sentence_index": sr.sentence_index,
                    "sentence_text": sr.sentence_text,
                    "evidence_status": sr.evidence_status,
                    "informativeness_score": round(float(sr.informativeness_score), 4),
                    "highest_similarity": round(float(sr.highest_similarity), 4),
                    "novelty_score": round(float(sr.novelty_score), 4),
                    "most_similar_sentence_id": sr.most_similar_sentence_id,
                    "most_similar_text": sr.most_similar_text,
                }
                for sr in summary.sentence_results
            ],
        }
