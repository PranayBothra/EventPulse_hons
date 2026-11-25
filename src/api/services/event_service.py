"""Event clustering, exploration, and emerging-topic orchestration service."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import numpy as np

from src.clustering.engine import TemporalEventClusteringEngine
from src.novelty.emerging_topic import (
    EventPulseClusterEmergenceTracker,
    RollingWindowEmergenceDetector,
)
from src.storage.repository import SQLiteRepository


class ClusteringInProgressError(RuntimeError):
    """Raised when a clustering run is requested while one is already active."""


class EventService:
    """Exposes SQLite event exploration, incremental clustering, and emerging-topic tracking."""

    def __init__(
        self,
        repository: SQLiteRepository,
        clustering_engine: TemporalEventClusteringEngine,
        model_id: str = "all-MiniLM-L6-v2",
    ):
        self.repo = repository
        self.clustering_engine = clustering_engine
        self.model_id = model_id
        self._clustering_lock = threading.Lock()
        self._clustering_running = False

    @property
    def is_clustering_running(self) -> bool:
        return self._clustering_running

    def list_events(
        self,
        limit: int = 50,
        status: Optional[str] = None,
        sort_by: str = "article_count",
    ) -> List[Dict[str, Any]]:
        return self.repo.get_top_events_with_publisher_count(
            limit=limit, status=status, sort_by=sort_by
        )

    def get_event_articles(self, event_id: str) -> Optional[List[Dict[str, Any]]]:
        ev = self.repo.get_event_by_id(event_id)
        if not ev:
            return None
        return self.repo.get_event_articles_with_novelty(event_id)

    def run_clustering(self, include_sample: bool = False) -> Dict[str, Any]:
        """Run incremental temporal event clustering on unassigned articles under mutual exclusion."""
        if not self._clustering_lock.acquire(blocking=False):
            raise ClusteringInProgressError("A clustering job is already running.")

        self._clustering_running = True
        try:
            is_sample_filter = None if include_sample else False
            decisions = self.clustering_engine.cluster_unassigned_articles(
                is_sample=is_sample_filter
            )
            new_events = sum(1 for d in decisions if d.is_new_event)
            return {
                "status": "completed",
                "articles_clustered": len(decisions),
                "new_events_created": new_events,
                "active_events_total": len(self.clustering_engine.active_events),
                "error_message": None,
            }
        finally:
            self._clustering_running = False
            self._clustering_lock.release()

    def detect_emerging_topics(
        self,
        detector: str = "online_cluster",
        article_window: int = 400,
        min_cluster_size: int = 2,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Run existing emerging-topic detector from src/novelty/emerging_topic.py over cached embeddings."""
        recent_desc = self.repo.get_articles(limit=article_window, offset=0, is_sample=False)
        if not recent_desc:
            return {
                "detector_used": detector,
                "articles_scanned": 0,
                "topics_found": 0,
                "topics": [],
            }

        articles_chrono = list(reversed(recent_desc))
        article_ids = [a["id"] for a in articles_chrono]
        emb_map = self.repo.get_embeddings_by_article_ids(article_ids, model_id=self.model_id)

        if not emb_map:
            return {
                "detector_used": detector,
                "articles_scanned": len(articles_chrono),
                "topics_found": 0,
                "topics": [],
            }

        det_norm = detector.strip().lower()
        if det_norm == "rolling_window":
            tracker: Any = RollingWindowEmergenceDetector()
            detector_label = "RollingWindowEmergenceDetector"
        else:
            tracker = EventPulseClusterEmergenceTracker(match_threshold=0.55)
            detector_label = "EventPulseClusterEmergenceTracker"

        step_to_article: Dict[int, Dict[str, Any]] = {}
        cluster_steps: Dict[int, List[int]] = {}
        embedded_count = 0
        onset_step = max(0, len(articles_chrono) // 2)

        for step_idx, art in enumerate(articles_chrono):
            blob = emb_map.get(art["id"])
            if not blob:
                continue
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-12:
                continue
            vec = vec / norm

            embedded_count += 1
            step_to_article[step_idx] = art
            out = tracker.process_one(
                step=step_idx,
                article_id=art["id"],
                vec=vec,
                is_synthetic=False,
                scenario_id=None,
                onset_step=onset_step,
            )
            cid = out.get("cluster_id")
            if cid is not None:
                cluster_steps.setdefault(int(cid), []).append(step_idx)

        topics: List[Dict[str, Any]] = []
        for cid, steps in cluster_steps.items():
            if len(steps) < min_cluster_size:
                continue
            member_arts = [step_to_article[s] for s in steps]
            latest = member_arts[-1]
            sources = sorted({a["source_name"] for a in member_arts if a.get("source_name")})
            topics.append(
                {
                    "cluster_id": cid,
                    "detector": detector_label,
                    "article_count": len(steps),
                    "seeded_before_onset": steps[0] < onset_step,
                    "first_step": steps[0],
                    "last_step": steps[-1],
                    "representative_article_id": latest["id"],
                    "representative_headline": latest["headline"],
                    "latest_published_at": latest.get("published_at"),
                    "sources": sources,
                }
            )

        topics.sort(key=lambda t: (t["article_count"], t["last_step"]), reverse=True)
        topics = topics[:limit]
        return {
            "detector_used": detector_label,
            "articles_scanned": embedded_count,
            "topics_found": len(topics),
            "topics": topics,
        }
