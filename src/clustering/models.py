"""Data models for events, clusters, and assignment decisions."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import numpy as np

from src.representation.article_representation import ArticleRepresentation


@dataclass
class EventCluster:
    """Represents a coherent real-world event and its evolving representation."""

    id: str
    title: str
    summary: str
    first_seen_at: str
    last_seen_at: str
    article_ids: List[int] = field(default_factory=list)
    article_count: int = 1
    centroid: Optional[np.ndarray] = None
    embedding_sum: Optional[np.ndarray] = None
    entities: Set[str] = field(default_factory=set)
    headline_tokens: Set[str] = field(default_factory=set)
    status: str = "EMERGING"

    def add_article(
        self,
        article: ArticleRepresentation,
        headline_tokens: Set[str],
        reference_epoch: Optional[float] = None,
    ) -> None:
        """Incrementally incorporate an assigned article into this event.

        Maintains a mathematically correct running unnormalized sum (Option A):
            S_new = S_old + v_new
            centroid = normalize(S_new)
        This guarantees equivalence with batch arithmetic mean direction without
        distortion from normalizing intermediate states.
        """
        self.article_ids.append(article.article_id)
        self.article_count = len(self.article_ids)

        # Update temporal bounds
        if article.published_at:
            if not self.last_seen_at or article.published_at > self.last_seen_at:
                self.last_seen_at = article.published_at
            if not self.first_seen_at or article.published_at < self.first_seen_at:
                self.first_seen_at = article.published_at

        # Accumulate entities and tokens
        self.entities.update(article.normalized_entities)
        self.headline_tokens.update(headline_tokens)

        # Update unnormalized embedding sum and unit centroid
        v_new = article.embedding.astype(np.float32)
        if self.embedding_sum is None:
            if self.centroid is not None and (self.article_count - 1) > 0:
                # If hydrated from persistent store with only centroid, scale by prior count
                self.embedding_sum = self.centroid.astype(np.float32) * float(self.article_count - 1) + v_new
            else:
                self.embedding_sum = v_new.copy()
        else:
            self.embedding_sum = self.embedding_sum + v_new

        norm = float(np.linalg.norm(self.embedding_sum))
        if norm > 1e-12:
            self.centroid = (self.embedding_sum / norm).astype(np.float32)
        else:
            # Handle degenerate opposing vectors: norm is zero
            self.centroid = np.zeros_like(self.embedding_sum, dtype=np.float32)

        # Update lifecycle status
        self.update_lifecycle_status(reference_epoch=reference_epoch)

    def rebuild_centroid(self, embeddings: List[np.ndarray]) -> None:
        """Safely recompute exact sum and unit centroid from all member embeddings."""
        if not embeddings:
            self.embedding_sum = None
            self.centroid = None
            return

        sum_vec = np.sum(embeddings, axis=0).astype(np.float32)
        self.embedding_sum = sum_vec
        norm = float(np.linalg.norm(sum_vec))
        if norm > 1e-12:
            self.centroid = (sum_vec / norm).astype(np.float32)
        else:
            self.centroid = np.zeros_like(sum_vec, dtype=np.float32)

    def update_lifecycle_status(
        self,
        reference_epoch: Optional[float] = None,
        active_window_hours: float = 48.0,
        dormant_window_hours: float = 168.0,
    ) -> str:
        """Update and return event lifecycle status based on temporal intervals.

        Partitions time elapsed delta = t_ref - t_last:
          - [0, active_window_hours]: EMERGING if count == 1, else ACTIVE
          - (active_window_hours, dormant_window_hours]: DORMANT
          - (dormant_window_hours, infinity): INACTIVE
          - RESOLVED: only if explicitly marked as concluded
        """
        if self.status == "RESOLVED":
            return self.status

        from src.representation.article_representation import parse_timestamp_to_epoch

        ref_epoch = (
            reference_epoch
            if reference_epoch is not None
            else datetime.now(timezone.utc).timestamp()
        )
        last_epoch = (
            parse_timestamp_to_epoch(self.last_seen_at)
            if self.last_seen_at
            else ref_epoch
        )

        elapsed_hours = max(0.0, (ref_epoch - last_epoch) / 3600.0)

        if elapsed_hours <= active_window_hours:
            self.status = "ACTIVE" if self.article_count >= 2 else "EMERGING"
        elif elapsed_hours <= dormant_window_hours:
            self.status = "DORMANT"
        else:
            self.status = "INACTIVE"

        return self.status


@dataclass
class AssignmentDecision:
    """Result of an article event assignment evaluation."""

    article_id: int
    event_id: str
    is_new_event: bool
    similarity_score: float
    is_ambiguous: bool = False
    competing_event_id: Optional[str] = None
    competing_score: Optional[float] = None
    score_breakdown: Dict[str, float] = field(default_factory=dict)
