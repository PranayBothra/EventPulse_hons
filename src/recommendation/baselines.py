"""Implementation of 4 independent recommendation baselines for comparative evaluation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from src.personalisation.user_profiles import UserProfileManager
from src.recommendation.ranking import (
    EventRecommendation,
    NoveltyAwareRecommender,
    RecommendationWeights,
)
from src.storage.repository import SQLiteRepository


class BaseRecommender(ABC):
    """Abstract base class for all recommendation models in EventPulse."""

    @abstractmethod
    def rank_events(
        self, user_id: str, current_epoch: Optional[float] = None, top_k: int = 10
    ) -> List[EventRecommendation]:
        """Rank events for a given user."""
        pass


class ChronologicalRecommender(BaseRecommender):
    """Baseline A: Ranks events strictly by temporal recency (publication date)."""

    def __init__(self, repository: SQLiteRepository):
        self.repo = repository

    def rank_events(
        self, user_id: str, current_epoch: Optional[float] = None, top_k: int = 10
    ) -> List[EventRecommendation]:
        query = """
        SELECT e.id, e.title, e.summary, e.last_seen_at,
               a.id AS article_id, a.headline, a.source_name, a.published_at
        FROM events e
        JOIN event_articles ea ON ea.event_id = e.id
        JOIN articles a ON ea.article_id = a.id
        WHERE e.article_count > 0
        GROUP BY e.id
        ORDER BY e.last_seen_at DESC
        LIMIT ?;
        """
        with self.repo._get_connection() as conn:
            rows = conn.execute(query, (top_k,)).fetchall()

        recommendations = []
        for r in rows:
            recommendations.append(
                EventRecommendation(
                    event_id=r["id"],
                    title=r["title"],
                    summary=r["summary"] or "",
                    representative_article_id=r["article_id"],
                    article_headline=r["headline"],
                    source_name=r["source_name"],
                    published_at=r["published_at"],
                    published_epoch=0.0,
                    score=1.0,
                    score_breakdown={"recency_rank": 1.0},
                    explanation="Recommended purely by publication recency (Baseline A: Chronological).",
                )
            )
        return recommendations


class ContentBasedRecommender(BaseRecommender):
    """Baseline B: Standard Content-Based Filtering using cosine similarity to user centroid.

    Does not incorporate temporal event clustering, novelty, freshness decay, or exposure penalties.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        profile_manager: Optional[UserProfileManager] = None,
        model_id: str = "all-MiniLM-L6-v2",
    ):
        self.repo = repository
        self.profile_manager = profile_manager or UserProfileManager(repository, model_id=model_id)
        self.model_id = model_id

    def rank_events(
        self, user_id: str, current_epoch: Optional[float] = None, top_k: int = 10
    ) -> List[EventRecommendation]:
        profile = self.profile_manager.get_profile(user_id)
        user_vec = profile.long_term_vector

        # If user has no history, fall back to chronological
        if user_vec is None:
            return ChronologicalRecommender(self.repo).rank_events(user_id, current_epoch, top_k)

        query = """
        SELECT e.id, e.title, e.summary, e.centroid_blob, e.last_seen_at,
               a.id AS article_id, a.headline, a.source_name, a.published_at
        FROM events e
        JOIN event_articles ea ON ea.event_id = e.id
        JOIN articles a ON ea.article_id = a.id
        WHERE e.article_count > 0 AND e.centroid_blob IS NOT NULL
        GROUP BY e.id
        LIMIT 100;
        """
        with self.repo._get_connection() as conn:
            rows = conn.execute(query).fetchall()

        scored = []
        for r in rows:
            blob = r["centroid_blob"]
            vec = np.frombuffer(blob, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            cos_sim = float(np.dot(user_vec, vec))
            scored.append((cos_sim, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        recommendations = []
        for sim, r in scored[:top_k]:
            recommendations.append(
                EventRecommendation(
                    event_id=r["id"],
                    title=r["title"],
                    summary=r["summary"] or "",
                    representative_article_id=r["article_id"],
                    article_headline=r["headline"],
                    source_name=r["source_name"],
                    published_at=r["published_at"],
                    published_epoch=0.0,
                    score=round(sim, 4),
                    score_breakdown={"content_similarity": round(sim, 4)},
                    explanation="Recommended based on static semantic similarity to past reading history (Baseline B: Content-Based).",
                )
            )
        return recommendations


class EventAwareWithoutNoveltyRecommender(BaseRecommender):
    """Baseline C: Event-aware recommender combining topic relevance and freshness, but without novelty awareness.

    Disables event novelty, user novelty, exposure penalty, and redundancy penalty.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        profile_manager: Optional[UserProfileManager] = None,
        model_id: str = "all-MiniLM-L6-v2",
    ):
        weights = RecommendationWeights(
            w_long=0.50,
            w_short=0.0,
            w_event_novelty=0.0,
            w_user_novelty=0.0,
            w_freshness=0.50,
            w_source_diversity=0.0,
            w_exposure_penalty=0.0,
            w_redundancy_penalty=0.0,
            mmr_lambda=1.0,  # No diversity reranking
        )
        self.engine = NoveltyAwareRecommender(
            repository=repository,
            profile_manager=profile_manager,
            weights=weights,
            model_id=model_id,
        )

    def rank_events(
        self, user_id: str, current_epoch: Optional[float] = None, top_k: int = 10
    ) -> List[EventRecommendation]:
        recs = self.engine.rank_events_for_user(
            user_id=user_id,
            current_epoch=current_epoch,
            top_k=top_k,
            apply_mmr=False,
        )
        for r in recs:
            r.explanation = "Recommended based on event topic match and freshness (Baseline C: Event-Aware without Novelty)."
        return recs


class ProposedEventPulseRecommender(BaseRecommender):
    """Proposed Model: Full temporal event intelligence with dynamic dual user profiling,

    incremental information novelty, exposure control, and MMR diversity reranking.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        profile_manager: Optional[UserProfileManager] = None,
        weights: Optional[RecommendationWeights] = None,
        model_id: str = "all-MiniLM-L6-v2",
    ):
        self.engine = NoveltyAwareRecommender(
            repository=repository,
            profile_manager=profile_manager,
            weights=weights or RecommendationWeights(),
            model_id=model_id,
        )

    def rank_events(
        self, user_id: str, current_epoch: Optional[float] = None, top_k: int = 10
    ) -> List[EventRecommendation]:
        return self.engine.rank_events_for_user(
            user_id=user_id,
            current_epoch=current_epoch,
            top_k=top_k,
            apply_mmr=True,
        )
