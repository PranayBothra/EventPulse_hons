"""Temporal Novelty-Aware Recommendation Engine with MMR Reranking for EventPulse."""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.personalisation.user_novelty import UserNoveltyEngine
from src.personalisation.user_profiles import UserProfile, UserProfileManager
from src.storage.repository import SQLiteRepository


class RecommendationMode(str, Enum):
    """Adaptive recommendation operational modes."""
    RELEVANCE_FOCUSED = "RELEVANCE_FOCUSED"
    BALANCED_INTELLIGENCE = "BALANCED_INTELLIGENCE"
    DISCOVERY_FOCUSED = "DISCOVERY_FOCUSED"


@dataclass
class RecommendationWeights:
    """Configurable weights for the multi-objective event ranking function."""
    w_long: float = 0.25
    w_short: float = 0.25
    w_event_novelty: float = 0.15
    w_user_novelty: float = 0.10
    w_freshness: float = 0.10
    w_source_diversity: float = 0.05
    w_exposure_penalty: float = 0.10
    w_redundancy_penalty: float = 0.10
    freshness_half_life_seconds: float = 86400.0  # 24 hours
    exposure_lambda: float = 0.5
    mmr_lambda: float = 0.80

    @classmethod
    def for_mode(cls, mode: RecommendationMode) -> "RecommendationWeights":
        """Factory method returning empirical parameter configurations for each adaptive mode."""
        if mode == RecommendationMode.RELEVANCE_FOCUSED:
            return cls(
                w_long=0.40,
                w_short=0.35,
                w_event_novelty=0.05,
                w_user_novelty=0.05,
                w_freshness=0.10,
                w_source_diversity=0.05,
                w_exposure_penalty=0.05,
                w_redundancy_penalty=0.05,
                mmr_lambda=0.95,
            )
        elif mode == RecommendationMode.DISCOVERY_FOCUSED:
            return cls(
                w_long=0.15,
                w_short=0.15,
                w_event_novelty=0.25,
                w_user_novelty=0.20,
                w_freshness=0.10,
                w_source_diversity=0.10,
                w_exposure_penalty=0.15,
                w_redundancy_penalty=0.20,
                mmr_lambda=0.65,
            )
        else:  # BALANCED_INTELLIGENCE (Recommended Default)
            return cls(
                w_long=0.25,
                w_short=0.25,
                w_event_novelty=0.15,
                w_user_novelty=0.10,
                w_freshness=0.10,
                w_source_diversity=0.05,
                w_exposure_penalty=0.10,
                w_redundancy_penalty=0.10,
                mmr_lambda=0.80,
            )


@dataclass
class EventRecommendation:
    """Represents a scored, explained event recommendation."""
    event_id: str
    title: str
    summary: str
    representative_article_id: int
    article_headline: str
    source_name: str
    published_at: str
    published_epoch: float
    score: float
    score_breakdown: Dict[str, float]
    explanation: str


class NoveltyAwareRecommender:
    """Personalised news recommender integrating temporal events, information novelty, and MMR."""

    def __init__(
        self,
        repository: SQLiteRepository,
        profile_manager: Optional[UserProfileManager] = None,
        novelty_engine: Optional[UserNoveltyEngine] = None,
        weights: Optional[RecommendationWeights] = None,
        mode: Optional[RecommendationMode] = None,
        model_id: str = "all-MiniLM-L6-v2",
    ):
        self.repo = repository
        self.model_id = model_id
        self.profile_manager = profile_manager or UserProfileManager(repository, model_id=model_id)
        self.novelty_engine = novelty_engine or UserNoveltyEngine(repository, model_id=model_id)
        if weights is not None:
            self.weights = weights
        elif mode is not None:
            self.weights = RecommendationWeights.for_mode(mode)
        else:
            self.weights = RecommendationWeights.for_mode(RecommendationMode.BALANCED_INTELLIGENCE)
        self.mode = mode or RecommendationMode.BALANCED_INTELLIGENCE
        self.freshness_decay_constant = math.log(2.0) / max(self.weights.freshness_half_life_seconds, 1.0)

    def _generate_explanation(self, breakdown: Dict[str, float]) -> str:
        """Create a transparent, score-grounded textual explanation."""
        reasons = []
        if breakdown.get("short_term_relevance", 0) > 0.65:
            reasons.append("matches your recent reading patterns")
        elif breakdown.get("long_term_relevance", 0) > 0.65:
            reasons.append("aligns with your enduring interests")

        if breakdown.get("event_novelty", 0) > 0.60:
            reasons.append(f"introduces key new developments (+{breakdown['event_novelty']:.2f} event novelty)")
        elif breakdown.get("user_novelty", 0) > 0.80:
            reasons.append("unseen topic you have not yet explored")

        if breakdown.get("freshness", 0) > 0.75:
            reasons.append("fresh breaking story")

        if breakdown.get("source_diversity", 0) >= 0.75:
            reasons.append("corroborated by multiple independent newsrooms")

        if breakdown.get("redundancy_penalty", 0) > 0.60:
            reasons.append("penalized for substantial content overlap with articles you already read")

        if not reasons:
            return "Recommended based on overall balance of topic relevance and freshness."

        return "Recommended because it " + "; ".join(reasons) + "."

    def rank_events_for_user(
        self,
        user_id: str,
        current_epoch: Optional[float] = None,
        top_k: int = 10,
        candidate_event_limit: int = 100,
        apply_mmr: bool = True,
    ) -> List[EventRecommendation]:
        """Rank candidate events for a given user using multi-objective novelty-aware scoring."""
        profile = self.profile_manager.get_profile(user_id)
        eval_epoch = current_epoch or datetime.now(timezone.utc).timestamp()

        # 1. Fetch candidate events from SQLite
        # Candidate events with at least 1 article
        query_candidates = """
        SELECT e.id, e.title, e.summary, e.last_seen_at, e.article_count, e.centroid_blob,
               e.key_entities, e.status
        FROM events e
        WHERE e.article_count > 0
        ORDER BY e.last_seen_at DESC
        LIMIT ?;
        """
        with self.repo._get_connection() as conn:
            event_rows = conn.execute(query_candidates, (candidate_event_limit,)).fetchall()

        if not event_rows:
            return []

        scored_candidates: List[Tuple[float, Dict[str, Any], np.ndarray, Dict[str, float]]] = []

        for row in event_rows:
            event_id = row["id"]
            centroid_blob = row["centroid_blob"]
            if centroid_blob:
                centroid = np.frombuffer(centroid_blob, dtype=np.float32)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
            else:
                centroid = np.zeros(384, dtype=np.float32)

            # Retrieve latest representative article for the event
            query_art = """
            SELECT a.id, a.headline, a.source_name, a.published_at,
                   an.event_novelty_score
            FROM event_articles ea
            JOIN articles a ON ea.article_id = a.id
            LEFT JOIN article_novelty an ON an.article_id = a.id AND an.event_id = ea.event_id
            WHERE ea.event_id = ?
            ORDER BY a.published_at DESC
            LIMIT 1;
            """
            with self.repo._get_connection() as conn:
                art_row = conn.execute(query_art, (event_id,)).fetchone()

            if not art_row:
                continue

            art_id = art_row["id"]
            headline = art_row["headline"]
            source_name = art_row["source_name"]
            published_at = art_row["published_at"]

            # Parse publication epoch
            try:
                dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                pub_epoch = dt.timestamp()
            except Exception:
                pub_epoch = eval_epoch

            # Compute individual score components:
            # 1. Long-term and short-term relevance
            lt_rel, st_rel, _ = self.profile_manager.compute_relevance_scores(profile, centroid)

            # 2. Event novelty score
            raw_event_nov = art_row["event_novelty_score"]
            event_novelty = float(raw_event_nov) if raw_event_nov is not None else 0.50

            # 3. User novelty score and redundancy penalty
            user_nov_result = self.novelty_engine.compute_user_novelty(user_id, art_id)
            user_novelty = user_nov_result.user_novelty_score
            redundancy_penalty = user_nov_result.redundancy_penalty

            # 4. Freshness decay: exp(-lambda * delta_t)
            delta_t = max(0.0, eval_epoch - pub_epoch)
            freshness = float(np.exp(-self.freshness_decay_constant * delta_t))

            # 5. Source diversity
            with self.repo._get_connection() as conn:
                source_count = conn.execute(
                    "SELECT COUNT(DISTINCT a.source_id) FROM event_articles ea JOIN articles a ON ea.article_id = a.id WHERE ea.event_id = ?;",
                    (event_id,)
                ).fetchone()[0]
            source_diversity = min(1.0, source_count / 4.0)

            # 6. Exposure penalty
            exposure_penalty = self.novelty_engine.compute_event_exposure_penalty(
                user_id, event_id, lambda_exposure=self.weights.exposure_lambda
            )

            # Multi-objective composite score
            w = self.weights
            composite_score = (
                w.w_long * lt_rel
                + w.w_short * st_rel
                + w.w_event_novelty * event_novelty
                + w.w_user_novelty * user_novelty
                + w.w_freshness * freshness
                + w.w_source_diversity * source_diversity
                - w.w_exposure_penalty * exposure_penalty
                - w.w_redundancy_penalty * redundancy_penalty
            )

            breakdown = {
                "long_term_relevance": round(lt_rel, 4),
                "short_term_relevance": round(st_rel, 4),
                "event_novelty": round(event_novelty, 4),
                "user_novelty": round(user_novelty, 4),
                "freshness": round(freshness, 4),
                "source_diversity": round(source_diversity, 4),
                "exposure_penalty": round(exposure_penalty, 4),
                "redundancy_penalty": round(redundancy_penalty, 4),
                "raw_composite": round(composite_score, 4),
            }

            candidate_meta = {
                "event_id": event_id,
                "title": row["title"],
                "summary": row["summary"] or "",
                "representative_article_id": art_id,
                "article_headline": headline,
                "source_name": source_name,
                "published_at": published_at,
                "published_epoch": pub_epoch,
            }

            scored_candidates.append((composite_score, candidate_meta, centroid, breakdown))

        # Sort candidates descending by composite score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 2. MMR Diversity Reranking
        if apply_mmr and len(scored_candidates) > 1:
            selected: List[Tuple[float, Dict[str, Any], np.ndarray, Dict[str, float]]] = []
            remaining = list(scored_candidates)
            mmr_lambda = self.weights.mmr_lambda

            while remaining and len(selected) < top_k:
                if not selected:
                    # Pick highest scoring candidate first
                    best = remaining.pop(0)
                    selected.append(best)
                else:
                    selected_centroids = np.vstack([s[2] for s in selected])
                    best_mmr_score = -float("inf")
                    best_idx = 0

                    for idx, (base_score, meta, centroid, breakdown) in enumerate(remaining):
                        # Max similarity to already selected centroids
                        sims = np.dot(selected_centroids, centroid)
                        max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
                        mmr_score = mmr_lambda * base_score - (1.0 - mmr_lambda) * max_sim

                        if mmr_score > best_mmr_score:
                            best_mmr_score = mmr_score
                            best_idx = idx

                    selected.append(remaining.pop(best_idx))
            final_pool = selected
        else:
            final_pool = scored_candidates[:top_k]

        # 3. Build formatted EventRecommendation objects
        recommendations: List[EventRecommendation] = []
        for score, meta, _, breakdown in final_pool:
            explanation = self._generate_explanation(breakdown)
            recommendations.append(
                EventRecommendation(
                    event_id=meta["event_id"],
                    title=meta["title"],
                    summary=meta["summary"],
                    representative_article_id=meta["representative_article_id"],
                    article_headline=meta["article_headline"],
                    source_name=meta["source_name"],
                    published_at=meta["published_at"],
                    published_epoch=meta["published_epoch"],
                    score=round(score, 4),
                    score_breakdown=breakdown,
                    explanation=explanation,
                )
            )

        return recommendations
