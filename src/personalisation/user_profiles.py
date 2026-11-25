"""Dynamic User Interest Modelling with dual long-term and short-term profiles."""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.storage.repository import SQLiteRepository


# Default half-life for short-term profile recency decay: 48 hours in seconds
DEFAULT_HALF_LIFE_SECONDS = 48.0 * 3600.0
LAMBDA_USER_DECAY = math.log(2.0) / DEFAULT_HALF_LIFE_SECONDS

# Standard interaction weights reflecting engagement strength.
# Note: Unclicked impressions are non-committal exposures (weight 0.0), not explicit negative feedback.
INTERACTION_WEIGHTS = {
    "READ": 2.0,
    "SAVE": 2.0,
    "LIKE": 1.5,
    "CLICK": 1.0,
    "IMPRESSION": 0.0,
    "DISLIKE": -1.5,
}


@dataclass
class UserProfile:
    """Represents a user's dual semantic interest profile."""
    user_id: str
    long_term_vector: Optional[np.ndarray] = None
    short_term_vector: Optional[np.ndarray] = None
    explicit_topics: List[str] = field(default_factory=list)
    interaction_count: int = 0
    last_interaction_epoch: Optional[float] = None

    @property
    def is_cold_start(self) -> bool:
        """User has insufficient interactions to establish a personalized centroid."""
        return self.interaction_count < 3 or self.long_term_vector is None


class UserProfileManager:
    """Maintains, updates, and computes user semantic profiles in SQLite."""

    def __init__(
        self,
        repository: SQLiteRepository,
        half_life_seconds: float = DEFAULT_HALF_LIFE_SECONDS,
        model_id: str = "all-MiniLM-L6-v2",
    ):
        self.repo = repository
        self.half_life_seconds = half_life_seconds
        self.decay_constant = math.log(2.0) / max(half_life_seconds, 1.0)
        self.model_id = model_id

    def get_profile(self, user_id: str) -> UserProfile:
        """Retrieve user profile from repository."""
        data = self.repo.get_user_profile(user_id)
        if not data:
            return UserProfile(user_id=user_id)

        lt_vec = None
        st_vec = None
        if data.get("long_term_profile_blob"):
            lt_vec = np.frombuffer(data["long_term_profile_blob"], dtype=np.float32)
            norm = np.linalg.norm(lt_vec)
            if norm > 0:
                lt_vec = lt_vec / norm

        if data.get("short_term_profile_blob"):
            st_vec = np.frombuffer(data["short_term_profile_blob"], dtype=np.float32)
            norm = np.linalg.norm(st_vec)
            if norm > 0:
                st_vec = st_vec / norm

        return UserProfile(
            user_id=user_id,
            long_term_vector=lt_vec,
            short_term_vector=st_vec,
            explicit_topics=data.get("explicit_topics", []),
            interaction_count=data.get("interaction_count", 0),
        )

    def record_interaction(
        self,
        user_id: str,
        article_id: int,
        interaction_type: str,
        interaction_epoch: float,
        event_id: Optional[str] = None,
        interaction_time: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        recompute_profile: bool = True,
    ) -> None:
        """Record an explicit or implicit user action and update profile state."""
        time_str = interaction_time or datetime.fromtimestamp(interaction_epoch, timezone.utc).isoformat()
        weight = INTERACTION_WEIGHTS.get(interaction_type.upper(), 1.0)

        # 1. Log interaction in database
        self.repo.record_user_interaction(
            user_id=user_id,
            article_id=article_id,
            event_id=event_id,
            interaction_type=interaction_type,
            weight=weight,
            interaction_time=time_str,
            interaction_epoch=interaction_epoch,
            metadata=metadata,
        )

        # 2. Mark article sentences as consumed by this user
        sentences = self.repo.get_article_sentences(article_id)
        if sentences:
            sentence_ids = [s["id"] for s in sentences]
            self.repo.record_user_consumed_sentences(
                user_id=user_id,
                sentence_ids=sentence_ids,
                consumed_time=time_str,
                consumed_epoch=interaction_epoch,
            )

        # 3. Update interest profile representations
        if recompute_profile:
            self.recompute_profile(user_id=user_id, current_epoch=interaction_epoch)

    def recompute_profile(
        self, user_id: str, current_epoch: Optional[float] = None
    ) -> UserProfile:
        """Recompute dual long-term and short-term interest centroids from full interaction history."""
        interactions = self.repo.get_user_interactions(user_id)
        if not interactions:
            return UserProfile(user_id=user_id)

        now_epoch = current_epoch or interactions[-1]["interaction_epoch"]

        # Collect article embeddings
        article_ids = list(set(r["article_id"] for r in interactions))
        embeddings_map = self.repo.get_embeddings_by_article_ids(article_ids, model_id=self.model_id)

        long_term_sum: Optional[np.ndarray] = None
        short_term_sum: Optional[np.ndarray] = None
        total_long_weight = 0.0
        total_short_weight = 0.0

        for r in interactions:
            aid = r["article_id"]
            blob = embeddings_map.get(aid)
            if blob is None:
                continue

            vec = np.frombuffer(blob, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            unit_vec = vec / norm

            w = r["interaction_weight"]
            # Only positive engagement signals contribute to interest centroids;
            # unclicked impressions (w=0.0) and negative actions (w<0) are excluded from positive centroid accumulation
            if w <= 0.01:
                continue

            if long_term_sum is None:
                long_term_sum = np.zeros_like(unit_vec)
                short_term_sum = np.zeros_like(unit_vec)

            # Long-term accumulation
            long_term_sum += w * unit_vec
            total_long_weight += abs(w)

            # Short-term recency decay
            delta_t = max(0.0, now_epoch - r["interaction_epoch"])
            decay = math.exp(-self.decay_constant * delta_t)
            short_weight = w * decay
            short_term_sum += short_weight * unit_vec
            total_short_weight += abs(short_weight)

        lt_vec = None
        st_vec = None

        if long_term_sum is not None and np.linalg.norm(long_term_sum) > 0:
            lt_vec = (long_term_sum / np.linalg.norm(long_term_sum)).astype(np.float32)

        if short_term_sum is not None and np.linalg.norm(short_term_sum) > 0:
            st_vec = (short_term_sum / np.linalg.norm(short_term_sum)).astype(np.float32)

        lt_blob = lt_vec.tobytes() if lt_vec is not None else None
        st_blob = st_vec.tobytes() if st_vec is not None else None

        self.repo.upsert_user_profile(
            user_id=user_id,
            long_term_blob=lt_blob,
            short_term_blob=st_blob,
        )

        return UserProfile(
            user_id=user_id,
            long_term_vector=lt_vec,
            short_term_vector=st_vec,
            interaction_count=len(interactions),
            last_interaction_epoch=interactions[-1]["interaction_epoch"],
        )

    def compute_relevance_scores(
        self, profile: UserProfile, target_embedding: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute long-term, short-term, and combined user relevance scores in [0.0, 1.0]."""
        if profile.is_cold_start:
            # Default prior for cold-start users
            return 0.5, 0.5, 0.5

        norm = np.linalg.norm(target_embedding)
        cand_vec = target_embedding / norm if norm > 0 else target_embedding

        # Long-term cosine similarity mapped to [0, 1]
        if profile.long_term_vector is not None:
            lt_cos = float(np.dot(profile.long_term_vector, cand_vec))
            lt_score = max(0.0, min(1.0, (lt_cos + 1.0) / 2.0))
        else:
            lt_score = 0.5

        # Short-term cosine similarity mapped to [0, 1]
        if profile.short_term_vector is not None:
            st_cos = float(np.dot(profile.short_term_vector, cand_vec))
            st_score = max(0.0, min(1.0, (st_cos + 1.0) / 2.0))
        else:
            st_score = lt_score

        # Weighted blend (60% short-term recency, 40% enduring long-term)
        combined = 0.40 * lt_score + 0.60 * st_score
        return round(lt_score, 4), round(st_score, 4), round(combined, 4)
