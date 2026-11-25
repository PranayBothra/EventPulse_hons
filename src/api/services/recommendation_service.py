"""Recommendation and user persona orchestration service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.personalisation.user_profiles import UserProfile, UserProfileManager
from src.recommendation.baselines import (
    ChronologicalRecommender,
    ContentBasedRecommender,
    EventAwareWithoutNoveltyRecommender,
    ProposedEventPulseRecommender,
)
from src.recommendation.ranking import (
    EventRecommendation,
    RecommendationMode,
    RecommendationWeights,
)
from src.storage.repository import SQLiteRepository

PRESET_PERSONAS: Dict[str, Dict[str, str]] = {
    "user_tech": {
        "display_name": "AI & Technology Specialist (Synthetic Persona)",
        "description": "Synthetic demonstration archetype — distinct from offline MIND benchmark evaluation logs.",
    },
    "user_geopolitics": {
        "display_name": "Geopolitics & Affairs (Synthetic Persona)",
        "description": "Synthetic demonstration archetype — distinct from offline MIND benchmark evaluation logs.",
    },
    "user_science": {
        "display_name": "Space & Science (Synthetic Persona)",
        "description": "Synthetic demonstration archetype — distinct from offline MIND benchmark evaluation logs.",
    },
    "user_cold_start": {
        "display_name": "Fresh User (Cold Start Archetype)",
        "description": "Synthetic cold-start archetype with zero prior engagement history.",
    },
}


class RecommendationService:
    """Orchestrates ranking algorithms, adaptive modes, and user feedback without modifying core math."""

    def __init__(
        self,
        repository: SQLiteRepository,
        profile_manager: UserProfileManager,
        model_id: str = "all-MiniLM-L6-v2",
    ):
        self.repo = repository
        self.profile_manager = profile_manager
        self.model_id = model_id

    @staticmethod
    def list_presets() -> List[Dict[str, Any]]:
        return [
            {
                "user_id": uid,
                "display_name": meta["display_name"],
                "description": meta["description"],
                "is_synthetic_demo": True,
            }
            for uid, meta in PRESET_PERSONAS.items()
        ]

    def get_user_profile_summary(self, user_id: str) -> Dict[str, Any]:
        profile: UserProfile = self.profile_manager.get_profile(user_id)
        consumed_ids = self.repo.get_user_consumed_sentence_ids(user_id)
        meta = PRESET_PERSONAS.get(user_id)
        return {
            "user_id": user_id,
            "display_name": meta["display_name"] if meta else None,
            "is_synthetic_demo": user_id in PRESET_PERSONAS,
            "interaction_count": profile.interaction_count,
            "consumed_sentences_count": len(consumed_ids),
            "is_cold_start": profile.is_cold_start,
            "explicit_topics": profile.explicit_topics,
        }

    def get_user_interactions(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        return self.repo.get_user_interactions(user_id, limit=limit)

    def record_interaction(
        self,
        user_id: str,
        article_id: int,
        event_id: str,
        interaction_type: str,
        interaction_epoch: float,
    ) -> Dict[str, Any]:
        self.profile_manager.record_interaction(
            user_id=user_id,
            article_id=article_id,
            event_id=event_id,
            interaction_type=interaction_type,
            interaction_epoch=interaction_epoch,
        )
        updated = self.get_user_profile_summary(user_id)
        return {
            "status": "recorded",
            "user_id": user_id,
            "article_id": article_id,
            "event_id": event_id,
            "interaction_type": interaction_type.upper(),
            "updated_interaction_count": updated["interaction_count"],
            "consumed_sentences_count": updated["consumed_sentences_count"],
            "is_cold_start": updated["is_cold_start"],
        }

    def recommend_events(
        self,
        user_id: str,
        algorithm: str = "EventPulse",
        mode: Optional[str] = None,
        top_k: int = 10,
        weight_overrides: Optional[Dict[str, Optional[float]]] = None,
    ) -> Dict[str, Any]:
        profile = self.profile_manager.get_profile(user_id)
        algo_norm = algorithm.strip()

        resolved_mode: Optional[str] = None
        if algo_norm == "EventPulse":
            rec_mode = RecommendationMode.RELEVANCE_FOCUSED
            if mode:
                rec_mode = RecommendationMode(mode.upper().strip())
            resolved_mode = rec_mode.value
            weights = RecommendationWeights.for_mode(rec_mode)
            if weight_overrides:
                for field_name, val in weight_overrides.items():
                    if val is not None and hasattr(weights, field_name):
                        setattr(weights, field_name, float(val))
            engine = ProposedEventPulseRecommender(
                repository=self.repo,
                profile_manager=self.profile_manager,
                weights=weights,
                model_id=self.model_id,
            )
        elif algo_norm == "Baseline_A":
            engine = ChronologicalRecommender(self.repo)
        elif algo_norm == "Baseline_B":
            engine = ContentBasedRecommender(
                self.repo, profile_manager=self.profile_manager, model_id=self.model_id
            )
        elif algo_norm == "Baseline_C":
            engine = EventAwareWithoutNoveltyRecommender(
                self.repo, profile_manager=self.profile_manager, model_id=self.model_id
            )
        else:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Choose from: EventPulse, Baseline_A, Baseline_B, Baseline_C."
            )

        recs: List[EventRecommendation] = engine.rank_events(user_id=user_id, top_k=top_k)
        return {
            "user_id": user_id,
            "algorithm": algo_norm,
            "mode": resolved_mode,
            "is_cold_start": profile.is_cold_start,
            "count": len(recs),
            "recommendations": [
                {
                    "rank": idx,
                    "event_id": r.event_id,
                    "title": r.title,
                    "summary": r.summary,
                    "representative_article_id": r.representative_article_id,
                    "article_headline": r.article_headline,
                    "source_name": r.source_name,
                    "published_at": r.published_at,
                    "published_epoch": r.published_epoch,
                    "score": r.score,
                    "score_breakdown": r.score_breakdown,
                    "explanation": r.explanation,
                }
                for idx, r in enumerate(recs, start=1)
            ],
        }
