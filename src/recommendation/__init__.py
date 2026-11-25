"""Recommendation engines, ranking algorithms, and baselines for EventPulse."""

from src.recommendation.ranking import (
    RecommendationMode,
    RecommendationWeights,
    EventRecommendation,
    NoveltyAwareRecommender,
)
from src.recommendation.baselines import (
    BaseRecommender,
    ChronologicalRecommender,
    ContentBasedRecommender,
    EventAwareWithoutNoveltyRecommender,
    ProposedEventPulseRecommender,
)

__all__ = [
    "RecommendationWeights",
    "EventRecommendation",
    "NoveltyAwareRecommender",
    "BaseRecommender",
    "ChronologicalRecommender",
    "ContentBasedRecommender",
    "EventAwareWithoutNoveltyRecommender",
    "ProposedEventPulseRecommender",
]
