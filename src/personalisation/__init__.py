"""Personalisation, dynamic user profiling, and user-level novelty modules for EventPulse."""

from src.personalisation.user_profiles import (
    UserProfile,
    UserProfileManager,
    DEFAULT_HALF_LIFE_SECONDS,
    INTERACTION_WEIGHTS,
)
from src.personalisation.user_novelty import UserNoveltyScore, UserNoveltyEngine

__all__ = [
    "UserProfile",
    "UserProfileManager",
    "DEFAULT_HALF_LIFE_SECONDS",
    "INTERACTION_WEIGHTS",
    "UserNoveltyScore",
    "UserNoveltyEngine",
]
