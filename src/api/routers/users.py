"""User persona, profile, and interaction history endpoints."""

from typing import List
from fastapi import APIRouter, Depends, Query

from src.api.dependencies import AppContainer, get_container
from src.api.schemas import PersonaSummary, UserInteractionItem, UserProfileResponse

router = APIRouter(prefix="/api/users", tags=["Users & Personas"])


@router.get("/presets", response_model=List[PersonaSummary])
def get_preset_personas(container: AppContainer = Depends(get_container)) -> List[PersonaSummary]:
    return [
        PersonaSummary(**p)
        for p in container.recommendation_service.list_presets()
    ]


@router.get("/{user_id}/profile", response_model=UserProfileResponse)
def get_user_profile(
    user_id: str,
    container: AppContainer = Depends(get_container),
) -> UserProfileResponse:
    data = container.recommendation_service.get_user_profile_summary(user_id)
    return UserProfileResponse(**data)


@router.get("/{user_id}/interactions", response_model=List[UserInteractionItem])
def get_user_interactions(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    container: AppContainer = Depends(get_container),
) -> List[UserInteractionItem]:
    rows = container.recommendation_service.get_user_interactions(user_id, limit=limit)
    return [UserInteractionItem(**r) for r in rows]
