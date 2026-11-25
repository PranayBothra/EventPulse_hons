"""Recommendation ranking and mode/weight configuration endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import AppContainer, get_container
from src.api.schemas import RecommendationsListResponse

router = APIRouter(prefix="/api", tags=["Recommendations"])


@router.get("/recommendations", response_model=RecommendationsListResponse)
def get_recommendations(
    user_id: str = Query(default="user_tech", min_length=1),
    algorithm: str = Query(default="EventPulse"),
    mode: Optional[str] = Query(default=None),
    top_k: int = Query(default=10, ge=1, le=50),
    w_long: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    w_short: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    w_event_novelty: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    w_user_novelty: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    w_freshness: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    w_redundancy_penalty: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    mmr_lambda: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    container: AppContainer = Depends(get_container),
) -> RecommendationsListResponse:
    weight_overrides = {
        "w_long": w_long,
        "w_short": w_short,
        "w_event_novelty": w_event_novelty,
        "w_user_novelty": w_user_novelty,
        "w_freshness": w_freshness,
        "w_redundancy_penalty": w_redundancy_penalty,
        "mmr_lambda": mmr_lambda,
    }
    try:
        res = container.recommendation_service.recommend_events(
            user_id=user_id,
            algorithm=algorithm,
            mode=mode,
            top_k=top_k,
            weight_overrides=weight_overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RecommendationsListResponse(**res)
