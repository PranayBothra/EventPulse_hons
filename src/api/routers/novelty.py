"""Sentence-level incremental novelty inspection endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import AppContainer, get_container
from src.api.schemas import EvolvingEventItem, NoveltyInspectionResponse

router = APIRouter(prefix="/api/novelty", tags=["Novelty Inspector"])


@router.get("/evolving-events", response_model=List[EvolvingEventItem])
def get_evolving_events(
    min_articles: int = Query(default=2, ge=2, le=100),
    limit: int = Query(default=30, ge=1, le=200),
    container: AppContainer = Depends(get_container),
) -> List[EvolvingEventItem]:
    rows = container.novelty_service.list_evolving_events(
        min_articles=min_articles, limit=limit
    )
    return [EvolvingEventItem(**r) for r in rows]


@router.get("/inspect", response_model=NoveltyInspectionResponse)
def inspect_novelty(
    event_id: str = Query(..., min_length=1),
    article_id: Optional[int] = Query(default=None, ge=1),
    persist: bool = Query(default=True),
    container: AppContainer = Depends(get_container),
) -> NoveltyInspectionResponse:
    res = container.novelty_service.inspect_article_novelty(
        event_id=event_id,
        article_id=article_id,
        persist=persist,
    )
    if res is None:
        raise HTTPException(
            status_code=404,
            detail=f"Event '{event_id}' or target article not found for novelty inspection.",
        )
    return NoveltyInspectionResponse(**res)
