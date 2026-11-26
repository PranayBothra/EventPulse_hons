"""Event explorer, clustering execution, and emerging topic endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import AppContainer, get_container
from src.api.services.event_service import ClusteringInProgressError
from src.api.schemas import (
    ClusteringRunResponse,
    EmergingTopicsResponse,
    EventArticleItem,
    EventSummaryItem,
)

router = APIRouter(prefix="/api", tags=["Events & Topics"])


@router.get("/events", response_model=List[EventSummaryItem])
def get_events(
    limit: int = Query(default=50, ge=1, le=200),
    status: Optional[str] = Query(default=None),
    sort_by: str = Query(default="article_count", pattern="^(article_count|last_seen_at)$"),
    container: AppContainer = Depends(get_container),
) -> List[EventSummaryItem]:
    rows = container.event_service.list_events(limit=limit, status=status, sort_by=sort_by)
    return [EventSummaryItem(**r) for r in rows]


@router.get("/events/{event_id}/articles", response_model=List[EventArticleItem])
def get_event_articles(
    event_id: str,
    container: AppContainer = Depends(get_container),
) -> List[EventArticleItem]:
    rows = container.event_service.get_event_articles(event_id)
    if rows is None:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")
    return [EventArticleItem(**{**r, "is_ambiguous": bool(r.get("is_ambiguous"))}) for r in rows]


@router.get("/topics", response_model=EmergingTopicsResponse)
def get_emerging_topics(
    detector: str = Query(default="online_cluster", pattern="^(online_cluster|rolling_window)$"),
    article_window: int = Query(default=400, ge=10, le=2000),
    min_cluster_size: int = Query(default=2, ge=1, le=50),
    limit: int = Query(default=20, ge=1, le=100),
    container: AppContainer = Depends(get_container),
) -> EmergingTopicsResponse:
    res = container.event_service.detect_emerging_topics(
        detector=detector,
        article_window=article_window,
        min_cluster_size=min_cluster_size,
        limit=limit,
    )
    return EmergingTopicsResponse(**res)


@router.post("/clustering/run", response_model=ClusteringRunResponse)
def post_run_clustering(
    include_sample: bool = Query(default=False),
    container: AppContainer = Depends(get_container),
) -> ClusteringRunResponse:
    try:
        res = container.event_service.run_clustering(include_sample=include_sample)
        if int(res.get("articles_clustered") or 0) > 0:
            container.live_sync_scheduler.bump_data_version()
        return ClusteringRunResponse(**res)
    except ClusteringInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {exc}") from exc
