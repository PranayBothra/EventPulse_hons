"""Unified and provider-specific news ingestion endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from src.api.dependencies import AppContainer, get_container
from src.api.schemas import (
    AutoSyncConfigRequest,
    FreeNewsAPIRunRequest,
    IngestionRunResponse,
    IngestionStatusResponse,
    LiveSyncStatusResponse,
)
from src.api.services.ingestion_service import IngestionInProgressError
from src.ingestion.freenewsapi_client import (
    FreeNewsAPIAuthError,
    FreeNewsAPIConfigError,
    FreeNewsAPIError,
    FreeNewsAPIRateLimitError,
    FreeNewsAPIRequestError,
    FreeNewsAPIValidationError,
)

router = APIRouter(prefix="/api/ingestion", tags=["Ingestion"])


@router.get("/status", response_model=IngestionStatusResponse)
def get_ingestion_status(
    container: AppContainer = Depends(get_container),
) -> IngestionStatusResponse:
    data = container.ingestion_service.get_status(
        clustering_running=container.event_service.is_clustering_running
    )
    live_sync = container.live_sync_scheduler.get_status()
    data["data_version"] = live_sync["data_version"]
    data["live_sync"] = live_sync
    return IngestionStatusResponse(**data)


@router.post("/auto-sync", response_model=LiveSyncStatusResponse)
def configure_auto_sync(
    payload: AutoSyncConfigRequest,
    container: AppContainer = Depends(get_container),
) -> LiveSyncStatusResponse:
    status = container.live_sync_scheduler.configure(
        paused=payload.paused,
        rss_interval_seconds=payload.rss_interval_seconds,
        freenewsapi_interval_seconds=payload.freenewsapi_interval_seconds,
        auto_cluster_enabled=payload.auto_cluster_enabled,
        auto_cluster_min_batch=payload.auto_cluster_min_batch,
    )
    return LiveSyncStatusResponse(**status)


@router.get("/history")
def get_ingestion_history(
    limit: int = Query(default=20, ge=1, le=200),
    container: AppContainer = Depends(get_container),
) -> List[Dict[str, Any]]:
    return container.ingestion_service.get_history(limit=limit)


@router.post("/freenewsapi/run", response_model=IngestionRunResponse)
def run_freenewsapi_ingestion(
    background_tasks: BackgroundTasks,
    payload: Optional[FreeNewsAPIRunRequest] = None,
    container: AppContainer = Depends(get_container),
) -> IngestionRunResponse:
    req = payload or FreeNewsAPIRunRequest()
    try:
        res = container.ingestion_service.run_freenewsapi(
            country=req.country,
            language=req.language,
            topics=req.topics,
            in_title=req.in_title,
            max_pages_per_topic=req.max_pages_per_topic,
            max_details_to_fetch=req.max_details_to_fetch,
        )
        if int(res.get("articles_inserted") or 0) > 0:
            container.live_sync_scheduler.bump_data_version()
            background_tasks.add_task(
                container.live_sync_scheduler.maybe_run_batch_clustering_sync
            )
        return IngestionRunResponse(**res)
    except IngestionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FreeNewsAPIConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FreeNewsAPIAuthError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except FreeNewsAPIValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FreeNewsAPIRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except (FreeNewsAPIRequestError, FreeNewsAPIError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/rss/run", response_model=IngestionRunResponse)
def run_rss_ingestion(
    background_tasks: BackgroundTasks,
    feed_id: Optional[str] = Query(default=None),
    use_sample: bool = Query(default=False),
    container: AppContainer = Depends(get_container),
) -> IngestionRunResponse:
    try:
        res = container.ingestion_service.run_rss(feed_id=feed_id, use_sample=use_sample)
        if int(res.get("articles_inserted") or 0) > 0:
            container.live_sync_scheduler.bump_data_version()
            background_tasks.add_task(
                container.live_sync_scheduler.maybe_run_batch_clustering_sync
            )
        return IngestionRunResponse(**res)
    except IngestionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RSS ingestion failed: {exc}") from exc


@router.post("/run", response_model=IngestionRunResponse)
def run_unified_ingestion(
    background_tasks: BackgroundTasks,
    provider: str = Query(default="rss", pattern="^(rss|freenewsapi)$"),
    feed_id: Optional[str] = Query(default=None),
    use_sample: bool = Query(default=False),
    country: Optional[str] = Query(default="IN"),
    language: Optional[str] = Query(default="en"),
    container: AppContainer = Depends(get_container),
) -> IngestionRunResponse:
    if provider == "freenewsapi":
        return run_freenewsapi_ingestion(
            background_tasks=background_tasks,
            payload=FreeNewsAPIRunRequest(country=country, language=language),
            container=container,
        )
    return run_rss_ingestion(
        background_tasks=background_tasks,
        feed_id=feed_id,
        use_sample=use_sample,
        container=container,
    )
