"""Unified and provider-specific news ingestion endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import AppContainer, get_container
from src.api.schemas import (
    FreeNewsAPIRunRequest,
    IngestionRunResponse,
    IngestionStatusResponse,
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
    return IngestionStatusResponse(**data)


@router.get("/history")
def get_ingestion_history(
    limit: int = Query(default=20, ge=1, le=200),
    container: AppContainer = Depends(get_container),
) -> List[Dict[str, Any]]:
    return container.ingestion_service.get_history(limit=limit)


@router.post("/freenewsapi/run", response_model=IngestionRunResponse)
def run_freenewsapi_ingestion(
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
    feed_id: Optional[str] = Query(default=None),
    use_sample: bool = Query(default=False),
    container: AppContainer = Depends(get_container),
) -> IngestionRunResponse:
    try:
        res = container.ingestion_service.run_rss(feed_id=feed_id, use_sample=use_sample)
        return IngestionRunResponse(**res)
    except IngestionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RSS ingestion failed: {exc}") from exc


@router.post("/run", response_model=IngestionRunResponse)
def run_unified_ingestion(
    provider: str = Query(default="rss", pattern="^(rss|freenewsapi)$"),
    feed_id: Optional[str] = Query(default=None),
    use_sample: bool = Query(default=False),
    country: Optional[str] = Query(default="IN"),
    language: Optional[str] = Query(default="en"),
    container: AppContainer = Depends(get_container),
) -> IngestionRunResponse:
    if provider == "freenewsapi":
        return run_freenewsapi_ingestion(
            payload=FreeNewsAPIRunRequest(country=country, language=language),
            container=container,
        )
    return run_rss_ingestion(feed_id=feed_id, use_sample=use_sample, container=container)
