"""Health and database statistics endpoints."""

from fastapi import APIRouter, Depends

from src.api.dependencies import AppContainer, get_container
from src.api.schemas import DatabaseStatsResponse, HealthResponse

router = APIRouter(prefix="/api", tags=["Health & Statistics"])


@router.get("/health", response_model=HealthResponse)
def get_health(container: AppContainer = Depends(get_container)) -> HealthResponse:
    db_ok = True
    try:
        container.repository.get_article_count()
    except Exception:
        db_ok = False
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        app_name=container.config.app.name,
        version=container.config.app.version,
        database_connected=db_ok,
        embedding_model=container.embedding_provider.model_id,
    )


@router.get("/stats", response_model=DatabaseStatsResponse)
def get_stats(container: AppContainer = Depends(get_container)) -> DatabaseStatsResponse:
    stats = container.repository.get_database_stats()
    return DatabaseStatsResponse(
        **stats,
        distinct_sources_count=len(stats.get("sources", {})),
    )
