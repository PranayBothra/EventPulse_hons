"""FastAPI application factory, lifespan manager, and router registry for EventPulse."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import build_app_container
from src.api.routers import (
    benchmarks,
    events,
    ingestion,
    interactions,
    novelty,
    recommendations,
    stats,
    users,
)
from src.config import get_project_root
from src.ingestion.freenewsapi_client import FreeNewsAPIClient
from src.representation.embedding import BaseEmbeddingProvider


def create_app(
    db_path: Optional[str] = None,
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    freenewsapi_client: Optional[FreeNewsAPIClient] = None,
    config_path: Optional[str] = None,
) -> FastAPI:
    """Create and configure the EventPulse FastAPI application."""

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        app_instance.state.container = build_app_container(
            db_path=db_path,
            embedding_provider=embedding_provider,
            freenewsapi_client=freenewsapi_client,
            config_path=config_path,
        )
        yield

    application = FastAPI(
        title="EventPulse API",
        version="0.1.0",
        description="Temporal Novelty-Aware Event Intelligence & Personalised News Recommendation Backend",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register REST routers
    application.include_router(stats.router)
    application.include_router(users.router)
    application.include_router(recommendations.router)
    application.include_router(interactions.router)
    application.include_router(events.router)
    application.include_router(novelty.router)
    application.include_router(benchmarks.router)
    application.include_router(ingestion.router)

    # Mount static frontend assets and serve SPA shell at '/'
    static_dir = get_project_root() / "static"
    index_html = static_dir / "index.html"
    if static_dir.is_dir():
        application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @application.get("/", include_in_schema=False)
    def root():
        if index_html.is_file():
            return FileResponse(str(index_html), media_type="text/html")
        return {
            "service": "EventPulse FastAPI Backend",
            "status": "running",
            "docs_url": "/docs",
            "openapi_url": "/openapi.json",
            "health_url": "/api/health",
        }

    return application


app = create_app()
