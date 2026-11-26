"""Shared application container and FastAPI dependency getters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Request

from src.api.services.event_service import EventService
from src.api.services.ingestion_service import IngestionService
from src.api.services.live_sync_scheduler import LiveSyncScheduler
from src.api.services.novelty_service import NoveltyService
from src.api.services.recommendation_service import RecommendationService
from src.clustering.engine import TemporalEventClusteringEngine
from src.config import GlobalConfig, load_config
from src.entities.extractor import NamedEntityExtractor
from src.ingestion.freenewsapi_client import FreeNewsAPIClient
from src.ingestion.pipeline import IngestionPipeline
from src.novelty.event_memory import EventMemoryManager
from src.novelty.novelty_aggregation import EventNoveltyEngine
from src.novelty.sentence_extraction import SentencePipeline
from src.personalisation.user_profiles import UserProfileManager
from src.representation.embedding import BaseEmbeddingProvider, get_embedding_provider
from src.storage.repository import SQLiteRepository


@dataclass
class AppContainer:
    """Reusable singleton services bound to FastAPI app.state via lifespan."""

    config: GlobalConfig
    repository: SQLiteRepository
    embedding_provider: BaseEmbeddingProvider
    profile_manager: UserProfileManager
    novelty_engine: EventNoveltyEngine
    clustering_engine: TemporalEventClusteringEngine
    ingestion_pipeline: IngestionPipeline
    recommendation_service: RecommendationService
    event_service: EventService
    novelty_service: NoveltyService
    ingestion_service: IngestionService
    live_sync_scheduler: LiveSyncScheduler


def build_app_container(
    db_path: Optional[str] = None,
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    freenewsapi_client: Optional[FreeNewsAPIClient] = None,
    config_path: Optional[str] = None,
    enable_live_sync: Optional[bool] = None,
) -> AppContainer:
    """Build and wire all reusable repositories, ML engines, and services."""
    cfg = load_config(config_path=config_path)
    resolved_db = db_path or cfg.database.path
    repo = SQLiteRepository(db_path=resolved_db, timeout=cfg.database.timeout_seconds)

    emb_provider = embedding_provider or get_embedding_provider(
        provider_type=cfg.embedding.provider,
        model_name=cfg.embedding.model_name,
    )
    model_id = emb_provider.model_id

    profile_manager = UserProfileManager(repo, model_id=model_id)
    memory_manager = EventMemoryManager(repo, model_id=model_id)
    sentence_pipeline = SentencePipeline(repo, embedding_provider=emb_provider)
    novelty_engine = EventNoveltyEngine(
        repository=repo,
        memory_manager=memory_manager,
        pipeline=sentence_pipeline,
        model_id=model_id,
    )
    clustering_engine = TemporalEventClusteringEngine(
        config=cfg,
        repository=repo,
        embedding_provider=emb_provider,
        entity_extractor=NamedEntityExtractor(use_spacy=True),
    )
    ingestion_pipeline = IngestionPipeline(config=cfg, repository=repo)

    rec_service = RecommendationService(repo, profile_manager=profile_manager, model_id=model_id)
    event_service = EventService(repo, clustering_engine=clustering_engine, model_id=model_id)
    novelty_service = NoveltyService(repo, novelty_engine=novelty_engine, memory_manager=memory_manager)
    ingestion_service = IngestionService(
        repository=repo,
        pipeline=ingestion_pipeline,
        freenewsapi_client=freenewsapi_client or FreeNewsAPIClient(),
    )
    live_sync_scheduler = LiveSyncScheduler(
        repository=repo,
        ingestion_service=ingestion_service,
        event_service=event_service,
        config=cfg.live_sync,
        enabled_override=enable_live_sync,
    )

    return AppContainer(
        config=cfg,
        repository=repo,
        embedding_provider=emb_provider,
        profile_manager=profile_manager,
        novelty_engine=novelty_engine,
        clustering_engine=clustering_engine,
        ingestion_pipeline=ingestion_pipeline,
        recommendation_service=rec_service,
        event_service=event_service,
        novelty_service=novelty_service,
        ingestion_service=ingestion_service,
        live_sync_scheduler=live_sync_scheduler,
    )


def get_container(request: Request) -> AppContainer:
    return request.app.state.container
