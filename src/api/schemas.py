"""Pydantic v2 request and response schemas for EventPulse FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str
    database_connected: bool
    embedding_model: str


class DatabaseStatsResponse(BaseModel):
    db_path: str
    db_size_bytes: int
    total_articles: int
    live_articles: int
    sample_articles: int
    total_feeds: int
    total_ingestion_runs: int
    total_events: int
    total_cached_embeddings: int
    embeddings_by_model: Dict[str, int] = Field(default_factory=dict)
    sources: Dict[str, int] = Field(default_factory=dict)
    distinct_sources_count: int


class PersonaSummary(BaseModel):
    user_id: str
    display_name: str
    description: str
    is_synthetic_demo: bool = True


class UserProfileResponse(BaseModel):
    user_id: str
    display_name: Optional[str] = None
    is_synthetic_demo: bool
    interaction_count: int
    consumed_sentences_count: int
    is_cold_start: bool
    explicit_topics: List[str] = Field(default_factory=list)


class UserInteractionItem(BaseModel):
    id: int
    user_id: str
    article_id: Optional[int] = None
    event_id: Optional[str] = None
    interaction_type: str
    interaction_weight: float
    interaction_time: str
    interaction_epoch: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InteractionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    article_id: int = Field(..., ge=1)
    event_id: str = Field(..., min_length=1)
    interaction_type: Literal["LIKE", "READ", "DISLIKE"]


class InteractionCreateResponse(BaseModel):
    status: str = "recorded"
    user_id: str
    article_id: int
    event_id: str
    interaction_type: str
    updated_interaction_count: int
    consumed_sentences_count: int
    is_cold_start: bool


class RecommendationItemResponse(BaseModel):
    rank: int
    event_id: str
    title: str
    summary: str
    representative_article_id: int
    article_headline: str
    source_name: str
    published_at: str
    published_epoch: float
    score: float
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    explanation: str


class RecommendationsListResponse(BaseModel):
    user_id: str
    algorithm: str
    mode: Optional[str] = None
    is_cold_start: bool
    count: int
    recommendations: List[RecommendationItemResponse] = Field(default_factory=list)


class EventSummaryItem(BaseModel):
    id: str
    title: str
    summary: Optional[str] = None
    article_count: int
    publisher_count: int
    first_seen_at: str
    last_seen_at: str
    status: str


class EventArticleItem(BaseModel):
    id: int
    headline: str
    source_id: str
    source_name: str
    published_at: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    provider: Optional[str] = "rss"
    external_id: Optional[str] = None
    event_novelty_score: Optional[float] = None
    novelty_ratio: Optional[float] = None
    similarity_score: Optional[float] = None
    is_ambiguous: bool = False


class EvolvingEventItem(BaseModel):
    id: str
    title: str
    article_count: int
    first_seen_at: str
    last_seen_at: str
    status: str


class SentenceNoveltyItemResponse(BaseModel):
    sentence_id: int
    article_id: int
    sentence_index: int
    sentence_text: str
    evidence_status: str
    informativeness_score: float
    highest_similarity: float
    novelty_score: float
    most_similar_sentence_id: Optional[int] = None
    most_similar_text: Optional[str] = None


class NoveltyInspectionResponse(BaseModel):
    article_id: int
    event_id: str
    article_headline: str
    source_name: str
    published_at: str
    event_novelty_score: float
    novel_sentence_count: int
    redundant_sentence_count: int
    insufficient_evidence_count: int
    total_sentence_count: int
    novelty_ratio: float
    prior_memory_sentence_count: int
    sentence_results: List[SentenceNoveltyItemResponse] = Field(default_factory=list)


class EmergingTopicItem(BaseModel):
    cluster_id: int
    detector: str
    article_count: int
    seeded_before_onset: bool
    first_step: int
    last_step: int
    representative_article_id: int
    representative_headline: str
    latest_published_at: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EmergingTopicsResponse(BaseModel):
    detector_used: str
    articles_scanned: int
    topics_found: int
    topics: List[EmergingTopicItem] = Field(default_factory=list)


class FreeNewsAPIRunRequest(BaseModel):
    country: Optional[str] = "IN"
    language: Optional[str] = "en"
    topics: Optional[List[str]] = None
    in_title: Optional[str] = None
    max_pages_per_topic: int = Field(default=1, ge=1, le=5)
    max_details_to_fetch: int = Field(default=20, ge=1, le=100)


class ProviderStatusResponse(BaseModel):
    provider: str
    enabled: bool
    credentials_configured: bool
    job_running: bool
    last_sync_at: Optional[str] = None
    last_status: Optional[str] = None
    last_error: Optional[str] = None
    articles_fetched: int = 0
    articles_inserted: int = 0
    duplicates_skipped: int = 0
    failed_records: int = 0
    quota: Dict[str, Any] = Field(default_factory=dict)


class IngestionStatusResponse(BaseModel):
    ingestion_job_running: bool
    clustering_job_running: bool
    freenewsapi: ProviderStatusResponse
    rss: ProviderStatusResponse


class IngestionRunResponse(BaseModel):
    provider: str
    status: str
    started_at: str
    completed_at: str
    articles_fetched: int
    articles_inserted: int
    duplicates_skipped: int
    failed_or_malformed: int
    details: Dict[str, Any] = Field(default_factory=dict)


class ClusteringRunResponse(BaseModel):
    status: str
    articles_clustered: int
    new_events_created: int
    active_events_total: int
    error_message: Optional[str] = None
