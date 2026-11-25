"""Data models for raw feed entries, processed articles, and pipeline statistics."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RawFeedItem(BaseModel):
    """Raw item extracted from an RSS/Atom feed."""

    title: Optional[str] = None
    link: Optional[str] = None
    description: Optional[str] = None
    published_raw: Optional[str] = None
    published_parsed: Optional[Any] = None
    source_id: str
    source_name: str
    retrieved_at: str
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedArticle(BaseModel):
    """Cleaned, standardized, and validated news article ready for persistence."""

    canonical_url: str
    url: str
    headline: str
    headline_clean: str
    headline_hash: str
    description: str = ""
    content_clean: str = ""
    source_id: str
    source_name: str
    published_at: str
    retrieved_at: str
    is_sample: bool = False
    provider: str = "rss"
    external_id: Optional[str] = None
    country: Optional[str] = None
    language: str = "en"
    dataset_origin: str = "event_corpus"
    timestamp_precision: str = "UNKNOWN"
    timestamp_source: Optional[str] = None
    timestamp_is_inferred: bool = False
    published_at_original: Optional[str] = None
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedIngestionResult(BaseModel):
    """Outcome and statistics for a single feed ingestion run."""

    feed_id: str
    feed_name: str
    status: str  # 'SUCCESS', 'PARTIAL', 'FAILED'
    items_fetched: int = 0
    items_stored: int = 0
    items_duplicate: int = 0
    items_malformed: int = 0
    error_message: Optional[str] = None


class PipelineIngestionSummary(BaseModel):
    """Overall summary of an ingestion pipeline execution run."""

    started_at: str
    completed_at: str
    total_feeds_processed: int = 0
    total_items_fetched: int = 0
    total_items_stored: int = 0
    total_items_duplicate: int = 0
    total_items_malformed: int = 0
    feed_results: List[FeedIngestionResult] = Field(default_factory=list)
    overall_status: str = "SUCCESS"
