"""News feed ingestion pipeline for EventPulse."""

from src.ingestion.models import (
    FeedIngestionResult,
    PipelineIngestionSummary,
    ProcessedArticle,
    RawFeedItem,
)
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.rss_client import RSSClient

__all__ = [
    "RawFeedItem",
    "ProcessedArticle",
    "FeedIngestionResult",
    "PipelineIngestionSummary",
    "RSSClient",
    "IngestionPipeline",
]
