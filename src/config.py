"""Configuration loading and validation for EventPulse."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "EventPulse"
    version: str = "0.1.0"
    environment: str = "development"


class DatabaseConfig(BaseModel):
    path: str = "data/eventpulse.db"
    timeout_seconds: int = 30
    pragmas: Dict[str, str] = Field(
        default_factory=lambda: {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "foreign_keys": "ON",
        }
    )


class IngestionConfig(BaseModel):
    request_timeout_seconds: int = 15
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EventPulseResearchBot/1.0"
    )
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    feeds_file: str = "config/feeds.yaml"
    sample_data_file: str = "data/sample/sample_articles.json"
    max_articles_per_feed: int = 50


class PreprocessingConfig(BaseModel):
    min_headline_chars: int = 10
    max_headline_chars: int = 300
    min_description_chars: int = 15
    normalize_unicode: bool = True
    strip_html_tags: bool = True
    remove_tracking_params: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: str = "logs/eventpulse.log"


class EmbeddingConfig(BaseModel):
    provider: str = "sentence-transformers"
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 16
    device: str = "cpu"
    headline_weight: int = 2


class ClusteringConfig(BaseModel):
    match_threshold: float = 0.48
    ambiguity_delta: float = 0.05
    ambiguity_floor: float = 0.42
    max_temporal_window_hours: float = 168.0
    temporal_half_life_hours: float = 48.0
    weight_semantic: float = 0.45
    weight_entity: float = 0.25
    weight_temporal: float = 0.20
    weight_headline: float = 0.10


class LiveSyncConfig(BaseModel):
    enabled: bool = True
    run_on_startup: bool = True
    startup_delay_seconds: int = 15
    rss_interval_seconds: int = 300
    freenewsapi_interval_seconds: int = 1800
    auto_cluster_enabled: bool = True
    auto_cluster_min_batch: int = 10
    freenewsapi_min_remaining_quota: int = 50


class GlobalConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    live_sync: LiveSyncConfig = Field(default_factory=LiveSyncConfig)


class FeedConfig(BaseModel):
    id: str
    name: str
    url: str
    category: str = "general"
    enabled: bool = True
    language: str = "en"


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> GlobalConfig:
    """Load and validate the global application configuration.

    Args:
        config_path: Path to config.yaml. Defaults to 'config/config.yaml'.

    Returns:
        GlobalConfig instance.
    """
    root = get_project_root()
    path = Path(config_path) if config_path else root / "config" / "config.yaml"

    if not path.is_file():
        # Fall back to default configuration
        return GlobalConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return GlobalConfig(**data)


def load_feeds(feeds_path: Optional[str] = None) -> List[FeedConfig]:
    """Load and validate the list of configured RSS feeds.

    Args:
        feeds_path: Path to feeds.yaml. Defaults to 'config/feeds.yaml'.

    Returns:
        List of FeedConfig instances.
    """
    root = get_project_root()
    path = Path(feeds_path) if feeds_path else root / "config" / "feeds.yaml"

    if not path.is_file():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_feeds = data.get("feeds", [])
    return [FeedConfig(**feed) for feed in raw_feeds]
