"""Internal article representation integrating text, dense embeddings, and named entities."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import numpy as np

from src.entities.extractor import NamedEntityExtractor
from src.preprocessing.date_parser import parse_datetime_to_utc
from src.representation.cache import EmbeddingCacheManager
from src.representation.embedding import BaseEmbeddingProvider


@dataclass
class ArticleRepresentation:
    """Standardized internal representation of an article for clustering and ranking."""

    article_id: int
    headline: str
    description: str
    source_id: str
    source_name: str
    source_url: str
    published_at: str
    retrieved_at: str
    published_epoch: float
    text_for_embedding: str
    text_hash: str
    embedding: np.ndarray
    entities: Dict[str, Set[str]]
    normalized_entities: Set[str]
    topic: Optional[str] = None
    is_sample: bool = False
    event_id: Optional[str] = None


def format_text_for_embedding(
    headline: str, description: Optional[str] = None, headline_weight: int = 2
) -> str:
    """Construct an intelligently weighted text representation for embedding.

    Headlines contain concentrated event-specific propositions; repeating the
    headline gives it higher attention weight compared to conversational or
    boilerplate text in descriptions.
    """
    clean_hl = (headline or "").strip()
    clean_desc = (description or "").strip()

    if not clean_desc:
        return clean_hl

    # Headline repeated according to headline_weight (e.g. 2x)
    hl_component = ". ".join([clean_hl] * max(1, headline_weight))
    return f"{hl_component}. {clean_desc}"


def parse_timestamp_to_epoch(iso_timestamp: Optional[str]) -> float:
    """Convert ISO-8601 UTC timestamp to POSIX epoch seconds."""
    if not iso_timestamp:
        return datetime.now(timezone.utc).timestamp()
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return datetime.now(timezone.utc).timestamp()


def build_article_representation(
    article_dict: Dict[str, Any],
    embedding_provider: BaseEmbeddingProvider,
    cache_manager: Optional[EmbeddingCacheManager] = None,
    entity_extractor: Optional[NamedEntityExtractor] = None,
    headline_weight: int = 2,
) -> ArticleRepresentation:
    """Build a complete ArticleRepresentation from a database article record.

    Handles embedding computation, persistent cache lookups, and entity extraction.
    """
    article_id = article_dict["id"]
    headline = article_dict.get("headline", "")
    description = article_dict.get("description", "")
    source_id = article_dict.get("source_id", "unknown")
    source_name = article_dict.get("source_name", source_id)
    source_url = article_dict.get("canonical_url") or article_dict.get("url", "")
    published_at = article_dict.get("published_at") or article_dict.get("retrieved_at")
    retrieved_at = article_dict.get("retrieved_at") or published_at
    is_sample = bool(article_dict.get("is_sample", False))

    published_epoch = parse_timestamp_to_epoch(published_at)

    # Format text for embedding & compute hash
    text_for_embedding = format_text_for_embedding(
        headline=headline,
        description=description,
        headline_weight=headline_weight,
    )
    text_hash = EmbeddingCacheManager.compute_text_hash(text_for_embedding)

    # Retrieve embedding from cache or compute
    embedding = None
    if cache_manager is not None:
        embedding = cache_manager.get(
            article_id=article_id,
            text_hash=text_hash,
            model_id=embedding_provider.model_id,
            expected_dim=embedding_provider.dimension,
        )

    if embedding is None:
        embedding = embedding_provider.embed_single(text_for_embedding)
        if cache_manager is not None:
            cache_manager.put(
                article_id=article_id,
                text_hash=text_hash,
                model_id=embedding_provider.model_id,
                embedding=embedding,
            )

    # Extract named entities
    if entity_extractor is None:
        entity_extractor = NamedEntityExtractor(use_spacy=True)

    entities = entity_extractor.extract(headline, description)
    normalized_entities = entity_extractor.extract_flat(headline, description)

    raw_meta = article_dict.get("raw_metadata")
    topic = None
    if isinstance(raw_meta, dict):
        topic = raw_meta.get("category")

    return ArticleRepresentation(
        article_id=article_id,
        headline=headline,
        description=description,
        source_id=source_id,
        source_name=source_name,
        source_url=source_url,
        published_at=published_at,
        retrieved_at=retrieved_at,
        published_epoch=published_epoch,
        text_for_embedding=text_for_embedding,
        text_hash=text_hash,
        embedding=embedding,
        entities=entities,
        normalized_entities=normalized_entities,
        topic=topic,
        is_sample=is_sample,
    )
