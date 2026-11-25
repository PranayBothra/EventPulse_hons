"""Article representation and embedding pipeline for EventPulse."""

from src.representation.article_representation import (
    ArticleRepresentation,
    build_article_representation,
)
from src.representation.cache import EmbeddingCacheManager
from src.representation.embedding import (
    BaseEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    TfidfEmbeddingProvider,
    get_embedding_provider,
)

__all__ = [
    "BaseEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "TfidfEmbeddingProvider",
    "get_embedding_provider",
    "EmbeddingCacheManager",
    "ArticleRepresentation",
    "build_article_representation",
]
