"""Persistent SQLite embedding cache manager."""

import hashlib
import logging
from typing import Optional
import numpy as np

from src.storage.repository import SQLiteRepository

logger = logging.getLogger("eventpulse.cache")


class EmbeddingCacheManager:
    """Manages persistent caching of dense vectors in SQLite to eliminate redundant inference."""

    def __init__(self, repository: SQLiteRepository):
        self.repository = repository

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute deterministic SHA-256 hash of the exact input string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(
        self, article_id: int, text_hash: str, model_id: str, expected_dim: int
    ) -> Optional[np.ndarray]:
        """Retrieve cached embedding vector.

        Returns:
            np.ndarray if cache hit and dimensions match; None on cache miss.
        """
        blob = self.repository.get_embedding(
            article_id=article_id, text_hash=text_hash, model_id=model_id
        )
        if blob is None:
            return None

        try:
            arr = np.frombuffer(blob, dtype=np.float32)
            if arr.shape[0] != expected_dim:
                logger.warning(
                    "Cached embedding dimension mismatch for article %d (expected %d, got %d). Invalidating.",
                    article_id,
                    expected_dim,
                    arr.shape[0],
                )
                return None
            return arr.copy()
        except Exception as e:
            logger.error("Failed to deserialize embedding blob for article %d: %s", article_id, e)
            return None

    def put(
        self,
        article_id: int,
        text_hash: str,
        model_id: str,
        embedding: np.ndarray,
    ) -> None:
        """Persist embedding vector to cache."""
        arr = embedding.astype(np.float32)
        blob = arr.tobytes()
        self.repository.save_embedding(
            article_id=article_id,
            text_hash=text_hash,
            model_id=model_id,
            dimension=arr.shape[0],
            embedding_blob=blob,
        )
