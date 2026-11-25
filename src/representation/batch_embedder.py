"""Batch embedding generator for large-scale corpus processing with memory management."""

import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import GlobalConfig, load_config
from src.representation.article_representation import format_text_for_embedding
from src.representation.cache import EmbeddingCacheManager
from src.representation.embedding import (
    BaseEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from src.storage.repository import SQLiteRepository

logger = logging.getLogger("eventpulse.batch_embedder")


class BatchEmbedder:
    """Manages chunked, high-throughput batch vector generation with SQLite float32 persistence."""

    def __init__(
        self,
        repository: Optional[SQLiteRepository] = None,
        provider: Optional[BaseEmbeddingProvider] = None,
        config: Optional[GlobalConfig] = None,
        batch_size: int = 128,
    ):
        self.config = config or load_config()
        self.repository = repository or SQLiteRepository(
            db_path=self.config.database.path,
            timeout=self.config.database.timeout_seconds,
        )
        e_cfg = self.config.embedding
        self.provider = provider or SentenceTransformerEmbeddingProvider(
            model_name=e_cfg.model_name,
            batch_size=min(batch_size, 64),
            device=e_cfg.device,
        )
        self.batch_size = batch_size
        self.headline_weight = e_cfg.headline_weight

    def get_uncached_articles(self) -> List[Dict[str, Any]]:
        """Fetch all articles from the database that do not yet have cached embeddings."""
        query = """
        SELECT a.id, a.headline, a.description
        FROM articles a
        LEFT JOIN article_embeddings ae ON a.id = ae.article_id
        WHERE ae.article_id IS NULL
        ORDER BY a.id ASC;
        """
        with self.repository._get_connection() as conn:
            rows = conn.execute(query).fetchall()
            return [dict(r) for r in rows]

    def embed_and_cache_all_uncached(
        self, max_articles: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute and persist embeddings for all uncached articles.

        Returns:
            Dictionary with execution statistics and validation results.
        """
        uncached = self.get_uncached_articles()
        if max_articles:
            uncached = uncached[:max_articles]

        total = len(uncached)
        if total == 0:
            logger.info("All articles already have cached embeddings. Nothing to embed.")
            return {"total_uncached": 0, "embedded": 0, "duration_seconds": 0.0}

        logger.info(
            "Starting batch embedding generation for %d uncached articles (batch_size=%d)...",
            total,
            self.batch_size,
        )
        t0 = time.time()
        embedded_count = 0
        model_id = self.provider.model_id
        expected_dim = self.provider.dimension

        for i in range(0, total, self.batch_size):
            chunk = uncached[i : i + self.batch_size]
            texts: List[str] = []
            hashes: List[str] = []
            ids: List[int] = []

            for art in chunk:
                formatted_text = format_text_for_embedding(
                    headline=art["headline"],
                    description=art.get("description"),
                    headline_weight=self.headline_weight,
                )
                th = EmbeddingCacheManager.compute_text_hash(formatted_text)
                texts.append(formatted_text)
                hashes.append(th)
                ids.append(art["id"])

            # Compute L2-normalized embeddings in batch
            vectors = self.provider.embed_texts(texts)

            # Strict validation
            assert vectors.shape[0] == len(chunk), "Batch count mismatch"
            assert vectors.shape[1] == expected_dim, f"Expected dim {expected_dim}, got {vectors.shape[1]}"
            assert not np.isnan(vectors).any(), "NaN detected in embeddings"
            assert not np.isinf(vectors).any(), "Inf detected in embeddings"

            # Check unit normalization: ||v|| ~ 1.0
            norms = np.linalg.norm(vectors, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-3), "Vectors are not unit normalized"

            # Prepare for persistence
            records_to_save: List[Tuple[int, str, str, int, bytes]] = []
            for art_id, th, vec in zip(ids, hashes, vectors):
                blob = vec.astype(np.float32).tobytes()
                records_to_save.append((art_id, th, model_id, expected_dim, blob))

            saved = self.repository.batch_save_embeddings(records_to_save)
            embedded_count += saved

            # Explicit memory cleanup
            del vectors, texts, hashes, records_to_save
            if (i // self.batch_size) % 10 == 0:
                gc.collect()

            elapsed = time.time() - t0
            rate = embedded_count / max(elapsed, 0.001)
            logger.info(
                "Progress: %d/%d (%.1f%%) | %.0f articles/sec",
                embedded_count,
                total,
                (embedded_count / total) * 100,
                rate,
            )

        duration = time.time() - t0
        logger.info(
            "Batch embedding complete: %d articles embedded and cached in %.2fs (%.0f arts/sec)",
            embedded_count,
            duration,
            embedded_count / max(duration, 0.001),
        )

        return {
            "total_uncached": total,
            "embedded": embedded_count,
            "duration_seconds": duration,
            "rate_per_second": embedded_count / max(duration, 0.001),
            "model_id": model_id,
            "dimension": expected_dim,
        }
