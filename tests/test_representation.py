"""Unit tests for embedding generation, caching, and article representations."""

import numpy as np
import pytest

from src.representation.article_representation import (
    build_article_representation,
    format_text_for_embedding,
    parse_timestamp_to_epoch,
)
from src.representation.cache import EmbeddingCacheManager
from src.representation.embedding import (
    BaseEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    TfidfEmbeddingProvider,
    get_embedding_provider,
)
from src.storage.repository import SQLiteRepository


class TestEmbeddingProviders:
    def test_tfidf_provider_dimension_and_norm(self):
        provider = TfidfEmbeddingProvider(dimension=128)
        assert provider.dimension == 128
        assert "tfidf" in provider.model_id

        texts = ["Headline about climate summit", "Another article on quantum tech"]
        embs = provider.embed_texts(texts)
        assert embs.shape == (2, 128)
        assert embs.dtype == np.float32

        # Check L2 unit normalization
        norm0 = np.linalg.norm(embs[0])
        norm1 = np.linalg.norm(embs[1])
        assert pytest.approx(norm0, rel=1e-3) == 1.0
        assert pytest.approx(norm1, rel=1e-3) == 1.0

    def test_empty_article_text_handling(self):
        provider = TfidfEmbeddingProvider(dimension=64)
        embs = provider.embed_texts(["", "   ", None])
        assert embs.shape == (3, 64)
        # Should not crash and should return valid float vectors
        assert not np.isnan(embs).any()

    def test_factory_fallback(self):
        provider = get_embedding_provider("tfidf")
        assert isinstance(provider, TfidfEmbeddingProvider)
        assert provider.dimension == 384


def insert_test_article(repo: SQLiteRepository, article_id: int, headline: str = "Test Headline") -> None:
    """Helper to insert a minimal valid article row for foreign key integrity in tests."""
    with repo._get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO articles (
                id, canonical_url, url, headline, headline_clean, headline_hash,
                description, content_clean, source_id, source_name, published_at, retrieved_at, is_sample, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_id,
                f"https://example.com/test/{article_id}",
                f"https://example.com/test/{article_id}",
                headline,
                headline.lower(),
                f"hash_{article_id}",
                f"Description for {article_id}",
                f"Content for {article_id}",
                "test_source",
                "Test Source",
                "2026-11-24T10:00:00Z",
                "2026-11-24T10:00:00Z",
                0,
                "2026-11-24T10:00:00Z",
            ),
        )
        conn.commit()


class TestEmbeddingCache:
    def test_cache_put_and_get(self, temp_db: SQLiteRepository):
        insert_test_article(temp_db, article_id=1)
        cache = EmbeddingCacheManager(temp_db)
        arr = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        text_hash = "abc123hash"
        model_id = "test_model"

        # First query should be cache miss
        cached = cache.get(article_id=1, text_hash=text_hash, model_id=model_id, expected_dim=4)
        assert cached is None

        # Put in cache
        cache.put(article_id=1, text_hash=text_hash, model_id=model_id, embedding=arr)

        # Retrieve
        cached = cache.get(article_id=1, text_hash=text_hash, model_id=model_id, expected_dim=4)
        assert cached is not None
        np.testing.assert_allclose(cached, arr, atol=1e-5)

    def test_cache_miss_on_hash_or_model_mismatch(self, temp_db: SQLiteRepository):
        insert_test_article(temp_db, article_id=2)
        cache = EmbeddingCacheManager(temp_db)
        arr = np.array([0.5, 0.5], dtype=np.float32)
        cache.put(article_id=2, text_hash="original_hash", model_id="model_a", embedding=arr)

        # Different text hash
        assert cache.get(article_id=2, text_hash="altered_hash", model_id="model_a", expected_dim=2) is None
        # Different model ID
        assert cache.get(article_id=2, text_hash="original_hash", model_id="model_b", expected_dim=2) is None
        # Dimension mismatch
        assert cache.get(article_id=2, text_hash="original_hash", model_id="model_a", expected_dim=128) is None

    def test_cache_multi_model_coexistence(self, temp_db: SQLiteRepository):
        """Verify that embeddings from different models coexist for the same article without overwriting."""
        insert_test_article(temp_db, article_id=3)
        cache = EmbeddingCacheManager(temp_db)
        arr_minilm = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        arr_harrier = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)

        # Store baseline MiniLM
        cache.put(article_id=3, text_hash="hash_3", model_id="all-MiniLM-L6-v2", embedding=arr_minilm)
        # Store modern Harrier
        cache.put(article_id=3, text_hash="hash_3", model_id="microsoft/harrier-oss-v1-270m", embedding=arr_harrier)

        # Both must be retrievable independently
        ret_minilm = cache.get(article_id=3, text_hash="hash_3", model_id="all-MiniLM-L6-v2", expected_dim=4)
        ret_harrier = cache.get(article_id=3, text_hash="hash_3", model_id="microsoft/harrier-oss-v1-270m", expected_dim=6)

        assert ret_minilm is not None
        assert ret_harrier is not None
        np.testing.assert_allclose(ret_minilm, arr_minilm, atol=1e-5)
        np.testing.assert_allclose(ret_harrier, arr_harrier, atol=1e-5)


class TestArticleRepresentation:
    def test_format_text_headline_weighting(self):
        hl = "Breaking News Event"
        desc = "Detailed explanation of the breaking news event."
        weighted = format_text_for_embedding(hl, desc, headline_weight=2)
        assert "Breaking News Event. Breaking News Event. Detailed explanation" in weighted

    def test_format_text_empty_description(self):
        weighted = format_text_for_embedding("Just A Headline", None)
        assert weighted == "Just A Headline"

    def test_parse_timestamp_to_epoch(self):
        iso_str = "2026-11-24T12:00:00Z"
        epoch = parse_timestamp_to_epoch(iso_str)
        assert epoch > 0
        # Invalid date falls back to current epoch without crashing
        assert parse_timestamp_to_epoch("invalid-date-string") > 0

    def test_build_representation_with_cache(self, temp_db: SQLiteRepository):
        insert_test_article(temp_db, article_id=10, headline="Historic Space Mission Launches")
        article_data = {
            "id": 10,
            "headline": "Historic Space Mission Launches",
            "description": "Astronauts lift off on lunar trajectory.",
            "canonical_url": "https://example.com/space/1",
            "source_id": "space_news",
            "published_at": "2026-11-24T10:00:00Z",
            "retrieved_at": "2026-11-24T10:05:00Z",
        }
        provider = TfidfEmbeddingProvider(dimension=64)
        cache = EmbeddingCacheManager(temp_db)

        rep1 = build_article_representation(
            article_dict=article_data,
            embedding_provider=provider,
            cache_manager=cache,
        )
        assert rep1.article_id == 10
        assert rep1.embedding.shape == (64,)
        assert rep1.published_epoch > 0

        # Second call should fetch from cache
        rep2 = build_article_representation(
            article_dict=article_data,
            embedding_provider=provider,
            cache_manager=cache,
        )
        np.testing.assert_allclose(rep1.embedding, rep2.embedding, atol=1e-5)
