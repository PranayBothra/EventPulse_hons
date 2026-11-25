"""Unit tests for BatchEmbedder."""

import numpy as np
import pytest

from src.representation.batch_embedder import BatchEmbedder
from src.representation.embedding import TfidfEmbeddingProvider
from src.storage.repository import SQLiteRepository


@pytest.fixture
def temp_repo_with_articles(tmp_path):
    db_path = tmp_path / "test.db"
    repo = SQLiteRepository(str(db_path))

    # Insert 5 test articles
    articles = [
        {
            "canonical_url": f"https://example.com/test-{i}",
            "url": f"https://example.com/test-{i}",
            "headline": f"Global economic summit announcement {i}",
            "headline_clean": f"Global economic summit announcement {i}",
            "headline_hash": f"hash_{i}",
            "description": f"Details about the economic summit {i}",
            "content_clean": f"Details about the economic summit {i}",
            "source_id": "test_source",
            "source_name": "Test Source",
            "published_at": "2026-09-01T12:00:00Z",
            "is_sample": 0,
        }
        for i in range(5)
    ]
    repo.batch_insert_articles(articles)
    return repo


def test_batch_embedder_flow(temp_repo_with_articles):
    repo = temp_repo_with_articles
    provider = TfidfEmbeddingProvider(dimension=64)
    embedder = BatchEmbedder(repository=repo, provider=provider, batch_size=2)

    uncached = embedder.get_uncached_articles()
    assert len(uncached) == 5

    res = embedder.embed_and_cache_all_uncached()
    assert res["embedded"] == 5

    # Check that subsequent call finds 0 uncached
    assert len(embedder.get_uncached_articles()) == 0

    # Verify cached embeddings
    for i in range(1, 6):
        blob = repo.get_embedding_by_article_id(i)
        assert blob is not None
        vec = np.frombuffer(blob, dtype=np.float32)
        assert vec.shape[0] == 64
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)
