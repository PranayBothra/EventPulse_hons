"""Unit tests for SQLite storage layer and repository."""

import pytest
from src.storage.repository import SQLiteRepository


class TestSQLiteRepository:
    def test_schema_initialized(self, temp_db: SQLiteRepository):
        stats = temp_db.get_database_stats()
        assert stats["total_articles"] == 0
        assert stats["total_feeds"] == 0
        assert stats["total_ingestion_runs"] == 0

    def test_upsert_and_update_feed(self, temp_db: SQLiteRepository):
        temp_db.upsert_feed(
            feed_id="test_feed",
            name="Test Feed",
            url="https://example.com/rss",
            category="world",
            enabled=True,
        )
        assert temp_db.get_database_stats()["total_feeds"] == 1

        temp_db.update_feed_status("test_feed", status="SUCCESS")
        with temp_db._get_connection() as conn:
            row = conn.execute("SELECT * FROM feeds WHERE id = 'test_feed'").fetchone()
            assert row["last_status"] == "SUCCESS"
            assert row["last_fetched_at"] is not None

    def test_insert_article_success(self, temp_db: SQLiteRepository):
        article = {
            "canonical_url": "https://example.com/news/1",
            "url": "https://example.com/news/1?utm_source=rss",
            "headline": "Historic Treaty Signed",
            "headline_clean": "historic treaty signed",
            "headline_hash": "hash_12345",
            "description": "Details about the historic treaty.",
            "content_clean": "Details about the historic treaty.",
            "source_id": "test_feed",
            "source_name": "Test News",
            "published_at": "2026-11-24T10:00:00Z",
            "retrieved_at": "2026-11-24T10:05:00Z",
            "is_sample": False,
        }
        inserted = temp_db.insert_article(article)
        assert inserted is True
        assert temp_db.get_article_count() == 1

        fetched = temp_db.get_article_by_canonical_url("https://example.com/news/1")
        assert fetched is not None
        assert fetched["headline"] == "Historic Treaty Signed"
        assert fetched["is_sample"] == 0

    def test_insert_duplicate_article_rejected(self, temp_db: SQLiteRepository):
        article = {
            "canonical_url": "https://example.com/news/dup",
            "url": "https://example.com/news/dup",
            "headline": "First Ingestion of Breaking Story",
            "headline_clean": "first ingestion of breaking story",
            "headline_hash": "hash_dup",
            "description": "Breaking story summary.",
            "source_id": "test_feed",
            "source_name": "Test News",
            "published_at": "2026-11-24T11:00:00Z",
            "retrieved_at": "2026-11-24T11:01:00Z",
            "is_sample": False,
        }
        first = temp_db.insert_article(article)
        assert first is True

        # Second attempt with same canonical URL
        second = temp_db.insert_article(article)
        assert second is False
        assert temp_db.get_article_count() == 1

    def test_batch_insert_idempotency(self, temp_db: SQLiteRepository):
        articles = [
            {
                "canonical_url": f"https://example.com/news/{i}",
                "url": f"https://example.com/news/{i}",
                "headline": f"News Headline {i}",
                "headline_clean": f"news headline {i}",
                "headline_hash": f"hash_{i}",
                "description": f"Description {i}",
                "source_id": "batch_feed",
                "source_name": "Batch Feed",
                "published_at": "2026-11-24T12:00:00Z",
                "retrieved_at": "2026-11-24T12:01:00Z",
                "is_sample": False,
            }
            for i in range(5)
        ]

        stored, dups = temp_db.batch_insert_articles(articles)
        assert stored == 5
        assert dups == 0
        assert temp_db.get_article_count() == 5

        # Repeated ingestion of identical batch
        stored_repeat, dups_repeat = temp_db.batch_insert_articles(articles)
        assert stored_repeat == 0
        assert dups_repeat == 5
        assert temp_db.get_article_count() == 5

    def test_record_ingestion_run(self, temp_db: SQLiteRepository):
        run_id = temp_db.record_ingestion_run(
            {
                "started_at": "2026-11-24T12:00:00Z",
                "completed_at": "2026-11-24T12:00:05Z",
                "source": "all_feeds",
                "items_fetched": 15,
                "items_stored": 12,
                "items_duplicate": 3,
                "items_malformed": 0,
                "status": "SUCCESS",
            }
        )
        assert run_id > 0
        runs = temp_db.get_latest_ingestion_runs(limit=5)
        assert len(runs) == 1
        assert runs[0]["items_stored"] == 12
