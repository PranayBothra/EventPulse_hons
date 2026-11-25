"""Unit tests for RSSClient, fallback loader, and IngestionPipeline."""

from unittest.mock import MagicMock, patch
import pytest

from src.config import FeedConfig, GlobalConfig
from src.ingestion.fallback import load_sample_articles
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.rss_client import RSSClient
from src.storage.repository import SQLiteRepository
from tests.conftest import MALFORMED_RSS_XML, SAMPLE_RSS_XML


class TestRSSClient:
    def test_parse_valid_rss(self):
        client = RSSClient()
        items = client._parse_with_feedparser(
            xml_content=SAMPLE_RSS_XML,
            feed_id="test_feed",
            feed_name="Test Feed",
            retrieved_at="2026-11-24T09:00:00Z",
        )
        assert len(items) == 2
        assert items[0].title == "Global Leaders Convene for Climate Summit in Geneva"
        assert items[0].link.startswith("https://testnews.example.com/world/climate-summit-2026")
        assert "carbon reduction" in items[0].description
        assert items[0].source_id == "test_feed"

    def test_element_tree_fallback_parser(self):
        client = RSSClient()
        items = client._parse_with_element_tree(
            xml_content=SAMPLE_RSS_XML,
            feed_id="test_feed",
            feed_name="Test Feed",
            retrieved_at="2026-11-24T09:00:00Z",
        )
        assert len(items) == 2
        assert items[1].title == "Breakthrough in Fusion Energy Net Gain Reported"
        assert items[1].link == "https://testnews.example.com/science/fusion-breakthrough"

    def test_handle_malformed_feed(self):
        client = RSSClient()
        items = client._parse_with_feedparser(
            xml_content=MALFORMED_RSS_XML,
            feed_id="broken_feed",
            feed_name="Broken Feed",
            retrieved_at="2026-11-24T09:00:00Z",
        )
        # Should parse entries without crashing even if fields are missing or dates invalid
        assert len(items) == 2


class TestFallbackLoader:
    def test_load_sample_articles(self):
        samples = load_sample_articles()
        assert len(samples) >= 3
        for s in samples:
            assert s.is_sample is True
            assert s.headline != ""
            assert s.canonical_url.startswith("http")
            assert s.published_at.endswith("Z")
            assert s.headline_hash != ""


class TestIngestionPipeline:
    def test_pipeline_with_sample_fallback(
        self, temp_db: SQLiteRepository, test_config: GlobalConfig
    ):
        pipeline = IngestionPipeline(config=test_config, repository=temp_db)
        summary = pipeline.run(use_sample_fallback=True)

        assert summary.overall_status == "SUCCESS"
        assert summary.total_items_stored > 0
        assert summary.total_items_duplicate == 0

        # Verify items stored in database are marked is_sample=1
        stats = temp_db.get_database_stats()
        assert stats["total_articles"] == summary.total_items_stored
        assert stats["sample_articles"] == summary.total_items_stored
        assert stats["live_articles"] == 0

        # Repeated run with sample data should result in 0 new stored, all duplicates
        second_summary = pipeline.run(use_sample_fallback=True)
        assert second_summary.total_items_stored == 0
        assert second_summary.total_items_duplicate == summary.total_items_stored
        assert temp_db.get_article_count() == stats["total_articles"]

    @patch("src.ingestion.rss_client.RSSClient.fetch_feed_raw")
    def test_pipeline_with_mocked_feed(
        self,
        mock_fetch: MagicMock,
        temp_db: SQLiteRepository,
        test_config: GlobalConfig,
        sample_feed: FeedConfig,
    ):
        mock_fetch.return_value = SAMPLE_RSS_XML

        # Patch load_feeds to return our sample feed
        with patch("src.ingestion.pipeline.load_feeds", return_value=[sample_feed]):
            pipeline = IngestionPipeline(config=test_config, repository=temp_db)
            summary = pipeline.run()

            assert summary.overall_status == "SUCCESS"
            assert summary.total_items_fetched == 2
            assert summary.total_items_stored == 2
            assert summary.total_items_duplicate == 0

            # Verify in DB
            assert temp_db.get_article_count() == 2
            stats = temp_db.get_database_stats()
            assert stats["live_articles"] == 2
            assert stats["sample_articles"] == 0

            # Second identical run must be idempotent: 0 stored, 2 duplicates
            second_summary = pipeline.run()
            assert second_summary.total_items_stored == 0
            assert second_summary.total_items_duplicate == 2
            assert temp_db.get_article_count() == 2
