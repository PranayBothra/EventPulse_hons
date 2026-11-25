"""Ingestion pipeline orchestrator for EventPulse."""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Set, Tuple

from src.config import FeedConfig, GlobalConfig, load_config, load_feeds
from src.ingestion.fallback import load_sample_articles
from src.ingestion.models import (
    FeedIngestionResult,
    PipelineIngestionSummary,
    ProcessedArticle,
    RawFeedItem,
)
from src.ingestion.rss_client import RSSClient
from src.preprocessing.date_parser import parse_datetime_to_utc
from src.preprocessing.deduplicator import (
    canonicalize_url,
    compute_headline_hash,
)
from src.preprocessing.text_cleaner import clean_html, normalize_text
from src.storage.repository import SQLiteRepository


class IngestionPipeline:
    """Coordinates feed fetching, preprocessing, deduplication, and persistence."""

    def __init__(
        self,
        config: Optional[GlobalConfig] = None,
        repository: Optional[SQLiteRepository] = None,
        rss_client: Optional[RSSClient] = None,
    ):
        self.config = config or load_config()
        self.repository = repository or SQLiteRepository(
            db_path=self.config.database.path,
            timeout=self.config.database.timeout_seconds,
        )
        self.rss_client = rss_client or RSSClient(
            user_agent=self.config.ingestion.user_agent,
            timeout=self.config.ingestion.request_timeout_seconds,
            max_retries=self.config.ingestion.max_retries,
            backoff_factor=self.config.ingestion.retry_backoff_seconds,
        )
        self.logger = logging.getLogger("eventpulse.pipeline")

    def run(
        self,
        feed_id: Optional[str] = None,
        use_sample_fallback: bool = False,
    ) -> PipelineIngestionSummary:
        """Execute the ingestion pipeline across configured feeds or fallback dataset.

        Args:
            feed_id: Optional specific feed ID to ingest. If None, ingests all enabled feeds.
            use_sample_fallback: If True, ingests the explicitly labelled sample dataset.

        Returns:
            PipelineIngestionSummary containing detailed run metrics.
        """
        started_at = datetime.now(timezone.utc).isoformat()
        self.logger.info("Starting EventPulse ingestion pipeline at %s", started_at)

        if use_sample_fallback:
            return self._run_sample_fallback(started_at)

        feeds = load_feeds(self.config.ingestion.feeds_file)
        if not feeds:
            self.logger.warning("No feeds loaded from %s; running fallback dataset.", self.config.ingestion.feeds_file)
            return self._run_sample_fallback(started_at)

        # Filter specific feed if requested
        if feed_id:
            feeds = [f for f in feeds if f.id == feed_id]
            if not feeds:
                raise ValueError(f"Configured feed with ID '{feed_id}' not found.")

        feed_results: List[FeedIngestionResult] = []
        total_fetched = 0
        total_stored = 0
        total_duplicates = 0
        total_malformed = 0

        # Register all feeds in storage
        for feed in feeds:
            self.repository.upsert_feed(
                feed_id=feed.id,
                name=feed.name,
                url=feed.url,
                category=feed.category,
                enabled=feed.enabled,
            )

        for feed in feeds:
            if not feed.enabled:
                self.logger.info("Skipping disabled feed: %s (%s)", feed.name, feed.id)
                continue

            result = self._process_single_feed(feed)
            feed_results.append(result)
            total_fetched += result.items_fetched
            total_stored += result.items_stored
            total_duplicates += result.items_duplicate
            total_malformed += result.items_malformed

        # If all live feeds failed and fetched 0 items, offer graceful fallback
        if total_fetched == 0 and not feed_id:
            self.logger.warning("All live feeds failed or returned 0 items. Triggering fallback sample dataset.")
            fallback_summary = self._run_sample_fallback(started_at)
            # Combine results
            fallback_summary.feed_results.extend(feed_results)
            return fallback_summary

        completed_at = datetime.now(timezone.utc).isoformat()
        overall_status = "SUCCESS"
        if any(r.status == "FAILED" for r in feed_results):
            overall_status = "PARTIAL" if total_stored > 0 else "FAILED"

        # Record overall ingestion run in SQLite
        self.repository.record_ingestion_run(
            {
                "started_at": started_at,
                "completed_at": completed_at,
                "source": feed_id or "all_configured_feeds",
                "items_fetched": total_fetched,
                "items_stored": total_stored,
                "items_duplicate": total_duplicates,
                "items_malformed": total_malformed,
                "status": overall_status,
                "error_message": None if overall_status == "SUCCESS" else "One or more feeds encountered issues",
            }
        )

        summary = PipelineIngestionSummary(
            started_at=started_at,
            completed_at=completed_at,
            total_feeds_processed=len(feed_results),
            total_items_fetched=total_fetched,
            total_items_stored=total_stored,
            total_items_duplicate=total_duplicates,
            total_items_malformed=total_malformed,
            feed_results=feed_results,
            overall_status=overall_status,
        )

        self.logger.info(
            "Ingestion pipeline complete: %d fetched, %d stored, %d duplicates, %d malformed (Status: %s)",
            total_fetched,
            total_stored,
            total_duplicates,
            total_malformed,
            overall_status,
        )
        return summary

    def _process_single_feed(self, feed: FeedConfig) -> FeedIngestionResult:
        """Fetch, preprocess, deduplicate, and persist entries from a single RSS feed."""
        self.logger.info("Processing feed: %s (%s)", feed.name, feed.url)
        fetched_count = 0
        stored_count = 0
        duplicate_count = 0
        malformed_count = 0
        error_msg = None
        status = "SUCCESS"

        try:
            raw_items = self.rss_client.parse_feed(
                feed_id=feed.id,
                feed_name=feed.name,
                url=feed.url,
                max_items=self.config.ingestion.max_articles_per_feed,
            )
            fetched_count = len(raw_items)
        except Exception as e:
            self.logger.error("Error fetching feed '%s': %s", feed.name, e)
            self.repository.update_feed_status(feed.id, status="ERROR")
            return FeedIngestionResult(
                feed_id=feed.id,
                feed_name=feed.name,
                status="FAILED",
                items_fetched=0,
                items_stored=0,
                items_duplicate=0,
                items_malformed=0,
                error_message=str(e),
            )

        # Preprocess and deduplicate
        valid_articles: List[ProcessedArticle] = []
        batch_seen_urls: Set[str] = set()
        batch_seen_hashes: Set[str] = set()

        for raw_item in raw_items:
            processed = self._preprocess_item(raw_item, feed)
            if processed is None:
                malformed_count += 1
                continue

            # In-batch duplicate check
            if (
                processed.canonical_url in batch_seen_urls
                or processed.headline_hash in batch_seen_hashes
            ):
                duplicate_count += 1
                continue

            batch_seen_urls.add(processed.canonical_url)
            batch_seen_hashes.add(processed.headline_hash)
            valid_articles.append(processed)

        # Batch insert into SQLite with persistent duplicate prevention
        if valid_articles:
            articles_to_insert = [a.model_dump() for a in valid_articles]
            batch_stored, batch_duplicates = self.repository.batch_insert_articles(
                articles_to_insert
            )
            stored_count += batch_stored
            duplicate_count += batch_duplicates

        status = "SUCCESS" if malformed_count == 0 else "PARTIAL"
        self.repository.update_feed_status(feed.id, status=status)

        return FeedIngestionResult(
            feed_id=feed.id,
            feed_name=feed.name,
            status=status,
            items_fetched=fetched_count,
            items_stored=stored_count,
            items_duplicate=duplicate_count,
            items_malformed=malformed_count,
            error_message=error_msg,
        )

    def _preprocess_item(
        self, item: RawFeedItem, feed: FeedConfig
    ) -> Optional[ProcessedArticle]:
        """Validate, normalize, and transform a raw item into a ProcessedArticle.

        Returns:
            ProcessedArticle if valid, None if malformed.
        """
        raw_title = item.title or ""
        raw_link = item.link or ""

        # Validate basic required fields
        if not raw_title.strip() or not raw_link.strip():
            self.logger.debug("Malformed item skipped: missing title or link")
            return None

        # Clean and normalize headline
        headline_clean = normalize_text(
            raw_title,
            normalize_unicode=self.config.preprocessing.normalize_unicode,
            strip_html=self.config.preprocessing.strip_html_tags,
            max_length=self.config.preprocessing.max_headline_chars,
        )

        if len(headline_clean) < self.config.preprocessing.min_headline_chars:
            self.logger.debug(
                "Headline too short (%d chars): '%s'",
                len(headline_clean),
                headline_clean,
            )
            return None

        # Canonicalize URL
        canonical = canonicalize_url(raw_link)
        if not canonical:
            return None

        # Clean description / content
        desc_clean = normalize_text(
            item.description,
            normalize_unicode=self.config.preprocessing.normalize_unicode,
            strip_html=self.config.preprocessing.strip_html_tags,
        )

        # Parse publication date to UTC with fallback to retrieved_at
        pub_dt_utc = parse_datetime_to_utc(
            item.published_parsed or item.published_raw
        )

        # Compute headline fingerprint hash
        headline_hash = compute_headline_hash(headline_clean)

        return ProcessedArticle(
            canonical_url=canonical,
            url=raw_link,
            headline=raw_title.strip(),
            headline_clean=headline_clean,
            headline_hash=headline_hash,
            description=desc_clean,
            content_clean=desc_clean,
            source_id=feed.id,
            source_name=feed.name,
            published_at=pub_dt_utc,
            retrieved_at=item.retrieved_at,
            is_sample=False,
            raw_metadata=item.raw_metadata,
        )

    def _run_sample_fallback(self, started_at: str) -> PipelineIngestionSummary:
        """Run ingestion using the explicitly labelled fallback sample dataset."""
        self.logger.info("Executing ingestion with explicitly labelled sample dataset.")
        sample_articles = load_sample_articles(self.config.ingestion.sample_data_file)
        stored, duplicates = self.repository.batch_insert_articles(
            [a.model_dump() for a in sample_articles]
        )

        completed_at = datetime.now(timezone.utc).isoformat()
        result = FeedIngestionResult(
            feed_id="sample_fallback_dataset",
            feed_name="Explicitly Labelled Sample Archive",
            status="SUCCESS",
            items_fetched=len(sample_articles),
            items_stored=stored,
            items_duplicate=duplicates,
            items_malformed=0,
            error_message=None,
        )

        self.repository.record_ingestion_run(
            {
                "started_at": started_at,
                "completed_at": completed_at,
                "source": "sample_fallback_dataset",
                "items_fetched": len(sample_articles),
                "items_stored": stored,
                "items_duplicate": duplicates,
                "items_malformed": 0,
                "status": "SUCCESS",
                "error_message": "Ingested explicitly labelled fallback sample data",
            }
        )

        return PipelineIngestionSummary(
            started_at=started_at,
            completed_at=completed_at,
            total_feeds_processed=1,
            total_items_fetched=len(sample_articles),
            total_items_stored=stored,
            total_items_duplicate=duplicates,
            total_items_malformed=0,
            feed_results=[result],
            overall_status="SUCCESS",
        )

    def ingest_processed_batch(
        self,
        articles: List[ProcessedArticle],
        source_label: str,
        feed_id: str,
        feed_name: str,
        fetched_count: int,
        malformed_count: int = 0,
        pre_skipped_duplicates: int = 0,
        started_at: Optional[str] = None,
    ) -> PipelineIngestionSummary:
        """Persist externally discovered articles through unified deduplication and storage."""
        start_ts = started_at or datetime.now(timezone.utc).isoformat()
        valid_articles: List[ProcessedArticle] = []
        seen_urls: Set[str] = set()
        seen_hashes: Set[str] = set()
        duplicate_count = pre_skipped_duplicates

        for art in articles:
            if art.canonical_url in seen_urls or art.headline_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_urls.add(art.canonical_url)
            seen_hashes.add(art.headline_hash)
            valid_articles.append(art)

        stored_count = 0
        if valid_articles:
            batch_stored, batch_dups = self.repository.batch_insert_articles(
                [a.model_dump() for a in valid_articles]
            )
            stored_count += batch_stored
            duplicate_count += batch_dups

        completed_at = datetime.now(timezone.utc).isoformat()
        overall_status = "SUCCESS" if malformed_count == 0 else ("PARTIAL" if stored_count > 0 or duplicate_count > 0 else "FAILED")

        self.repository.record_ingestion_run(
            {
                "started_at": start_ts,
                "completed_at": completed_at,
                "source": source_label,
                "items_fetched": fetched_count,
                "items_stored": stored_count,
                "items_duplicate": duplicate_count,
                "items_malformed": malformed_count,
                "status": overall_status,
                "error_message": None,
            }
        )

        result = FeedIngestionResult(
            feed_id=feed_id,
            feed_name=feed_name,
            status=overall_status,
            items_fetched=fetched_count,
            items_stored=stored_count,
            items_duplicate=duplicate_count,
            items_malformed=malformed_count,
        )
        return PipelineIngestionSummary(
            started_at=start_ts,
            completed_at=completed_at,
            total_feeds_processed=1,
            total_items_fetched=fetched_count,
            total_items_stored=stored_count,
            total_items_duplicate=duplicate_count,
            total_items_malformed=malformed_count,
            feed_results=[result],
            overall_status=overall_status,
        )
