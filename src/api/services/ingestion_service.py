"""Unified news ingestion orchestration for RSS feeds and FreeNewsAPI.io."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.ingestion.freenewsapi_client import (
    DEFAULT_COUNTRY,
    DEFAULT_LANGUAGE,
    DEFAULT_TOPICS,
    FreeNewsAPIClient,
    FreeNewsAPIConfigError,
    FreeNewsAPIError,
    is_freenewsapi_configured,
)
from src.ingestion.pipeline import IngestionPipeline
from src.storage.repository import SQLiteRepository

logger = logging.getLogger("eventpulse.api.ingestion")


class IngestionInProgressError(RuntimeError):
    """Raised when an ingestion job is triggered while one is already active."""


class IngestionService:
    """Coordinates RSS and FreeNewsAPI ingestion through EventPulse's shared pipeline."""

    def __init__(
        self,
        repository: SQLiteRepository,
        pipeline: IngestionPipeline,
        freenewsapi_client: Optional[FreeNewsAPIClient] = None,
    ):
        self.repo = repository
        self.pipeline = pipeline
        self.freenewsapi_client = freenewsapi_client or FreeNewsAPIClient()
        self._lock = threading.Lock()
        self._running = False
        self._state: Dict[str, Dict[str, Any]] = {
            "freenewsapi": {
                "enabled": True,
                "last_sync_at": None,
                "last_status": None,
                "last_error": None,
                "articles_fetched": 0,
                "articles_inserted": 0,
                "duplicates_skipped": 0,
                "failed_records": 0,
                "quota": {},
            },
            "rss": {
                "enabled": True,
                "last_sync_at": None,
                "last_status": None,
                "last_error": None,
                "articles_fetched": 0,
                "articles_inserted": 0,
                "duplicates_skipped": 0,
                "failed_records": 0,
                "quota": {},
            },
        }
        self._hydrate_initial_state()

    def _hydrate_initial_state(self) -> None:
        """Seed status fields from persisted SQLite ingestion_runs on startup."""
        try:
            runs = self.repo.get_latest_ingestion_runs(limit=20)
        except Exception:
            return
        for r in runs:
            src = str(r.get("source") or "")
            key = "freenewsapi" if src.startswith("freenewsapi") else "rss"
            if self._state[key]["last_sync_at"] is None:
                self._state[key]["last_sync_at"] = r.get("completed_at") or r.get("started_at")
                self._state[key]["last_status"] = r.get("status")
                self._state[key]["last_error"] = r.get("error_message")
                self._state[key]["articles_fetched"] = int(r.get("items_fetched") or 0)
                self._state[key]["articles_inserted"] = int(r.get("items_stored") or 0)
                self._state[key]["duplicates_skipped"] = int(r.get("items_duplicate") or 0)
                self._state[key]["failed_records"] = int(r.get("items_malformed") or 0)

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self, clustering_running: bool = False) -> Dict[str, Any]:
        fn_cred = bool(self.freenewsapi_client._explicit_key) or is_freenewsapi_configured(
            self.freenewsapi_client._env_path
        )
        fn_state = dict(self._state["freenewsapi"])
        if self.freenewsapi_client.last_quota:
            fn_state["quota"] = dict(self.freenewsapi_client.last_quota)

        return {
            "ingestion_job_running": self._running,
            "clustering_job_running": clustering_running,
            "freenewsapi": {
                "provider": "freenewsapi",
                "credentials_configured": fn_cred,
                "job_running": self._running,
                **fn_state,
            },
            "rss": {
                "provider": "rss",
                "credentials_configured": True,
                "job_running": self._running,
                **self._state["rss"],
            },
        }

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.repo.get_latest_ingestion_runs(limit=limit)

    def run_rss(self, feed_id: Optional[str] = None, use_sample: bool = False) -> Dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise IngestionInProgressError("An ingestion job is already running.")

        self._running = True
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            summary = self.pipeline.run(feed_id=feed_id, use_sample_fallback=use_sample)
            self._state["rss"].update(
                {
                    "last_sync_at": summary.completed_at,
                    "last_status": summary.overall_status,
                    "last_error": None,
                    "articles_fetched": summary.total_items_fetched,
                    "articles_inserted": summary.total_items_stored,
                    "duplicates_skipped": summary.total_items_duplicate,
                    "failed_records": summary.total_items_malformed,
                }
            )
            return {
                "provider": "rss",
                "status": summary.overall_status,
                "started_at": summary.started_at,
                "completed_at": summary.completed_at,
                "articles_fetched": summary.total_items_fetched,
                "articles_inserted": summary.total_items_stored,
                "duplicates_skipped": summary.total_items_duplicate,
                "failed_or_malformed": summary.total_items_malformed,
                "details": {
                    "feeds_processed": summary.total_feeds_processed,
                    "feed_id_filter": feed_id,
                    "use_sample": use_sample,
                },
            }
        except Exception as exc:
            self._state["rss"].update(
                {
                    "last_sync_at": datetime.now(timezone.utc).isoformat(),
                    "last_status": "FAILED",
                    "last_error": str(exc),
                }
            )
            raise
        finally:
            self._running = False
            self._lock.release()

    def run_freenewsapi(
        self,
        country: Optional[str] = DEFAULT_COUNTRY,
        language: Optional[str] = DEFAULT_LANGUAGE,
        topics: Optional[List[str]] = None,
        in_title: Optional[str] = None,
        max_pages_per_topic: int = 1,
        max_details_to_fetch: int = 20,
    ) -> Dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise IngestionInProgressError("An ingestion job is already running.")

        self._running = True
        started_at = datetime.now(timezone.utc).isoformat()
        target_topics = topics if topics is not None else list(DEFAULT_TOPICS)
        try:
            discovery = self.freenewsapi_client.discover_and_fetch_new_articles(
                repository=self.repo,
                country=country,
                language=language,
                topics=target_topics,
                in_title=in_title,
                max_pages_per_topic=max_pages_per_topic,
                max_details_to_fetch=max_details_to_fetch,
            )
            processed_articles = discovery["processed_articles"]
            failed_total = discovery["details_failed"] + discovery["malformed_skipped"]
            summary = self.pipeline.ingest_processed_batch(
                articles=processed_articles,
                source_label=f"freenewsapi:{country or 'ALL'}:{language or 'all'}",
                feed_id="freenewsapi_live",
                feed_name="FreeNewsAPI.io Live Stream",
                fetched_count=discovery["listings_fetched"],
                malformed_count=failed_total,
                pre_skipped_duplicates=discovery["duplicates_skipped_before_details"],
                started_at=started_at,
            )
            status_val = summary.overall_status
            if discovery["listings_fetched"] == 0:
                status_val = "EMPTY"

            self._state["freenewsapi"].update(
                {
                    "last_sync_at": summary.completed_at,
                    "last_status": status_val,
                    "last_error": None,
                    "articles_fetched": discovery["listings_fetched"],
                    "articles_inserted": summary.total_items_stored,
                    "duplicates_skipped": summary.total_items_duplicate,
                    "failed_records": failed_total,
                    "quota": discovery.get("quota", {}),
                }
            )
            return {
                "provider": "freenewsapi",
                "status": status_val,
                "started_at": summary.started_at,
                "completed_at": summary.completed_at,
                "articles_fetched": discovery["listings_fetched"],
                "articles_inserted": summary.total_items_stored,
                "duplicates_skipped": summary.total_items_duplicate,
                "failed_or_malformed": failed_total,
                "details": {
                    "country": discovery["country"],
                    "language": discovery["language"],
                    "topics": discovery["topics"],
                    "topic_listing_counts": discovery["topic_listing_counts"],
                    "unique_discovered": discovery["unique_discovered"],
                    "details_requested": discovery["details_requested"],
                    "quota": discovery.get("quota", {}),
                },
            }
        except FreeNewsAPIConfigError as exc:
            self._state["freenewsapi"].update(
                {
                    "last_sync_at": datetime.now(timezone.utc).isoformat(),
                    "last_status": "UNCONFIGURED",
                    "last_error": str(exc),
                }
            )
            raise
        except FreeNewsAPIError as exc:
            self._state["freenewsapi"].update(
                {
                    "last_sync_at": datetime.now(timezone.utc).isoformat(),
                    "last_status": "FAILED",
                    "last_error": str(exc),
                }
            )
            raise
        finally:
            self._running = False
            self._lock.release()
