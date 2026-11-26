"""Background live-sync scheduler for periodic news ingestion and batch-thresholded clustering."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.api.services.event_service import ClusteringInProgressError, EventService
from src.api.services.ingestion_service import IngestionInProgressError, IngestionService
from src.config import LiveSyncConfig
from src.ingestion.freenewsapi_client import FreeNewsAPIConfigError, FreeNewsAPIError
from src.storage.repository import SQLiteRepository

logger = logging.getLogger("eventpulse.api.live_sync")


def _iso_from_epoch(epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


class LiveSyncScheduler:
    """Runs periodic RSS/FreeNewsAPI ingestion and batch-triggered clustering in the background."""

    def __init__(
        self,
        repository: SQLiteRepository,
        ingestion_service: IngestionService,
        event_service: EventService,
        config: Optional[LiveSyncConfig] = None,
        enabled_override: Optional[bool] = None,
    ):
        self.repo = repository
        self.ingestion_service = ingestion_service
        self.event_service = event_service
        self.config = config or LiveSyncConfig()

        self._enabled = (
            bool(enabled_override) if enabled_override is not None else bool(self.config.enabled)
        )
        self._paused = False
        self._rss_interval = max(60, int(self.config.rss_interval_seconds))
        self._fn_interval = max(300, int(self.config.freenewsapi_interval_seconds))
        self._auto_cluster_enabled = bool(self.config.auto_cluster_enabled)
        self._auto_cluster_min_batch = max(1, int(self.config.auto_cluster_min_batch))
        self._fn_min_quota = max(0, int(self.config.freenewsapi_min_remaining_quota))

        self._state_lock = threading.Lock()
        self._data_version = 1
        self._last_auto_sync_at: Optional[str] = None
        self._last_auto_cluster_at: Optional[str] = None
        self._last_auto_cluster_count: int = 0
        self._last_error: Optional[str] = None

        self._next_rss_epoch: Optional[float] = None
        self._next_fn_epoch: Optional[float] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event: Optional[asyncio.Event] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def bump_data_version(self) -> int:
        """Increment monotonic content version whenever new articles, events, or interactions land."""
        with self._state_lock:
            self._data_version += 1
            return self._data_version

    def configure(
        self,
        paused: Optional[bool] = None,
        rss_interval_seconds: Optional[int] = None,
        freenewsapi_interval_seconds: Optional[int] = None,
        auto_cluster_enabled: Optional[bool] = None,
        auto_cluster_min_batch: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update live-sync runtime settings without restarting the server."""
        now = time.time()
        with self._state_lock:
            if paused is not None:
                self._paused = bool(paused)
            if rss_interval_seconds is not None:
                self._rss_interval = max(60, int(rss_interval_seconds))
                if self._next_rss_epoch is not None:
                    self._next_rss_epoch = min(self._next_rss_epoch, now + self._rss_interval)
            if freenewsapi_interval_seconds is not None:
                self._fn_interval = max(300, int(freenewsapi_interval_seconds))
                if self._next_fn_epoch is not None:
                    self._next_fn_epoch = min(self._next_fn_epoch, now + self._fn_interval)
            if auto_cluster_enabled is not None:
                self._auto_cluster_enabled = bool(auto_cluster_enabled)
            if auto_cluster_min_batch is not None:
                self._auto_cluster_min_batch = max(1, int(auto_cluster_min_batch))
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Return lightweight scheduler telemetry for the UI status endpoint."""
        try:
            pending_unassigned = self.repo.get_unassigned_article_count(is_sample=False)
        except Exception:
            pending_unassigned = 0

        with self._state_lock:
            return {
                "enabled": self._enabled,
                "active": self.is_running and not self._paused,
                "paused": self._paused,
                "rss_interval_seconds": self._rss_interval,
                "freenewsapi_interval_seconds": self._fn_interval,
                "auto_cluster_enabled": self._auto_cluster_enabled,
                "auto_cluster_min_batch": self._auto_cluster_min_batch,
                "pending_unassigned_articles": pending_unassigned,
                "data_version": self._data_version,
                "next_rss_sync_at": _iso_from_epoch(self._next_rss_epoch) if (self._enabled and not self._paused) else None,
                "next_freenewsapi_sync_at": _iso_from_epoch(self._next_fn_epoch) if (self._enabled and not self._paused) else None,
                "last_auto_sync_at": self._last_auto_sync_at,
                "last_auto_cluster_at": self._last_auto_cluster_at,
                "last_auto_cluster_count": self._last_auto_cluster_count,
                "last_error": self._last_error,
            }

    async def start(self) -> None:
        """Start background periodic loop inside FastAPI lifespan."""
        if not self._enabled or self.is_running:
            return
        now = time.time()
        delay = max(1, int(self.config.startup_delay_seconds))
        with self._state_lock:
            if self.config.run_on_startup:
                self._next_rss_epoch = now + delay
                self._next_fn_epoch = now + delay + 5
            else:
                self._next_rss_epoch = now + self._rss_interval
                self._next_fn_epoch = now + self._fn_interval

        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._loop(), name="eventpulse-live-sync")
        logger.info(
            "Background live-sync started (rss=%ss, freenewsapi=%ss, cluster_batch>=%s).",
            self._rss_interval,
            self._fn_interval,
            self._auto_cluster_min_batch,
        )

    async def stop(self) -> None:
        """Stop background periodic loop cleanly on server shutdown."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Background live-sync stopped.")

    async def _loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self.run_due_tasks()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Unexpected error in live-sync loop: %s", exc)
                with self._state_lock:
                    self._last_error = str(exc)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

    def _should_run_freenewsapi(self) -> bool:
        status = self.ingestion_service.get_status(
            clustering_running=self.event_service.is_clustering_running
        )
        fn = status.get("freenewsapi", {})
        if not fn.get("credentials_configured"):
            return False
        quota = fn.get("quota") or {}
        remaining = quota.get("daily_remaining")
        if remaining is not None and int(remaining) <= self._fn_min_quota:
            return False
        return True

    def maybe_run_batch_clustering_sync(self) -> int:
        """Run clustering synchronously if auto-clustering is enabled and batch threshold is met."""
        with self._state_lock:
            enabled = self._auto_cluster_enabled
            cluster_batch = self._auto_cluster_min_batch
        if not enabled:
            return 0

        pending_count = self.repo.get_unassigned_article_count(is_sample=False)
        if pending_count < cluster_batch:
            return 0

        try:
            c_res = self.event_service.run_clustering(include_sample=False)
            clustered_total = int(c_res.get("articles_clustered") or 0)
            if clustered_total > 0:
                with self._state_lock:
                    self._last_auto_cluster_at = datetime.now(timezone.utc).isoformat()
                    self._last_auto_cluster_count = clustered_total
                self.bump_data_version()
            return clustered_total
        except ClusteringInProgressError:
            logger.debug("Skipped batch auto-clustering; clustering already active.")
            return 0
        except Exception as exc:
            logger.warning("Background batch clustering failed: %s", exc)
            with self._state_lock:
                self._last_error = f"Batch clustering error: {exc}"
            return 0

    async def run_due_tasks(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Execute any scheduled ingestion or batch-thresholded clustering tasks that are due."""
        current = time.time() if now is None else now
        with self._state_lock:
            if not self._enabled or self._paused:
                return {"skipped": True, "reason": "disabled_or_paused"}
            rss_due = self._next_rss_epoch is None or current >= self._next_rss_epoch
            fn_due = self._next_fn_epoch is None or current >= self._next_fn_epoch
            rss_interval = self._rss_interval
            fn_interval = self._fn_interval
            auto_cluster_enabled = self._auto_cluster_enabled
            cluster_batch = self._auto_cluster_min_batch

        inserted_total = 0
        clustered_total = 0

        if rss_due:
            with self._state_lock:
                self._next_rss_epoch = current + rss_interval
            try:
                res = await asyncio.to_thread(self.ingestion_service.run_rss)
                inserted_total += int(res.get("articles_inserted") or 0)
                with self._state_lock:
                    self._last_auto_sync_at = datetime.now(timezone.utc).isoformat()
                    self._last_error = None
            except IngestionInProgressError:
                logger.debug("Skipped scheduled RSS sync; another ingestion job is running.")
            except Exception as exc:
                logger.warning("Scheduled RSS sync failed: %s", exc)
                with self._state_lock:
                    self._last_error = f"RSS sync error: {exc}"

        if fn_due:
            with self._state_lock:
                self._next_fn_epoch = current + fn_interval
            if self._should_run_freenewsapi():
                try:
                    res = await asyncio.to_thread(
                        self.ingestion_service.run_freenewsapi,
                        max_pages_per_topic=1,
                        max_details_to_fetch=10,
                    )
                    inserted_total += int(res.get("articles_inserted") or 0)
                    with self._state_lock:
                        self._last_auto_sync_at = datetime.now(timezone.utc).isoformat()
                        self._last_error = None
                except (IngestionInProgressError, FreeNewsAPIConfigError):
                    pass
                except FreeNewsAPIError as exc:
                    logger.warning("Scheduled FreeNewsAPI sync skipped/failed: %s", exc)
                    with self._state_lock:
                        self._last_error = f"FreeNewsAPI sync error: {exc}"
                except Exception as exc:
                    logger.warning("Scheduled FreeNewsAPI sync error: %s", exc)
                    with self._state_lock:
                        self._last_error = f"FreeNewsAPI sync error: {exc}"

        if inserted_total > 0:
            self.bump_data_version()

        # Batch-thresholded background clustering: avoid heavy compute on 1-2 stray articles
        pending_count = await asyncio.to_thread(
            self.repo.get_unassigned_article_count, is_sample=False
        )
        if auto_cluster_enabled and pending_count >= cluster_batch:
            clustered_total = await asyncio.to_thread(self.maybe_run_batch_clustering_sync)

        return {
            "skipped": False,
            "rss_ran": rss_due,
            "freenewsapi_ran": fn_due,
            "articles_inserted": inserted_total,
            "articles_clustered": clustered_total,
            "pending_unassigned": pending_count,
        }
