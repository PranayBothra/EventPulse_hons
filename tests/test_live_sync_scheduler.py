"""Tests for background LiveSyncScheduler, batch-thresholded clustering, and auto-sync API endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.services.event_service import EventService
from src.api.services.ingestion_service import IngestionInProgressError, IngestionService
from src.api.services.live_sync_scheduler import LiveSyncScheduler
from src.config import LiveSyncConfig
from src.ingestion.freenewsapi_client import FreeNewsAPIClient
from src.representation.embedding import TfidfEmbeddingProvider
from src.storage.repository import SQLiteRepository


def _insert_dummy_article(repo: SQLiteRepository, idx: int, is_sample: bool = False) -> int:
    return repo.insert_article(
        {
            "canonical_url": f"https://example.com/live/{idx}",
            "url": f"https://example.com/live/{idx}",
            "headline": f"Live Update Story Headline Number {idx}",
            "headline_clean": f"Live Update Story Headline Number {idx}",
            "headline_hash": f"live_hash_{idx}",
            "description": f"Live description text for story {idx}.",
            "content_clean": f"Live description text for story {idx}.",
            "source_id": "test_rss",
            "source_name": "Test RSS",
            "published_at": "2026-11-26T00:00:00Z",
            "retrieved_at": "2026-11-26T00:01:00Z",
            "is_sample": is_sample,
            "provider": "rss",
        }
    )


def test_batch_thresholded_auto_clustering_and_data_version(tmp_path):
    repo = SQLiteRepository(db_path=str(tmp_path / "sync_batch.db"))
    for i in range(3):
        _insert_dummy_article(repo, i, is_sample=False)

    assert repo.get_unassigned_article_count(is_sample=False) == 3

    ing_service = MagicMock(spec=IngestionService)
    ing_service.run_rss.return_value = {"articles_inserted": 1}
    ing_service.get_status.return_value = {
        "freenewsapi": {"credentials_configured": False, "quota": {}}
    }

    ev_service = MagicMock(spec=EventService)
    ev_service.is_clustering_running = False
    ev_service.run_clustering.return_value = {"articles_clustered": 5}

    scheduler = LiveSyncScheduler(
        repository=repo,
        ingestion_service=ing_service,
        event_service=ev_service,
        config=LiveSyncConfig(
            enabled=True,
            rss_interval_seconds=300,
            freenewsapi_interval_seconds=1800,
            auto_cluster_min_batch=5,
        ),
    )

    # Below batch threshold (3 < 5): RSS runs, data_version bumps, clustering is skipped
    res1 = asyncio.run(scheduler.run_due_tasks(now=1000.0))
    assert res1["rss_ran"] is True
    assert res1["articles_inserted"] == 1
    assert res1["articles_clustered"] == 0
    assert ev_service.run_clustering.call_count == 0
    assert scheduler.get_status()["data_version"] == 2

    # Add 2 more unassigned articles to reach batch threshold (5 >= 5)
    _insert_dummy_article(repo, 10, is_sample=False)
    _insert_dummy_article(repo, 11, is_sample=False)
    assert repo.get_unassigned_article_count(is_sample=False) == 5

    res2 = asyncio.run(scheduler.run_due_tasks(now=1305.0))
    assert res2["rss_ran"] is True
    assert res2["articles_clustered"] == 5
    assert ev_service.run_clustering.call_count == 1
    assert scheduler.get_status()["data_version"] == 4
    assert scheduler.get_status()["last_auto_cluster_count"] == 5

    # When auto_cluster_enabled=False, clustering stays skipped even if pending >= batch
    scheduler.configure(auto_cluster_enabled=False)
    res3 = asyncio.run(scheduler.run_due_tasks(now=1610.0))
    assert res3["articles_clustered"] == 0
    assert ev_service.run_clustering.call_count == 1


def test_scheduler_skips_when_locked_or_quota_low(tmp_path):
    repo = SQLiteRepository(db_path=str(tmp_path / "sync_guard.db"))

    ing_service = MagicMock(spec=IngestionService)
    ing_service.run_rss.side_effect = IngestionInProgressError("busy")
    ing_service.get_status.return_value = {
        "freenewsapi": {
            "credentials_configured": True,
            "quota": {"daily_remaining": 10},
        }
    }

    ev_service = MagicMock(spec=EventService)
    ev_service.is_clustering_running = False

    scheduler = LiveSyncScheduler(
        repository=repo,
        ingestion_service=ing_service,
        event_service=ev_service,
        config=LiveSyncConfig(
            enabled=True,
            freenewsapi_min_remaining_quota=50,
            auto_cluster_min_batch=10,
        ),
    )

    res = asyncio.run(scheduler.run_due_tasks(now=2000.0))
    assert res["rss_ran"] is True
    assert res["freenewsapi_ran"] is True
    assert ing_service.run_freenewsapi.call_count == 0
    assert scheduler.get_status()["last_error"] is None


def test_auto_sync_api_endpoints_and_lifespan(tmp_path):
    db_file = tmp_path / "api_sync.db"
    SQLiteRepository(db_path=str(db_file))
    provider = TfidfEmbeddingProvider(dimension=384)

    app = create_app(
        db_path=str(db_file),
        embedding_provider=provider,
        freenewsapi_client=FreeNewsAPIClient(api_key="", min_request_interval_seconds=0.0),
        enable_live_sync=True,
    )
    with TestClient(app) as client:
        # Modify delay/pause immediately so no external RSS call runs during quick test
        r_pause = client.post(
            "/api/ingestion/auto-sync",
            json={
                "paused": True,
                "rss_interval_seconds": 120,
                "auto_cluster_enabled": False,
                "auto_cluster_min_batch": 15,
            },
        )
        assert r_pause.status_code == 200
        ls = r_pause.json()
        assert ls["enabled"] is True
        assert ls["paused"] is True
        assert ls["rss_interval_seconds"] == 120
        assert ls["auto_cluster_enabled"] is False
        assert ls["auto_cluster_min_batch"] == 15

        r_status = client.get("/api/ingestion/status")
        assert r_status.status_code == 200
        body = r_status.json()
        assert body["data_version"] >= 1
        assert body["live_sync"]["paused"] is True
        assert body["live_sync"]["auto_cluster_enabled"] is False
        assert body["live_sync"]["auto_cluster_min_batch"] == 15

        r_resume = client.post(
            "/api/ingestion/auto-sync",
            json={"paused": False, "auto_cluster_enabled": True},
        )
        assert r_resume.status_code == 200
        assert r_resume.json()["paused"] is False
        assert r_resume.json()["auto_cluster_enabled"] is True
        assert r_resume.json()["active"] is True
