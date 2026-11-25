"""Automated tests for FastAPI endpoints and FreeNewsAPI.io client (100% mocked HTTP)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.ingestion.freenewsapi_client import (
    FreeNewsAPIAuthError,
    FreeNewsAPIClient,
    FreeNewsAPIConfigError,
    FreeNewsAPIRateLimitError,
    FreeNewsAPIRequestError,
    FreeNewsAPIValidationError,
    is_freenewsapi_configured,
    load_freenewsapi_key,
)
from src.representation.embedding import TfidfEmbeddingProvider
from src.storage.repository import SQLiteRepository


@pytest.fixture
def seeded_env(tmp_path):
    """Create an isolated SQLite DB seeded with realistic articles, embeddings, sentences, and events."""
    db_file = tmp_path / "test_eventpulse.db"
    repo = SQLiteRepository(db_path=str(db_file))
    provider = TfidfEmbeddingProvider(dimension=384)
    model_id = provider.model_id

    art1 = {
        "canonical_url": "https://example.com/news/ai-chip-launch",
        "url": "https://example.com/news/ai-chip-launch",
        "headline": "Quantum Core Announces Next-Gen AI Accelerator Chip in Bengaluru",
        "headline_clean": "Quantum Core Announces Next-Gen AI Accelerator Chip in Bengaluru",
        "headline_hash": "hash_ai_chip_1",
        "description": "Quantum Core unveiled a 3nm neural processor aimed at low-power edge inference.",
        "content_clean": "Quantum Core unveiled a 3nm neural processor aimed at low-power edge inference. Industry analysts expect shipments in Q3.",
        "source_id": "tech_wire",
        "source_name": "Tech Wire India",
        "published_at": "2026-11-24T08:00:00Z",
        "retrieved_at": "2026-11-24T08:05:00Z",
        "is_sample": False,
        "provider": "rss",
    }
    art2 = {
        "canonical_url": "https://example.com/news/ai-chip-benchmark",
        "url": "https://example.com/news/ai-chip-benchmark",
        "headline": "Benchmarks Confirm 40 Percent Efficiency Gain for Quantum Core Chip",
        "headline_clean": "Benchmarks Confirm 40 Percent Efficiency Gain for Quantum Core Chip",
        "headline_hash": "hash_ai_chip_2",
        "description": "Independent tests show the new Bengaluru-designed accelerator cuts energy draw by 40 percent.",
        "content_clean": "Quantum Core unveiled a 3nm neural processor aimed at low-power edge inference. Independent tests confirm a 40 percent power reduction compared to prior generations.",
        "source_id": "science_daily",
        "source_name": "Science Daily",
        "published_at": "2026-11-24T12:00:00Z",
        "retrieved_at": "2026-11-24T12:05:00Z",
        "is_sample": False,
        "provider": "rss",
    }
    art3 = {
        "canonical_url": "https://example.com/news/space-probe-orbit",
        "url": "https://example.com/news/space-probe-orbit",
        "headline": "ISRO Lunar Polar Probe Enters Stable Mapping Orbit Successfully",
        "headline_clean": "ISRO Lunar Polar Probe Enters Stable Mapping Orbit Successfully",
        "headline_hash": "hash_space_1",
        "description": "Ground telemetry confirms the polar orbiter began high-resolution surface imaging.",
        "content_clean": "Ground telemetry confirms the polar orbiter began high-resolution surface imaging. Mission control reported nominal solar array deployment.",
        "source_id": "space_chronicle",
        "source_name": "Space Chronicle",
        "published_at": "2026-11-24T15:00:00Z",
        "retrieved_at": "2026-11-24T15:05:00Z",
        "is_sample": False,
        "provider": "rss",
    }

    for art in (art1, art2, art3):
        repo.insert_article(art)

    db_arts = list(reversed(repo.get_articles(limit=10)))
    vecs = provider.embed_texts([f"{a['headline_clean']} {a['description']}" for a in db_arts])
    for art, vec in zip(db_arts, vecs):
        repo.save_embedding(
            article_id=art["id"],
            text_hash=art["headline_hash"],
            model_id=model_id,
            dimension=384,
            embedding_blob=vec.tobytes(),
        )

    # Create Event 1 with 2 articles, Event 2 with 1 article
    cent1 = (vecs[0] + vecs[1]) / max(float(np.linalg.norm(vecs[0] + vecs[1])), 1e-12)
    repo.create_event(
        event_id="evt_ai_chip",
        title=db_arts[0]["headline"],
        summary=db_arts[0]["description"],
        first_seen_at=db_arts[0]["published_at"],
        last_seen_at=db_arts[1]["published_at"],
        centroid_blob=cent1.astype(np.float32).tobytes(),
        key_entities=["Quantum Core", "Bengaluru"],
        status="ACTIVE",
    )
    repo.assign_article_to_event("evt_ai_chip", db_arts[0]["id"], 1.0, False)
    repo.assign_article_to_event("evt_ai_chip", db_arts[1]["id"], 0.88, False)
    repo.update_event(
        event_id="evt_ai_chip",
        last_seen_at=db_arts[1]["published_at"],
        article_count=2,
        centroid_blob=cent1.astype(np.float32).tobytes(),
        key_entities=["Quantum Core", "Bengaluru"],
        status="ACTIVE",
    )

    repo.create_event(
        event_id="evt_space_probe",
        title=db_arts[2]["headline"],
        summary=db_arts[2]["description"],
        first_seen_at=db_arts[2]["published_at"],
        last_seen_at=db_arts[2]["published_at"],
        centroid_blob=vecs[2].astype(np.float32).tobytes(),
        key_entities=["ISRO"],
        status="EMERGING",
    )
    repo.assign_article_to_event("evt_space_probe", db_arts[2]["id"], 1.0, False)

    # Save sentences and prior event memory for art1 in evt_ai_chip
    s1_texts = [
        db_arts[0]["headline"],
        "Quantum Core unveiled a 3nm neural processor aimed at low-power edge inference.",
        "Subscribe to our daily newsletter for more tech updates.",
    ]
    s1_vecs = provider.embed_texts(s1_texts)
    s1_ids = repo.save_article_sentences([
        {
            "article_id": db_arts[0]["id"],
            "sentence_index": idx,
            "sentence_hash": f"s1_{idx}",
            "sentence_text": txt,
            "is_information_bearing": True,
            "informativeness_score": 0.9 if idx < 2 else 0.1,
            "named_entities": ["Quantum Core"],
            "has_numbers": idx == 1,
            "has_dates": False,
            "numerical_tokens": ["3"] if idx == 1 else [],
            "token_count": len(txt.split()),
        }
        for idx, txt in enumerate(s1_texts)
    ])
    repo.batch_save_sentence_embeddings([
        (sid, model_id, vec.astype(np.float32).tobytes()) for sid, vec in zip(s1_ids, s1_vecs)
    ])
    repo.save_event_memory_sentences([
        {
            "event_id": "evt_ai_chip",
            "sentence_id": sid,
            "article_id": db_arts[0]["id"],
            "published_at": db_arts[0]["published_at"],
            "published_epoch": datetime.fromisoformat(db_arts[0]["published_at"].replace("Z", "+00:00")).timestamp(),
            "novelty_score": 1.0 if idx < 2 else 0.1,
            "evidence_status": "POTENTIALLY_NEW" if idx < 2 else "INSUFFICIENT_EVIDENCE",
            "most_similar_sentence_id": None,
            "highest_similarity": 0.0,
        }
        for idx, sid in enumerate(s1_ids)
    ])

    return {
        "db_path": str(db_file),
        "repo": repo,
        "provider": provider,
        "articles": db_arts,
    }


def _make_mock_freenews_client(handler, api_key: str = "test-secret-key", max_retries: int = 2) -> FreeNewsAPIClient:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return FreeNewsAPIClient(
        api_key=api_key,
        max_retries=max_retries,
        backoff_seconds=0.01,
        min_request_interval_seconds=0.0,
        http_client=client,
    )


# =============================================================================
# 1. FastAPI Core Endpoints Tests
# =============================================================================

def test_api_health_stats_and_users(seeded_env):
    app = create_app(
        db_path=seeded_env["db_path"],
        embedding_provider=seeded_env["provider"],
        freenewsapi_client=FreeNewsAPIClient(api_key="", min_request_interval_seconds=0.0),
    )
    with TestClient(app) as client:
        r_root = client.get("/")
        assert r_root.status_code == 200
        assert "text/html" in r_root.headers.get("content-type", "")
        assert "EventPulse" in r_root.text

        r_js = client.get("/static/js/app.js")
        assert r_js.status_code == 200
        assert "bootstrap" in r_js.text

        r_docs = client.get("/docs")
        assert r_docs.status_code == 200

        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"
        assert r_health.json()["database_connected"] is True

        r_stats = client.get("/api/stats")
        assert r_stats.status_code == 200
        st = r_stats.json()
        assert st["total_articles"] == 3
        assert st["total_events"] == 2
        assert st["distinct_sources_count"] == 3

        r_presets = client.get("/api/users/presets")
        assert r_presets.status_code == 200
        ids = {p["user_id"] for p in r_presets.json()}
        assert ids == {"user_tech", "user_geopolitics", "user_science", "user_cold_start"}
        assert all(p["is_synthetic_demo"] is True for p in r_presets.json())

        r_prof = client.get("/api/users/user_tech/profile")
        assert r_prof.status_code == 200
        assert r_prof.json()["is_cold_start"] is True
        assert r_prof.json()["interaction_count"] == 0


def test_api_recommendations_algorithms_weights_and_interactions(seeded_env):
    app = create_app(
        db_path=seeded_env["db_path"],
        embedding_provider=seeded_env["provider"],
        freenewsapi_client=FreeNewsAPIClient(api_key="", min_request_interval_seconds=0.0),
    )
    art1_id = seeded_env["articles"][0]["id"]
    with TestClient(app) as client:
        # 1. Default mode should be RELEVANCE_FOCUSED
        r_ep = client.get("/api/recommendations", params={"user_id": "user_tech", "algorithm": "EventPulse", "top_k": 5})
        assert r_ep.status_code == 200
        data_ep = r_ep.json()
        assert data_ep["mode"] == "RELEVANCE_FOCUSED"
        assert data_ep["count"] == 2
        assert "score_breakdown" in data_ep["recommendations"][0]
        assert data_ep["recommendations"][0]["explanation"]

        # 2. Switch algorithms + modes + custom weights
        for algo in ("Baseline_A", "Baseline_B", "Baseline_C"):
            r_b = client.get("/api/recommendations", params={"user_id": "user_tech", "algorithm": algo})
            assert r_b.status_code == 200
            assert r_b.json()["algorithm"] == algo

        r_custom = client.get(
            "/api/recommendations",
            params={
                "user_id": "user_tech",
                "algorithm": "EventPulse",
                "mode": "DISCOVERY_FOCUSED",
                "w_long": 0.10,
                "w_event_novelty": 0.40,
                "mmr_lambda": 0.60,
            },
        )
        assert r_custom.status_code == 200
        assert r_custom.json()["mode"] == "DISCOVERY_FOCUSED"

        r_bad = client.get("/api/recommendations", params={"algorithm": "UnknownAlgo"})
        assert r_bad.status_code == 400

        # 3. Record LIKE, READ, DISLIKE interactions & check persistence
        for action in ("LIKE", "READ", "DISLIKE"):
            r_int = client.post(
                "/api/interactions",
                json={
                    "user_id": "user_tech",
                    "article_id": art1_id,
                    "event_id": "evt_ai_chip",
                    "interaction_type": action,
                },
            )
            assert r_int.status_code == 200

        r_prof = client.get("/api/users/user_tech/profile")
        assert r_prof.status_code == 200
        assert r_prof.json()["interaction_count"] == 3
        assert r_prof.json()["consumed_sentences_count"] == 3

        r_hist = client.get("/api/users/user_tech/interactions", params={"limit": 10})
        assert r_hist.status_code == 200
        assert len(r_hist.json()) == 3


def test_api_events_novelty_topics_and_benchmarks(seeded_env):
    app = create_app(
        db_path=seeded_env["db_path"],
        embedding_provider=seeded_env["provider"],
        freenewsapi_client=FreeNewsAPIClient(api_key="", min_request_interval_seconds=0.0),
    )
    with TestClient(app) as client:
        # Events listing
        r_evs = client.get("/api/events")
        assert r_evs.status_code == 200
        evs = r_evs.json()
        assert evs[0]["id"] == "evt_ai_chip"
        assert evs[0]["article_count"] == 2
        assert evs[0]["publisher_count"] == 2

        # Event article timeline
        r_ev_arts = client.get("/api/events/evt_ai_chip/articles")
        assert r_ev_arts.status_code == 200
        assert len(r_ev_arts.json()) == 2
        assert client.get("/api/events/missing_evt/articles").status_code == 404

        # Novelty inspector
        r_evol = client.get("/api/novelty/evolving-events")
        assert r_evol.status_code == 200
        assert len(r_evol.json()) == 1
        assert r_evol.json()[0]["id"] == "evt_ai_chip"

        r_nov = client.get("/api/novelty/inspect", params={"event_id": "evt_ai_chip"})
        assert r_nov.status_code == 200
        nov = r_nov.json()
        assert nov["prior_memory_sentence_count"] == 3
        assert nov["total_sentence_count"] >= 2
        statuses = {s["evidence_status"] for s in nov["sentence_results"]}
        assert statuses.issubset({"POTENTIALLY_NEW", "REPEATED_INFORMATION", "INSUFFICIENT_EVIDENCE"})

        # Emerging topics
        r_topics = client.get("/api/topics", params={"detector": "online_cluster", "min_cluster_size": 1})
        assert r_topics.status_code == 200
        assert r_topics.json()["articles_scanned"] == 3
        assert r_topics.json()["topics_found"] >= 1

        # Offline benchmarks
        r_mind = client.get("/api/benchmarks/mind")
        assert r_mind.status_code == 200
        assert r_mind.json()["evaluation_scope"] == "offline_benchmark"
        r_corp = client.get("/api/benchmarks/corpus-novelty")
        assert r_corp.status_code == 200
        assert r_corp.json()["evaluation_scope"] == "offline_benchmark"


# =============================================================================
# 2. FreeNewsAPI Client & Ingestion Endpoint Tests (Mocked HTTP)
# =============================================================================

def test_freenewsapi_env_loading_and_missing_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FREENEWSAPI_API_KEY=dummy_env_value\n", encoding="utf-8")
    monkeypatch.delenv("FREENEWSAPI_API_KEY", raising=False)
    assert is_freenewsapi_configured(env_path=env_file) is True
    assert load_freenewsapi_key(env_path=env_file) == "dummy_env_value"

    monkeypatch.setenv("FREENEWSAPI_API_KEY", "")
    client = FreeNewsAPIClient(api_key="", min_request_interval_seconds=0.0)
    with pytest.raises(FreeNewsAPIConfigError):
        client.fetch_news_listing()


def test_freenewsapi_auth_error_rate_limit_timeout_json_and_outage(monkeypatch):
    # 1. 401/403 Auth error scrubs secret key
    def auth_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Invalid key super_secret_token"})

    c_auth = _make_mock_freenews_client(auth_handler, api_key="super_secret_token")
    with pytest.raises(FreeNewsAPIAuthError) as exc_info:
        c_auth.fetch_news_listing()
    assert "super_secret_token" not in str(exc_info.value)

    # 2. Pagination parameter constraint (cannot mix cursor and offset)
    with pytest.raises(FreeNewsAPIValidationError):
        c_auth.fetch_news_listing(cursor="abc", offset=10)

    # 3. Transient 429 then success
    calls = {"n": 0}
    def retry_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0.01"}, json={"error": "Too fast"})
        return httpx.Response(
            200,
            headers={"x-ratelimit-limit-day": "5000", "x-ratelimit-remaining-day": "4999"},
            json={"data": [{"uuid": "u1", "title": "Valid Live Headline Sample", "published_at": "2026-11-24T10:00:00Z", "publisher": "Hindu"}], "meta": {"has_more": False}},
        )

    c_retry = _make_mock_freenews_client(retry_handler, max_retries=2)
    items, _ = c_retry.fetch_news_listing()
    assert len(items) == 1
    assert c_retry.last_quota["daily_remaining"] == 4999

    # 4. Daily quota exhaustion (0 remaining -> immediate FreeNewsAPIRateLimitError)
    def quota_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"x-ratelimit-remaining-day": "0"}, json={"error": "Daily limit reached"})

    c_quota = _make_mock_freenews_client(quota_handler)
    with pytest.raises(FreeNewsAPIRateLimitError):
        c_quota.fetch_news_listing()

    # 5. Timeout & Invalid JSON & 503 Outage
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    c_to = _make_mock_freenews_client(timeout_handler, max_retries=0)
    with pytest.raises(FreeNewsAPIRequestError):
        c_to.fetch_news_listing()

    def bad_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-valid-json{{{")

    c_json = _make_mock_freenews_client(bad_json_handler, max_retries=0)
    with pytest.raises(FreeNewsAPIRequestError):
        c_json.fetch_news_listing()

    def outage_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    c_out = _make_mock_freenews_client(outage_handler, max_retries=1)
    with pytest.raises(FreeNewsAPIRequestError):
        c_out.fetch_news_listing()


def test_freenewsapi_pagination_dedup_ingestion_and_rss_independence(seeded_env):
    detail_calls: List[str] = []

    def api_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        # Ensure forbidden deprecated parameters are never emitted
        for bad_param in ("q", "in_body", "in_subtitle"):
            assert bad_param not in params

        if path.endswith("/news"):
            assert params.get("country") == "IN"
            assert params.get("language") == "en"
            cursor = params.get("cursor")
            if cursor is None:
                return httpx.Response(
                    200,
                    headers={"x-ratelimit-limit-day": "5000", "x-ratelimit-remaining-day": "4990"},
                    json={
                        "data": [
                            {"uuid": "uuid-new-1", "title": "India Semiconductor Mission Approves New Fab Facility", "published_at": "2026-11-24T20:00:00Z", "publisher": "The Hindu"}
                        ],
                        "meta": {"has_more": True, "next_cursor": "page2_tok", "returned": 1, "page_size": 1},
                    },
                )
            elif cursor == "page2_tok":
                return httpx.Response(
                    200,
                    headers={"x-ratelimit-limit-day": "5000", "x-ratelimit-remaining-day": "4989"},
                    json={
                        "data": [
                            {"uuid": "uuid-new-2", "title": "ISRO Conducts Deep Space Cryogenic Engine Ignition Test", "published_at": "2026-11-24T21:00:00Z", "publisher": "Indian Express"}
                        ],
                        "meta": {"has_more": False, "next_cursor": None, "returned": 1, "page_size": 1},
                    },
                )
        elif path.endswith("/details"):
            uid = params["uuid"]
            detail_calls.append(uid)
            return httpx.Response(
                200,
                json={
                    "data": {
                        "uuid": uid,
                        "title": "India Semiconductor Mission Approves New Fab Facility" if uid == "uuid-new-1" else "ISRO Conducts Deep Space Cryogenic Engine Ignition Test",
                        "publisher": "The Hindu" if uid == "uuid-new-1" else "Indian Express",
                        "original_url": f"https://news.example.in/{uid}?utm_source=api",
                        "published_at": "2026-11-24T20:00:00Z",
                        "incipit": "Major domestic technology milestone announced in New Delhi today.",
                        "body": "Major domestic technology milestone announced in New Delhi today with industry partners.",
                        "topics": ["technology"],
                        "countries": ["IN"],
                        "languages": ["en"],
                        "authors": ["Staff Reporter"],
                    }
                },
            )
        return httpx.Response(404, json={"error": "Not found"})

    fn_client = _make_mock_freenews_client(api_handler)
    app = create_app(
        db_path=seeded_env["db_path"],
        embedding_provider=seeded_env["provider"],
        freenewsapi_client=fn_client,
    )
    with TestClient(app) as client:
        # 1. First FreeNewsAPI ingestion fetches both paginated items + details and stores them
        r1 = client.post(
            "/api/ingestion/freenewsapi/run",
            json={"country": "IN", "language": "en", "topics": ["technology"], "max_pages_per_topic": 2},
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["articles_fetched"] == 2
        assert d1["articles_inserted"] == 2
        assert d1["duplicates_skipped"] == 0
        assert len(detail_calls) == 2

        # 2. Second identical run skips /details via SQLite external_id check!
        detail_calls.clear()
        r2 = client.post(
            "/api/ingestion/freenewsapi/run",
            json={"country": "IN", "language": "en", "topics": ["technology"], "max_pages_per_topic": 2},
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["articles_inserted"] == 0
        assert d2["duplicates_skipped"] == 2
        assert len(detail_calls) == 0  # Avoided redundant /details API calls!

        # 3. Cluster newly ingested FreeNewsAPI articles with existing EventPulse clustering
        r_clust = client.post("/api/clustering/run")
        assert r_clust.status_code == 200
        assert r_clust.json()["articles_clustered"] == 2

        # 4. Status & history check (never expose raw api key)
        r_status = client.get("/api/ingestion/status")
        assert r_status.status_code == 200
        st = r_status.json()
        assert st["freenewsapi"]["credentials_configured"] is True
        assert st["freenewsapi"]["quota"]["daily_remaining"] == 4989
        assert "test-secret-key" not in json.dumps(st)

        r_hist = client.get("/api/ingestion/history")
        assert r_hist.status_code == 200
        assert len(r_hist.json()) >= 2

        # 5. RSS sample ingestion works independently even if FreeNewsAPI client is broken/missing key
        app.state.container.ingestion_service.freenewsapi_client = FreeNewsAPIClient(api_key="", min_request_interval_seconds=0.0)
        r_fn_fail = client.post("/api/ingestion/freenewsapi/run")
        assert r_fn_fail.status_code == 503

        r_rss = client.post("/api/ingestion/rss/run", params={"use_sample": True})
        assert r_rss.status_code == 200
        assert r_rss.json()["provider"] == "rss"
        assert r_rss.json()["articles_fetched"] > 0
