"""Unit tests for GDELTClient."""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.gdelt_client import GDELTClient


@pytest.fixture
def sample_gdelt_raw():
    return {
        "url": "https://www.reuters.com/technology/ai-advances-2026-09-01/",
        "title": "New AI models show rapid advances in multimodal reasoning",
        "seendate": "20260901T143000Z",
        "socialimage": "https://reuters.com/img/ai.jpg",
        "domain": "reuters.com",
        "language": "English",
        "sourcecountry": "United States",
    }


def test_normalize_article_success(sample_gdelt_raw):
    client = GDELTClient(rate_limit_delay=0.01)
    normalized = client.normalize_article(sample_gdelt_raw, query_topic="artificial intelligence")

    assert normalized is not None
    assert normalized["canonical_url"] == "https://www.reuters.com/technology/ai-advances-2026-09-01"
    assert normalized["headline"] == sample_gdelt_raw["title"]
    assert normalized["headline_clean"] == sample_gdelt_raw["title"]
    assert len(normalized["headline_hash"]) == 64
    assert normalized["source_id"] == "gdelt_reuters.com"
    assert normalized["source_name"] == "reuters.com"
    assert normalized["published_at"] == "2026-09-01T14:30:00+00:00"
    assert normalized["seen_at"] == "20260901T143000Z"
    assert normalized["dataset_origin"] == "event_corpus"
    assert normalized["content_availability"] == "HEADLINE_ONLY"
    assert normalized["language"] == "English"
    assert normalized["country"] == "United States"


def test_normalize_article_missing_fields():
    client = GDELTClient(rate_limit_delay=0.01)
    assert client.normalize_article({}, "test") is None
    assert client.normalize_article({"url": "http://example.com"}, "test") is None
    assert client.normalize_article({"title": "Headline only"}, "test") is None


def test_fetch_articles_success(sample_gdelt_raw):
    client = GDELTClient(rate_limit_delay=0.01)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"articles": [sample_gdelt_raw]}

    with patch("requests.get", return_value=mock_resp) as mock_get:
        records = client.fetch_articles("artificial intelligence", max_records=10)
        assert len(records) == 1
        assert records[0]["headline"] == sample_gdelt_raw["title"]
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "query" in kwargs["params"]
        assert kwargs["params"]["mode"] == "artlist"
        assert kwargs["params"]["format"] == "json"


def test_fetch_articles_handles_429():
    client = GDELTClient(rate_limit_delay=0.01, max_retries=2)
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Too Many Requests"

    with patch("requests.get", return_value=mock_resp), patch("time.sleep") as mock_sleep:
        records = client.fetch_articles("climate change", max_records=5)
        assert records == []
        assert mock_sleep.called
