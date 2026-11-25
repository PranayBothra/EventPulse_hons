"""Unit tests for HistoricalCorpusLoader."""

from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.historical_corpus_loader import HistoricalCorpusLoader
from src.storage.repository import SQLiteRepository


@pytest.fixture
def temp_repo(tmp_path):
    db_path = tmp_path / "test.db"
    return SQLiteRepository(str(db_path))


def test_ingest_categorized_news(temp_repo):
    loader = HistoricalCorpusLoader(repository=temp_repo)

    mock_jsonl = (
        b'{"link": "https://huffpost.com/entry/tech-news-1", "headline": "Tech Giants Invest in AI", "short_description": "Massive capital flows into frontier AI models.", "date": "2022-09-01", "category": "TECH", "authors": "AP"}\n'
        b'{"link": "https://huffpost.com/entry/energy-news-2", "headline": "Clean Energy Grid Expands", "short_description": "Solar and wind generation hit new records.", "date": "2022-09-02", "category": "ENVIRONMENT", "authors": "Reuters"}\n'
    )

    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_jsonl.splitlines()

    with patch("urllib.request.urlopen", return_value=mock_resp):
        stored, dup = loader.ingest_categorized_news(max_articles=10, batch_size=5)
        assert stored == 2
        assert dup == 0

        # Duplicate check on second ingestion of same items
        mock_resp2 = MagicMock()
        mock_resp2.__enter__.return_value = mock_jsonl.splitlines()
        with patch("urllib.request.urlopen", return_value=mock_resp2):
            stored2, dup2 = loader.ingest_categorized_news(max_articles=10, batch_size=5)
            assert stored2 == 0
            assert dup2 == 2


def test_ingest_temporal_headlines(temp_repo):
    loader = HistoricalCorpusLoader(repository=temp_repo)

    mock_csv = (
        b"publish_date,headline_text\n"
        b"20211230,global markets rally on year end optimism\n"
        b"20211231,new climate research warns on ocean temperatures\n"
    )

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_csv
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        stored, dup = loader.ingest_temporal_headlines(max_articles=10, batch_size=5)
        assert stored == 2
        assert dup == 0
