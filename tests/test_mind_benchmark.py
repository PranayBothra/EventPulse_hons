"""Unit tests for MIND benchmark loader."""

import os
from pathlib import Path
import pytest

from src.benchmarks.mind_loader import MINDLoader
from src.storage.repository import SQLiteRepository


@pytest.fixture
def temp_mind_env(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)

    news_content = (
        "N1001\tnews\tworld\tPeace treaty signed\tHistoric summit concludes\thttps://msn.com/news/1\t[]\t[]\n"
        "N1002\ttech\tai\tBreakthrough in quantum AI\tNew computing model announced\thttps://msn.com/tech/2\t[]\t[]\n"
    )
    with open(train_dir / "news.tsv", "w", encoding="utf-8") as f:
        f.write(news_content)

    behaviors_content = (
        "1\tU1001\t11/11/2019 9:00:00 AM\tN1001\tN1001-1 N1002-0\n"
        "2\tU1002\t11/11/2019 9:15:00 AM\tN1002\tN1001-0 N1002-1\n"
    )
    with open(train_dir / "behaviors.tsv", "w", encoding="utf-8") as f:
        f.write(behaviors_content)

    db_path = tmp_path / "test.db"
    repo = SQLiteRepository(str(db_path))

    return str(tmp_path), repo


def test_mind_loader_parse_news(temp_mind_env):
    mind_dir, repo = temp_mind_env
    loader = MINDLoader(base_dir=mind_dir, repository=repo)

    items = loader.parse_news_file(split="train")
    assert len(items) == 2
    assert items[0]["news_id"] == "N1001"
    assert items[0]["category"] == "news"
    assert items[0]["title"] == "Peace treaty signed"
    assert items[1]["news_id"] == "N1002"
    assert items[1]["category"] == "tech"


def test_mind_loader_parse_behaviors(temp_mind_env):
    mind_dir, repo = temp_mind_env
    loader = MINDLoader(base_dir=mind_dir, repository=repo)

    behaviors = loader.parse_behaviors_file(split="train")
    assert len(behaviors) == 2
    assert behaviors[0]["impression_id"] == 1
    assert behaviors[0]["user_id"] == "U1001"
    assert "N1001-1" in behaviors[0]["impressions"]


def test_mind_loader_ingest_split(temp_mind_env):
    mind_dir, repo = temp_mind_env
    loader = MINDLoader(base_dir=mind_dir, repository=repo)

    result = loader.ingest_split(split="train")
    assert result["news_stored"] == 2
    assert result["behaviors_stored"] == 2

    stats = repo.get_benchmark_stats()
    assert stats["news_total"] == 2
    assert stats["impressions_total"] == 2
    assert stats["unique_users"] == 2
