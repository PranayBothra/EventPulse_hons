"""MIND (Microsoft News Dataset) benchmark loader and ingestion for EventPulse.

Parses official MIND-small dataset files (news.tsv, behaviors.tsv) from
data/benchmarks/mind/ and persists them into dedicated SQLite benchmark tables
(benchmark_news, benchmark_impressions), strictly decoupled from event_corpus.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.storage.repository import SQLiteRepository

logger = logging.getLogger("eventpulse.benchmarks.mind")


class MINDLoader:
    """Loader for Microsoft News Dataset (MIND) benchmark."""

    def __init__(
        self,
        base_dir: str = "data/benchmarks/mind",
        repository: Optional[SQLiteRepository] = None,
    ):
        self.base_dir = Path(base_dir)
        self.repository = repository or SQLiteRepository()

    def parse_news_file(
        self, split: str = "train", max_records: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Parse news.tsv from MIND split directory.

        Args:
            split: 'train' or 'dev'.
            max_records: Maximum rows to parse.

        Returns:
            List of parsed news items.
        """
        news_path = self.base_dir / split / "news.tsv"
        if not news_path.exists():
            logger.error("MIND news file not found: %s", news_path)
            return []

        news_items: List[Dict[str, Any]] = []
        with open(news_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                if max_records and idx >= max_records:
                    break
                if len(row) < 5:
                    continue

                news_id = row[0].strip()
                category = row[1].strip() if len(row) > 1 else ""
                subcategory = row[2].strip() if len(row) > 2 else ""
                title = row[3].strip() if len(row) > 3 else ""
                abstract = row[4].strip() if len(row) > 4 else ""
                url = row[5].strip() if len(row) > 5 else ""
                title_entities = row[6].strip() if len(row) > 6 else ""
                abstract_entities = row[7].strip() if len(row) > 7 else ""

                if not news_id or not title:
                    continue

                news_items.append(
                    {
                        "news_id": news_id,
                        "category": category,
                        "subcategory": subcategory,
                        "title": title,
                        "abstract": abstract,
                        "url": url,
                        "title_entities": title_entities,
                        "abstract_entities": abstract_entities,
                        "split": split,
                    }
                )

        logger.info("Parsed %d news items from MIND %s split", len(news_items), split)
        return news_items

    def parse_behaviors_file(
        self, split: str = "train", max_records: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Parse behaviors.tsv from MIND split directory.

        Args:
            split: 'train' or 'dev'.
            max_records: Maximum rows to parse.

        Returns:
            List of parsed impression interactions.
        """
        behaviors_path = self.base_dir / split / "behaviors.tsv"
        if not behaviors_path.exists():
            logger.error("MIND behaviors file not found: %s", behaviors_path)
            return []

        impressions: List[Dict[str, Any]] = []
        with open(behaviors_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                if max_records and idx >= max_records:
                    break
                if len(row) < 5:
                    continue

                try:
                    impression_id = int(row[0].strip())
                except ValueError:
                    continue

                user_id = row[1].strip()
                time_str = row[2].strip()
                history = row[3].strip()
                imp_list = row[4].strip()

                impressions.append(
                    {
                        "impression_id": impression_id,
                        "user_id": user_id,
                        "time": time_str,
                        "history": history,
                        "impressions": imp_list,
                        "split": split,
                    }
                )

        logger.info(
            "Parsed %d behavior logs from MIND %s split", len(impressions), split
        )
        return impressions

    def ingest_split(
        self,
        split: str = "train",
        max_news: Optional[int] = 10000,
        max_behaviors: Optional[int] = 10000,
        batch_size: int = 1000,
    ) -> Dict[str, int]:
        """Ingest MIND news and behaviors into the dedicated benchmark tables.

        Args:
            split: 'train' or 'dev'.
            max_news: Maximum news items to ingest.
            max_behaviors: Maximum behavior logs to ingest.
            batch_size: Database commit batch chunk size.

        Returns:
            Dictionary with counts of ingested news and behaviors.
        """
        logger.info("Starting ingestion for MIND %s split...", split)
        news_items = self.parse_news_file(split=split, max_records=max_news)
        behaviors = self.parse_behaviors_file(split=split, max_records=max_behaviors)

        inserted_news = 0
        for i in range(0, len(news_items), batch_size):
            chunk = news_items[i : i + batch_size]
            inserted_news += self.repository.batch_insert_benchmark_news(chunk)

        inserted_behaviors = 0
        for i in range(0, len(behaviors), batch_size):
            chunk = behaviors[i : i + batch_size]
            inserted_behaviors += self.repository.batch_insert_benchmark_impressions(
                chunk
            )

        logger.info(
            "Ingested MIND %s split: %d news, %d behaviors",
            split,
            inserted_news,
            inserted_behaviors,
        )
        return {
            "split": split,
            "news_stored": inserted_news,
            "behaviors_stored": inserted_behaviors,
        }
