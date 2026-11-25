"""Fallback dataset loader for offline execution and testing.

Provides explicitly labelled sample articles without fabricating live news.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.ingestion.models import ProcessedArticle
from src.preprocessing.date_parser import parse_datetime_to_utc
from src.preprocessing.deduplicator import (
    canonicalize_url,
    compute_headline_hash,
)
from src.preprocessing.text_cleaner import clean_html, normalize_text

logger = logging.getLogger("eventpulse.fallback")


def load_sample_articles(
    sample_file_path: Optional[str] = None,
) -> List[ProcessedArticle]:
    """Load and validate explicitly labelled sample articles from fallback JSON.

    Args:
        sample_file_path: Path to sample JSON file.
                          Defaults to 'data/sample/sample_articles.json'.

    Returns:
        List of ProcessedArticle instances with is_sample=True.
    """
    if sample_file_path:
        path = Path(sample_file_path)
    else:
        # Resolve relative to repo root
        repo_root = Path(__file__).resolve().parent.parent.parent
        path = repo_root / "data" / "sample" / "sample_articles.json"

    if not path.is_file():
        logger.warning("Sample fallback file not found at: %s", path)
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)
    except Exception as e:
        logger.error("Failed to parse sample articles file %s: %s", path, e)
        return []

    now_utc = datetime.now(timezone.utc).isoformat()
    processed_samples: List[ProcessedArticle] = []

    for item in raw_entries:
        headline = item.get("headline", "").strip()
        url = item.get("url", "").strip()
        if not headline or not url:
            continue

        headline_clean = normalize_text(headline)
        canonical = canonicalize_url(url)
        headline_hash = compute_headline_hash(headline_clean)
        desc_clean = normalize_text(clean_html(item.get("description", "")))
        published_at = parse_datetime_to_utc(item.get("published_at"))

        article = ProcessedArticle(
            canonical_url=canonical,
            url=url,
            headline=headline,
            headline_clean=headline_clean,
            headline_hash=headline_hash,
            description=desc_clean,
            content_clean=desc_clean,
            source_id=item.get("source_id", "sample_archive"),
            source_name=item.get("source_name", "Explicit Sample Archive"),
            published_at=published_at,
            retrieved_at=now_utc,
            is_sample=True,  # Explicitly flagged as sample
            raw_metadata={
                "category": item.get("category", "general"),
                "fallback_source": True,
            },
        )
        processed_samples.append(article)

    logger.info(
        "Loaded %d explicitly labelled fallback sample articles from %s",
        len(processed_samples),
        path,
    )
    return processed_samples
