"""GDELT DOC 2.0 API client for historical news collection in EventPulse.

Connects to the official GDELT DOC 2.0 API (https://api.gdeltproject.org/api/v2/doc/doc)
with rate limiting, exponential backoff, topic queries, date-window partitioning,
content-availability classification, and full metadata provenance.
"""

import json
import logging
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests

from src.preprocessing.deduplicator import canonicalize_url, compute_headline_hash
from src.preprocessing.text_cleaner import clean_html, normalize_text

logger = logging.getLogger("eventpulse.gdelt")

GDELT_DOC_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_TOPICS = [
    "artificial intelligence",
    "quantum computing",
    "climate change",
    "electric vehicles",
    "global economy",
    "biotechnology",
    "space exploration",
    "cybersecurity",
    "geopolitics",
    "public health",
]


class GDELTClient:
    """Robust client for GDELT DOC 2.0 API."""

    def __init__(
        self,
        base_url: str = GDELT_DOC_BASE_URL,
        rate_limit_delay: float = 6.0,
        request_timeout: int = 25,
        max_retries: int = 3,
        user_agent: str = "EventPulse-Research/1.0 (Academic Research; news-event-study)",
    ):
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.last_request_time: float = 0.0

    def _enforce_rate_limit(self) -> None:
        """Enforce polite minimum delay between consecutive GDELT requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_duration = self.rate_limit_delay - elapsed
            logger.debug("Rate limiting: sleeping for %.2fs", sleep_duration)
            time.sleep(sleep_duration)

    def fetch_articles(
        self,
        query: str,
        max_records: int = 250,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        language: str = "eng",
    ) -> List[Dict[str, Any]]:
        """Fetch news articles matching a query from GDELT DOC 2.0 API.

        Args:
            query: Topic or keyword query string.
            max_records: Number of records to request (max 250).
            start_datetime: Format 'YYYYMMDDHHMMSS' or None.
            end_datetime: Format 'YYYYMMDDHHMMSS' or None.
            language: Source language filter (default 'eng').

        Returns:
            List of normalized article dictionaries ready for persistence.
        """
        full_query = f"{query} sourcelang:{language}" if language else query
        params: Dict[str, str] = {
            "query": full_query,
            "mode": "artlist",
            "maxrecords": str(min(max_records, 250)),
            "format": "json",
        }
        if start_datetime:
            params["startdatetime"] = start_datetime
        if end_datetime:
            params["enddatetime"] = end_datetime

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

        records: List[Dict[str, Any]] = []

        for attempt in range(1, self.max_retries + 1):
            self._enforce_rate_limit()
            try:
                self.last_request_time = time.time()
                response = requests.get(
                    self.base_url,
                    params=params,
                    headers=headers,
                    timeout=self.request_timeout,
                )

                if response.status_code == 200:
                    try:
                        data = response.json()
                        raw_articles = data.get("articles", [])
                        logger.info(
                            "GDELT fetch success for '%s': retrieved %d articles",
                            query,
                            len(raw_articles),
                        )
                        for item in raw_articles:
                            normalized = self.normalize_article(item, query)
                            if normalized:
                                records.append(normalized)
                        return records
                    except json.JSONDecodeError:
                        logger.warning(
                            "GDELT response was not valid JSON: %s",
                            response.text[:200],
                        )
                        return []

                elif response.status_code == 429:
                    backoff = self.rate_limit_delay * (2**attempt)
                    logger.warning(
                        "GDELT 429 Too Many Requests (attempt %d/%d). Backing off %.1fs...",
                        attempt,
                        self.max_retries,
                        backoff,
                    )
                    time.sleep(backoff)

                else:
                    logger.warning(
                        "GDELT returned HTTP %d on attempt %d/%d: %s",
                        response.status_code,
                        attempt,
                        self.max_retries,
                        response.text[:200],
                    )
                    time.sleep(self.rate_limit_delay)

            except requests.RequestException as exc:
                logger.warning(
                    "GDELT request exception on attempt %d/%d: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(self.rate_limit_delay * attempt)

        logger.error(
            "Failed to fetch GDELT articles for query '%s' after %d attempts",
            query,
            self.max_retries,
        )
        return records

    def normalize_article(
        self, raw_item: Dict[str, Any], query_topic: str
    ) -> Optional[Dict[str, Any]]:
        """Normalize a raw GDELT JSON article record into EventPulse schema format."""
        url = raw_item.get("url", "").strip()
        title = raw_item.get("title", "").strip()

        if not url or not title:
            return None

        canonical = canonicalize_url(url)
        clean_title = normalize_text(clean_html(title))
        if not clean_title:
            return None

        headline_hash = compute_headline_hash(clean_title)

        # Parse GDELT seendate format: YYYYMMDDTHHMMSSZ
        seendate_raw = raw_item.get("seendate", "")
        published_at: Optional[str] = None
        if seendate_raw:
            try:
                # e.g. 20260902T011500Z
                dt = datetime.strptime(seendate_raw, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=timezone.utc
                )
                published_at = dt.isoformat()
            except ValueError:
                published_at = datetime.now(timezone.utc).isoformat()
        else:
            published_at = datetime.now(timezone.utc).isoformat()

        domain = raw_item.get("domain", "")
        if not domain:
            try:
                domain = urllib.parse.urlparse(url).netloc.lower()
            except Exception:
                domain = "gdelt.org"

        now = datetime.now(timezone.utc).isoformat()

        return {
            "canonical_url": canonical,
            "url": url,
            "headline": title,
            "headline_clean": clean_title,
            "headline_hash": headline_hash,
            "description": clean_title,  # GDELT DOC API provides title without description
            "content_clean": clean_title,
            "source_id": f"gdelt_{domain}",
            "source_name": domain,
            "published_at": published_at,
            "seen_at": seendate_raw or published_at,
            "retrieved_at": now,
            "is_sample": 0,
            "dataset_origin": "event_corpus",
            "content_availability": "HEADLINE_ONLY",
            "language": raw_item.get("language", "English"),
            "country": raw_item.get("sourcecountry"),
            "raw_metadata": json.dumps(
                {
                    "gdelt_query": query_topic,
                    "seendate": seendate_raw,
                    "socialimage": raw_item.get("socialimage"),
                    "domain": domain,
                    "language": raw_item.get("language"),
                    "sourcecountry": raw_item.get("sourcecountry"),
                }
            ),
        }

    def collect_corpus(
        self,
        topics: Optional[List[str]] = None,
        max_per_topic: int = 100,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Collect articles across multiple topics with rate-limiting.

        Args:
            topics: List of topic queries (default: DEFAULT_TOPICS).
            max_per_topic: Maximum articles to fetch per topic.
            start_datetime: Optional date filter.
            end_datetime: Optional date filter.

        Returns:
            List of all successfully normalized articles.
        """
        topic_list = topics or DEFAULT_TOPICS
        all_articles: List[Dict[str, Any]] = []

        logger.info(
            "Starting GDELT multi-topic collection across %d topics", len(topic_list)
        )
        for idx, topic in enumerate(topic_list, start=1):
            logger.info(
                "GDELT [%d/%d] Collecting topic: '%s'", idx, len(topic_list), topic
            )
            items = self.fetch_articles(
                query=topic,
                max_records=max_per_topic,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )
            all_articles.extend(items)
            logger.info(
                "GDELT [%d/%d] Topic '%s' yielded %d articles (cumulative: %d)",
                idx,
                len(topic_list),
                topic,
                len(items),
                len(all_articles),
            )

        return all_articles
