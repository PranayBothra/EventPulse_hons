"""FreeNewsAPI.io v1 client for discovery, detail retrieval, and EventPulse normalization."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from dotenv import load_dotenv

from src.config import get_project_root
from src.ingestion.models import ProcessedArticle
from src.preprocessing.date_parser import parse_datetime_to_utc
from src.preprocessing.deduplicator import canonicalize_url, compute_headline_hash
from src.preprocessing.text_cleaner import normalize_text
from src.storage.repository import SQLiteRepository

logger = logging.getLogger("eventpulse.freenewsapi")

DEFAULT_BASE_URL = "https://api.freenewsapi.io/v1"
DEFAULT_COUNTRY = "IN"
DEFAULT_LANGUAGE = "en"
DEFAULT_TOPICS: Tuple[str, ...] = ("technology", "science", "business", "world", "politics")


class FreeNewsAPIError(Exception):
    """Base error for FreeNewsAPI client failures."""


class FreeNewsAPIConfigError(FreeNewsAPIError):
    """Raised when FREENEWSAPI_API_KEY is missing or unconfigured."""


class FreeNewsAPIAuthError(FreeNewsAPIError):
    """Raised on HTTP 401/403 authentication or authorization errors."""


class FreeNewsAPINotFoundError(FreeNewsAPIError):
    """Raised on HTTP 404 when a requested endpoint or article UUID is not found."""


class FreeNewsAPIValidationError(FreeNewsAPIError):
    """Raised on invalid query parameter combinations or HTTP 400 validation errors."""


class FreeNewsAPIRateLimitError(FreeNewsAPIError):
    """Raised on HTTP 429 rate-limiting or daily quota exhaustion."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        quota: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.quota = quota or {}


class FreeNewsAPIRequestError(FreeNewsAPIError):
    """Raised on timeouts, connection errors, invalid JSON, or 5xx responses."""


@dataclass
class FreeNewsListingItem:
    """Lightweight article entry returned by GET /v1/news."""

    uuid: str
    title: str
    published_at: Optional[str] = None
    publisher: str = "Unknown Publisher"


@dataclass
class FreeNewsPaginationMeta:
    """Pagination and filter metadata returned by GET /v1/news."""

    page_size: int = 10
    returned: int = 0
    has_more: bool = False
    offset: Optional[int] = None
    next_offset: Optional[int] = None
    next_cursor: Optional[str] = None
    order_by: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FreeNewsArticleDetail:
    """Full article payload returned by GET /v1/details."""

    uuid: str
    title: str
    publisher: str
    original_url: str
    published_at: Optional[str] = None
    incipit: str = ""
    body: str = ""
    thumbnail: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    countries: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)


def load_freenewsapi_key(env_path: Optional[Path] = None) -> str:
    """Load FREENEWSAPI_API_KEY from environment / root .env without mutating or logging it."""
    target_env = env_path or (get_project_root() / ".env")
    if target_env.is_file():
        load_dotenv(dotenv_path=target_env, override=False)
    return os.getenv("FREENEWSAPI_API_KEY", "").strip()


def is_freenewsapi_configured(env_path: Optional[Path] = None) -> bool:
    """Return True if a non-empty FREENEWSAPI_API_KEY is available."""
    return bool(load_freenewsapi_key(env_path=env_path))


def _sanitize_error_text(text: str, api_key: str) -> str:
    """Scrub API credential substrings before surfacing or logging error text."""
    clean = text or ""
    if api_key:
        clean = clean.replace(api_key, "***REDACTED***")
    return clean[:500]


def _slugify_source_id(publisher: str) -> str:
    """Derive a stable EventPulse source_id for a FreeNewsAPI publisher string."""
    slug = re.sub(r"[^a-z0-9]+", "_", (publisher or "freenewsapi").lower()).strip("_")
    return f"freenewsapi_{slug[:48]}" if slug else "freenewsapi_unknown"


class FreeNewsAPIClient:
    """HTTP client for FreeNewsAPI.io v1 endpoints with quota tracking, retries, and SQLite dedup."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 15.0,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        min_request_interval_seconds: float = 0.4,
        http_client: Optional[httpx.Client] = None,
        env_path: Optional[Path] = None,
    ):
        self._explicit_key = api_key
        self._env_path = env_path
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self._external_client = http_client
        self._last_request_ts = 0.0
        self.last_quota: Dict[str, Any] = {}

    def _resolve_api_key(self) -> str:
        key = self._explicit_key if self._explicit_key is not None else load_freenewsapi_key(self._env_path)
        key = (key or "").strip()
        if not key:
            raise FreeNewsAPIConfigError(
                "FREENEWSAPI_API_KEY is not configured. Set it in the root .env file or environment."
            )
        return key

    def _throttle(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.min_request_interval_seconds:
            time.sleep(self.min_request_interval_seconds - elapsed)

    @staticmethod
    def _extract_quota_headers(headers: httpx.Headers) -> Dict[str, Any]:
        quota: Dict[str, Any] = {}
        for h_key, out_key in (
            ("x-ratelimit-limit-day", "daily_limit"),
            ("x-ratelimit-remaining-day", "daily_remaining"),
            ("x-ratelimit-reset-day", "daily_reset_utc"),
            ("retry-after", "retry_after"),
        ):
            val = headers.get(h_key)
            if val is not None:
                if out_key in ("daily_limit", "daily_remaining"):
                    try:
                        quota[out_key] = int(val)
                    except ValueError:
                        quota[out_key] = val
                elif out_key == "retry_after":
                    try:
                        quota[out_key] = float(val)
                    except ValueError:
                        quota[out_key] = val
                else:
                    quota[out_key] = val
        return quota

    def _request_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        api_key = self._resolve_api_key()
        headers = {"x-api-key": api_key, "Accept": "application/json"}

        attempt = 0
        while True:
            self._throttle()
            try:
                if self._external_client is not None:
                    resp = self._external_client.get(
                        f"{self.base_url}{path}",
                        params=params,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                else:
                    with httpx.Client(timeout=self.timeout_seconds) as client:
                        resp = client.get(
                            f"{self.base_url}{path}",
                            params=params,
                            headers=headers,
                        )
                self._last_request_ts = time.monotonic()
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    continue
                raise FreeNewsAPIRequestError(f"FreeNewsAPI request timed out on {path}: {exc}") from exc
            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    continue
                raise FreeNewsAPIRequestError(f"FreeNewsAPI connection error on {path}: {exc}") from exc

            quota = self._extract_quota_headers(resp.headers)
            if quota:
                self.last_quota = quota

            status = resp.status_code
            if status == 200:
                try:
                    payload = resp.json()
                except ValueError as exc:
                    raise FreeNewsAPIRequestError(f"Invalid JSON received from FreeNewsAPI {path}") from exc
                if not isinstance(payload, dict):
                    raise FreeNewsAPIRequestError(f"Unexpected JSON root type from FreeNewsAPI {path}")
                return payload

            safe_body = _sanitize_error_text(resp.text, api_key)
            if status in (401, 403):
                raise FreeNewsAPIAuthError(
                    f"FreeNewsAPI authentication/authorization error (HTTP {status}): {safe_body}"
                )
            if status == 400:
                raise FreeNewsAPIValidationError(f"FreeNewsAPI rejected parameters (HTTP 400): {safe_body}")
            if status == 404:
                raise FreeNewsAPINotFoundError(f"FreeNewsAPI resource not found (HTTP 404): {safe_body}")
            if status == 429:
                daily_rem = quota.get("daily_remaining")
                if isinstance(daily_rem, int) and daily_rem <= 0:
                    raise FreeNewsAPIRateLimitError(
                        "FreeNewsAPI daily quota exhausted (0 remaining requests today).",
                        retry_after=quota.get("retry_after"),
                        quota=quota,
                    )
                retry_wait = quota.get("retry_after")
                wait_sec = float(retry_wait) if isinstance(retry_wait, (int, float)) else (self.backoff_seconds * (2 ** attempt))
                wait_sec = min(max(wait_sec, 0.2), 10.0)
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(wait_sec)
                    continue
                raise FreeNewsAPIRateLimitError(
                    f"FreeNewsAPI rate limit exceeded after retries (HTTP 429): {safe_body}",
                    retry_after=quota.get("retry_after"),
                    quota=quota,
                )

            if status >= 500 and attempt < self.max_retries:
                attempt += 1
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                continue

            raise FreeNewsAPIRequestError(f"FreeNewsAPI error (HTTP {status}) on {path}: {safe_body}")

    def fetch_news_listing(
        self,
        country: Optional[str] = DEFAULT_COUNTRY,
        language: Optional[str] = DEFAULT_LANGUAGE,
        topic: Optional[str] = None,
        in_title: Optional[str] = None,
        cursor: Optional[str] = None,
        offset: Optional[int] = None,
        page_size: Optional[int] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> Tuple[List[FreeNewsListingItem], FreeNewsPaginationMeta]:
        """Call GET /v1/news adhering to documented pagination and filter rules."""
        if cursor is not None and offset is not None:
            raise FreeNewsAPIValidationError(
                "FreeNewsAPI does not allow combining 'cursor' and 'offset' pagination in a single request."
            )

        params: Dict[str, Any] = {}
        if country:
            params["country"] = country.upper()
        if language:
            params["language"] = language.lower()
        if topic:
            params["topic"] = topic.strip().lower()
        if in_title:
            params["in_title"] = in_title.strip()
        if cursor is not None:
            params["cursor"] = cursor
        if offset is not None:
            params["offset"] = offset
        if page_size is not None:
            params["page_size"] = page_size
        if published_after:
            params["published_after"] = published_after
        if published_before:
            params["published_before"] = published_before
        if order_by:
            params["order_by"] = order_by

        payload = self._request_json("/news", params)
        raw_list = payload.get("data")
        if raw_list is None:
            raw_list = []
        if not isinstance(raw_list, list):
            raise FreeNewsAPIRequestError("Invalid 'data' field in /v1/news response; expected array.")

        items: List[FreeNewsListingItem] = []
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            uid = str(entry.get("uuid") or "").strip()
            title = str(entry.get("title") or "").strip()
            if not uid or not title:
                continue
            items.append(
                FreeNewsListingItem(
                    uuid=uid,
                    title=title,
                    published_at=entry.get("published_at"),
                    publisher=str(entry.get("publisher") or "Unknown Publisher"),
                )
            )

        raw_meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        meta = FreeNewsPaginationMeta(
            page_size=int(raw_meta.get("page_size") or len(items) or 10),
            returned=int(raw_meta.get("returned") if raw_meta.get("returned") is not None else len(items)),
            has_more=bool(raw_meta.get("has_more", False)),
            offset=raw_meta.get("offset"),
            next_offset=raw_meta.get("next_offset"),
            next_cursor=raw_meta.get("next_cursor"),
            order_by=raw_meta.get("order_by"),
            filters=raw_meta.get("filters") if isinstance(raw_meta.get("filters"), dict) else {},
        )
        return items, meta

    def fetch_article_details(self, uuid: str) -> FreeNewsArticleDetail:
        """Call GET /v1/details?uuid=... for a single article UUID."""
        clean_uuid = (uuid or "").strip()
        if not clean_uuid:
            raise FreeNewsAPIValidationError("Article UUID is required for /v1/details.")

        payload = self._request_json("/details", {"uuid": clean_uuid})
        raw = payload.get("data")
        if not isinstance(raw, dict):
            raise FreeNewsAPIRequestError(f"Invalid /v1/details response for uuid={clean_uuid}")

        return FreeNewsArticleDetail(
            uuid=str(raw.get("uuid") or clean_uuid),
            title=str(raw.get("title") or "").strip(),
            publisher=str(raw.get("publisher") or "Unknown Publisher").strip(),
            original_url=str(raw.get("original_url") or raw.get("url") or "").strip(),
            published_at=raw.get("published_at"),
            incipit=str(raw.get("incipit") or ""),
            body=str(raw.get("body") or ""),
            thumbnail=raw.get("thumbnail"),
            authors=list(raw.get("authors") or []),
            topics=list(raw.get("topics") or []),
            countries=list(raw.get("countries") or []),
            languages=list(raw.get("languages") or []),
        )

    @staticmethod
    def normalize_detail_to_article(
        detail: FreeNewsArticleDetail,
        retrieved_at: Optional[str] = None,
        min_headline_chars: int = 10,
        max_headline_chars: int = 300,
    ) -> Optional[ProcessedArticle]:
        """Normalize a FreeNewsArticleDetail payload into EventPulse's ProcessedArticle schema."""
        if not detail.title or not detail.original_url:
            return None

        headline_clean = normalize_text(
            detail.title,
            normalize_unicode=True,
            strip_html=True,
            max_length=max_headline_chars,
        )
        if len(headline_clean) < min_headline_chars:
            return None

        canonical = canonicalize_url(detail.original_url)
        if not canonical:
            return None

        raw_desc = detail.incipit or detail.body
        desc_clean = normalize_text(raw_desc, normalize_unicode=True, strip_html=True)
        body_clean = normalize_text(detail.body or detail.incipit, normalize_unicode=True, strip_html=True)
        pub_dt_utc = parse_datetime_to_utc(detail.published_at)
        headline_hash = compute_headline_hash(headline_clean)

        primary_country = detail.countries[0] if detail.countries else None
        primary_language = detail.languages[0] if detail.languages else "en"
        fetched_iso = retrieved_at or datetime.now(timezone.utc).isoformat()

        return ProcessedArticle(
            canonical_url=canonical,
            url=detail.original_url,
            headline=detail.title,
            headline_clean=headline_clean,
            headline_hash=headline_hash,
            description=desc_clean,
            content_clean=body_clean,
            source_id=_slugify_source_id(detail.publisher),
            source_name=detail.publisher,
            published_at=pub_dt_utc,
            retrieved_at=fetched_iso,
            is_sample=False,
            provider="freenewsapi",
            external_id=detail.uuid,
            country=primary_country,
            language=primary_language,
            dataset_origin="live_freenewsapi",
            timestamp_precision="SECOND" if detail.published_at else "UNKNOWN",
            timestamp_source="freenewsapi" if detail.published_at else "fallback_utc",
            timestamp_is_inferred=not bool(detail.published_at),
            published_at_original=detail.published_at,
            raw_metadata={
                "provider": "freenewsapi",
                "uuid": detail.uuid,
                "topics": detail.topics,
                "authors": detail.authors,
                "countries": detail.countries,
                "languages": detail.languages,
                "thumbnail": detail.thumbnail,
            },
        )

    def discover_and_fetch_new_articles(
        self,
        repository: SQLiteRepository,
        country: Optional[str] = DEFAULT_COUNTRY,
        language: Optional[str] = DEFAULT_LANGUAGE,
        topics: Optional[Sequence[str]] = DEFAULT_TOPICS,
        in_title: Optional[str] = None,
        max_pages_per_topic: int = 1,
        max_details_to_fetch: int = 25,
    ) -> Dict[str, Any]:
        """Execute two-stage discovery -> SQLite deduplication -> selective /details retrieval.

        Returns raw telemetry and normalized ProcessedArticle objects without fabricating empty hits.
        """
        topic_list: List[Optional[str]] = (
            [t.strip().lower() for t in topics if t and t.strip()]
            if topics
            else [None]
        )

        discovered_by_uuid: Dict[str, FreeNewsListingItem] = {}
        topic_counts: Dict[str, int] = {}
        listings_seen_total = 0

        for topic_val in topic_list:
            cursor: Optional[str] = None
            t_label = topic_val or "all"
            t_count = 0
            for _page in range(max_pages_per_topic):
                items, meta = self.fetch_news_listing(
                    country=country,
                    language=language,
                    topic=topic_val,
                    in_title=in_title,
                    cursor=cursor,
                )
                listings_seen_total += len(items)
                t_count += len(items)
                for it in items:
                    if it.uuid not in discovered_by_uuid:
                        discovered_by_uuid[it.uuid] = it
                if not meta.has_more or not meta.next_cursor:
                    break
                cursor = meta.next_cursor
            topic_counts[t_label] = t_count

        candidate_uuids = list(discovered_by_uuid.keys())
        existing_uuids = repository.get_existing_external_ids("freenewsapi", candidate_uuids)

        new_candidates: List[FreeNewsListingItem] = []
        skipped_existing = 0
        for uid, item in discovered_by_uuid.items():
            if uid in existing_uuids:
                skipped_existing += 1
                continue
            h_clean = normalize_text(item.title, normalize_unicode=True, strip_html=True, max_length=300)
            if len(h_clean) >= 10:
                h_hash = compute_headline_hash(h_clean)
                if repository.is_duplicate_headline(h_hash):
                    skipped_existing += 1
                    continue
            new_candidates.append(item)

        to_fetch = new_candidates[:max_details_to_fetch]
        processed: List[ProcessedArticle] = []
        failed_details = 0
        malformed_details = 0
        now_iso = datetime.now(timezone.utc).isoformat()

        for cand in to_fetch:
            try:
                detail = self.fetch_article_details(cand.uuid)
            except (FreeNewsAPINotFoundError, FreeNewsAPIRequestError) as exc:
                logger.warning("Failed fetching /details for uuid=%s: %s", cand.uuid, exc)
                failed_details += 1
                continue
            norm_art = self.normalize_detail_to_article(detail, retrieved_at=now_iso)
            if norm_art is None:
                malformed_details += 1
                continue
            processed.append(norm_art)

        return {
            "country": country,
            "language": language,
            "topics": [t for t in topic_list if t is not None],
            "topic_listing_counts": topic_counts,
            "listings_fetched": listings_seen_total,
            "unique_discovered": len(discovered_by_uuid),
            "duplicates_skipped_before_details": skipped_existing,
            "details_requested": len(to_fetch),
            "details_failed": failed_details,
            "malformed_skipped": malformed_details,
            "processed_articles": processed,
            "quota": dict(self.last_quota),
        }
