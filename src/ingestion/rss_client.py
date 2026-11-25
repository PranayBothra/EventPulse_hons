"""Resilient RSS/Atom client with retry logic, timeout handling, and malformed XML recovery."""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.ingestion.models import RawFeedItem


class RSSClient:
    """Client for fetching and parsing RSS and Atom news feeds."""

    def __init__(
        self,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EventPulseResearchBot/1.0",
        timeout: int = 15,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.logger = logging.getLogger("eventpulse.rss_client")

        # Configure session with exponential backoff retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
            }
        )

    def fetch_feed_raw(self, url: str) -> str:
        """Fetch raw XML content from a feed URL.

        Args:
            url: RSS/Atom endpoint URL.

        Returns:
            Raw response text.

        Raises:
            requests.RequestException on network or HTTP error.
        """
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def parse_feed(
        self,
        feed_id: str,
        feed_name: str,
        url: str,
        max_items: Optional[int] = None,
    ) -> List[RawFeedItem]:
        """Fetch and parse feed entries into RawFeedItem instances.

        Args:
            feed_id: Unique feed identifier.
            feed_name: Human-readable feed name.
            url: Feed endpoint URL.
            max_items: Optional upper limit on entries to return.

        Returns:
            List of RawFeedItem instances.
        """
        retrieved_at = datetime.now(timezone.utc).isoformat()
        try:
            xml_text = self.fetch_feed_raw(url)
        except Exception as e:
            self.logger.error("Failed to fetch feed '%s' (%s): %s", feed_name, url, e)
            raise

        items = self._parse_with_feedparser(
            xml_text, feed_id, feed_name, retrieved_at
        )

        # If feedparser returned nothing or bozo error on valid content, attempt XML fallback
        if not items and xml_text.strip().startswith("<"):
            self.logger.warning(
                "feedparser produced 0 items for '%s'; attempting ElementTree fallback",
                feed_name,
            )
            items = self._parse_with_element_tree(
                xml_text, feed_id, feed_name, retrieved_at
            )

        if max_items and len(items) > max_items:
            items = items[:max_items]

        self.logger.info(
            "Parsed %d items from '%s' (%s)", len(items), feed_name, feed_id
        )
        return items

    def _parse_with_feedparser(
        self, xml_content: str, feed_id: str, feed_name: str, retrieved_at: str
    ) -> List[RawFeedItem]:
        """Parse feed using python feedparser library."""
        parsed = feedparser.parse(xml_content)
        items: List[RawFeedItem] = []

        for entry in parsed.entries:
            # Extract headline / title
            title = entry.get("title")

            # Extract link / URL
            link = entry.get("link")

            # Extract description / summary
            description = entry.get("summary") or entry.get("description")
            if not description and "content" in entry and entry.content:
                description = entry.content[0].get("value", "")

            # Extract published timestamp
            published_raw = entry.get("published") or entry.get("updated")
            published_parsed = entry.get("published_parsed") or entry.get(
                "updated_parsed"
            )

            raw_meta = {
                "author": entry.get("author"),
                "tags": [t.get("term") for t in entry.get("tags", []) if t.get("term")],
                "guid": entry.get("id"),
            }

            item = RawFeedItem(
                title=title,
                link=link,
                description=description,
                published_raw=published_raw,
                published_parsed=published_parsed,
                source_id=feed_id,
                source_name=feed_name,
                retrieved_at=retrieved_at,
                raw_metadata=raw_meta,
            )
            items.append(item)

        return items

    def _parse_with_element_tree(
        self, xml_content: str, feed_id: str, feed_name: str, retrieved_at: str
    ) -> List[RawFeedItem]:
        """Fallback XML parsing using standard library ElementTree."""
        items: List[RawFeedItem] = []
        try:
            root = ET.fromstring(xml_content)
        except Exception as e:
            self.logger.error("ElementTree parsing also failed: %s", e)
            return items

        # RSS 2.0 channel -> item
        channel = root.find("channel")
        if channel is not None:
            for item_elem in channel.findall("item"):
                title = (
                    item_elem.findtext("title")
                    if item_elem.find("title") is not None
                    else None
                )
                link = (
                    item_elem.findtext("link")
                    if item_elem.find("link") is not None
                    else None
                )
                desc = (
                    item_elem.findtext("description")
                    if item_elem.find("description") is not None
                    else None
                )
                pub_date = (
                    item_elem.findtext("pubDate")
                    if item_elem.find("pubDate") is not None
                    else None
                )

                items.append(
                    RawFeedItem(
                        title=title,
                        link=link,
                        description=desc,
                        published_raw=pub_date,
                        published_parsed=None,
                        source_id=feed_id,
                        source_name=feed_name,
                        retrieved_at=retrieved_at,
                        raw_metadata={"parsed_via": "ElementTree_fallback"},
                    )
                )
            return items

        # Atom feed -> entry
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry_elem in root.findall("atom:entry", ns) or root.findall("entry"):
            title = entry_elem.findtext("title")
            link_elem = entry_elem.find("link")
            link = (
                link_elem.get("href")
                if link_elem is not None
                else entry_elem.findtext("link")
            )
            summary = entry_elem.findtext("summary") or entry_elem.findtext("content")
            updated = entry_elem.findtext("updated") or entry_elem.findtext("published")

            items.append(
                RawFeedItem(
                    title=title,
                    link=link,
                    description=summary,
                    published_raw=updated,
                    published_parsed=None,
                    source_id=feed_id,
                    source_name=feed_name,
                    retrieved_at=retrieved_at,
                    raw_metadata={"parsed_via": "ElementTree_Atom_fallback"},
                )
            )

        return items
