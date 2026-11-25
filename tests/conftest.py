"""Pytest configuration and shared fixtures for EventPulse."""

import os
import tempfile
from typing import Generator
import pytest

from src.config import FeedConfig, GlobalConfig
from src.storage.repository import SQLiteRepository

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test News Channel</title>
    <link>https://testnews.example.com</link>
    <description>Sample RSS feed for testing</description>
    <item>
      <title>Global Leaders Convene for Climate Summit in Geneva</title>
      <link>https://testnews.example.com/world/climate-summit-2026?utm_source=rss&amp;utm_medium=feed#comments</link>
      <description>&lt;p&gt;Delegates from over 120 countries gathered to discuss &lt;b&gt;carbon reduction&lt;/b&gt; targets.&amp;nbsp;&lt;/p&gt;</description>
      <pubDate>Mon, 24 Nov 2026 08:30:00 GMT</pubDate>
      <guid>test-guid-001</guid>
    </item>
    <item>
      <title>Breakthrough in Fusion Energy Net Gain Reported</title>
      <link>https://testnews.example.com/science/fusion-breakthrough</link>
      <description>Scientists announced sustained energy output exceeding input levels.</description>
      <pubDate>Sun, 23 Nov 2026 14:15:00 GMT</pubDate>
      <guid>test-guid-002</guid>
    </item>
  </channel>
</rss>
"""

MALFORMED_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Broken Channel</title>
    <item>
      <!-- Missing title -->
      <link>https://testnews.example.com/broken-item-1</link>
      <description>Item without title</description>
    </item>
    <item>
      <title>Item with Broken Date</title>
      <link>https://testnews.example.com/broken-date</link>
      <description>Date is totally invalid</description>
      <pubDate>Not a valid date at all</pubDate>
    </item>
  </channel>
</rss>
"""


@pytest.fixture
def temp_db() -> Generator[SQLiteRepository, None, None]:
    """Provide a fresh SQLiteRepository pointing to a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    repo = SQLiteRepository(db_path=temp_path)
    yield repo

    # Cleanup temporary database files
    for ext in ["", "-wal", "-shm"]:
        p = temp_path + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


@pytest.fixture
def test_config(temp_db: SQLiteRepository) -> GlobalConfig:
    """Provide test configuration using the temporary database."""
    cfg = GlobalConfig()
    cfg.database.path = str(temp_db.db_path)
    cfg.ingestion.max_articles_per_feed = 10
    return cfg


@pytest.fixture
def sample_feed() -> FeedConfig:
    """Provide a standard test feed configuration."""
    return FeedConfig(
        id="test_channel",
        name="Test News Channel",
        url="https://testnews.example.com/rss.xml",
        category="general",
        enabled=True,
    )
