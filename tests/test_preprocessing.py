"""Unit tests for preprocessing modules: text cleaning, date parsing, and deduplication."""

from datetime import datetime, timezone
import time
import pytest

from src.preprocessing.date_parser import parse_datetime_to_utc
from src.preprocessing.deduplicator import (
    are_headlines_near_duplicates,
    canonicalize_url,
    compute_headline_hash,
    normalize_headline_for_comparison,
)
from src.preprocessing.text_cleaner import clean_html, normalize_text


class TestTextCleaner:
    def test_clean_html_strips_tags(self):
        html = "<p>This is a <b>bold</b> statement with <a href='https://example.com'>a link</a>.</p>"
        result = clean_html(html)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "bold" in result
        assert "a link" in result

    def test_clean_html_unescapes_entities(self):
        html = "Scientists &amp; Researchers declare &quot;Victory&quot; &#39;Today&#39;&nbsp;!"
        result = clean_html(html)
        assert "&amp;" not in result
        assert "&quot;" not in result
        assert '&' in result
        assert '"Victory"' in result
        assert "'Today'" in result

    def test_clean_html_empty_input(self):
        assert clean_html(None) == ""
        assert clean_html("") == ""

    def test_normalize_text_whitespace_and_unicode(self):
        text = "  Paragraph\nwith   multiple\t\tspaces \u201csmart quotes\u201d and non\xa0breaking space.  "
        normalized = normalize_text(text)
        assert "\n" not in normalized
        assert "\t" not in normalized
        assert "smart quotes" in normalized
        assert normalized.startswith("Paragraph with multiple spaces")

    def test_normalize_text_truncation(self):
        text = "This is a long sentence that should be safely truncated."
        truncated = normalize_text(text, max_length=20)
        assert len(truncated) <= 20


class TestDateParser:
    def test_parse_rfc822_date(self):
        date_str = "Mon, 24 Nov 2026 08:30:00 GMT"
        utc_str = parse_datetime_to_utc(date_str)
        assert utc_str == "2026-11-24T08:30:00Z"

    def test_parse_iso8601_date(self):
        date_str = "2026-11-24T14:45:00+02:00"
        utc_str = parse_datetime_to_utc(date_str)
        # 14:45 +02:00 -> 12:45 UTC
        assert utc_str == "2026-11-24T12:45:00Z"

    def test_parse_struct_time(self):
        st = time.struct_time((2026, 11, 24, 10, 0, 0, 0, 328, 0))
        utc_str = parse_datetime_to_utc(st)
        assert utc_str.startswith("2026-11-24T")

    def test_parse_missing_date_falls_back(self):
        fallback = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        utc_str = parse_datetime_to_utc(None, fallback_dt=fallback)
        assert utc_str == "2026-01-01T12:00:00Z"

    def test_parse_malformed_date_falls_back(self):
        fallback = datetime(2026, 5, 10, 9, 30, 0, tzinfo=timezone.utc)
        utc_str = parse_datetime_to_utc("not a valid date string", fallback_dt=fallback)
        assert utc_str == "2026-05-10T09:30:00Z"


class TestDeduplicator:
    def test_canonicalize_url_strips_tracking(self):
        raw = "https://example.com/world/news?utm_source=twitter&utm_medium=social&utm_campaign=breaking&id=123"
        canonical = canonicalize_url(raw)
        assert "utm_source" not in canonical
        assert "utm_medium" not in canonical
        assert "utm_campaign" not in canonical
        assert "id=123" in canonical
        assert canonical == "https://example.com/world/news?id=123"

    def test_canonicalize_url_strips_fragment_and_trailing_slash(self):
        raw = "https://example.com/articles/science-update/#comments-section"
        canonical = canonicalize_url(raw)
        assert not canonical.endswith("/")
        assert "#" not in canonical
        assert canonical == "https://example.com/articles/science-update"

    def test_canonicalize_url_sorts_query_parameters(self):
        url1 = "https://example.com/search?b=2&a=1"
        url2 = "https://example.com/search?a=1&b=2"
        assert canonicalize_url(url1) == canonicalize_url(url2)

    def test_compute_headline_hash_ignores_publisher_suffix(self):
        h1 = "Global Climate Summit Concludes With Landmark Treaty - BBC News"
        h2 = "Global Climate Summit Concludes With Landmark Treaty | The Guardian"
        h3 = "global climate summit concludes with landmark treaty"

        hash1 = compute_headline_hash(h1)
        hash2 = compute_headline_hash(h2)
        hash3 = compute_headline_hash(h3)

        assert hash1 == hash2 == hash3

    def test_are_headlines_near_duplicates(self):
        h1 = "NASA Confirms Discovery of Water Ice on South Pole of Moon"
        h2 = "NASA Confirms Water Ice Discovered on South Pole of the Moon"
        h3 = "Stock Markets Rally Following Major Technology Earnings Surge"

        assert are_headlines_near_duplicates(h1, h2, threshold=0.7)
        assert not are_headlines_near_duplicates(h1, h3, threshold=0.7)
