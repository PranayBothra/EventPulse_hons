"""Text normalization, HTML cleaning, and whitespace sanitization."""

import html
import re
import unicodedata
from typing import Optional
from bs4 import BeautifulSoup

# Regex fallback for HTML tag removal
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_html(raw_html: Optional[str]) -> str:
    """Strip HTML tags and unescape HTML entities.

    Args:
        raw_html: String potentially containing HTML formatting.

    Returns:
        Clean plain text string.
    """
    if not raw_html:
        return ""

    text = str(raw_html)

    # Use BeautifulSoup with lxml/html.parser
    try:
        soup = BeautifulSoup(text, "html.parser")
        # Remove script and style elements
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()
        text = soup.get_text(separator=" ")
    except Exception:
        # Fallback to regex if parser fails
        text = TAG_RE.sub(" ", text)

    # Unescape HTML entities (&amp;, &nbsp;, etc.)
    text = html.unescape(text)

    return text


def normalize_text(
    text: Optional[str],
    normalize_unicode: bool = True,
    strip_html: bool = True,
    max_length: Optional[int] = None,
) -> str:
    """Normalize text by cleaning HTML, normalizing Unicode, and standardizing whitespace.

    Args:
        text: Input raw text string.
        normalize_unicode: Apply NFKC normalization if True.
        strip_html: Strip HTML tags if True.
        max_length: Optional character truncation boundary.

    Returns:
        Clean, standardized text.
    """
    if not text:
        return ""

    cleaned = str(text)

    if strip_html:
        cleaned = clean_html(cleaned)

    if normalize_unicode:
        # Normalizes smart quotes, non-breaking spaces (\xa0), em-dashes, etc.
        cleaned = unicodedata.normalize("NFKC", cleaned)

    # Replace newlines, tabs, and multiple spaces with a single space
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()

    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()

    return cleaned
