"""Deduplication and canonicalization for URLs and headlines."""

import hashlib
import re
import string
from typing import Optional, Set
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Common tracking / marketing / feed query parameters to strip
TRACKING_PARAMS: Set[str] = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "fbclid",
    "gclid",
    "msclkid",
    "mc_eid",
    "ref",
    "referrer",
    "rss",
    "feedtype",
    "at_medium",
    "at_campaign",
    "at_custom1",
    "at_custom2",
    "at_custom3",
    "at_custom4",
    "ocid",
    "cmp",
    "xtor",
    "ns_mchannel",
    "ns_source",
    "ns_campaign",
}

# Regex to strip common media source suffixes
SOURCE_SUFFIX_RE = re.compile(
    r"\s*(?:-|–|—|\|)\s*(?:BBC\s*(?:News)?|The Guardian|NPR(?:\s*News)?|Al Jazeera|Ars Technica|Reuters|AP News|CNN|The New York Times)\s*$",
    re.IGNORECASE,
)

PUNCTUATION_TRANS = str.maketrans("", "", string.punctuation)

STOPWORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "were", "be", "been", "by", "with", "from", "as",
    "that", "this", "it", "its", "at", "into", "over", "after"
}


def canonicalize_url(raw_url: str) -> str:
    """Canonicalize a URL by normalizing scheme, host, path, and stripping tracking parameters.

    Args:
        raw_url: Original article URL.

    Returns:
        Deterministic canonical URL string.
    """
    if not raw_url:
        return ""

    parsed = urlparse(raw_url.strip())

    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    netloc = parsed.netloc.lower()

    # Normalize path: collapse duplicate slashes and strip trailing slash
    path = re.sub(r"/{2,}", "/", parsed.path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # Filter out tracking query parameters and sort remaining parameters
    cleaned_query_pairs = []
    if parsed.query:
        query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
        for key, val in query_pairs:
            if key.lower() not in TRACKING_PARAMS:
                cleaned_query_pairs.append((key, val))
        cleaned_query_pairs.sort(key=lambda x: (x[0], x[1]))

    new_query = urlencode(cleaned_query_pairs)

    # Omit fragment (#...)
    canonical = urlunparse((scheme, netloc, path, "", new_query, ""))
    return canonical


def normalize_headline_for_comparison(headline: Optional[str]) -> str:
    """Normalize a headline by removing publisher tags, punctuation, and standardizing casing."""
    if not headline:
        return ""

    text = str(headline).strip()
    # Remove publisher branding suffixes (e.g. "- BBC News")
    text = SOURCE_SUFFIX_RE.sub("", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(PUNCTUATION_TRANS)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def compute_headline_hash(headline: Optional[str]) -> str:
    """Compute SHA-256 fingerprint hash for a normalized headline."""
    norm = normalize_headline_for_comparison(headline)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def are_headlines_near_duplicates(
    headline_a: str,
    headline_b: str,
    threshold: float = 0.85,
) -> bool:
    """Determine whether two headlines are near-duplicates using token Jaccard similarity.

    Args:
        headline_a: First headline.
        headline_b: Second headline.
        threshold: Jaccard similarity threshold (0.0 to 1.0).

    Returns:
        True if headlines share token similarity >= threshold.
    """
    norm_a = normalize_headline_for_comparison(headline_a)
    norm_b = normalize_headline_for_comparison(headline_b)

    if not norm_a or not norm_b:
        return False

    if norm_a == norm_b:
        return True

    tokens_a = {w for w in norm_a.split() if w not in STOPWORDS}
    tokens_b = {w for w in norm_b.split() if w not in STOPWORDS}

    if not tokens_a or not tokens_b:
        return False

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    if union == 0:
        return False

    jaccard = intersection / union
    return jaccard >= threshold
