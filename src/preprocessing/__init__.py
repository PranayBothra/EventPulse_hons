"""Preprocessing pipeline for text cleaning, date parsing, and deduplication."""

from src.preprocessing.date_parser import parse_datetime_to_utc
from src.preprocessing.deduplicator import (
    canonicalize_url,
    compute_headline_hash,
    are_headlines_near_duplicates,
)
from src.preprocessing.text_cleaner import clean_html, normalize_text

__all__ = [
    "clean_html",
    "normalize_text",
    "parse_datetime_to_utc",
    "canonicalize_url",
    "compute_headline_hash",
    "are_headlines_near_duplicates",
]
