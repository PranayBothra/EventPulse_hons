"""Robust timestamp parsing and timezone normalization."""

from datetime import datetime, timezone
import time
from typing import Any, Optional, Union
from dateutil import parser as date_parser


def parse_datetime_to_utc(
    raw_date: Optional[Union[str, datetime, time.struct_time]],
    fallback_dt: Optional[datetime] = None,
) -> str:
    """Parse a date representation into a standardized ISO-8601 UTC string.

    Args:
        raw_date: Raw date string, datetime object, or struct_time.
        fallback_dt: Datetime to use if parsing fails or input is None.
                     Defaults to current UTC time.

    Returns:
        ISO-8601 UTC formatted string (e.g., '2026-11-24T03:00:00Z').
    """
    if fallback_dt is None:
        fallback_dt = datetime.now(timezone.utc)
    elif fallback_dt.tzinfo is None:
        fallback_dt = fallback_dt.replace(tzinfo=timezone.utc)
    else:
        fallback_dt = fallback_dt.astimezone(timezone.utc)

    if raw_date is None or raw_date == "":
        return fallback_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # If it is a time.struct_time (e.g. from feedparser)
    if isinstance(raw_date, time.struct_time):
        try:
            timestamp = time.mktime(raw_date)
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return fallback_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # If already a datetime
    if isinstance(raw_date, datetime):
        if raw_date.tzinfo is None:
            dt = raw_date.replace(tzinfo=timezone.utc)
        else:
            dt = raw_date.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Otherwise parse string
    date_str = str(raw_date).strip()
    if not date_str:
        return fallback_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        dt = date_parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError, OverflowError):
        # Gracefully fall back on malformed dates
        return fallback_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
