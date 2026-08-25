"""One date parser for the CSV exports.

Notes and tasks each carried their own version of this, one of which returned
naive datetimes and one tz-aware, which is how comparison bugs get in. This one
always returns tz-aware UTC, or None.

The activity layer keeps its own vectorised pandas parser: same format, but a
per-element Python call on a large log is two orders of magnitude slower.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

# Node/JS Date.toString(), e.g.
# "Fri Aug 07 2026 10:42:27 GMT+0000 (Coordinated Universal Time)"
_TZ_SUFFIX_RE = re.compile(r"\s*\(.*?\)\s*$")
_FORMATS = (
    "%a %b %d %Y %H:%M:%S GMT%z",
    "%a %b %d %Y %H:%M:%S %Z%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%m/%d/%Y",
)

MIN_DATE = datetime.min.replace(tzinfo=timezone.utc)


def parse_date(value: str | None) -> datetime | None:
    """Parse an export timestamp into tz-aware UTC, or None."""
    if not value or not str(value).strip():
        return None
    cleaned = _TZ_SUFFIX_RE.sub("", str(value).strip())
    for fmt in _FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        from dateutil import parser as dateutil_parser

        dt = dateutil_parser.parse(cleaned)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None
