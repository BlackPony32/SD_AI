"""Figure-level grounding: every number the model writes must trace back to one it was given."""

from __future__ import annotations

import re

from .logging_setup import get_log

log = get_log("grounding")

# Numbers that carry no claim on their own.
TRIVIAL = {0.0, 1.0, 2.0, 100.0}

# Tokens we never treat as figures: ISO dates/times, dd/mm/yyyy, UUIDs, ranges
# inside identifiers, and anything glued to a letter (e.g. "Q3", "p90", "v2").
_SKIP_SPANS = re.compile(
    r"""
    \d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?   # ISO date/datetime
  | \d{1,2}/\d{1,2}/\d{2,4}                            # dd/mm/yyyy
  | \b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b
  | \b\d{2}:\d{2}(?::\d{2})?\b                         # clock time
    """,
    re.X | re.I,
)

# A figure: optional sign, digits with optional thousands separators, optional
# decimal part. Must not be immediately preceded/followed by a word character.
_FIGURE = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])"
                     r"|(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w])")


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", "").lstrip("+"))
    except ValueError:
        return None


def extract_figures(text: str) -> list[tuple[str, float]]:
    """Return (verbatim_token, value) for every figure in `text`."""
    masked = _SKIP_SPANS.sub(lambda m: " " * len(m.group(0)), str(text or ""))
    out: list[tuple[str, float]] = []
    for match in _FIGURE.finditer(masked):
        token = match.group(0)
        value = _to_float(token)
        if value is not None:
            out.append((token, value))
    return out


def _allowed_values(allowed: str) -> set[float]:
    """Every value the model may echo, plus the forms it may legitimately be
    written in (rounded, and percentage form for fractions)."""
    values: set[float] = set(TRIVIAL)
    for _, value in extract_figures(allowed):
        values.add(value)
        values.add(-value)
        for places in (0, 1, 2):
            values.add(round(value, places))
        if 0.0 < abs(value) <= 1.0:                 # rate -> percentage
            pct = value * 100.0
            values.update({pct, round(pct, 1), round(pct, 2), round(pct)})
    return values


def _matches(value: float, allowed: set[float]) -> bool:
    if value in allowed:
        return True
    tolerance = max(0.01, abs(value) * 0.005)
    return any(abs(value - candidate) <= tolerance for candidate in allowed)


def grounding_check(raw: str, allowed: str) -> list[str]:
    """Numeric tokens in `raw` with no support in `allowed`.

    Returns the verbatim tokens (duplicates preserved) so the repair prompt can
    quote them back to the model exactly as it wrote them.
    """
    if not raw:
        return []
    if not allowed:
        log.warning("  grounding: no allowed payload supplied; skipping check")
        return []

    permitted = _allowed_values(allowed)
    unsupported = [token for token, value in extract_figures(raw)
                   if not _matches(value, permitted)]
    return unsupported
