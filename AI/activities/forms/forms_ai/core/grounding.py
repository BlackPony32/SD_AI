"""Figure-level grounding: every number the model writes must trace back to a
number it was given.

This is a *compatible* implementation of the `grounding_check(raw, allowed)`
contract that `llm.py` already calls - if the project already ships one, delete
this file and keep the original. The semantics assumed by llm.py are:

    grounding_check(model_output, allowed_payload) -> list[str]

returning the numeric tokens in `model_output` that cannot be justified by
`allowed_payload`. An empty list means "fully grounded".

Design notes
------------
* The check is deliberately *lexical*, not semantic. It cannot tell you the
  model drew the wrong conclusion; it can tell you the model produced a figure
  nobody gave it, which is the failure mode that destroys trust in a report.
* Rounding drift is tolerated (0.5% or 0.01, whichever is larger) because the
  payload is rounded before serialisation.
* A fraction in the payload (0.42) justifies its percentage form (42, 42.0)
  because writing rates as percentages is normal prose.
* Dates, ISO timestamps, UUIDs and version-like tokens are skipped - they are
  identifiers, not claims.
* The real defence is upstream: precompute every figure the model might want to
  derive (deltas, percent changes, ratios) so it never has to do arithmetic.
  See prompts.build_allowed_payload.
"""

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
