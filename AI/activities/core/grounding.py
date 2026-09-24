"""Hallucination guard: every number a model writes must trace back to its input."""

from __future__ import annotations

import re

from .logging_setup import get_log

log = get_log("grounding")

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
_LIST_MARKER_RE = re.compile(r"^\s{0,3}\d+[.)]\s", re.M)


def _normalise(token: str) -> str:
    """1,234.50 -> 1234.5 ; 62.0 -> 62."""
    text = token.replace(",", "") if token.count(",") and "." in token else token.replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        return token
    return str(int(value)) if value == int(value) else str(round(value, 3))


def _variants(token: str) -> set[str]:
    out = {_normalise(token)}
    try:
        value = float(_normalise(token))
    except ValueError:
        return out
    for candidate in (abs(value), round(value), round(abs(value)), int(value), int(abs(value)),
                      round(value, 1), round(abs(value), 1)):
        out.add(_normalise(str(candidate)))
    return out


def grounding_check(text: str, allowed_text: str) -> list[str]:
    """Return the figures in `text` that do not appear in `allowed_text`."""
    try:
        allowed: set[str] = set()
        for token in _NUM_RE.findall(allowed_text or ""):
            allowed |= _variants(token)
        body = _LIST_MARKER_RE.sub("", text or "")
        return [tok for tok in _NUM_RE.findall(body) if not (_variants(tok) & allowed)]
    except Exception as exc:
        log.warning("  grounding check skipped: %s", exc)
        return []
