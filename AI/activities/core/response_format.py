"""Reshape a ReportResult into the delivery payload.

Wire format:

    {
      "sections": {"<section title slug>_report": "<that section's markdown>", ...},
      "report":   "<all sections joined, in order>",
      "uuid":     "<request id>",
      "metadata": {"topic": ..., "status": ..., "usage": ..., ...}
    }

`sections` is a flat title -> markdown map so a consumer can pull one block by
name without walking a list. Everything that is not a section body or the joined
report is bookkeeping, so it goes under `metadata` rather than sitting beside the
content. `report` is always the concatenation of `sections.values()`, so the two
views cannot disagree.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

# Keys that stay at the top level; everything else in the result is metadata.
_TOP_LEVEL = ("sections", "report")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

SECTION_KEY_SUFFIX = "_report"


def slugify(text: str) -> str:
    """"Orders and revenue by salesperson" -> "orders_and_revenue_by_salesperson"."""
    return _NON_ALNUM.sub("_", str(text or "").strip().lower()).strip("_")


def _as_dict(obj: Any) -> dict:
    """Accept a ReportResult, a dataclass Section, or a plain dict."""
    if isinstance(obj, Mapping):
        return dict(obj)
    for attr in ("to_dict", "as_dict", "dict", "model_dump"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, Mapping):
                    return dict(out)
            except Exception:
                pass
    return dict(getattr(obj, "__dict__", {}) or {})


def _section_body(section: Mapping[str, Any]) -> str:
    """The heading-inclusive markdown, so a section reads on its own.

    Emptiness is judged on the body rather than on the rendered markdown: a
    section whose body never got written still renders a heading, and a heading
    with nothing under it is a missing section, not a blank entry.
    """
    inner = None
    for field in ("body", "text", "content"):
        value = section.get(field)
        if isinstance(value, str):
            inner = value
            break
    if inner is not None and not inner.strip():
        return ""

    markdown = section.get("markdown")
    if isinstance(markdown, str) and markdown.strip():
        return markdown.strip()
    return (inner or "").strip()


def _section_key(section: Mapping[str, Any], taken: set[str]) -> str:
    """Title first -- the key is meant to be readable -- falling back to the
    internal key, then to a positional name. Collisions get a numeric suffix so
    two sections can never overwrite each other."""
    base = slugify(section.get("title")) or slugify(section.get("key")) or "section"
    if not base.endswith(SECTION_KEY_SUFFIX):
        base += SECTION_KEY_SUFFIX
    key, n = base, 2
    while key in taken:
        key, n = f"{base}_{n}", n + 1
    taken.add(key)
    return key


def build_sections_map(sections: Iterable[Any]) -> dict[str, str]:
    """[Section, ...] -> {"<title>_report": markdown}, order preserved."""
    out: dict[str, str] = {}
    taken: set[str] = set()
    for section in sections or []:
        data = _as_dict(section)
        body = _section_body(data)
        if not body:  # an empty section is a missing section, not a blank entry
            continue
        out[_section_key(data, taken)] = body
    return out


def to_payload(result: Any, uuid: str | None = None, **extra_metadata: Any) -> dict:
    """Build the payload from a ReportResult (or its dict form).

    Anything the result carries beyond `sections` and `report` -- status, topic,
    usage, timings, ungrounded figures, analytics errors -- lands in `metadata`
    untouched, so adding a field upstream needs no change here.
    """
    data = _as_dict(result)

    sections = build_sections_map(data.get("sections") or [])
    if not sections:
        # A result with no sections is a failure result: `build_report` returns
        # one carrying only a plain-English explanation, and so does the guard
        # in `run_all`. Rebuilding `report` from an empty map would throw that
        # explanation away and hand the caller an empty string, so the message
        # becomes the single section instead and the invariant still holds.
        message = data.get("report")
        if isinstance(message, str) and message.strip():
            key = (slugify(data.get("topic")) or "report") + SECTION_KEY_SUFFIX
            sections = {key: message.strip()}

    report = "\n\n".join(sections.values())

    metadata = {k: v for k, v in data.items() if k not in _TOP_LEVEL}
    metadata["section_order"] = list(sections)
    metadata.update(extra_metadata)

    payload: dict[str, Any] = {"sections": sections, "report": report}
    if uuid is not None:
        payload["uuid"] = uuid
    payload["metadata"] = metadata
    return payload