"""Markdown helpers shared by the renderers, plus the heading splitter that
turns a model-written report into addressable sections.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def money(value: float | None) -> str:
    """Always two decimals, thousands-separated, matching the salesperson export
    format exactly ($965,809.16, not $965,810)."""
    return "-" if value is None else f"${value:,.2f}"


_REF_CODE_RE = re.compile(r"\s*[\[(]\s*T\d{2,4}\s*[\])]")
_EG_PREFIX_RE = re.compile(r"^(?:e\.?g\.?|example[s]?:?|for example,?)\s*", re.I)


def clean_bullet(text: str) -> str:
    """Strip a stray internal ref code and a leading "e.g." a model added out of
    habit, then collapse the whitespace that leaves behind."""
    text = _REF_CODE_RE.sub("", str(text or ""))
    text = _EG_PREFIX_RE.sub("", text.strip())
    return re.sub(r"\s{2,}", " ", text).strip(" -")


def bullets(items: list, key: str = "text", limit: int = 5) -> list[str]:
    """Accept either ["string"] or [{"text": "..."}] -- models return both."""
    out, seen = [], set()
    for item in (items or [])[:limit]:
        text = item.get(key) if isinstance(item, dict) else item
        text = clean_bullet(text)
        dedup_key = re.sub(r"[^a-z0-9]", "", text.lower())[:80]
        if text and dedup_key not in seen:
            seen.add(dedup_key)
            out.append(text)
    return out


def dedupe(lines: list[str]) -> list[str]:
    """Drop near-identical lines two agents both thought of."""
    seen, out = set(), []
    for line in lines:
        key = re.sub(r"[^a-z0-9 ]", "", line.lower())[:60]
        if key and key not in seen:
            seen.add(key)
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Heading splitting -- used to carve a model-written report into sections
# ---------------------------------------------------------------------------

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_BOLD_LINE_RE = re.compile(r"^\*\*(.+?)\*\*:?$")


def normalise_title(title: str) -> str:
    """"## **Action Items:**" and "Action items" compare equal."""
    text = re.sub(r"[*_`#]", "", str(title or "")).strip().rstrip(":")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def split_headings(markdown: str) -> list[dict]:
    """Split markdown into [{level, title, body}] blocks.

    A standalone bold line counts as a heading: half the prompts in this project
    ask for `**Key findings**` rather than `## Key findings`, and a reader sees
    no difference between the two.
    """
    blocks: list[dict] = []
    current: dict | None = None
    for line in (markdown or "").splitlines():
        atx = _ATX_RE.match(line.strip())
        bold = _BOLD_LINE_RE.match(line.strip())
        if atx:
            current = {"level": len(atx.group(1)), "title": atx.group(2).strip(), "lines": []}
            blocks.append(current)
        elif bold:
            current = {"level": 3, "title": bold.group(1).strip(), "lines": []}
            blocks.append(current)
        elif current is not None:
            current["lines"].append(line)
        elif line.strip():
            current = {"level": 0, "title": "", "lines": [line]}
            blocks.append(current)
    return [{"level": b["level"], "title": b["title"], "body": "\n".join(b["lines"]).strip()}
            for b in blocks]


def find_heading_body(markdown: str, wanted: str) -> str | None:
    """Body of the section whose heading matches `wanted`, nested sub-headings
    included, or None when the heading is absent."""
    target = normalise_title(wanted)
    blocks = split_headings(markdown)
    for i, block in enumerate(blocks):
        if normalise_title(block["title"]) != target:
            continue
        parts = [block["body"]]
        for nested in blocks[i + 1:]:
            if nested["level"] <= block["level"]:
                break
            heading = "#" * nested["level"] + " " + nested["title"]
            parts += [heading, nested["body"]]
        return "\n\n".join(p for p in parts if p).strip()
    return None
