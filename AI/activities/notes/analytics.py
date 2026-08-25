"""Deterministic layer for the notes report: cleanup, scoring, statistics.

Raw CRM notes are mostly noise -- keyboard mashing, "test note 3", the same
sentence pasted twice. Everything countable happens here so the agent only ever
reads notes worth reading:

    1. Cleanup    -> drop junk, boilerplate and duplicate notes
    2. Scoring    -> rank what is left so the highest-signal notes go first
    3. Statistics -> a quantitative summary that grounds the agent's report

A flag never silently deletes a note without counting it: `filter_important_notes`
returns its own statistics so every run can show how much noise was discarded.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from typing import Any, Optional, TypedDict

from ..core.dates import MIN_DATE, parse_date
from ..core.logging_setup import get_log

log = get_log("notes_analytics")


class Note(TypedDict):
    id: str
    distributor_name: Optional[str]
    representativeDuplicate_name: Optional[str]
    text: str
    createdAt: str
    updatedAt: str


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

_FIELDS = ("id", "distributor_name", "representativeDuplicate_name",
           "text", "createdAt", "updatedAt")


def load_notes_from_csv(path: str) -> list[Note]:
    """Read a notes CSV into Note dicts, every field stripped and never None."""
    notes: list[Note] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            notes.append({k: (row.get(k) or "").strip() for k in _FIELDS})  # type: ignore[misc]
    log.info("Loaded %d raw notes from %s", len(notes), path)
    return notes


# ---------------------------------------------------------------------------
# 2. Cleanup and scoring
# ---------------------------------------------------------------------------

MIN_CHARS = 8                  # catches "d", "Gh", "hjjd"
MAX_REPEAT_RUN = 6             # catches "kkkgffghjghgf..." mashing
MIN_VOWEL_RATIO = 0.15         # mashing has almost no vowels
MIN_UNIQUE_CHAR_RATIO = 0.12   # long low-variety strings = mashing/repetition
LOREM_MARKER = "lorem ipsum"

FILLER_PATTERNS = [
    r"^\s*test\s*note\s*\d*\s*$",
    r"^\s*test\s*\d*\.?\d*\s*$",
    r"^\s*new\s*note\s*$",
    r"^\s*note\s*\d*\s*$",
    r"^\s*n\/?a\s*$",
    r"^\s*(asdf|qwerty|qwertyuio|hjjd|jskd)[a-z]*\s*$",
]

# Weighted by how much a term implies something needs doing about it.
ACTION_KEYWORDS = {
    "invoice": 3, "overdue": 4, "unpaid": 4, "payment": 3, "bill": 3,
    "credit": 2, "refund": 2, "owe": 3, "balance": 2,
    "complaint": 4, "damaged": 3, "short": 2, "delay": 3, "cancel": 3,
    "issue": 2, "problem": 3, "dispute": 3, "churn": 4, "competitor": 3,
    "expand": 2, "increase": 2, "trial": 2, "opportunity": 3,
    "follow up": 3, "follow-up": 3, "schedule": 2, "reschedule": 2,
    "meeting": 2, "deadline": 3, "asap": 3, "urgent": 4,
}

MONEY_RE = re.compile(r"[\$€£]\s?\d+|\b\d+[.,]\d{2}\b", re.I)
DATE_MENTION_RE = re.compile(
    r"\b\d{1,2}[/.-]\d{1,2}([/.-]\d{2,4})?\b|\b(mon|tue|wed|thu|fri|sat|sun)\w*\b", re.I)


def is_low_information(text: str) -> bool:
    """Empty strings, boilerplate and keyboard-mashing test data."""
    t = text.strip()
    if len(t) < MIN_CHARS:
        return True
    lower = t.lower()

    if LOREM_MARKER in lower:
        return True
    if any(re.match(p, lower) for p in FILLER_PATTERNS):
        return True
    if re.search(r"(.{1,4})\1{" + str(MAX_REPEAT_RUN) + ",}", lower):
        return True

    letters = [c for c in lower if c.isalpha()]
    if len(letters) > 15 and sum(1 for c in letters if c in "aeiou") / len(letters) < MIN_VOWEL_RATIO:
        return True

    return len(lower) > 30 and len(set(lower)) / len(lower) < MIN_UNIQUE_CHAR_RATIO


def importance_score(text: str) -> int:
    """Rank surviving notes so the highest-signal ones are seen first."""
    lower = text.lower()
    score = sum(w for kw, w in ACTION_KEYWORDS.items() if kw in lower)
    if MONEY_RE.search(text):
        score += 3
    if DATE_MENTION_RE.search(lower):
        score += 1
    score += min(len(text) // 80, 3)  # reward substance, capped
    return score


def filter_important_notes(notes: list[Note], *, min_score: int = 0,
                           max_notes: Optional[int] = None) -> tuple[list[Note], dict[str, int]]:
    """Drop junk and duplicates, then keep the highest-signal notes first.

    Returns (kept, stats). `stats` is logged every run so it is always visible
    how much of the file was noise."""
    seen: set[str] = set()
    stats = {"input": len(notes), "dropped_low_info": 0, "dropped_duplicate": 0,
             "dropped_low_score": 0, "kept": 0}
    scored: list[tuple[int, Note]] = []

    for note in notes:
        text = (note.get("text") or "").strip()
        if not text or is_low_information(text):
            stats["dropped_low_info"] += 1
            continue

        norm = re.sub(r"\s+", " ", text.lower())
        if norm in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(norm)

        score = importance_score(text)
        if score < min_score:
            stats["dropped_low_score"] += 1
            continue
        scored.append((score, note))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    if max_notes is not None:
        scored = scored[:max_notes]

    kept = [n for _, n in scored]
    stats["kept"] = len(kept)
    log.info("Note filtering stats: %s", stats)
    return kept, stats


# ---------------------------------------------------------------------------
# 3. Statistics
# ---------------------------------------------------------------------------

def compute_notes_statistics(raw_notes: list[Note], kept_notes: list[Note],
                             filter_stats: dict[str, int]) -> dict[str, Any]:
    """Quantitative summary of the filtered notes: grounds the agent's report and
    doubles as a sanity check on the pipeline."""
    dates = [d for d in (parse_date(n["createdAt"]) for n in kept_notes) if d]
    per_distributor = Counter(n["distributor_name"] for n in kept_notes if n.get("distributor_name"))
    per_rep = Counter(n["representativeDuplicate_name"] for n in kept_notes
                      if n.get("representativeDuplicate_name"))
    per_month = Counter(d.strftime("%Y-%m") for d in dates)

    keyword_hits: Counter = Counter()
    money_mentions = 0
    for n in kept_notes:
        lower = n["text"].lower()
        for kw in ACTION_KEYWORDS:
            if kw in lower:
                keyword_hits[kw] += 1
        if MONEY_RE.search(n["text"]):
            money_mentions += 1

    avg_len = (round(sum(len(n["text"]) for n in kept_notes) / len(kept_notes), 1)
               if kept_notes else 0)

    return {
        "total_raw_notes": len(raw_notes),
        "total_kept_notes": len(kept_notes),
        "filter_stats": filter_stats,
        "date_range": {
            "earliest": min(dates).strftime("%Y-%m-%d") if dates else None,
            "latest": max(dates).strftime("%Y-%m-%d") if dates else None,
        },
        "notes_per_distributor": dict(per_distributor.most_common()),
        "notes_per_rep": dict(per_rep.most_common()),
        "notes_per_month": dict(sorted(per_month.items())),
        "top_keyword_hits": dict(keyword_hits.most_common(10)),
        "money_mentions": money_mentions,
        "avg_note_length_chars": avg_len,
    }


def stats_to_text_block(stats: dict[str, Any]) -> str:
    """The stats as compact text for the agent's instructions. Doubles as the
    allow-list for the grounding check, so every figure the agent may quote has
    to appear here."""
    return "\n".join([
        f"- Raw notes received: {stats['total_raw_notes']}",
        f"- Notes kept after cleanup: {stats['total_kept_notes']} "
        f"(dropped {stats['filter_stats']['dropped_low_info']} low-information, "
        f"{stats['filter_stats']['dropped_duplicate']} duplicates)",
        f"- Date range covered: {stats['date_range']['earliest']} to {stats['date_range']['latest']}",
        f"- Notes per distributor: {stats['notes_per_distributor']}",
        f"- Notes per rep: {stats['notes_per_rep']}",
        f"- Notes containing a monetary figure: {stats['money_mentions']}",
        f"- Most common flagged terms: {stats['top_keyword_hits']}",
    ])


def build_notes_context(notes: list[Note]) -> str:
    """Group the filtered notes by distributor/rep, newest first, into a compact
    block the agent can read directly."""
    groups: dict[str, list[Note]] = {}
    for n in notes:
        key = (n.get("distributor_name") or n.get("representativeDuplicate_name")
               or "Unassigned")
        groups.setdefault(key, []).append(n)

    blocks = []
    for group_name, group_notes in groups.items():
        group_notes.sort(key=lambda n: parse_date(n["createdAt"]) or MIN_DATE, reverse=True)
        lines = [f"### {group_name}"]
        lines += [f"- [{n['createdAt']}] {n['text'].strip()}" for n in group_notes]
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
