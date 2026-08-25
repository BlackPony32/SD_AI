"""Deterministic layer for task-backlog analysis.

Everything that can be counted is counted here, in code. The LLM never counts;
it only explains. Two payloads come out, deliberately disjoint:

    build_metrics_payload(metrics)        -> AGENT A (numbers)
    build_pairs_payload(metrics, corpus)  -> AGENT B (title+description text)

Design rules:
  1. Load everything, completed included: completed tasks are the denominator of
     every rate that matters. Filtering them out destroys the analysis.
  2. Never silently drop a row. "Noise" is a flag on a task, not a deletion.
  3. Compute rates, not just counts. 14 overdue means nothing until you know
     whether the owner holds 20 tasks or 200.
  4. Analyse the TASK (status, dates, owner, priority) and the TITLE+DESCRIPTION
     PAIR separately: different questions, different agents.
  5. Nothing here raises. A failing section is replaced by a default, logged and
     recorded in metrics["_errors"], and the rest of the report still ships.

Stdlib only (csv, re, statistics). pandas is not required.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..core.dates import parse_date
from ..core.logging_setup import get_log

log = get_log("task_analytics")


def _make_safe(errors: list[dict], stage: str) -> Callable:
    """A `safe(section, default, fn, *args)` runner that logs and records
    failures instead of propagating them, so one broken column degrades exactly
    one section rather than the whole report."""

    def safe(section: str, default: Any, fn: Callable, *args, **kwargs) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            log.exception("  FAILED: %s.%s -> %s: %s", stage, section, type(exc).__name__, exc)
            errors.append({"stage": stage, "section": section,
                           "error": f"{type(exc).__name__}: {exc}"})
            return default

    return safe


# ---------------------------------------------------------------------------
# 1. Loading & parsing
# ---------------------------------------------------------------------------

CLOSED_STATUSES = {"COMPLETED", "DONE", "CANCELLED", "CANCELED", "ARCHIVED"}


def _clean(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def _enrich_row(r: dict) -> dict:
    """Attach normalised `_`-prefixed fields to one raw CSV row."""
    r["_title_raw"] = r.get("title") or ""
    r["_title"] = _clean(r.get("title"))
    r["_desc"] = _clean(r.get("description"))
    r["_status"] = (r.get("status") or "UNKNOWN").strip().upper()
    r["_priority"] = (r.get("priority") or "UNSET").strip().upper()
    r["_rep"] = _clean(r.get("representative_name")) or None
    r["_distributor"] = _clean(r.get("assignedDistributor_name")) or None
    r["_owner"] = r["_rep"] or r["_distributor"]
    r["_due"] = parse_date(r.get("dueDate"))
    r["_created"] = parse_date(r.get("createdAt"))
    r["_open"] = r["_status"] not in CLOSED_STATUSES
    r["_norm_title"] = normalize_text(r["_title"])
    r["_norm_desc"] = normalize_text(r["_desc"])
    r["_account"] = extract_account(r["_title"]) or extract_account(r["_desc"])
    return r


def load_tasks(csv_path: str | Path) -> tuple[list[dict], dict]:
    """Load and normalise every row, completed included. A malformed row is
    dropped and counted, never allowed to abort the load."""
    t0 = time.perf_counter()
    path = Path(csv_path)
    log.info("STAGE 1/3 load: reading %s", path)
    if not path.exists():
        raise FileNotFoundError(f"task file not found: {path}")

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    tasks, skipped = [], 0
    for i, r in enumerate(rows):
        try:
            tasks.append(_enrich_row(r))
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                log.warning("  row %s skipped (%s: %s)", i + 2, type(exc).__name__, exc)

    missing = [c for c in ("title", "status", "dueDate", "createdAt") if c not in columns]
    if missing:
        log.warning("  expected columns absent from export: %s", missing)

    report = {"rows_in_file": len(rows), "rows_loaded": len(tasks), "rows_skipped": skipped,
              "missing_columns": missing,
              "unparsed_due_dates": sum(1 for t in tasks if t.get("dueDate") and not t["_due"])}
    log.info("STAGE 1/3 load: OK -- %s rows loaded, %s skipped, %.2fs",
             len(tasks), skipped, time.perf_counter() - t0)
    if not tasks:
        log.error("  file contained no usable rows")
    return tasks, report


# ---------------------------------------------------------------------------
# 2. Text normalisation, accounts, title/description quality
# ---------------------------------------------------------------------------

_GENERIC_TOKENS = {
    "check", "checks", "call", "calls", "follow", "followup", "up", "visit", "review",
    "delivery", "deliver", "payment", "pay", "invoice", "order", "reorder", "restock",
    "stock", "meeting", "meet", "task", "todo", "note", "issue", "problem", "confirm",
    "weekly", "monthly", "daily", "quarterly", "biweekly", "urgent", "asap", "new",
    "account", "customer", "store", "this", "that", "them", "email", "office", "manager",
}

_ACCOUNT_AFTER_PREP_RE = re.compile(
    r"\b(?:at|from|to|for|with|w/)\s+([A-Z0-9][\w'&.#-]*(?:\s+[A-Z0-9#][\w'&.#-]*){0,3})"
)
_ACCOUNT_STANDALONE_RE = re.compile(r"^[A-Z][\w'&.#-]*(?:\s+[A-Z0-9#][\w'&.#-]*){1,3}$")
_ACCOUNT_SEGMENT_RE = re.compile(r"^[A-Z0-9][\w'&.# -]{2,}$")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, so 'Riverside  Grocery'
    and 'riverside grocery' collide."""
    t = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _looks_generic(candidate: str) -> bool:
    toks = [t for t in re.split(r"\W+", candidate.lower()) if t]
    return not toks or all(t in _GENERIC_TOKENS for t in toks)


def _canonical_account(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip(" .,-").title()


def extract_account(text: str) -> str | None:
    """Heuristic store/account extraction. Handles 'Fresh Mart - wrong order',
    'Fix issue at Northside Pharmacy', 'Collect payment from Main St Convenience',
    and bare 'Green Valley Market'."""
    s = _clean(text)
    if not s:
        return None
    for part in (p.strip() for p in re.split(r"\s*[-–—|:]\s*", s)):
        if 1 <= len(part.split()) <= 4 and _ACCOUNT_SEGMENT_RE.match(part) and not _looks_generic(part):
            if len(part.split()) > 1 or (part[0].isupper() and len(part) > 3):
                return _canonical_account(part)
    m = _ACCOUNT_AFTER_PREP_RE.search(s)
    if m and not _looks_generic(m.group(1)):
        return _canonical_account(m.group(1))
    if _ACCOUNT_STANDALONE_RE.match(s) and not _looks_generic(s):
        return _canonical_account(s)
    return None


_LOW_INFO_TITLES = {
    "todo", "task", "urgent", "asap", "important", "note", "misc", "n a", "na",
    "follow up", "followup", "check in", "call", "reminder", "pending review",
    "fix this", "circle back", "not sure yet", "need more info", "double check this",
    "check email for details", "follow up needed",
}
_TEST_RE = re.compile(r"\b(test|testing|asdf|qwerty|lorem ipsum|dummy|placeholder)\b", re.I)
_KEYMASH_RE = re.compile(r"^[b-df-hj-np-tv-z]{3,8}$", re.I)
# Any letter in any script, not just [a-z] -- see pair_quality.
_WORD_START_RE = re.compile(r"^[^\W\d_]+", re.UNICODE)


def pair_quality(title: str, description: str) -> dict[str, Any]:
    """Grade the (title, description) PAIR, not the title alone: could someone
    who did not write this task pick it up and act on it?

    Flags never remove a task, they only label it, so counts stay reconcilable
    with the source file."""
    t, d = _clean(title), _clean(description)
    nt, nd = normalize_text(t), normalize_text(d)
    flags: list[str] = []

    if not t:
        flags.append("empty_title")
    if nt in _LOW_INFO_TITLES or len(nt) <= 3:
        flags.append("vague_title")
    if _TEST_RE.search(f"{t} {d}") or _KEYMASH_RE.match(nt):
        flags.append("placeholder_text")
    if not d:
        flags.append("no_description")
    elif nd == nt or nd in nt:
        flags.append("description_echoes_title")
    if t and t.isupper() and len(t) > 12:
        flags.append("all_caps")

    # The pair that actually blocks work: nothing usable in the title and no
    # description to fall back on.
    if ("vague_title" in flags or "empty_title" in flags) and not d:
        flags.append("unactionable_pair")

    # Unicode-aware on purpose: [a-z] would score every Cyrillic or accented
    # title as non-actionable.
    has_verb = bool(_WORD_START_RE.match(nt)) and len(nt.split()) >= 2
    return {"flags": flags, "has_verb_start": has_verb,
            "title_words": len(nt.split()), "desc_words": len(nd.split())}


# ---------------------------------------------------------------------------
# 3. Categories (rule layer -- the LLM widens this, it does not replace it)
# ---------------------------------------------------------------------------

CATEGORY_RULES: dict[str, list[str]] = {
    "Payment / Collection": [r"\bpayment\b", r"\binvoice\b", r"\bcollect", r"\bbalance\b",
                             r"\bbilling\b", r"\bpaid\b", r"\bowe[sd]?\b", r"\bcredit\b"],
    "Complaint / Issue": [r"\bcomplaint\b", r"\bissue\b", r"\bproblem\b", r"\bbroken\b",
                          r"\bdamaged?\b", r"\bwrong\b", r"\bmissing\b", r"\bcooler\b",
                          r"\bfix\b", r"\bexpired?\b", r"\bshort(age|ed)?\b", r"\breturn"],
    "Order / Restock": [r"\bre-?order", r"\brestock\b", r"\bstock\b", r"\binventory\b",
                        r"\bsku\b", r"\bshelf\b", r"\btop ?up\b", r"\bnew (product|line)\b"],
    "Delivery / Fulfillment": [r"\bdeliver", r"\bdrop[- ]?off\b", r"\bship", r"\bpickup\b",
                               r"\broute\b", r"\bfreight\b"],
    "Pricing / Promo": [r"\bpric(e|ing)\b", r"\bquote\b", r"\bdiscount\b", r"\bpromo",
                        r"\bdeal\b", r"\bprice sheet\b"],
    "Contract / Admin": [r"\bcontract\b", r"\brenewal\b", r"\bagreement\b", r"\bpaperwork\b",
                         r"\bsign\b", r"\bterms\b"],
    "Visit / Store Check": [r"\bvisit\b", r"\bstore check\b", r"\bwalk[- ]?through\b",
                            r"\bstop by\b", r"\bshelf check\b", r"\bmerchandis"],
    "Meeting / Conversation": [r"\btalk to\b", r"\bmeet(ing)?\b", r"\bdiscuss\b",
                               r"\bcall (them|him|her|owner|customer|back)\b", r"\bintroduce\b"],
    "Follow-up / Check-in": [r"\bfollow[- ]?up\b", r"\bcheck[- ]?in\b", r"\btouch base\b",
                             r"\bcircle back\b", r"\bconfirm\b", r"\breschedule\b", r"\bremind"],
}
_CATEGORY_PATTERNS = {c: [re.compile(p, re.I) for p in ps] for c, ps in CATEGORY_RULES.items()}


def categorize(title: str, description: str) -> str:
    """Score-based, not first-match-wins: the category with the most pattern hits
    wins, so 'Collect payment from X - delivery was wrong' lands sensibly."""
    text = f"{title or ''} {description or ''}"
    scores = {c: sum(1 for p in ps if p.search(text)) for c, ps in _CATEGORY_PATTERNS.items()}
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] else "Uncategorized"


# ---------------------------------------------------------------------------
# 4. Metrics -- part A: the TASK (status, dates, owner, priority)
# ---------------------------------------------------------------------------

def _rate(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def _pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None


def _bucket_age(days: int) -> str:
    if days <= 7:
        return "1-7d"
    if days <= 30:
        return "8-30d"
    if days <= 90:
        return "31-90d"
    return "90d+"


def annotate(tasks: list[dict], now: datetime) -> None:
    """Second enrichment pass: the fields that need `now` or the rule engines --
    category, pair quality, days overdue, age. Guarded per task."""
    failures = 0
    for t in tasks:
        try:
            t["_category"] = categorize(t["_title"], t["_desc"])
        except Exception:
            t["_category"] = "Uncategorized"
            failures += 1
        try:
            t["_quality"] = pair_quality(t.get("_title_raw", t["_title"]), t["_desc"])
        except Exception:
            t["_quality"] = {"flags": [], "has_verb_start": False,
                             "title_words": 0, "desc_words": 0}
            failures += 1
        try:
            t["_overdue_days"] = ((now - t["_due"]).days
                                  if (t["_open"] and t["_due"] and t["_due"] < now) else None)
            t["_age_days"] = (now - t["_created"]).days if t["_created"] else None
        except Exception:
            t["_overdue_days"], t["_age_days"] = None, None
            failures += 1
    if failures:
        log.warning("  annotate: %s per-task field failures (defaults applied)", failures)


def section_overview(tasks, open_tasks, done_tasks, overdue) -> dict[str, Any]:
    """How big the backlog is, how late it is, how much nobody owns."""
    return {
        "total_tasks": len(tasks),
        "open_tasks": len(open_tasks),
        "completed_tasks": len(done_tasks),
        "completion_rate": _rate(len(done_tasks), len(tasks)),
        "open_overdue": len(overdue),
        "overdue_share_of_open_pct": _pct(len(overdue), len(open_tasks)),
        "open_without_due_date": sum(1 for t in open_tasks if not t["_due"]),
        "open_unowned": sum(1 for t in open_tasks if not t["_owner"]),
        "distinct_owners": len({t["_owner"] for t in tasks if t["_owner"]}),
        "statuses": dict(Counter(t["_status"] for t in tasks)),
    }


def section_aging(open_tasks, overdue, overdue_days) -> dict[str, Any]:
    """Separates 'a bit late' from 'abandoned': 90d+ items are dead work still
    carried as a commitment."""
    return {
        "overdue_buckets": dict(Counter(_bucket_age(d) for d in overdue_days)),
        "overdue_days_median": statistics.median(overdue_days) if overdue_days else None,
        "overdue_days_max": max(overdue_days) if overdue_days else None,
        "open_created_over_90d_ago": sum(1 for t in open_tasks if (t.get("_age_days") or 0) > 90),
        "oldest_overdue": [
            {"title": t["_title"], "priority": t["_priority"],
             "owner": t["_owner"] or "(unassigned)", "days_overdue": t["_overdue_days"]}
            for t in sorted(overdue, key=lambda x: -(x["_overdue_days"] or 0))[:5]
        ],
    }


def section_by_priority(tasks) -> dict[str, Any]:
    """The priority-inversion test: if HIGH does not beat LOW here, the priority
    field is decorative."""
    out = {}
    for prio in sorted({t["_priority"] for t in tasks}):
        sub = [t for t in tasks if t["_priority"] == prio]
        sub_open = [t for t in sub if t["_open"]]
        out[prio] = {
            "total": len(sub),
            "open": len(sub_open),
            "completion_rate": _rate(len(sub) - len(sub_open), len(sub)),
            "overdue_open": sum(1 for t in sub_open if t.get("_overdue_days") is not None),
        }
    return out


def section_by_category(tasks) -> dict[str, Any]:
    """Same shape per work type: which kind of work the team quietly never
    finishes."""
    cats: dict[str, dict[str, Any]] = {}
    for t in tasks:
        c = cats.setdefault(t["_category"], {"total": 0, "open": 0, "overdue": 0, "high_open": 0})
        c["total"] += 1
        c["open"] += t["_open"]
        c["overdue"] += t.get("_overdue_days") is not None
        c["high_open"] += t["_priority"] == "HIGH" and t["_open"]
    for c in cats.values():
        c["completion_rate"] = _rate(c["total"] - c["open"], c["total"])
    return dict(sorted(cats.items(), key=lambda kv: -kv[1]["total"]))


def section_by_owner(tasks, min_sample: int) -> dict[str, Any]:
    """Per-owner workload and reliability. Tasks with no representative bucket as
    `(distributor: X)` or `(no owner)`, visible but never mistaken for a person.
    `reliable_sample` says when an agent may comment on an individual."""
    people: dict[str, dict[str, Any]] = {}
    for t in tasks:
        owner = t["_rep"] or (f"(distributor: {t['_distributor']})" if t["_distributor"]
                              else "(no owner)")
        p = people.setdefault(owner, {"total": 0, "open": 0, "overdue": 0,
                                      "high_open": 0, "oldest_overdue_days": 0})
        p["total"] += 1
        p["open"] += t["_open"]
        p["overdue"] += t.get("_overdue_days") is not None
        p["high_open"] += t["_priority"] == "HIGH" and t["_open"]
        if t.get("_overdue_days"):
            p["oldest_overdue_days"] = max(p["oldest_overdue_days"], t["_overdue_days"])
    for p in people.values():
        p["completion_rate"] = _rate(p["total"] - p["open"], p["total"])
        p["overdue_rate_of_open"] = _rate(p["overdue"], p["open"])
        p["reliable_sample"] = p["total"] >= min_sample
    return dict(sorted(people.items(), key=lambda kv: -kv[1]["open"]))


def section_accounts(tasks, top_n: int = 10) -> dict[str, Any]:
    """Rollup by customer/store parsed out of the text: turns 500 scattered rows
    into 'these six accounts are where the trouble is'."""
    accounts: dict[str, dict[str, Any]] = {}
    for t in tasks:
        if not t.get("_account"):
            continue
        a = accounts.setdefault(t["_account"], {"total": 0, "open": 0, "overdue": 0,
                                                "high_open": 0, "categories": Counter()})
        a["total"] += 1
        a["open"] += t["_open"]
        a["overdue"] += t.get("_overdue_days") is not None
        a["high_open"] += t["_priority"] == "HIGH" and t["_open"]
        a["categories"][t["_category"]] += 1
    ranked = sorted(accounts.items(), key=lambda kv: (-kv[1]["open"], -kv[1]["total"]))[:top_n]
    return {k: {**v, "categories": dict(v["categories"].most_common(3))} for k, v in ranked}


# ---------------------------------------------------------------------------
# 5. Metrics -- part B: the TITLE + DESCRIPTION PAIR
# ---------------------------------------------------------------------------

def find_duplicate_pairs(tasks: list[dict], open_only: bool = True) -> list[dict]:
    """Group by (normalised title, normalised description), with ids so the agent
    can cite a group without inventing anything."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in tasks:
        if open_only and not t["_open"]:
            continue
        if t["_norm_title"]:
            groups[(t["_norm_title"], t["_norm_desc"][:80])].append(t)

    out = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        out.append({
            "title": rows[0]["_title"],
            "count": len(rows),
            "owners": sorted({r["_owner"] or "(unassigned)" for r in rows}),
            "ids": [r.get("id") for r in rows[:5]],
        })
    return sorted(out, key=lambda g: -g["count"])


def section_text_pairs(tasks, open_tasks) -> dict[str, Any]:
    """Whether the backlog is legible: do the two text fields together say enough
    to act on, does the description add anything, and how much of the queue is
    the same sentence written twice."""
    total = len(tasks)
    flags = Counter(f for t in tasks for f in t["_quality"]["flags"])

    # Same title, different descriptions -> the title alone is ambiguous.
    by_title: dict[str, set[str]] = defaultdict(set)
    for t in tasks:
        if t["_norm_title"]:
            by_title[t["_norm_title"]].add(t["_norm_desc"])
    ambiguous = sorted(((nt, len(ds)) for nt, ds in by_title.items() if len(ds) > 2),
                       key=lambda kv: -kv[1])[:5]

    title_words = [t["_quality"]["title_words"] for t in tasks if t["_quality"]["title_words"]]
    desc_words = [t["_quality"]["desc_words"] for t in tasks if t["_quality"]["desc_words"]]
    dupes = find_duplicate_pairs(tasks)

    return {
        "pairs_total": total,
        "pair_flag_counts": dict(flags.most_common()),
        "no_description_pct": _pct(flags.get("no_description", 0), total),
        "vague_title_count": flags.get("vague_title", 0),
        "unactionable_pair_count": flags.get("unactionable_pair", 0),
        "unactionable_examples": sorted({
            t["_title"] for t in open_tasks
            if "unactionable_pair" in t["_quality"]["flags"]})[:6],
        "description_echoes_title_count": flags.get("description_echoes_title", 0),
        "verb_start_pct": _pct(sum(1 for t in tasks if t["_quality"]["has_verb_start"]), total),
        "median_title_words": statistics.median(title_words) if title_words else None,
        "median_desc_words": statistics.median(desc_words) if desc_words else None,
        "ambiguous_titles": [{"title": nt, "distinct_descriptions": n} for nt, n in ambiguous],
        "duplicate_group_count": len(dupes),
        "duplicate_extra_rows": sum(g["count"] - 1 for g in dupes),
        "duplicate_examples": dupes[:5],
        "open_unowned_examples": [t["_title"] for t in open_tasks if not t["_owner"]][:5],
    }


def build_pair_corpus(tasks: list[dict], max_units: int = 180,
                      open_only: bool = True) -> list[dict]:
    """Collapse the open backlog into distinct title+description PAIRS: identical
    text repeated 8 times becomes one unit with count=8, which saves tokens and
    hands the agent the repetition signal explicitly. Ordered by count then
    lateness, so a truncated tail is the least consequential."""
    units: dict[str, dict[str, Any]] = {}
    for t in tasks:
        if open_only and not t["_open"]:
            continue
        key = f"{t['_norm_title']}|{t['_norm_desc'][:60]}"
        if not key.strip("|"):
            continue
        u = units.setdefault(key, {
            "title": t["_title"], "desc": t["_desc"], "count": 0,
            "priority": t["_priority"], "max_overdue": 0,
        })
        u["count"] += 1
        u["max_overdue"] = max(u["max_overdue"], t.get("_overdue_days") or 0)
        if t["_priority"] == "HIGH":
            u["priority"] = "HIGH"

    ordered = sorted(units.values(), key=lambda u: (-u["count"], -u["max_overdue"]))[:max_units]
    for i, u in enumerate(ordered, 1):
        u["ref"] = f"T{i:03d}"
    return ordered


# ---------------------------------------------------------------------------
# 6. Metrics assembly
# ---------------------------------------------------------------------------

def compute_metrics(tasks: list[dict], now: datetime | None = None,
                    min_sample: int = 15) -> dict[str, Any]:
    """Every section runs through `safe(...)`: a failure yields that section's
    default and appends to metrics["_errors"] instead of raising.

    `min_sample` guards per-person rates -- a 40% completion rate off 5 tasks is
    noise, and a model will happily name-and-shame on it."""
    t0 = time.perf_counter()
    errors: list[dict] = []
    safe = _make_safe(errors, "metrics")

    now = now or datetime.now(timezone.utc)
    log.info("STAGE 2/3 metrics: computing over %s tasks (as_of=%s)", len(tasks), now.date())
    annotate(tasks, now)

    open_tasks = [t for t in tasks if t["_open"]]
    done_tasks = [t for t in tasks if not t["_open"]]
    overdue = [t for t in open_tasks if t.get("_overdue_days") is not None]
    overdue_days = sorted(t["_overdue_days"] for t in overdue)

    metrics: dict[str, Any] = {"as_of": now.date().isoformat()}
    metrics["overview"] = safe("overview", {}, section_overview,
                               tasks, open_tasks, done_tasks, overdue)
    metrics["aging"] = safe("aging", {}, section_aging, open_tasks, overdue, overdue_days)
    metrics["by_priority"] = safe("by_priority", {}, section_by_priority, tasks)
    metrics["by_category"] = safe("by_category", {}, section_by_category, tasks)
    metrics["by_owner"] = safe("by_owner", {}, section_by_owner, tasks, min_sample)
    metrics["accounts"] = safe("accounts", {}, section_accounts, tasks)
    metrics["text_pairs"] = safe("text_pairs", {}, section_text_pairs, tasks, open_tasks)
    metrics["min_sample_for_person_rates"] = min_sample

    if errors:
        metrics["_errors"] = errors
        metrics["partial"] = True
        log.warning("STAGE 2/3 metrics: %s section(s) failed: %s",
                    len(errors), [e["section"] for e in errors])
    else:
        log.info("STAGE 2/3 metrics: OK -- %.2fs", time.perf_counter() - t0)

    try:
        return json.loads(json.dumps(metrics, default=str))  # JSON-safe once, here
    except Exception as exc:
        log.exception("  metrics not JSON-serialisable: %s", exc)
        return {"as_of": str(now.date()), "partial": True,
                "_errors": errors + [{"stage": "metrics", "section": "serialisation",
                                      "error": str(exc)}]}


# ---------------------------------------------------------------------------
# 7. Prompt payloads -- what each agent actually receives
# ---------------------------------------------------------------------------
# Agent A gets numbers with no free text; agent B gets text with no totals.
# Neither can drift into the other's job.

_TOP_OWNERS = 6
_TOP_ACCOUNTS = 6
_TOP_CATEGORIES = 6


def build_metrics_payload(metrics: dict) -> dict:
    """Compact copy of the metrics for AGENT A. Never mutates the original.

    "Uncategorized" is not a business category, it is every task the rules failed
    to classify -- a legibility problem, not a kind of work. It moves to
    `data_quality` and out of the ranking, so a real category is never crowded
    out of the top N by an artifact of the rule engine."""
    tp = metrics.get("text_pairs") or {}
    by_category = dict(metrics.get("by_category") or {})
    uncategorized = by_category.pop("Uncategorized", None)

    data_quality = {k: tp.get(k) for k in
                    ("no_description_pct", "vague_title_count", "unactionable_pair_count",
                     "duplicate_group_count", "duplicate_extra_rows")}
    if uncategorized:
        data_quality["tasks_without_a_clear_work_type"] = uncategorized.get("total")
        data_quality["open_without_a_clear_work_type"] = uncategorized.get("open")

    return {
        "as_of": metrics.get("as_of"),
        "overview": metrics.get("overview") or {},
        "aging": metrics.get("aging") or {},
        "by_priority": metrics.get("by_priority") or {},
        "by_category": dict(list(by_category.items())[:_TOP_CATEGORIES]),
        "by_owner": dict(list((metrics.get("by_owner") or {}).items())[:_TOP_OWNERS]),
        "accounts": dict(list((metrics.get("accounts") or {}).items())[:_TOP_ACCOUNTS]),
        "data_quality": data_quality,
        "min_sample_for_person_rates": metrics.get("min_sample_for_person_rates"),
        "partial": metrics.get("partial", False),
    }


def build_pairs_payload(metrics: dict, corpus: list[dict]) -> dict:
    """Compact copy of the title/description evidence for AGENT B: the pair text
    and the pair-level quality figures, and nothing about backlog size, owners or
    completion rates -- those belong to agent A.

    `duplicate_titles` carries title+count rather than the full group, so agent B
    can name a real repeated task without seeing owners or ids."""
    tp = metrics.get("text_pairs") or {}
    quality = {k: tp.get(k) for k in
               ("no_description_pct", "vague_title_count", "unactionable_pair_count",
                "unactionable_examples", "description_echoes_title_count",
                "verb_start_pct", "median_title_words", "median_desc_words",
                "ambiguous_titles", "duplicate_group_count", "duplicate_extra_rows")}
    quality["duplicate_titles"] = [{"title": g["title"], "count": g["count"]}
                                   for g in (tp.get("duplicate_examples") or [])]
    return {
        "pair_quality": quality,
        "pairs": [{"ref": u["ref"], "title": u["title"], "desc": u["desc"],
                   "n": u["count"], "priority": u["priority"], "overdue_days": u["max_overdue"]}
                  for u in corpus],
    }


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------

def analyze_tasks_file(csv_path: str | Path, output_dir: str | None = None,
                       now: datetime | None = None, corpus_size: int = 180) -> dict[str, Any]:
    """Load -> metrics -> pair corpus -> dump. Only an unreadable or empty file
    aborts; anything softer is logged and survived."""
    t0 = time.perf_counter()
    errors: list[dict] = []
    safe = _make_safe(errors, "analyze")

    tasks, load_report = load_tasks(csv_path)   # raises only if the file is unusable
    metrics = safe("compute_metrics", {"partial": True}, compute_metrics, tasks, now=now)
    metrics["source_file"] = load_report

    corpus = safe("pair_corpus", [], build_pair_corpus, tasks, corpus_size)
    log.info("STAGE 3/3 corpus: %s distinct title+description pairs from %s rows",
             len(corpus), len(tasks))

    result = {"metrics": metrics, "pair_corpus": corpus, "task_count": len(tasks),
              "analytics_errors": errors + metrics.get("_errors", [])}

    if output_dir:
        def _dump():
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
            (out / "pair_corpus.json").write_text(json.dumps(corpus, indent=2, default=str))
            return True

        safe("write_output", False, _dump)

    log.info("deterministic layer finished in %.2fs (%s error(s))",
             time.perf_counter() - t0, len(result["analytics_errors"]))
    return result
