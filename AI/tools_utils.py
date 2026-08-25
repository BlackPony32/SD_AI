from __future__ import annotations

import difflib
import functools
import re
import statistics
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

# ===========================================================================
# 1. Configuration
# ===========================================================================

#: Where the per-user exports live. Overridable with `set_data_root`.
DATA_ROOT = Path("data")

#: Logical dataset -> the filenames it may have been exported under, best first.
#: Resolution is prefix/substring based on top of this, so `raw_file_notes.csv`,
#: `notes_2026.csv` and `cleaned_notes.csv` all resolve to "notes" without
#: needing to be listed.
DATASET_FILENAMES: dict[str, tuple[str, ...]] = {
    "notes": ("raw_file_notes.csv", "cleaned_notes.csv"),
    "activities": ("raw_file_activities.csv", "cleaned_activities.csv"),
    "tasks": ("raw_file_tasks.csv", "tasks_synthetic_3.csv", "cleaned_tasks.csv"),
    "orders": ("raw_file_orders.csv", "cleaned_orders.csv"),
}

#: The default window when the caller names no period at all.
DEFAULT_PERIOD = "last month"

#: Below this many records on either side of a comparison, a percentage is
#: arithmetic rather than evidence, and is flagged as such.
MIN_SAMPLE = 30

#: Per-person rates below this many records are not reported as performance.
MIN_SAMPLE_PER_PERSON = 10

#: Rows sharing one exact creation second at or above this count were imported,
#: not performed by a human.
BULK_SECOND_MIN = 5

#: Recency bands, in days, for calling someone active / cooling / dormant.
DORMANT_DAYS = (30, 90, 180)

TRUE_STRINGS = {"true", "t", "yes", "y", "1"}
NULLISH = {"", "nan", "none", "null", "n/a", "na", "<na>", "nat"}

# --- Activities -----------------------------------------------------------

#: Explicit type -> category. Checked before the keyword rules below, so a type
#: whose name misleads (CREDIT_MEMO_ADDED reads commercial, behaves like risk)
#: can be pinned.
ACTIVITY_CATEGORIES: dict[str, str] = {
    "ORDER_ADDED": "commercial",
    "ORDER_CANCELED": "risk",
    "CREDIT_MEMO_ADDED": "risk",
    "CREDIT_MEMO_VOIDED": "risk",
    "CREDIT_MEMO_DELETED": "risk",
    "CUSTOMER_MERGED": "admin",
    "NOTE_ADDED": "engagement",
    "COMMENT_ADDED": "engagement",
    "TASK_ADDED": "engagement",
    "TASK_COMPLETED": "engagement",
    "CHECKED_IN": "engagement",
    "PHOTO_GROUP_ADDED": "engagement",
}

#: First matching group wins. Ordered so that reversal words (CANCEL, VOID) beat
#: the noun they attach to, which is why ORDER_CANCELED lands in "risk" and not
#: "commercial". A type this file has never seen still lands somewhere sensible.
ACTIVITY_CATEGORY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("CANCEL", "VOID", "DELET", "REFUND", "RETURN", "CREDIT_MEMO", "DISPUTE",
      "FAIL", "REJECT", "CHARGEBACK"), "risk"),
    (("MERGE", "IMPORT", "SYNC", "MIGRAT", "ARCHIV", "SETTING", "PERMISSION",
      "LOGIN", "EXPORT"), "admin"),
    (("ORDER", "INVOICE", "PAYMENT", "QUOTE", "CART", "CHECKOUT"), "commercial"),
    (("TASK", "NOTE", "COMMENT", "CHECK", "PHOTO", "VISIT", "CALL", "EMAIL",
      "MESSAGE", "MEETING"), "engagement"),
)

ACTIVITY_CATEGORY_LABELS = {
    "commercial": "Commercial",
    "engagement": "Engagement",
    "risk": "Risk",
    "admin": "Admin",
    "other": "Other",
}
ACTIVITY_CATEGORY_DESCRIPTIONS = {
    "commercial": "orders and money moving",
    "engagement": "notes, tasks, comments, check-ins",
    "risk": "cancellations, credit memos, voids",
    "admin": "record keeping, merges, integrations",
    "other": "not classified by the rule engine",
}

#: (opening type, closing type, plural label, what the ratio means). These are
#: cohort ratios over the same window, not per-item matching -- the export
#: carries no parent id linking a completion back to its creation.
ACTIVITY_WORKFLOW_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("TASK_ADDED", "TASK_COMPLETED", "tasks", "completion"),
    ("ORDER_ADDED", "ORDER_CANCELED", "orders", "cancellation"),
    ("CREDIT_MEMO_ADDED", "CREDIT_MEMO_VOIDED", "credit memos", "reversal"),
)

#: Words the generic title-caser gets wrong.
LABEL_OVERRIDES = {
    "QUICKBOOKS": "QuickBooks",
    "SHOPIFY": "Shopify",
    "DISTRIBUTOR": "Distributor (back office)",
    "REPRESENTATIVE": "Sales representative",
    "ORDER_DIRECT": "Direct order (customer app)",
    "SALES": "Sales rep app",
    "API": "API",
    "SMS": "SMS",
    "PDF": "PDF",
    "SKU": "SKU",
}

# --- Notes ----------------------------------------------------------------

NOTE_MIN_CHARS = 8            # "d", "Gh", "hjjd"
NOTE_MAX_REPEAT_RUN = 6       # "kkkgffghjghgf..." keyboard mashing
NOTE_MIN_VOWEL_RATIO = 0.15   # mashed text has almost no vowels
NOTE_MIN_UNIQUE_RATIO = 0.12  # long, low-variety strings are repetition

NOTE_FILLER_PATTERNS = (
    # "test note", "test note 3", "Test note 14.06", "test 2.1"
    r"^\s*test\s*note\b[\s\d.,/:-]*$",
    r"^\s*test(?:ing)?\b[\s\d.,/:-]*$",
    r"^\s*new\s*note\s*$",
    r"^\s*note\s*\d*\s*$",
    r"^\s*n\/?a\s*$",
    r"^\s*(?:asdf|qwerty|qwertyuio|hjjd|jskd|rtrt)[a-z]*\s*$",
    r"^\s*\W+\s*$",
)

#: Weighted by how strongly the term implies something needs doing about it.
#: Used both to rank notes and to summarise what the notes are *about*.
NOTE_ACTION_KEYWORDS: dict[str, int] = {
    "invoice": 3, "overdue": 4, "unpaid": 4, "payment": 3, "bill": 3,
    "credit": 2, "refund": 2, "owe": 3, "balance": 2, "collect": 3,
    "complaint": 4, "damaged": 3, "short": 2, "delay": 3, "cancel": 3,
    "issue": 2, "problem": 3, "dispute": 3, "churn": 4, "competitor": 3,
    "expand": 2, "increase": 2, "trial": 2, "opportunity": 3, "upsell": 3,
    "follow up": 3, "follow-up": 3, "schedule": 2, "reschedule": 2,
    "meeting": 2, "deadline": 3, "asap": 3, "urgent": 4, "escalate": 4,
}

#: Coarse buckets over those keywords, so a summary can say "half the notes are
#: about money" instead of listing thirty terms.
NOTE_THEMES: dict[str, tuple[str, ...]] = {
    "Money / collections": ("invoice", "overdue", "unpaid", "payment", "bill",
                            "credit", "refund", "owe", "balance", "collect"),
    "Problems / complaints": ("complaint", "damaged", "short", "delay", "cancel",
                              "issue", "problem", "dispute", "escalate"),
    "Churn risk": ("churn", "competitor"),
    "Growth": ("expand", "increase", "trial", "opportunity", "upsell"),
    "Scheduling": ("follow up", "follow-up", "schedule", "reschedule",
                   "meeting", "deadline"),
    "Urgency": ("asap", "urgent"),
}

MONEY_RE = re.compile(r"[\$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d+[.,]\d{2}\b")
DATE_MENTION_RE = re.compile(
    r"\b\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b|\b(?:mon|tue|wed|thu|fri|sat|sun)\w*\b", re.I)

# --- Tasks ----------------------------------------------------------------

TASK_CLOSED_STATUSES = {"COMPLETED", "DONE", "CLOSED", "CANCELLED", "CANCELED", "ARCHIVED"}

#: Score-based, not first-match-wins: the category with the most pattern hits
#: takes the task, so "Collect payment from X - delivery was wrong" lands on the
#: dominant subject rather than whichever rule happens to be listed first.
TASK_CATEGORY_RULES: dict[str, tuple[str, ...]] = {
    "Payment / collection": (r"\bpayment\b", r"\binvoice\b", r"\bcollect", r"\bbalance\b",
                             r"\bbilling\b", r"\bpaid\b", r"\bowe[sd]?\b", r"\bcredit\b"),
    "Complaint / issue": (r"\bcomplaint\b", r"\bissue\b", r"\bproblem\b", r"\bbroken\b",
                          r"\bdamaged?\b", r"\bwrong\b", r"\bmissing\b", r"\bcooler\b",
                          r"\bfix\b", r"\bexpired?\b", r"\bshort(?:age|ed)?\b", r"\breturn"),
    "Order / restock": (r"\bre-?order", r"\brestock\b", r"\bstock\b", r"\binventory\b",
                        r"\bsku\b", r"\bshelf\b", r"\btop ?up\b", r"\bnew (?:product|line)\b"),
    "Delivery": (r"\bdeliver", r"\bdrop[- ]?off\b", r"\bship", r"\bpickup\b",
                 r"\broute\b", r"\bfreight\b"),
    "Pricing / promo": (r"\bpric(?:e|ing)\b", r"\bquote\b", r"\bdiscount\b", r"\bpromo",
                        r"\bdeal\b", r"\bprice sheet\b"),
    "Contract / admin": (r"\bcontract\b", r"\brenewal\b", r"\bagreement\b", r"\bpaperwork\b",
                         r"\bsign\b", r"\bterms\b"),
    "Store visit": (r"\bvisit\b", r"\bstore check\b", r"\bwalk[- ]?through\b",
                    r"\bstop by\b", r"\bshelf check\b", r"\bmerchandis"),
    "Meeting / conversation": (r"\btalk to\b", r"\bmeet(?:ing)?\b", r"\bdiscuss\b",
                               r"\bcall (?:them|him|her|owner|customer|back)\b", r"\bintroduce\b"),
    "Follow-up": (r"\bfollow[- ]?up\b", r"\bcheck[- ]?in\b", r"\btouch base\b",
                  r"\bcircle back\b", r"\bconfirm\b", r"\breschedule\b", r"\bremind"),
}
_TASK_CATEGORY_PATTERNS = {c: [re.compile(p, re.I) for p in ps]
                           for c, ps in TASK_CATEGORY_RULES.items()}

#: Titles that name no subject. A task called "follow up" is a reminder that
#: something exists, not a description of what to do.
TASK_LOW_INFO_TITLES = {
    "todo", "task", "urgent", "asap", "important", "note", "misc", "n a", "na",
    "follow up", "followup", "check in", "checkin", "call", "reminder",
    "pending review", "fix this", "circle back", "not sure yet", "need more info",
    "double check this", "check email for details", "follow up needed", "tbd",
}
TASK_PLACEHOLDER_RE = re.compile(
    r"\b(?:test|testing|asdf|qwerty|lorem ipsum|dummy|placeholder)\b", re.I)
TASK_KEYMASH_RE = re.compile(r"^[b-df-hj-np-tv-z]{3,8}$", re.I)

#: Unicode-aware on purpose: an `[a-z]` check would score every Cyrillic or
#: accented title as unactionable.
WORD_START_RE = re.compile(r"^[^\W\d_]+", re.UNICODE)

#: Store/account extraction out of free text.
_ACCOUNT_AFTER_PREP_RE = re.compile(
    r"\b(?:at|from|to|for|with|w/)\s+([A-Z0-9][\w'&.#-]*(?:\s+[A-Z0-9#][\w'&.#-]*){0,3})")
_ACCOUNT_STANDALONE_RE = re.compile(r"^[A-Z][\w'&.#-]*(?:\s+[A-Z0-9#][\w'&.#-]*){1,3}$")
_ACCOUNT_SEGMENT_RE = re.compile(r"^[A-Z0-9][\w'&.# -]{2,}$")
_GENERIC_TOKENS = {
    "check", "checks", "call", "calls", "follow", "followup", "up", "visit", "review",
    "delivery", "deliver", "payment", "pay", "invoice", "order", "reorder", "restock",
    "stock", "meeting", "meet", "task", "todo", "note", "issue", "problem", "confirm",
    "weekly", "monthly", "daily", "quarterly", "biweekly", "urgent", "asap", "new",
    "account", "customer", "store", "this", "that", "them", "email", "office", "manager",
    "routine", "shelf", "expiry", "dates", "details", "upcoming", "drop", "performance",
}

# --- Orders ---------------------------------------------------------------

ORDER_SALESPERSON_FIELD = "salesDuplicate_name"
ORDER_AMOUNT_FIELD = "totalAmount"
UNASSIGNED_SALESPERSON = "Unassigned / direct"
#: Third-party orders arrive through an outside channel, so counting them as
#: team output overstates what the team sold. Excluded by default, but the
#: exclusion is always reported rather than applied silently.
ORDER_TYPES_EXCLUDED = ("THIRD_PARTY",)


def set_data_root(path: str | Path) -> None:
    """Point every tool at a different export directory (tests, other tenants)."""
    global DATA_ROOT
    DATA_ROOT = Path(path)


# ===========================================================================
# 2. Failure envelope
# ===========================================================================

class ToolError(Exception):
    """A failure the caller can act on: a bad filter, a missing file, an empty
    period. Carries a hint that names the next thing to try.

    Distinguished from an unexpected exception because the message is written
    for the agent, and the traceback is not attached.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def render(self) -> str:
        out = f"**No result.** {self.message}"
        if self.hint:
            out += f"\n\n*Try:* {self.hint}"
        return out


def tool_guard(fn: Callable[..., str]) -> Callable[..., str]:
    """Guarantee a tool returns a readable string.

    A `ToolError` renders as a short, actionable message. Anything else renders
    as an error block with the exception type and a truncated traceback, because
    an agent that is told *why* a call failed can usually fix the call, whereas
    a bare "error" leads it to retry the same thing.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return fn(*args, **kwargs)
        except ToolError as exc:
            return exc.render()
        except Exception as exc:  # noqa: BLE001 - a tool must never propagate
            tb = "".join(traceback.format_exc()).strip().splitlines()
            return (f"**Tool failed.** `{fn.__name__}` raised "
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"This is a bug in the tool, not in your call — the arguments you passed "
                    f"were accepted. Report it or try a narrower query.\n\n"
                    f"```\n" + "\n".join(tb[-6:]) + "\n```")

    return wrapper


# ===========================================================================
# 3. Loading
# ===========================================================================

_CACHE: dict[tuple[str, float, int], pd.DataFrame] = {}


def resolve_path(user_id: str, dataset: str) -> Path:
    """Find the CSV for one logical dataset under `DATA_ROOT/<user_id>/`.

    Tries the known filenames first, then any *.csv whose stem contains the
    dataset name, so exports keep working when someone renames
    `activities.csv` to `raw_file_activities_2026.csv`.
    """
    if dataset not in DATASET_FILENAMES:
        raise ToolError(f"Unknown dataset '{dataset}'.",
                        f"one of: {', '.join(sorted(DATASET_FILENAMES))}")

    base = DATA_ROOT / str(user_id)
    if not base.is_dir():
        available = sorted(p.name for p in DATA_ROOT.iterdir()) if DATA_ROOT.is_dir() else []
        raise ToolError(
            f"No data directory for user '{user_id}' (looked in `{base}`).",
            (f"known user ids: {', '.join(available[:15])}" if available
             else f"create `{base}` and place the CSV exports in it"))

    for name in DATASET_FILENAMES[dataset]:
        candidate = base / name
        if candidate.is_file():
            return candidate

    stem_hits = sorted(p for p in base.glob("*.csv") if dataset[:-1] in p.stem.lower())
    if stem_hits:
        return stem_hits[0]

    present = sorted(p.name for p in base.glob("*.csv"))
    raise ToolError(
        f"No {dataset} file for user '{user_id}'. Looked for "
        f"{', '.join(DATASET_FILENAMES[dataset])} in `{base}`.",
        (f"files actually present: {', '.join(present)}" if present
         else f"`{base}` contains no CSV files at all"))


def read_csv_cached(path: Path) -> pd.DataFrame:
    """Read a CSV once per (path, mtime, size).

    An agent asks four or five questions of the same export in a row; without
    this, each one re-parses the file and re-runs the timestamp conversion. The
    key includes mtime and size so a file replaced between calls is re-read.
    Returns a defensive copy — callers add derived columns freely.
    """
    stat = path.stat()
    key = (str(path), stat.st_mtime, stat.st_size)
    if key not in _CACHE:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (df[col].astype("string").str.strip()
                           .replace({v: pd.NA for v in ("", "nan", "None", "null", "NULL")}))
        _CACHE.clear()          # one export at a time; keeps memory flat
        _CACHE[key] = df
    return _CACHE[key].copy()


def clear_cache() -> None:
    """Drop the parsed-CSV cache. Call after rewriting an export mid-session."""
    _CACHE.clear()


def require_columns(df: pd.DataFrame, required: Sequence[str], dataset: str) -> None:
    """Fail with the actual column list rather than a KeyError three frames down."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ToolError(
            f"The {dataset} export is missing required column(s): {', '.join(missing)}.",
            f"columns present: {', '.join(df.columns[:25])}")


# ===========================================================================
# 4. Timestamps
# ===========================================================================

#: These exports store Node's `Date.toString()`:
#: "Fri Aug 07 2026 10:42:27 GMT+0000 (Coordinated Universal Time)".
#: Slicing to the offset keeps pandas on its C parser, which is roughly two
#: orders of magnitude faster than letting it guess per row.
_JS_FORMAT = "%a %b %d %Y %H:%M:%S GMT%z"
_JS_LEN = 33
_PAREN_RE = re.compile(r"\(.*\)")


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Vectorised timestamp parsing, UTC, with a flexible retry for stragglers.

    Anything unparseable becomes NaT rather than raising — a single malformed
    row must not cost the whole answer. Callers report the NaT count.
    """
    text = series.astype("string")
    out = pd.to_datetime(text.str.slice(0, _JS_LEN), format=_JS_FORMAT,
                         errors="coerce", utc=True)
    missing = out.isna() & text.notna()
    if bool(missing.any()):
        retry = text[missing].str.replace(_PAREN_RE, "", regex=True).str.strip()
        out.loc[missing] = pd.to_datetime(retry, errors="coerce", utc=True, format="mixed")
    return out


def to_utc(value: datetime | date | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def fmt_date(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def fmt_datetime(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


# ===========================================================================
# 5. Periods
# ===========================================================================

@dataclass
class Period:
    """A resolved time window, plus everything needed to explain it.

    `start` is inclusive, `end` exclusive. `anchor` is the date the rolling
    windows were measured back from — the newest record in the file, not today,
    because these are exports and "the last 30 days" of a file that stops in
    August means the 30 days before it stopped.
    """

    start: pd.Timestamp | None            # None = open-ended (no lower bound)
    end: pd.Timestamp | None              # None = open-ended (no upper bound)
    label: str                            # human phrasing, e.g. "last 30 days"
    spec: str                             # what the caller actually passed
    anchor: pd.Timestamp                  # reference date the window ran back from
    anchor_is_file_max: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def is_all_time(self) -> bool:
        return self.start is None and self.end is None

    @property
    def days(self) -> int | None:
        if self.start is None or self.end is None:
            return None
        return max(int((self.end - self.start).total_seconds() // 86400), 0)

    def mask(self, timestamps: pd.Series) -> pd.Series:
        """Boolean mask over a tz-aware datetime series. NaT is always False."""
        keep = timestamps.notna()
        if self.start is not None:
            keep &= timestamps >= self.start
        if self.end is not None:
            keep &= timestamps < self.end
        return keep

    def previous(self) -> "Period | None":
        """The equally long window immediately before this one, for comparisons.

        Returns None for all-time and for open-ended windows, where "the period
        before" has no meaning.
        """
        if self.start is None or self.end is None:
            return None
        length = self.end - self.start
        return Period(start=self.start - length, end=self.start,
                      label=f"the {self.days} days before that", spec="(derived)",
                      anchor=self.anchor, anchor_is_file_max=self.anchor_is_file_max)

    def describe(self) -> str:
        if self.is_all_time:
            return "all time"
        if self.start is None:
            return f"everything before {fmt_date(self.end)}"
        if self.end is None:
            return f"{fmt_date(self.start)} onwards"
        return f"{fmt_date(self.start)} to {fmt_date(self.end - timedelta(seconds=1))}"


_ALL_TIME_WORDS = {"all", "all time", "alltime", "everything", "ever", "lifetime",
                   "full", "entire period", "entire", "total", "no limit", "none"}

_RELATIVE_UNITS = {
    "d": 1, "day": 1, "days": 1,
    "w": 7, "week": 7, "weeks": 7,
    "m": 30, "mo": 30, "month": 30, "months": 30,
    "q": 91, "quarter": 91, "quarters": 91,
    "y": 365, "year": 365, "years": 365,
}

_MONTH_NAMES = {m.lower(): i for i, m in enumerate(
    ("January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"), start=1)}
_MONTH_NAMES.update({m[:3]: i for m, i in list(_MONTH_NAMES.items())})

_RANGE_SPLIT_RE = re.compile(r"\s*(?:\.\.|\.\.\.|--|—|:|\bto\b|\bthrough\b|\buntil\b)\s*", re.I)
_ISO_DAY_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_ISO_MONTH_RE = re.compile(r"^(\d{4})[-/](\d{1,2})$")
_YEAR_RE = re.compile(r"^(\d{4})$")
_QUARTER_RE = re.compile(r"^(?:q([1-4])\s*(\d{4})|(\d{4})\s*q([1-4]))$", re.I)
_MONTH_YEAR_RE = re.compile(r"^([a-z]{3,9})\.?\s+(\d{4})$", re.I)
_RELATIVE_RE = re.compile(
    r"^(?:last|past|previous|trailing|prev)?\s*(\d+)?\s*"
    r"(d|w|m|q|y|day|days|week|weeks|month|months|quarter|quarters|year|years)$", re.I)


def _parse_day(token: str) -> pd.Timestamp | None:
    token = token.strip()
    m = _ISO_DAY_RE.match(token)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        return to_utc(datetime(y, mo, d))
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d %b %Y", "%b %d %Y", "%d %B %Y"):
        try:
            return to_utc(datetime.strptime(token, fmt))
        except ValueError:
            continue
    return None


def _month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = to_utc(datetime(year, month, 1))
    end = to_utc(datetime(year + (month == 12), (month % 12) + 1, 1))
    return start, end


def _parse_boundary(token: str, *, end_of: bool) -> pd.Timestamp | None:
    """Parse one side of a range. `end_of` makes a partial date exclusive-end:
    "2026-03" as an end means the end of March, not its first instant.
    """
    token = token.strip()
    day = _parse_day(token)
    if day is not None:
        return day + timedelta(days=1) if end_of else day
    m = _ISO_MONTH_RE.match(token)
    if m:
        start, end = _month_bounds(int(m.group(1)), int(m.group(2)))
        return end if end_of else start
    m = _MONTH_YEAR_RE.match(token)
    if m and m.group(1).lower() in _MONTH_NAMES:
        start, end = _month_bounds(int(m.group(2)), _MONTH_NAMES[m.group(1).lower()])
        return end if end_of else start
    m = _YEAR_RE.match(token)
    if m:
        year = int(m.group(1))
        return to_utc(datetime(year + 1, 1, 1)) if end_of else to_utc(datetime(year, 1, 1))
    return None


def resolve_period(spec: str | None, anchor: pd.Timestamp,
                   *, default: str = DEFAULT_PERIOD) -> Period:
    """Turn whatever the caller wrote into a concrete window.

    The grammar is deliberately forgiving, because the agent is relaying a
    person's phrasing rather than filling in a form. All of these work:

        None, "", "default"      -> the default window (last month)
        "all", "all time"        -> no bounds at all
        "last month"             -> trailing 30 days from the anchor
        "last 6 weeks", "90d"    -> any <number><unit>, with or without "last"
        "this month", "mtd"      -> calendar month containing the anchor, to date
        "this quarter", "qtd"    -> calendar quarter containing the anchor
        "this year", "ytd"       -> calendar year containing the anchor
        "previous month"         -> the whole calendar month before the anchor's
        "2026-06", "June 2026"   -> that calendar month
        "2026", "Q2 2026"        -> that calendar year / quarter
        "2026-01-01..2026-03-31" -> explicit range (also "to", ":", "--")
        "since 2026-01-01"       -> open-ended forwards
        "before 2026-01-01"      -> open-ended backwards

    Rolling windows run back from `anchor`, which is the newest timestamp in the
    file rather than today. Calendar windows ("this month") are also taken
    relative to the anchor, so asking an export that ends in August about "this
    month" returns August and not an empty current month.

    Raises ToolError with the accepted forms if the spec cannot be parsed —
    never silently falls back, because silently returning the wrong window is
    the one failure mode an agent cannot detect.
    """
    raw = (spec or "").strip()
    if not raw or raw.lower() in {"default", "auto"}:
        raw = default
    text = re.sub(r"\s+", " ", raw.lower()).strip(" .")
    notes: list[str] = []

    def build(start, end, label, extra: list[str] | None = None) -> Period:
        return Period(start=start, end=end, label=label, spec=raw, anchor=anchor,
                      notes=notes + (extra or []))

    if text in _ALL_TIME_WORDS:
        return build(None, None, "all time")

    # --- open-ended -------------------------------------------------------
    m = re.match(r"^(?:since|after|from)\s+(.+)$", text)
    if m:
        start = _parse_boundary(m.group(1), end_of=False)
        if start is None:
            raise ToolError(f"Could not read a date out of '{m.group(1)}'.",
                            "use YYYY-MM-DD, YYYY-MM, or a month name with a year")
        return build(start, None, f"since {fmt_date(start)}")

    # "before X" excludes X; "until / up to / through X" includes it. The
    # distinction is the difference between a whole year and a year plus a day,
    # and an agent relaying a person's phrasing will use both.
    m = re.match(r"^(before|until|up to|through)\s+(.+)$", text)
    if m:
        inclusive = m.group(1) != "before"
        end = _parse_boundary(m.group(2), end_of=inclusive)
        if end is None:
            raise ToolError(f"Could not read a date out of '{m.group(2)}'.",
                            "use YYYY-MM-DD, YYYY-MM, or a month name with a year")
        shown = end if not inclusive else end - timedelta(days=1)
        return build(None, end,
                     f"up to and including {fmt_date(shown)}" if inclusive
                     else f"before {fmt_date(shown)}")

    # --- explicit range ---------------------------------------------------
    parts = [p for p in _RANGE_SPLIT_RE.split(text) if p]
    if len(parts) == 2:
        start = _parse_boundary(parts[0], end_of=False)
        end = _parse_boundary(parts[1], end_of=True)
        if start is not None and end is not None:
            if end <= start:
                raise ToolError(
                    f"The range '{raw}' ends before it starts "
                    f"({fmt_date(start)} → {fmt_date(end)}).",
                    "put the earlier date first")
            return build(start, end, f"{fmt_date(start)} to {fmt_date(end - timedelta(days=1))}")

    # --- calendar-to-date -------------------------------------------------
    if text in {"today", "this day"}:
        start = anchor.normalize()
        return build(start, start + timedelta(days=1), "the anchor day")
    if text in {"this week", "wtd", "week to date"}:
        start = (anchor - timedelta(days=int(anchor.dayofweek))).normalize()
        return build(start, anchor.normalize() + timedelta(days=1), "this week to date")
    if text in {"this month", "mtd", "month to date", "current month"}:
        start, _ = _month_bounds(anchor.year, anchor.month)
        return build(start, anchor.normalize() + timedelta(days=1),
                     f"{anchor:%B %Y} to date")
    if text in {"this quarter", "qtd", "quarter to date", "current quarter"}:
        q = (anchor.month - 1) // 3
        start = to_utc(datetime(anchor.year, q * 3 + 1, 1))
        return build(start, anchor.normalize() + timedelta(days=1),
                     f"Q{q + 1} {anchor.year} to date")
    if text in {"this year", "ytd", "year to date", "current year"}:
        start = to_utc(datetime(anchor.year, 1, 1))
        return build(start, anchor.normalize() + timedelta(days=1), f"{anchor.year} to date")

    # --- previous whole calendar unit -------------------------------------
    if text in {"previous month", "last calendar month", "previous calendar month"}:
        first, _ = _month_bounds(anchor.year, anchor.month)
        prev_end = first
        prev_start = _month_bounds(prev_end.year - (prev_end.month == 1),
                                   12 if prev_end.month == 1 else prev_end.month - 1)[0]
        return build(prev_start, prev_end, f"{prev_start:%B %Y}")
    if text in {"previous year", "last calendar year", "previous calendar year"}:
        return build(to_utc(datetime(anchor.year - 1, 1, 1)),
                     to_utc(datetime(anchor.year, 1, 1)), str(anchor.year - 1))
    if text in {"previous quarter", "last calendar quarter"}:
        q = (anchor.month - 1) // 3
        this_start = to_utc(datetime(anchor.year, q * 3 + 1, 1))
        prev_q, prev_y = (q - 1, anchor.year) if q else (3, anchor.year - 1)
        return build(to_utc(datetime(prev_y, prev_q * 3 + 1, 1)), this_start,
                     f"Q{prev_q + 1} {prev_y}")

    # --- named calendar periods ------------------------------------------
    m = _QUARTER_RE.match(text.replace(" ", "") if "q" in text else text)
    if m:
        q = int(m.group(1) or m.group(4))
        year = int(m.group(2) or m.group(3))
        return build(to_utc(datetime(year, (q - 1) * 3 + 1, 1)),
                     to_utc(datetime(year + (q == 4), 1 if q == 4 else q * 3 + 1, 1)),
                     f"Q{q} {year}")
    for regex, extract in ((_ISO_MONTH_RE, lambda mm: (int(mm.group(1)), int(mm.group(2)))),
                           (_MONTH_YEAR_RE, lambda mm: (int(mm.group(2)),
                                                        _MONTH_NAMES.get(mm.group(1).lower(), 0)))):
        mm = regex.match(text)
        if mm:
            year, month = extract(mm)
            if 1 <= month <= 12:
                start, end = _month_bounds(year, month)
                return build(start, end, f"{start:%B %Y}")
    mm = _YEAR_RE.match(text)
    if mm:
        year = int(mm.group(1))
        return build(to_utc(datetime(year, 1, 1)), to_utc(datetime(year + 1, 1, 1)), str(year))

    # --- single explicit day ---------------------------------------------
    day = _parse_day(text)
    if day is not None:
        return build(day, day + timedelta(days=1), fmt_date(day))

    # --- rolling relative window -----------------------------------------
    m = _RELATIVE_RE.match(text.replace("-", " ").strip())
    if m:
        count = int(m.group(1) or 1)
        unit = m.group(2).lower()
        days = count * _RELATIVE_UNITS[unit]
        if days <= 0:
            raise ToolError(f"'{raw}' resolves to a window of zero days.",
                            "use a positive number, e.g. 'last 30 days'")
        end = anchor + timedelta(seconds=1)      # anchor row is inside its own window
        start = end - timedelta(days=days)
        if unit in {"d", "day", "days"}:
            label = f"last {days} days"
        else:
            noun = {"w": "week", "m": "month", "mo": "month",
                    "q": "quarter", "y": "year"}.get(unit, unit.rstrip("s"))
            label = (f"last {noun} ({days} days)" if count == 1
                     else f"last {count} {noun}s ({days} days)")
        return build(start, end, label)

    raise ToolError(
        f"Could not understand the period '{raw}'.",
        "use 'last 30 days', 'last 3 months', 'this month', 'previous month', "
        "'2026-06', 'June 2026', 'Q2 2026', '2026', '2026-01-01..2026-03-31', "
        "'since 2026-01-01', or 'all time'")


def period_coverage_note(period: Period, timestamps: pd.Series,
                         matched: int, total: int) -> list[str]:
    """Warn when the window and the data barely overlap.

    An agent asking about July on a file that stops in May gets zero rows; unless
    it is told the file stops in May it will report "no activity in July" as a
    business finding rather than a data problem.
    """
    notes: list[str] = []
    valid = timestamps.dropna()
    if valid.empty:
        return ["The file contains no parseable timestamps at all."]

    lo, hi = valid.min(), valid.max()
    if matched == 0 and not period.is_all_time:
        notes.append(
            f"**No rows fall in this time window at all.** The file covers {fmt_date(lo)} to "
            f"{fmt_date(hi)}; the requested window is {period.describe()}. This is a "
            f"coverage gap, not a business finding.")
    elif 0 < matched < total * 0.02 and not period.is_all_time:
        notes.append(
            f"The time window alone leaves {matched} of {total} rows, before any other "
            f"filter is applied (file covers {fmt_date(lo)} to {fmt_date(hi)}).")

    if period.anchor_is_file_max:
        stale = (pd.Timestamp.now(tz="UTC") - hi).days
        if stale > 45:
            notes.append(
                f"Windows are measured back from the newest record in the file "
                f"({fmt_date(hi)}), not from today — this export is {stale} days old, so "
                f"\"last month\" means the last month *of the data*.")
    return notes


# ===========================================================================
# 6. Fuzzy filters
# ===========================================================================

def normalize_key(value: Any) -> str:
    """Casefold and strip punctuation so 'Maria  Gonzalez' == 'maria gonzalez'."""
    text = re.sub(r"[^\w\s]", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def resolve_value(requested: str, candidates: Iterable[Any], *, field_name: str,
                  allow_multiple: bool = False) -> list[str]:
    """Match what the caller typed against the values actually in the column.

    Four passes, most precise first: exact, case/punctuation-insensitive,
    substring, then difflib similarity. This is what lets an agent pass "maria",
    "ORDER_ADDED", "order added" or "orders" and land on the right value without
    having had to list the column first.

    Raises ToolError naming the closest real values when nothing matches — an
    empty result set from a typo is otherwise indistinguishable from a genuine
    zero, and the agent will report the typo as a finding.
    """
    pool = [str(c) for c in candidates if str(c) not in NULLISH]
    uniq = sorted(set(pool))
    want = str(requested).strip()
    if not want:
        return []

    if want in uniq:
        return [want]

    norm_want = normalize_key(want)
    by_norm: dict[str, list[str]] = {}
    for value in uniq:
        by_norm.setdefault(normalize_key(value), []).append(value)
    if norm_want in by_norm:
        return by_norm[norm_want]

    # ORDER_ADDED / "order added" / "order-added" all normalise together.
    snake = norm_want.replace(" ", "_").upper()
    if snake in uniq:
        return [snake]

    partial = [v for v in uniq if norm_want and norm_want in normalize_key(v)]
    if partial:
        return partial if allow_multiple or len(partial) == 1 else sorted(partial)

    close = difflib.get_close_matches(norm_want, list(by_norm), n=3, cutoff=0.7)
    if close:
        hits = [v for key in close for v in by_norm[key]]
        if len(hits) == 1 or allow_multiple:
            return hits
        raise ToolError(
            f"'{requested}' is ambiguous for {field_name}.",
            f"did you mean one of: {', '.join(hits[:6])}?")

    sample = ", ".join(uniq[:12]) or "(the column is empty)"
    raise ToolError(
        f"No {field_name} matches '{requested}'.",
        f"values present in this export: {sample}"
        + (f" … and {len(uniq) - 12} more" if len(uniq) > 12 else ""))


def parse_bool(value: Any, *, default: bool = False) -> bool:
    """Accept the many ways an LLM writes a boolean, including 'true' as a string."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_STRINGS


def parse_int(value: Any, *, default: int, minimum: int = 1,
              maximum: int = 500, name: str = "limit") -> int:
    """Coerce a numeric argument, clamped, tolerating '25', 25.0 and None."""
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        raise ToolError(f"`{name}` must be a whole number, got '{value}'.",
                        f"pass an integer between {minimum} and {maximum}") from None
    return max(minimum, min(maximum, parsed))


def split_multi(value: Any) -> list[str]:
    """Split 'HIGH, MEDIUM' or ['HIGH','MEDIUM'] into a list of tokens."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in re.split(r"[,;|]", str(value)) if part.strip()]


# ===========================================================================
# 7. Domain enrichment
# ===========================================================================

def humanize(value: Any) -> str:
    """TASK_ADDED -> 'Task added'. The reader never sees a database enum.

    Values that are already human (a person's name) pass through untouched, so
    this is safe to apply to any column.
    """
    if value is None or pd.isna(value):
        return "—"
    raw = str(value).strip()
    if raw in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[raw]
    if not re.fullmatch(r"[A-Z0-9_]+", raw):
        return raw
    words = [w for w in raw.split("_") if w]
    if not words:
        return raw
    return " ".join([LABEL_OVERRIDES.get(words[0], words[0].capitalize())]
                    + [LABEL_OVERRIDES.get(w, w.lower()) for w in words[1:]])


# --- Activities -----------------------------------------------------------

def categorize_activity(activity_type: Any) -> str:
    """Map an activity type to a coarse category, including types never seen."""
    key = str(activity_type or "").strip().upper()
    if key in ACTIVITY_CATEGORIES:
        return ACTIVITY_CATEGORIES[key]
    for keywords, category in ACTIVITY_CATEGORY_RULES:
        if any(k in key for k in keywords):
            return category
    return "other"


def load_activities(user_id: str) -> tuple[pd.DataFrame, list[str]]:
    """Load and enrich the activity log. Returns (frame, data-quality notes).

    Derived columns: `_ts`, `_type`, `_type_label`, `_category`, `_actor`,
    `_actor_kind`, `_channel`, `_customer`, `_hour`, `_weekday`, `_is_weekend`.

    `_actor` resolution matters and is easy to get wrong. These exports carry a
    named representative on only a small minority of rows; everything else
    records the *channel* that produced the event (DISTRIBUTOR back office,
    QUICKBOOKS sync). Naming the channel as if it were a person is the single
    most misleading thing this dataset invites, so `_actor_kind` marks which one
    each row is and the tools surface the split.
    """
    path = resolve_path(user_id, "activities")
    df = read_csv_cached(path)
    require_columns(df, ["type", "createdAt"], "activities")
    notes: list[str] = []

    df["_ts"] = parse_timestamps(df["createdAt"])
    unparsed = int(df["_ts"].isna().sum())
    if unparsed:
        notes.append(f"{unparsed} row(s) had an unreadable createdAt and are excluded "
                     f"from every dated figure.")
        df = df[df["_ts"].notna()].copy()
    if df.empty:
        raise ToolError("No activity rows survived timestamp parsing.",
                        "check the createdAt column format in the export")

    df["_type"] = df["type"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["_type_label"] = df["_type"].map(humanize)
    df["_category"] = df["_type"].map(categorize_activity)
    df["_category_label"] = df["_category"].map(ACTIVITY_CATEGORY_LABELS)

    named = (df["representativeDuplicate_name"].astype("string")
             if "representativeDuplicate_name" in df.columns
             else pd.Series(pd.NA, index=df.index, dtype="string"))
    channel = (df["createdBy"].astype("string") if "createdBy" in df.columns
               else pd.Series(pd.NA, index=df.index, dtype="string"))

    df["_actor"] = named.fillna(channel).fillna("Unattributed").astype(str)
    df["_actor_kind"] = pd.Series(
        ["person" if pd.notna(n) else ("channel" if pd.notna(c) else "none")
         for n, c in zip(named, channel)], index=df.index, dtype="string")
    df["_actor_label"] = [humanize(a) for a in df["_actor"]]
    df["_channel"] = channel.fillna("unknown").astype(str)
    df["_customer"] = (df["appCustomer_name"].astype("string")
                       if "appCustomer_name" in df.columns else pd.NA)

    person_rows = int((df["_actor_kind"] == "person").sum())
    if person_rows < len(df):
        notes.append(
            f"Only {person_rows} of {len(df)} events name an individual "
            f"({person_rows / len(df) * 100:.0f}%); the rest record the channel that "
            f"produced the event (back office, integrations). Per-person activity cannot "
            f"be read from this file beyond those named rows — use the orders export for "
            f"salesperson attribution.")

    dt = df["_ts"].dt
    df["_hour"] = dt.hour.astype("int16")
    df["_weekday"] = dt.day_name().astype(str)
    df["_is_weekend"] = dt.dayofweek.ge(5)
    df["_date"] = dt.normalize()

    bulk = df["_ts"].value_counts()
    bulk_rows = int(bulk[bulk >= BULK_SECOND_MIN].sum())
    if bulk_rows:
        notes.append(
            f"{bulk_rows} row(s) ({bulk_rows / len(df) * 100:.1f}%) share an exact creation "
            f"second with {BULK_SECOND_MIN}+ others — those were imported in bulk, not "
            f"performed, and inflate whatever day they land on.")

    return df.sort_values("_ts").reset_index(drop=True), notes


# --- Notes ----------------------------------------------------------------

def is_low_information(text: str) -> bool:
    """True for empty strings, boilerplate and keyboard mashing.

    These exports are roughly half junk — "test note", "rtrtrtr", and one note
    that is the string "kkgffghjghgfghj" repeated for 1,300 characters. Reading
    those as customer signal is worse than reading nothing.
    """
    t = (text or "").strip()
    if len(t) < NOTE_MIN_CHARS:
        return True
    lower = t.lower()
    if "lorem ipsum" in lower:
        return True
    if any(re.match(p, lower) for p in NOTE_FILLER_PATTERNS):
        return True
    if re.search(r"(.{1,4})\1{" + str(NOTE_MAX_REPEAT_RUN) + r",}", lower):
        return True
    letters = [c for c in lower if c.isalpha()]
    if len(letters) > 15 and sum(c in "aeiou" for c in letters) / len(letters) < NOTE_MIN_VOWEL_RATIO:
        return True
    return len(lower) > 30 and len(set(lower)) / len(lower) < NOTE_MIN_UNIQUE_RATIO


def note_importance(text: str) -> int:
    """Rank surviving notes so the highest-signal ones surface first."""
    lower = (text or "").lower()
    score = sum(w for kw, w in NOTE_ACTION_KEYWORDS.items() if kw in lower)
    if MONEY_RE.search(text or ""):
        score += 3
    if DATE_MENTION_RE.search(lower):
        score += 1
    return score + min(len(text or "") // 80, 3)   # reward substance, capped


def note_themes(text: str) -> list[str]:
    lower = (text or "").lower()
    return [theme for theme, terms in NOTE_THEMES.items() if any(t in lower for t in terms)]


def load_notes(user_id: str) -> tuple[pd.DataFrame, list[str]]:
    """Load and enrich CRM notes. Returns (frame, data-quality notes).

    Derived columns: `_ts`, `_text`, `_author`, `_author_kind`, `_is_noise`,
    `_score`, `_themes`, `_has_money`, `_len`, `_dupe_of`.

    Junk is *flagged*, never dropped, so the counts stay reconcilable with the
    source file and a caller who wants the raw picture can still get it.
    """
    path = resolve_path(user_id, "notes")
    df = read_csv_cached(path)
    require_columns(df, ["text", "createdAt"], "notes")
    notes: list[str] = []

    df["_ts"] = parse_timestamps(df["createdAt"])
    unparsed = int(df["_ts"].isna().sum())
    if unparsed:
        notes.append(f"{unparsed} note(s) had an unreadable createdAt and are excluded "
                     f"from dated figures.")

    df["_text"] = df["text"].fillna("").astype(str).str.strip()
    rep = (df["representativeDuplicate_name"].astype("string")
           if "representativeDuplicate_name" in df.columns
           else pd.Series(pd.NA, index=df.index, dtype="string"))
    dist = (df["distributor_name"].astype("string")
            if "distributor_name" in df.columns
            else pd.Series(pd.NA, index=df.index, dtype="string"))
    df["_author"] = rep.fillna(dist).fillna("Unattributed").astype(str)
    df["_author_kind"] = ["rep" if pd.notna(r) else ("distributor" if pd.notna(d) else "none")
                          for r, d in zip(rep, dist)]

    df["_is_noise"] = [is_low_information(t) for t in df["_text"]]
    df["_score"] = [0 if noise else note_importance(t)
                    for t, noise in zip(df["_text"], df["_is_noise"])]
    df["_themes"] = [note_themes(t) if not n else []
                     for t, n in zip(df["_text"], df["_is_noise"])]
    df["_has_money"] = [bool(MONEY_RE.search(t)) for t in df["_text"]]
    df["_len"] = df["_text"].str.len()

    # Exact-duplicate detection on normalised text: the same note pasted twice
    # is one observation, not two, and inflates any per-account count.
    seen: dict[str, Any] = {}
    dupe_of: list[Any] = []
    for idx, text in zip(df.index, df["_text"]):
        key = re.sub(r"\s+", " ", text.lower())
        if not key or df.at[idx, "_is_noise"]:
            dupe_of.append(pd.NA)
            continue
        dupe_of.append(seen.get(key, pd.NA))
        seen.setdefault(key, idx)
    df["_dupe_of"] = dupe_of

    noise_n = int(df["_is_noise"].sum())
    dupe_n = int(df["_dupe_of"].notna().sum())
    if noise_n or dupe_n:
        notes.append(
            f"{noise_n} of {len(df)} notes are test data or keyboard mashing and "
            f"{dupe_n} are exact duplicates of an earlier note; both are excluded by "
            f"default (pass include_noise=true to see them).")
    return df, notes


# --- Tasks ----------------------------------------------------------------

def categorize_task(title: str, description: str) -> str:
    text = f"{title or ''} {description or ''}"
    scores = {c: sum(1 for p in ps if p.search(text)) for c, ps in _TASK_CATEGORY_PATTERNS.items()}
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] else "Uncategorized"


def extract_account(text: str) -> str | None:
    """Pull a store/account name out of a task title or description.

    Handles "Fresh Mart - wrong order", "Fix issue at Northside Pharmacy",
    "Collect payment from Main St Convenience" and a bare "Green Valley Market".
    Heuristic by nature: it is a rollup aid, not an authoritative key, and the
    tools label it as such wherever it is shown.
    """
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return None

    def generic(candidate: str) -> bool:
        toks = [t for t in re.split(r"\W+", candidate.lower()) if t]
        return not toks or all(t in _GENERIC_TOKENS for t in toks)

    def canonical(name: str) -> str:
        return re.sub(r"\s+", " ", name).strip(" .,-").title()

    for part in (p.strip() for p in re.split(r"\s*[-–—|:]\s*", s)):
        if (1 <= len(part.split()) <= 4 and _ACCOUNT_SEGMENT_RE.match(part)
                and not generic(part)):
            if len(part.split()) > 1 or (part[0].isupper() and len(part) > 3):
                return canonical(part)
    m = _ACCOUNT_AFTER_PREP_RE.search(s)
    if m and not generic(m.group(1)):
        return canonical(m.group(1))
    if _ACCOUNT_STANDALONE_RE.match(s) and not generic(s):
        return canonical(s)
    return None


def task_quality_flags(title: str, description: str) -> list[str]:
    """Grade the (title, description) PAIR, not the title alone.

    The question is whether someone who did not write the task could pick it up
    and act on it. A vague title with a good description is fine; a vague title
    with no description is a reminder that something exists, and
    `unactionable_pair` is the flag that says so.
    """
    t = re.sub(r"\s+", " ", str(title or "")).strip()
    d = re.sub(r"\s+", " ", str(description or "")).strip()
    nt, nd = normalize_key(t), normalize_key(d)
    flags: list[str] = []

    if not t:
        flags.append("empty_title")
    if nt in TASK_LOW_INFO_TITLES or len(nt) <= 3:
        flags.append("vague_title")
    if TASK_PLACEHOLDER_RE.search(f"{t} {d}") or TASK_KEYMASH_RE.match(nt):
        flags.append("placeholder_text")
    if not d:
        flags.append("no_description")
    elif nd == nt or nd in nt:
        flags.append("description_echoes_title")
    if t and t.isupper() and len(t) > 12:
        flags.append("all_caps")
    if ("vague_title" in flags or "empty_title" in flags) and not d:
        flags.append("unactionable_pair")
    return flags


def load_tasks(user_id: str, *, as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load and enrich the task backlog. Returns (frame, data-quality notes).

    Derived columns: `_ts` (created), `_due`, `_title`, `_desc`, `_status`,
    `_priority`, `_owner`, `_owner_kind`, `_is_open`, `_overdue_days`,
    `_age_days`, `_category`, `_account`, `_flags`, `_dupe_key`.

    Completed tasks are loaded, never filtered out: they are the denominator of
    every completion rate, and a backlog view that hides them can only report
    how much work exists, not how much gets done.

    Overdue is measured against `as_of`, which defaults to the newest createdAt
    in the file. On a stale export that keeps "overdue" meaningful instead of
    marking the entire backlog late by however long the export has been sitting.
    """
    path = resolve_path(user_id, "tasks")
    df = read_csv_cached(path)
    require_columns(df, ["title", "status"], "tasks")
    notes: list[str] = []

    df["_ts"] = (parse_timestamps(df["createdAt"]) if "createdAt" in df.columns
                 else pd.Series(pd.NaT, index=df.index))
    df["_due"] = (parse_timestamps(df["dueDate"]) if "dueDate" in df.columns
                  else pd.Series(pd.NaT, index=df.index))

    ref = as_of or (df["_ts"].max() if df["_ts"].notna().any() else pd.Timestamp.now(tz="UTC"))

    df["_title"] = df["title"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["_desc"] = (df["description"].fillna("").astype(str)
                   .str.replace(r"\s+", " ", regex=True).str.strip()
                   if "description" in df.columns else "")
    df["_status"] = df["status"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["_priority"] = (df["priority"].fillna("UNSET").astype(str).str.strip().str.upper()
                       if "priority" in df.columns else "UNSET")

    rep = (df["representative_name"].astype("string") if "representative_name" in df.columns
           else pd.Series(pd.NA, index=df.index, dtype="string"))
    dist = (df["assignedDistributor_name"].astype("string")
            if "assignedDistributor_name" in df.columns
            else pd.Series(pd.NA, index=df.index, dtype="string"))
    df["_owner"] = rep.fillna(dist).fillna("(unassigned)").astype(str)
    df["_owner_kind"] = ["rep" if pd.notna(r) else ("distributor" if pd.notna(d) else "none")
                         for r, d in zip(rep, dist)]

    df["_is_open"] = ~df["_status"].isin(TASK_CLOSED_STATUSES)
    # Nullable Int64, not plain int: a task with no due date has *no* overdue
    # value, and a plain list would make pandas coerce the column to float and
    # turn "not overdue" into NaN — which compares and formats as a number.
    # Callers must still guard with pd.isna before int().
    df["_overdue_days"] = pd.array(
        [int((ref - due).days) if (op and pd.notna(due) and due < ref) else None
         for op, due in zip(df["_is_open"], df["_due"])], dtype="Int64")
    df["_age_days"] = pd.array(
        [int((ref - ts).days) if pd.notna(ts) else None for ts in df["_ts"]], dtype="Int64")

    df["_category"] = [categorize_task(t, d) for t, d in zip(df["_title"], df["_desc"])]
    df["_account"] = [extract_account(t) or extract_account(d)
                      for t, d in zip(df["_title"], df["_desc"])]
    df["_flags"] = [task_quality_flags(t, d) for t, d in zip(df["_title"], df["_desc"])]
    df["_dupe_key"] = [f"{normalize_key(t)}|{normalize_key(d)[:80]}"
                       for t, d in zip(df["_title"], df["_desc"])]

    dupe_groups = df[df["_title"].ne("")].groupby("_dupe_key").size()
    extra = int((dupe_groups[dupe_groups > 1] - 1).sum())
    if extra:
        notes.append(f"{extra} task(s) repeat the title *and* description of another task "
                     f"({int((dupe_groups > 1).sum())} duplicate groups) — most are the "
                     f"recurring-task generator, but they double-count in any per-owner total.")

    unowned = int((df["_owner_kind"] == "none").sum())
    if unowned:
        notes.append(f"{unowned} of {len(df)} tasks name neither a representative nor a "
                     f"distributor and are grouped as '(unassigned)'.")
    if "dueDate" in df.columns:
        no_due = int(df["_due"].isna().sum())
        if no_due:
            notes.append(f"{no_due} task(s) have no usable due date and can never be "
                         f"counted as overdue.")
    notes.append(f"Overdue is measured against {fmt_date(ref)}, the newest createdAt in "
                 f"the file, not today.")
    return df, notes


def load_orders(user_id: str, *, exclude_third_party: bool = True
                ) -> tuple[pd.DataFrame, list[str]]:
    """Load the order book. Returns (frame, data-quality notes).

    Derived columns: `_ts`, `_amount`, `_salesperson`, `_qty`, `_status`,
    `_payment_status`, `_customer`, `_is_third_party`.

    This is the only complete salesperson attribution in the export set — the
    activity log leaves the representative blank on order-creation rows — which
    is why the activity statistics tool reaches into it for the revenue table.
    """
    path = resolve_path(user_id, "orders")
    df = read_csv_cached(path)
    require_columns(df, ["createdAt"], "orders")
    notes: list[str] = []

    df["_ts"] = parse_timestamps(df["createdAt"])
    df = df[df["_ts"].notna()].copy()
    df["_amount"] = pd.to_numeric(df.get(ORDER_AMOUNT_FIELD), errors="coerce").fillna(0.0)
    df["_qty"] = pd.to_numeric(df.get("totalQuantity"), errors="coerce").fillna(0)

    df["_is_third_party"] = (df["type"].astype("string").isin(ORDER_TYPES_EXCLUDED)
                             if "type" in df.columns else False)
    if exclude_third_party and bool(pd.Series(df["_is_third_party"]).any()):
        n = int(df["_is_third_party"].sum())
        revenue = float(df.loc[df["_is_third_party"], "_amount"].sum())
        notes.append(f"{n} third-party order(s) worth {money(revenue)} are excluded — they "
                     f"were placed through an outside channel rather than by the team.")
        df = df[~df["_is_third_party"]].copy()

    sales = (df[ORDER_SALESPERSON_FIELD].astype("string")
             if ORDER_SALESPERSON_FIELD in df.columns
             else pd.Series(pd.NA, index=df.index, dtype="string"))
    df["_salesperson"] = sales.fillna(UNASSIGNED_SALESPERSON).astype(str)
    df["_status"] = df.get("orderStatus", pd.Series("UNKNOWN", index=df.index)).astype(str)
    df["_payment_status"] = df.get("paymentStatus",
                                   pd.Series("UNKNOWN", index=df.index)).astype(str)
    df["_customer"] = df.get("customer_name", pd.Series(pd.NA, index=df.index)).astype("string")

    unassigned = int((df["_salesperson"] == UNASSIGNED_SALESPERSON).sum())
    if unassigned:
        notes.append(f"{unassigned} of {len(df)} orders carry no salesperson and are grouped "
                     f"as '{UNASSIGNED_SALESPERSON}'.")
    return df.sort_values("_ts").reset_index(drop=True), notes


# ===========================================================================
# 8. Markdown
# ===========================================================================

def money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def money_short(value: Any) -> str:
    """Compact currency for table cells: $1.2M, $45.3K, $812."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    for cutoff, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= cutoff:
            return f"${v / cutoff:,.1f}{suffix}"
    return f"${v:,.0f}"


def pct(numerator: Any, denominator: Any, *, digits: int = 1) -> str:
    try:
        d = float(denominator)
        if d == 0:
            return "—"
        return f"{float(numerator) / d * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def change(current: float | int | None, previous: float | int | None,
           *, sample: int | None = None) -> str:
    """Period-over-period change with an honest basis rather than a fake 100%.

    A move from zero is not a percentage, and a percentage computed on a handful
    of records is arithmetic rather than evidence. Both cases are labelled: an
    agent shown "+340%" with no qualifier will write it into a summary as though
    it means something.
    """
    cur = float(current or 0)
    prev = float(previous or 0)
    if prev == 0 and cur == 0:
        return "no activity either period"
    if prev == 0:
        return f"up from zero (0 → {cur:,.0f})"
    if cur == 0:
        return "dropped to zero (-100%)"
    delta = (cur - prev) / abs(prev) * 100
    flag = " *(few records — directional only)*" if (sample or min(cur, prev)) < MIN_SAMPLE else ""
    return f"{delta:+.1f}%{flag}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
             *, align: Sequence[str] | None = None, empty: str = "*No rows.*") -> str:
    """Render a Markdown table, escaping pipes so free text cannot break it."""
    if not rows:
        return empty

    def cell(value: Any) -> str:
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    align = align or ["---"] * len(headers)
    out = ["| " + " | ".join(cell(h) for h in headers) + " |",
           "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def truncate(text: Any, width: int = 60) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= width else s[: width - 1].rstrip() + "…"


def header_block(title: str, period: Period, *, matched: int, total: int,
                 unit: str = "records", filters: dict[str, Any] | None = None,
                 caveats: Sequence[str] = ()) -> str:
    """The standard preamble every tool emits.

    It answers, before any number appears, the four things an agent has to know
    to use the number correctly: what window this is, how many records it covers
    out of how many exist, which filters were applied, and what is wrong with
    the data. Reports built on tool output are only as honest as this block.
    """
    lines = [f"## {title}", ""]
    scope = (f"**Period:** {period.label}" if period.is_all_time
             else f"**Period:** {period.label} — {period.describe()}")
    if period.spec and normalize_key(period.spec) != normalize_key(period.label):
        scope += f"  *(from `{period.spec}`)*"
    lines.append(scope)
    lines.append(f"**Coverage:** {matched:,} of {total:,} {unit} "
                 f"({pct(matched, total)} of the file)")

    active = {k: v for k, v in (filters or {}).items()
              if v not in (None, "", [], False) and str(v).lower() != "none"}
    if active:
        lines.append("**Filters:** " + ", ".join(
            f"{k}={', '.join(map(str, v)) if isinstance(v, list) else v}"
            for k, v in active.items()))

    real = [c for c in caveats if c]
    if real:
        lines += ["", "> **Read this first**"] + [f"> - {c}" for c in real]
    return "\n".join(lines)


def bullet_list(items: Iterable[str], *, empty: str = "") -> str:
    rendered = [f"- {i}" for i in items if i]
    return "\n".join(rendered) if rendered else empty


def describe_distribution(values: Sequence[float], unit: str = "",
                          *, currency: bool = False) -> str:
    """Mean *and* median together, with the gap called out when it is large.

    Order values in this data run from $0 to $450,000; a mean on its own is a
    number nothing in the file resembles, and an agent handed only a mean will
    describe a typical order as one that has never been placed.
    """
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return "—"
    mean, median = statistics.fmean(vals), statistics.median(vals)
    show = (lambda v: money(v)) if currency else (lambda v: f"{v:,.1f}{unit}")
    out = f"mean {show(mean)}, median {show(median)}"
    if median > 0 and mean / median > 2:
        out += (f" (the mean is {mean / median:.1f}× the median, so a few large values "
                f"dominate — plan against the median)")
    return out


def counter_table(counter: Counter, headers: tuple[str, str, str], total: int,
                  limit: int = 10) -> str:
    rows = [[humanize(k), f"{v:,}", pct(v, total)] for k, v in counter.most_common(limit)]
    return md_table(list(headers), rows, align=["---", "---:", "---:"])


def sample_rows_note(n: int, threshold: int = MIN_SAMPLE_PER_PERSON) -> str:
    return (f"Rows covering fewer than {threshold} records are marked `*` — their rates are "
            f"arithmetic, not performance." if n else "")