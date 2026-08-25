"""Deterministic layer for the activity report.

    analyze_activities_file(csv, orders_csv) -> {
        "metrics":                 dict,        # -> Statistics Analyst
        "representative_profiles": list[dict],  # -> Situation Writer
        "display_tables":          dict,        # title + intro + markdown, code-rendered
        "analytics_errors":        list[dict],  # a broken section never kills the run
    }

Design rules:
  1. Schema-tolerant: nothing hardcodes a name or an activity-type list. Types
     are discovered from the data and categorised by explicit map first, keyword
     rule second, so a new INVOICE_VOIDED lands in risk_signal, not "other".
  2. One pass, vectorised: sections read precomputed boolean window columns and a
     single groupby/crosstab. Cost is O(rows), not O(rows x types x people).
  3. Raw enums stay in the metrics as stable keys; every one carries a human
     `label` for the tables and prompts.
  4. Numbers stay numbers in `metrics`; formatting happens once, in render_tables.
  5. pct_change returns None with a `basis` when there is no baseline, so the
     report can say "up from none" instead of inventing a percentage.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..core.markdown import md_table, money

# ---------------------------------------------------------------------------
# Configuration -- tune here, not in the body.
# ---------------------------------------------------------------------------

TYPE_FIELD = "type"
CREATED_FIELD = "createdAt"
UPDATED_FIELD = "updatedAt"

# First non-empty column wins; the source column is reported alongside the
# value, so a channel bucket is never presented as a named person.
REPRESENTATIVE_PRIORITY = ("representativeDuplicate_name", "createdBy")
CHANNEL_FIELD = "createdBy"

# Orders file (optional): the only complete salesperson attribution available.
ORDER_SALESPERSON_FIELD = "salesDuplicate_name"
ORDER_AMOUNT_FIELD = "totalAmount"
UNASSIGNED_SALESPERSON = "Unassigned / Direct"
# Third-party orders come through an outside channel, so counting them would
# overstate what the team itself sold.
ORDER_TYPE_FIELD = "type"
ORDER_TYPES_EXCLUDED = ("THIRD_PARTY",)

MIN_SAMPLE = 30           # events per side below which a % change is noise
BULK_SECOND_MIN = 5       # rows sharing one exact second => import artifact
MASS_UPDATE_SHARE = 0.5   # share of updatedAt on one date => migration stamp
DORMANT_DAYS = (30, 90, 180)          # active / cooling / dormant / churned
OUTLIER_Z = 3.5           # Iglewicz-Hoaglin modified z on monthly counts
AUTOMATION_SHARE = 0.75   # one activity owns >75% of an hour...
AUTOMATION_LIFT = 2.0     # ...and >2x its own overall share => not organic
AUTOMATION_MIN_EVENTS = 10

CAP_REPRESENTATIVES = 25  # in metrics (full set stays in the profiles list)
CAP_TYPES = 20
CAP_MONTHS = 24
CAP_EXAMPLES = 3
SALESPERSON_DISPLAY_LIMIT = 20   # rows rendered; the JSON keeps everyone

ACTIVITY_CATEGORIES = {
    "ORDER_ADDED": "commercial",
    "ORDER_CANCELED": "risk_signal",
    "CREDIT_MEMO_ADDED": "risk_signal",
    "CREDIT_MEMO_VOIDED": "risk_signal",
    "CREDIT_MEMO_DELETED": "risk_signal",
    "CUSTOMER_MERGED": "administrative",
    "NOTE_ADDED": "customer_engagement",
    "COMMENT_ADDED": "customer_engagement",
    "TASK_ADDED": "customer_engagement",
    "TASK_COMPLETED": "customer_engagement",
    "CHECKED_IN": "customer_engagement",
    "PHOTO_GROUP_ADDED": "customer_engagement",
}

# First matching group wins, so CREDIT_MEMO_* and *_CANCELED are risk signals
# before the ORDER/INVOICE rule can claim them.
CATEGORY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("CANCEL", "VOID", "DELET", "REFUND", "RETURN", "CREDIT_MEMO", "DISPUTE", "FAIL"), "risk_signal"),
    (("MERGE", "IMPORT", "SYNC", "MIGRAT", "ARCHIV", "SETTING", "PERMISSION"), "administrative"),
    (("ORDER", "INVOICE", "PAYMENT", "QUOTE", "CART"), "commercial"),
    (("TASK", "NOTE", "COMMENT", "CHECK", "PHOTO", "VISIT", "CALL", "EMAIL", "MESSAGE", "MEETING"), "customer_engagement"),
)

CATEGORY_LABELS = {
    "commercial": "Commercial",
    "customer_engagement": "Engagement",
    "risk_signal": "Risk",
    "administrative": "Admin",
    "other": "Other",
}
CATEGORY_DESCRIPTIONS = {
    "commercial": "Order creation",
    "customer_engagement": "Notes, tasks, comments, check-ins",
    "risk_signal": "Cancellations, credit memos, voids",
    "administrative": "Record keeping and merges",
    "other": "Uncategorised",
}

# Words the generic title-caser gets wrong; anything else is derived from the enum.
LABEL_OVERRIDES = {
    "QUICKBOOKS": "QuickBooks",
    "DISTRIBUTOR": "Distributor (back office)",
    "REPRESENTATIVE": "Sales representative",
    "SHOPIFY": "Shopify",
    "API": "API",
    "SMS": "SMS",
    "PDF": "PDF",
}

# (opening type, closing type, label, relation). Cohort ratios, not per-item
# matching: the log carries no parent id. `relation` keeps the wording honest --
# only a completion rate implies a backlog.
WORKFLOW_PAIRS = (
    ("TASK_ADDED", "TASK_COMPLETED", "tasks", "completion"),
    ("ORDER_ADDED", "ORDER_CANCELED", "orders", "cancellation"),
    ("CREDIT_MEMO_ADDED", "CREDIT_MEMO_VOIDED", "credit memos", "reversal"),
)

# name -> (days_back_start, days_back_end) measured from the reference date
WINDOWS: dict[str, tuple[int, int | None]] = {
    "last_30d": (30, None),
    "prev_30d": (60, 30),
    "last_90d": (90, None),
    "prev_90d": (180, 90),
    "last_182d": (182, None),
    "last_365d": (365, None),
    "prev_365d": (730, 365),
}

# Node/JS Date.toString(). Slicing to the offset keeps pandas on its C parser,
# roughly two orders of magnitude faster; misses are retried flexibly.
_JS_DATE_FORMAT = "%a %b %d %Y %H:%M:%S GMT%z"
_JS_DATE_LEN = 33
_PAREN_RE = re.compile(r"\(.*\)")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class ErrorLog:
    """Collects section failures so one bad section costs a section, not the run."""

    def __init__(self) -> None:
        self.errors: list[dict] = []

    @contextmanager
    def guard(self, section: str):
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            self.errors.append({"section": section, "error": f"{type(exc).__name__}: {exc}"})


def humanize(value: str | None) -> str:
    """TASK_ADDED -> 'Task Added'. The reader never sees a database enum."""
    if value is None:
        return ""
    raw = str(value).strip()
    if raw in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[raw]
    if not re.fullmatch(r"[A-Z0-9_]+", raw):
        return raw  # already human, e.g. a person's name
    return " ".join(LABEL_OVERRIDES.get(w, w.capitalize()) for w in raw.split("_") if w)


def pct_change(current: float | None, previous: float | None) -> dict:
    """Percentage change with an honest `basis` instead of a fake 100%.

    basis is one of: ok | no_baseline (something from nothing) |
    no_activity (nothing from nothing) | dropped_to_zero.
    """
    cur, prev = float(current or 0), float(previous or 0)
    if prev == 0 and cur == 0:
        return {"pct": None, "basis": "no_activity"}
    if prev == 0:
        return {"pct": None, "basis": "no_baseline"}
    if cur == 0:
        return {"pct": -100.0, "basis": "dropped_to_zero"}
    return {"pct": round((cur - prev) / abs(prev) * 100.0, 1), "basis": "ok"}


def reliability(*counts: float) -> str:
    """Rule of thumb: a period-over-period % needs ~30 events on the thinner side
    before it means anything. Surfaced so the writer hedges correctly."""
    n = min([float(c or 0) for c in counts] or [0])
    if n == 0:
        return "no_data"
    return "low_sample_low_confidence" if n < MIN_SAMPLE else "adequate"


def _entropy(counts: Iterable[float]) -> float:
    """Normalised Shannon entropy, 0 (does one thing) .. 1 (evenly spread)."""
    vals = np.asarray([c for c in counts if c > 0], dtype=float)
    if vals.size <= 1:
        return 0.0
    p = vals / vals.sum()
    return round(float(-(p * np.log(p)).sum() / math.log(vals.size)), 3)


def _modified_z(values: np.ndarray) -> np.ndarray:
    """Iglewicz-Hoaglin robust z-score (median/MAD). Immune to the very spikes it
    is meant to find, unlike a mean/std z-score."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        scale = np.mean(np.abs(values - med)) * 1.253314
        return np.zeros_like(values, dtype=float) if scale == 0 else (values - med) / scale
    return 0.6745 * (values - med) / mad


def _iso(ts) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).isoformat()


def _num(x) -> Any:
    """numpy scalar -> plain python, so json.dumps never chokes."""
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return round(float(x), 4)
    return x


def categorize(activity_type: str) -> str:
    if activity_type in ACTIVITY_CATEGORIES:
        return ACTIVITY_CATEGORIES[activity_type]
    upper = str(activity_type).upper()
    for keywords, category in CATEGORY_RULES:
        if any(k in upper for k in keywords):
            return category
    return "other"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def parse_js_timestamps(series: pd.Series) -> pd.Series:
    """Fast path for JS date strings, with a flexible retry for the stragglers."""
    text = series.astype("string")
    out = pd.to_datetime(text.str.slice(0, _JS_DATE_LEN), format=_JS_DATE_FORMAT,
                         errors="coerce", utc=True)
    missing = out.isna() & text.notna()
    if missing.any():
        retry = text[missing].str.replace(_PAREN_RE, "", regex=True).str.strip()
        out.loc[missing] = pd.to_datetime(retry, errors="coerce", utc=True)
    return out


def load_activities(path: str | Path | pd.DataFrame, errors: ErrorLog | None = None) -> pd.DataFrame:
    """Read and normalise the activity log. Missing optional columns are
    synthesised so every downstream section can assume they exist."""
    errors = errors or ErrorLog()
    df = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    for col in df.columns[df.dtypes == object]:
        df[col] = df[col].astype("string").str.strip().replace(
            {"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})

    if TYPE_FIELD not in df.columns:
        errors.errors.append({"section": "load",
                              "error": f"missing '{TYPE_FIELD}' column; all events typed UNKNOWN"})
        df[TYPE_FIELD] = "UNKNOWN"
    df[TYPE_FIELD] = df[TYPE_FIELD].fillna("UNKNOWN").astype(str)

    if CREATED_FIELD not in df.columns:
        raise ValueError(f"activity file has no '{CREATED_FIELD}' column - nothing can be dated")

    df["datetime"] = parse_js_timestamps(df[CREATED_FIELD])
    df["updated_dt"] = (parse_js_timestamps(df[UPDATED_FIELD])
                        if UPDATED_FIELD in df.columns else pd.NaT)

    unparsed = int(df["datetime"].isna().sum())
    if unparsed:
        errors.errors.append({"section": "load",
                              "error": f"{unparsed} row(s) had an unparseable {CREATED_FIELD} and were dropped"})
        df = df[df["datetime"].notna()].copy()
    if df.empty:
        raise ValueError("no activity rows survived timestamp parsing")

    # Representative resolution: first populated column in the priority list wins.
    who = pd.Series(pd.NA, index=df.index, dtype="string")
    source = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in REPRESENTATIVE_PRIORITY:
        if col not in df.columns:
            continue
        fill = who.isna() & df[col].notna()
        who = who.mask(fill, df[col].astype("string"))
        source = source.mask(fill, col)
    df["representative"] = who.fillna("Unattributed").astype(str)
    df["representative_source"] = source.fillna("none").astype(str)

    df["channel"] = df[CHANNEL_FIELD].astype(str) if CHANNEL_FIELD in df.columns else "unknown"
    df["category"] = df[TYPE_FIELD].map(categorize).astype(str)

    dt = df["datetime"].dt
    df["hour_utc"] = dt.hour.astype("int16")
    df["day_of_week"] = dt.day_name().astype(str)
    df["is_weekend"] = dt.dayofweek.ge(5)
    df["date"] = dt.date
    df["month"] = dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    return df.sort_values("datetime").reset_index(drop=True)


def attach_windows(df: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    """One boolean column per window. Every later section sums these instead of
    re-slicing the frame, which is what keeps the whole thing single-pass."""
    age_days = (ref - df["datetime"]).dt.total_seconds() / 86400.0
    df["age_days"] = age_days
    for name, (start, end) in WINDOWS.items():
        mask = age_days < start
        if end is not None:
            mask &= age_days >= end
        df[f"w_{name}"] = mask
    return df


_WCOLS = [f"w_{w}" for w in WINDOWS]


def _window_frame(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Counts per key per window, in one groupby."""
    grouped = df.groupby(key, observed=True)[_WCOLS].sum()
    grouped.columns = list(WINDOWS)
    grouped["total"] = df.groupby(key, observed=True).size()
    return grouped.sort_values("total", ascending=False)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def overview_section(df: pd.DataFrame, ref: pd.Timestamp) -> dict:
    first, last = df["datetime"].min(), df["datetime"].max()
    span = max((last - first).days + 1, 1)
    active_days = int(df["date"].nunique())
    daily = df.groupby("date", observed=True).size()
    counts = {w: int(df[f"w_{w}"].sum()) for w in WINDOWS}

    return {
        "total_events": int(len(df)),
        "distinct_types": int(df[TYPE_FIELD].nunique()),
        "distinct_representatives": int(df["representative"].nunique()),
        "first_event": _iso(first),
        "last_event": _iso(last),
        "span_days": span,
        "active_days": active_days,
        "active_day_coverage_pct": round(active_days / span * 100, 1),
        "events_per_active_day_mean": round(float(daily.mean()), 2),
        "events_per_active_day_median": round(float(daily.median()), 2),
        "busiest_day": {"date": str(daily.idxmax()), "events": int(daily.max())},
        "weekend_share_pct": round(float(df["is_weekend"].mean()) * 100, 1),
        "window_counts": counts,
        "volume_change": {
            "mom": {**pct_change(counts["last_30d"], counts["prev_30d"]),
                    "reliability": reliability(counts["last_30d"], counts["prev_30d"])},
            "qoq": {**pct_change(counts["last_90d"], counts["prev_90d"]),
                    "reliability": reliability(counts["last_90d"], counts["prev_90d"])},
            "yoy": {**pct_change(counts["last_365d"], counts["prev_365d"]),
                    "reliability": reliability(counts["last_365d"], counts["prev_365d"])},
        },
        # Zero by construction; the useful figure is staleness against the real clock.
        "days_since_last_event_vs_reference": int((ref - last).days),
        "days_since_last_event_vs_today": int((pd.Timestamp.now("UTC") - last).days),
    }


def type_section(df: pd.DataFrame) -> dict:
    frame = _window_frame(df, TYPE_FIELD)
    total = int(len(df))
    first_last = df.groupby(TYPE_FIELD, observed=True)["datetime"].agg(["min", "max"])

    recent_total = int(df["w_last_90d"].sum())
    # A thin trailing window makes every share a large fraction of a tiny number,
    # so the mix-shift column carries its own health warning.
    mix_shift_reliability = reliability(recent_total)

    rows = []
    for name, r in frame.head(CAP_TYPES).iterrows():
        share_recent = (r["last_90d"] / max(recent_total, 1)) * 100
        share_all = r["total"] / total * 100
        category = categorize(name)
        rows.append({
            "type": name,
            "label": humanize(name),
            "category": category,
            "category_label": CATEGORY_LABELS.get(category, category),
            "count": int(r["total"]),
            "share_pct": round(share_all, 1),
            "count_last_30d": int(r["last_30d"]),
            "count_last_90d": int(r["last_90d"]),
            "share_last_90d_pct": round(share_recent, 1),
            # What the team stopped doing, not just that they did less.
            "share_shift_pp_vs_alltime": round(share_recent - share_all, 1),
            "share_shift_reliability": mix_shift_reliability,
            "mom": {**pct_change(r["last_30d"], r["prev_30d"]),
                    "reliability": reliability(r["last_30d"], r["prev_30d"])},
            "qoq": {**pct_change(r["last_90d"], r["prev_90d"]),
                    "reliability": reliability(r["last_90d"], r["prev_90d"])},
            "first_seen": _iso(first_last.loc[name, "min"]),
            "last_seen": _iso(first_last.loc[name, "max"]),
        })

    cat_counts = df.groupby("category", observed=True).size().sort_values(ascending=False)
    cat_windows = _window_frame(df, "category")
    nested = df.groupby(["category", TYPE_FIELD], observed=True).size()
    by_category = {}
    for cat, count in cat_counts.items():
        types = nested.loc[cat].sort_values(ascending=False)
        by_category[cat] = {
            "label": CATEGORY_LABELS.get(cat, cat),
            "description": CATEGORY_DESCRIPTIONS.get(cat, ""),
            "count": int(count),
            "share_pct": round(count / total * 100, 1),
            "count_last_90d": int(cat_windows.loc[cat, "last_90d"]),
            "qoq": {**pct_change(cat_windows.loc[cat, "last_90d"], cat_windows.loc[cat, "prev_90d"]),
                    "reliability": reliability(cat_windows.loc[cat, "last_90d"],
                                               cat_windows.loc[cat, "prev_90d"])},
            "activities": {humanize(t): {"count": int(c),
                                         "share_within_category_pct": round(c / count * 100, 1)}
                           for t, c in types.items()},
        }

    return {
        "by_type": rows,
        "types_truncated": bool(len(frame) > CAP_TYPES),
        "events_in_last_90d": recent_total,
        "mix_shift_reliability": mix_shift_reliability,
        "activities_with_no_recent_use": [r["label"] for r in rows if r["count_last_90d"] == 0],
        "type_diversity_index": _entropy(frame["total"].to_numpy()),
        "by_category": by_category,
    }


def representative_section(df: pd.DataFrame, ref: pd.Timestamp) -> tuple[dict, list[dict]]:
    """Per-representative behavioural profiles -- the part that differs file to
    file. Returns (summary_for_metrics, full_profiles)."""
    total = int(len(df))
    frame = _window_frame(df, "representative")
    agg = df.groupby("representative", observed=True).agg(
        first_seen=("datetime", "min"),
        last_seen=("datetime", "max"),
        active_days=("date", "nunique"),
        distinct_types=(TYPE_FIELD, "nunique"),
        weekend_share=("is_weekend", "mean"),
        median_hour=("hour_utc", "median"),
    )
    mix = pd.crosstab(df["representative"], df[TYPE_FIELD])
    cat_mix = pd.crosstab(df["representative"], df["category"])
    source = df.groupby("representative", observed=True)["representative_source"].agg(
        lambda s: s.value_counts().index[0])
    gaps = df.groupby("representative", observed=True)["datetime"].diff().dt.total_seconds() / 86400.0
    median_gap = gaps.groupby(df["representative"]).median()

    profiles: list[dict] = []
    for name, r in frame.iterrows():
        a = agg.loc[name]
        tenure = max((a["last_seen"] - a["first_seen"]).days + 1, 1)
        recency = int((ref - a["last_seen"]).days)
        status = ("active" if recency <= DORMANT_DAYS[0] else
                  "cooling" if recency <= DORMANT_DAYS[1] else
                  "dormant" if recency <= DORMANT_DAYS[2] else "churned")
        row_mix = mix.loc[name]
        top = row_mix[row_mix > 0].sort_values(ascending=False)
        is_person = bool(source.get(name) == REPRESENTATIVE_PRIORITY[0])
        profiles.append({
            "representative": name,
            "label": humanize(name),
            "identified_by": source.get(name, "none"),
            # A channel bucket is a group or a machine, never an individual.
            "is_named_person": is_person,
            "kind": "Named person" if is_person else "Channel / integration",
            "events": int(r["total"]),
            "share_of_all_events_pct": round(r["total"] / total * 100, 1),
            "first_seen": _iso(a["first_seen"]),
            "last_seen": _iso(a["last_seen"]),
            "days_since_last_event": recency,
            "status": status,
            "tenure_days": int(tenure),
            "active_days": int(a["active_days"]),
            "active_day_coverage_pct": round(a["active_days"] / tenure * 100, 1),
            "events_per_active_day": round(r["total"] / max(int(a["active_days"]), 1), 2),
            "median_days_between_events": (None if pd.isna(median_gap.get(name))
                                           else round(float(median_gap.get(name)), 2)),
            "distinct_activities": int(a["distinct_types"]),
            "activity_diversity_index": _entropy(row_mix.to_numpy()),
            "main_activity": humanize(top.index[0]) if len(top) else None,
            "main_activity_share_pct": round(float(top.iloc[0] / r["total"] * 100), 1) if len(top) else 0.0,
            "top_activities": {humanize(t): int(c) for t, c in top.head(5).items()},
            "category_mix": {CATEGORY_LABELS.get(c, c): int(v)
                             for c, v in cat_mix.loc[name].items() if v > 0},
            "events_last_30d": int(r["last_30d"]),
            "events_last_90d": int(r["last_90d"]),
            "mom": {**pct_change(r["last_30d"], r["prev_30d"]),
                    "reliability": reliability(r["last_30d"], r["prev_30d"])},
            "qoq": {**pct_change(r["last_90d"], r["prev_90d"]),
                    "reliability": reliability(r["last_90d"], r["prev_90d"])},
            "weekend_share_pct": round(float(a["weekend_share"]) * 100, 1),
            "median_hour_utc": _num(a["median_hour"]),
        })

    status_counts = pd.Series([p["status"] for p in profiles]).value_counts().to_dict()
    shares = np.array([p["share_of_all_events_pct"] for p in profiles], dtype=float)
    summary = {
        "representative_count": len(profiles),
        "identification_priority": list(REPRESENTATIVE_PRIORITY),
        "named_person_count": int(sum(p["is_named_person"] for p in profiles)),
        "named_person_event_count": int(sum(p["events"] for p in profiles if p["is_named_person"])),
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        # HHI on volume share: 1.0 = one source generates everything.
        "concentration_hhi": round(float(((shares / 100) ** 2).sum()), 3),
        "top_share_pct": round(float(shares.max()), 1) if len(shares) else 0.0,
        "top_representatives": profiles[:CAP_REPRESENTATIVES],
        "truncated": bool(len(profiles) > CAP_REPRESENTATIVES),
        "by_channel": {humanize(k): int(v) for k, v in
                       df.groupby("channel", observed=True).size().sort_values(ascending=False).items()},
    }
    return summary, profiles


def salesperson_orders_section(orders_path: str | Path, activity_ref: pd.Timestamp) -> dict:
    """Orders and revenue per salesperson, straight from the order book.

    The activity log cannot supply this: it records only part of all orders and
    leaves the representative blank on order-creation rows.

    Windows run back from whichever source is more recent, orders or activity --
    anchoring on orders alone would shrink the trailing windows whenever the
    order export stops earlier than the log.

    Third-party orders are excluded but their count and revenue are kept in the
    section, so the exclusion is visible rather than silent."""
    orders = pd.read_csv(orders_path)
    orders.columns = [str(c).strip().lstrip("\ufeff") for c in orders.columns]
    orders["datetime"] = parse_js_timestamps(orders[CREATED_FIELD])
    orders = orders[orders["datetime"].notna()].copy()
    orders["amount"] = pd.to_numeric(orders.get(ORDER_AMOUNT_FIELD), errors="coerce").fillna(0.0)

    excluded_orders = excluded_revenue = 0
    if ORDER_TYPE_FIELD in orders.columns:
        is_excluded = orders[ORDER_TYPE_FIELD].astype("string").isin(ORDER_TYPES_EXCLUDED)
        excluded_orders = int(is_excluded.sum())
        excluded_revenue = round(float(orders.loc[is_excluded, "amount"].sum()), 2)
        orders = orders[~is_excluded].copy()

    if ORDER_SALESPERSON_FIELD not in orders.columns:
        raise ValueError(f"orders file has no '{ORDER_SALESPERSON_FIELD}' column")
    orders["salesperson"] = (orders[ORDER_SALESPERSON_FIELD].astype("string").str.strip()
                             .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})
                             .fillna(UNASSIGNED_SALESPERSON).astype(str))

    ref = max(orders["datetime"].max(), activity_ref)
    age = (ref - orders["datetime"]).dt.total_seconds() / 86400.0
    orders["w_30"] = age < 30
    orders["w_prev30"] = (age >= 30) & (age < 60)
    orders["w_90"] = age < 90
    # Money per window as columns, so the rollup is one groupby-sum.
    for w in ("w_30", "w_prev30", "w_90"):
        orders[f"amt_{w}"] = orders["amount"].where(orders[w], 0.0)

    stats = orders.groupby("salesperson", observed=True).agg(
        orders_total=("amount", "size"),
        revenue_total=("amount", "sum"),
        revenue_median=("amount", "median"),
        largest_order=("amount", "max"),
        orders_30=("w_30", "sum"),
        orders_prev30=("w_prev30", "sum"),
        orders_90=("w_90", "sum"),
        revenue_30=("amt_w_30", "sum"),
        revenue_prev30=("amt_w_prev30", "sum"),
        revenue_90=("amt_w_90", "sum"),
        first_order=("datetime", "min"),
        last_order=("datetime", "max"),
    )

    total_orders, total_revenue = int(len(orders)), float(orders["amount"].sum())
    rows = []
    for name, s in stats.iterrows():
        rows.append({
            "salesperson": name,
            "is_attributed": name != UNASSIGNED_SALESPERSON,
            "orders_total": int(s["orders_total"]),
            "revenue_total": round(float(s["revenue_total"]), 2),
            "avg_order_value": round(float(s["revenue_total"]) / max(int(s["orders_total"]), 1), 2),
            "median_order_value": round(float(s["revenue_median"]), 2),
            "largest_order": round(float(s["largest_order"]), 2),
            "orders_last_90d": int(s["orders_90"]),
            "revenue_last_90d": round(float(s["revenue_90"]), 2),
            "orders_last_30d": int(s["orders_30"]),
            "revenue_last_30d": round(float(s["revenue_30"]), 2),
            "orders_mom": {**pct_change(s["orders_30"], s["orders_prev30"]),
                           "reliability": reliability(s["orders_30"], s["orders_prev30"])},
            "revenue_mom": {**pct_change(s["revenue_30"], s["revenue_prev30"]),
                            "reliability": reliability(s["orders_30"], s["orders_prev30"])},
            "share_of_orders_pct": round(int(s["orders_total"]) / total_orders * 100, 1),
            "share_of_revenue_pct": (round(float(s["revenue_total"]) / total_revenue * 100, 1)
                                     if total_revenue else 0.0),
            "first_order": _iso(s["first_order"]),
            "last_order": _iso(s["last_order"]),
            "days_since_last_order": int((ref - s["last_order"]).days),
        })
    rows.sort(key=lambda r: r["revenue_total"], reverse=True)

    unassigned = next((r for r in rows if not r["is_attributed"]), None)
    amounts = orders["amount"]
    mean_aov = total_revenue / max(total_orders, 1)
    median_aov = float(amounts.median())
    return {
        "source": f"orders file, {ORDER_SALESPERSON_FIELD} (the activity log does not carry "
                  f"salesperson attribution on order-creation rows)",
        "excluded_order_types": list(ORDER_TYPES_EXCLUDED),
        "excluded_orders": excluded_orders,
        "excluded_revenue": excluded_revenue,
        "reference_date_utc": _iso(ref),
        "activity_reference_date_utc": _iso(activity_ref),
        "orders_date_range": [_iso(orders["datetime"].min()), _iso(orders["datetime"].max())],
        "salesperson_count": sum(1 for r in rows if r["is_attributed"]),
        "totals": {
            "orders": total_orders,
            "revenue": round(total_revenue, 2),
            "avg_order_value": round(mean_aov, 2),
            "median_order_value": round(median_aov, 2),
            "orders_last_90d": int(orders["w_90"].sum()),
            "revenue_last_90d": round(float(orders["amt_w_90"].sum()), 2),
            "orders_last_30d": int(orders["w_30"].sum()),
            "revenue_last_30d": round(float(orders["amt_w_30"].sum()), 2),
            "orders_mom": {**pct_change(int(orders["w_30"].sum()), int(orders["w_prev30"].sum())),
                           "reliability": reliability(int(orders["w_30"].sum()),
                                                      int(orders["w_prev30"].sum()))},
            "revenue_mom": {**pct_change(float(orders["amt_w_30"].sum()),
                                         float(orders["amt_w_prev30"].sum())),
                            "reliability": reliability(int(orders["w_30"].sum()),
                                                       int(orders["w_prev30"].sum()))},
        },
        "concentration": {
            "top3_revenue_share_pct": (round(sum(r["revenue_total"] for r in rows[:3])
                                             / total_revenue * 100, 1) if total_revenue else 0.0),
            "unassigned_orders": unassigned["orders_total"] if unassigned else 0,
            "unassigned_share_pct": unassigned["share_of_orders_pct"] if unassigned else 0.0,
        },
        # Mean is meaningless next to a median three orders of magnitude smaller,
        # so both are reported and the gap is measured.
        "revenue_skew": {
            "mean_order_value": round(mean_aov, 2),
            "median_order_value": round(median_aov, 2),
            "largest_order": round(float(amounts.max()), 2),
            "mean_to_median_ratio": round(mean_aov / max(median_aov, 0.01), 1),
        },
        "by_salesperson": rows,
    }


def _salesperson_summary_rows(salesperson_orders: dict) -> tuple[list[dict], dict]:
    """Reshape into the entire-period / last-3-months / last-month / MoM layout
    the business already uses. Shared by the table renderer and the CSV export so
    the two cannot disagree. Pure reshape: nothing is recomputed."""
    rows = [{
        "salesperson": r["salesperson"],
        "orders_entire_period": r["orders_total"],
        "revenue_entire_period": r["revenue_total"],
        "orders_last_3_months": r["orders_last_90d"],
        "revenue_last_3_months": r["revenue_last_90d"],
        "orders_last_month": r["orders_last_30d"],
        "revenue_last_month": r["revenue_last_30d"],
        "revenue_change_mom": r["revenue_mom"],
    } for r in salesperson_orders.get("by_salesperson", [])]

    t = salesperson_orders.get("totals", {})
    totals = {
        "salesperson": "Total",
        "orders_entire_period": t.get("orders"),
        "revenue_entire_period": t.get("revenue"),
        "orders_last_3_months": t.get("orders_last_90d"),
        "revenue_last_3_months": t.get("revenue_last_90d"),
        "orders_last_month": t.get("orders_last_30d"),
        "revenue_last_month": t.get("revenue_last_30d"),
        "revenue_change_mom": t.get("revenue_mom"),
    }
    return rows, totals


def workflow_section(df: pd.DataFrame) -> dict:
    counts = df[TYPE_FIELD].value_counts()
    recent = df.loc[df["w_last_90d"], TYPE_FIELD].value_counts()
    pairs = {}
    for opened_type, closed_type, label, relation in WORKFLOW_PAIRS:
        opened, closed = int(counts.get(opened_type, 0)), int(counts.get(closed_type, 0))
        if opened == 0 and closed == 0:
            continue
        pairs[label] = {
            "opening_activity": humanize(opened_type),
            "closing_activity": humanize(closed_type),
            "relation": relation,
            "opened": opened,
            "closed": closed,
            "rate_pct": round(closed / opened * 100, 1) if opened else None,
            "rate_means": {"completion": "share of created items later marked done",
                           "cancellation": "share of created items later cancelled",
                           "reversal": "share of created items later voided"}[relation],
            # Only a completion flow leaves a backlog behind.
            "net_open_implied": (opened - closed) if relation == "completion" else None,
            "opened_last_90d": int(recent.get(opened_type, 0)),
            "closed_last_90d": int(recent.get(closed_type, 0)),
        }
    return {
        "pairs": pairs,
        "method_note": ("Cohort ratio, not per-item matching: the log carries no parent id "
                        "linking a completion back to its creation, so the rate compares totals "
                        "over the same period and will misstate it if items routinely close in a "
                        "later period than they open."),
    }


def temporal_section(df: pd.DataFrame) -> dict:
    monthly = df.groupby("month", observed=True).size()
    full_index = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_index, fill_value=0)
    values = monthly.to_numpy(dtype=float)

    z = _modified_z(values)
    anomalies = [{"month": str(full_index[i]), "events": int(values[i]),
                  "modified_z": round(float(z[i]), 2),
                  "kind": "spike" if z[i] > 0 else "drought"}
                 for i in np.where(np.abs(z) > OUTLIER_Z)[0]]

    tail = values[-12:]
    slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) >= 3 else 0.0
    half = len(tail) // 2
    first_half, second_half = float(tail[:half].sum()), float(tail[half:].sum())

    hourly = df["hour_utc"].value_counts().reindex(range(24), fill_value=0).sort_index().to_numpy()
    total = int(hourly.sum())
    best_windows = {}
    for size in (1, 3, 4):
        sums = [int(hourly[[(s + i) % 24 for i in range(size)]].sum()) for s in range(24)]
        start = int(np.argmax(sums))
        best_windows[f"{size}h"] = {
            "start_hour_utc": start, "end_hour_utc": (start + size) % 24,
            "events": sums[start],
            "share_pct": round(sums[start] / total * 100, 1) if total else 0.0,
        }

    return {
        "timezone": "UTC",
        "monthly_counts": {str(p): int(v) for p, v in
                           zip(full_index[-CAP_MONTHS:], values[-CAP_MONTHS:])},
        "monthly_median": round(float(np.median(values)), 1),
        "months_covered": int(len(full_index)),
        "months_with_zero_events": int((values == 0).sum()),
        "trend_last_12m": {
            "slope_events_per_month": round(slope, 2),
            "direction": "rising" if slope > 0.5 else "falling" if slope < -0.5 else "flat",
            "first_half_events": int(first_half),
            "second_half_events": int(second_half),
            **pct_change(second_half, first_half),
            "reliability": reliability(first_half, second_half),
        },
        "anomalous_months": anomalies[:CAP_EXAMPLES * 2],
        "hourly_counts": {int(h): int(c) for h, c in enumerate(hourly)},
        "peak_hour_utc": int(np.argmax(hourly)) if total else None,
        "best_contiguous_windows": best_windows,
        "day_of_week_counts": {str(k): int(v) for k, v in df["day_of_week"].value_counts().items()},
        "note": ("Hours are UTC as stored. Convert to the tenant's local timezone before drawing "
                 "any staffing or scheduling conclusion."),
    }


def automation_section(df: pd.DataFrame) -> dict:
    """Hours and seconds that look machine-generated rather than human.

    Two tells: an hour where one normally-rare activity owns most of the volume,
    measured as lift against that activity's own share rather than raw dominance;
    and an exact second shared by many rows, which is an import, not typing."""
    total = int(len(df))
    overall = df[TYPE_FIELD].value_counts(normalize=True)
    per_hour = df.groupby(["hour_utc", TYPE_FIELD], observed=True).size()
    hour_totals = df.groupby("hour_utc", observed=True).size()

    suspected = []
    for hour, hour_total in hour_totals.items():
        if hour_total < AUTOMATION_MIN_EVENTS:
            continue
        top = per_hour.loc[hour].sort_values(ascending=False)
        share = float(top.iloc[0]) / hour_total
        baseline = float(overall.get(top.index[0], 1e-4))
        lift = share / baseline
        if share > AUTOMATION_SHARE and lift > AUTOMATION_LIFT:
            suspected.append({"hour_utc": int(hour), "dominant_activity": humanize(top.index[0]),
                              "dominant_share_pct": round(share * 100, 1),
                              "baseline_share_pct": round(baseline * 100, 1),
                              "lift_vs_baseline": round(lift, 1),
                              "events_in_hour": int(hour_total)})

    per_second = df["datetime"].value_counts()
    bulk = per_second[per_second >= BULK_SECOND_MIN]
    bulk_rows = int(bulk.sum())
    return {
        "automation_suspected_hours": suspected,
        "bulk_import_seconds": [{"timestamp": _iso(ts), "rows": int(n)}
                                for ts, n in bulk.head(CAP_EXAMPLES).items()],
        "bulk_import_row_count": bulk_rows,
        "bulk_import_share_pct": round(bulk_rows / total * 100, 1) if total else 0.0,
    }


def data_quality_section(df: pd.DataFrame, raw_columns: list[str], ref: pd.Timestamp,
                         automation: dict, orders_available: bool, orders_error: str | None) -> list[dict]:
    """Structured notes (code + severity + evidence) rather than prose blobs, so
    the agent can cite one without reformatting it and the caller can filter."""
    notes: list[dict] = []
    total = len(df)

    def add(code, severity, message, **evidence):
        notes.append({"code": code, "severity": severity, "message": message, "evidence": evidence})

    empty_cols = [c for c in raw_columns if c in df.columns and df[c].isna().all()]
    if empty_cols:
        add("empty_columns", "info",
            f"{len(empty_cols)} column(s) are empty in every row and carry no information: "
            f"{', '.join(empty_cols)}.", columns=empty_cols)

    constant = [c for c in raw_columns
                if c in df.columns and df[c].notna().any() and df[c].nunique(dropna=True) == 1]
    if constant:
        add("constant_columns", "info",
            f"{len(constant)} column(s) hold a single value throughout "
            f"({', '.join(constant)}), so this file is a single-tenant slice.", columns=constant)

    if not orders_available:
        if orders_error:
            add("orders_file_failed", "high",
                f"An orders file was supplied but salesperson statistics could not be computed "
                f"from it ({orders_error}) -- the 'Orders and revenue by salesperson' table is "
                f"not included below.",
                error=orders_error)
        else:
            add("no_orders_file", "high",
                "No orders file was supplied, so this report has no salesperson order or revenue "
                "statistics -- the 'Orders and revenue by salesperson' table is not included "
                "below. Pass an orders CSV to add it.")

    unattributed = int((df["representative_source"] == "none").sum())
    channel_only = int((df["representative_source"] == CHANNEL_FIELD).sum())
    named = total - unattributed - channel_only
    if named < total:
        redirect = ("see the salesperson table below for order and revenue figures by person "
                    "instead" if orders_available else
                    "supply an orders file to get salesperson-level order and revenue figures "
                    "from the order book instead")
        add("weak_representative_attribution", "high",
            f"Only {named} of {total} events ({named / total * 100:.0f}%) name a representative; "
            f"{channel_only} are attributed to a channel bucket and {unattributed} to nothing at "
            f"all. Per-person activity cannot be read from this file beyond those named rows -- "
            f"{redirect}.",
            named_events=named, channel_only=channel_only, unattributed=unattributed)

    if df["updated_dt"].notna().any():
        upd_dates = df["updated_dt"].dt.date.value_counts()
        top_date, top_n = upd_dates.index[0], int(upd_dates.iloc[0])
        if top_n / total >= MASS_UPDATE_SHARE:
            lag = (df["updated_dt"] - df["datetime"]).dt.total_seconds() / 86400.0
            add("updated_at_mass_rewrite", "high",
                f"{top_n} of {total} rows ({top_n / total * 100:.0f}%) share the same "
                f"{UPDATED_FIELD} date ({top_date}), with a median lag of {lag.median():.0f} days "
                f"after creation. That is a migration or backfill stamp, not an edit time -- do "
                f"not use {UPDATED_FIELD} to measure how long anything took.",
                shared_date=str(top_date), rows=top_n,
                median_lag_days=round(float(lag.median()), 1))

    if automation.get("bulk_import_row_count"):
        ex = (automation.get("bulk_import_seconds") or [{}])[0]
        add("bulk_import_rows", "medium",
            f"{automation['bulk_import_row_count']} rows "
            f"({automation['bulk_import_share_pct']}% of the file) share an exact creation second "
            f"with {BULK_SECOND_MIN}+ other rows (largest: {ex.get('rows')} rows at "
            f"{ex.get('timestamp')}). These were imported, not performed, and inflate whatever "
            f"day they land on.", **automation)

    recent, prior = int(df["w_last_30d"].sum()), int(df["w_prev_30d"].sum())
    if recent < MIN_SAMPLE:
        add("sparse_recent_window", "high",
            f"Only {recent} events fall in the trailing 30 days (vs {prior} in the 30 before "
            f"that), so month-over-month percentages rest on a handful of rows. Every comparison "
            f"carries a reliability flag; treat anything marked low confidence as directional.",
            last_30d=recent, prev_30d=prior, threshold=MIN_SAMPLE)

    last = df["datetime"].max()
    add("reference_date", "info",
        f"The reference date is the newest timestamp in the file ({last.date()}), not today. All "
        f"windows are measured backwards from it.",
        reference_date=_iso(ref), latest_event=_iso(last), days_stale=int((ref - last).days))

    dupes = int(df.duplicated(subset=[TYPE_FIELD, "representative", CREATED_FIELD]).sum())
    if dupes:
        add("possible_duplicate_events", "medium",
            f"{dupes} row(s) repeat the same activity, representative and creation second as "
            f"another row. Some are legitimate batch actions, but they double-count in any "
            f"per-event rate.", duplicate_rows=dupes)

    add("timezone", "info",
        "All timestamps are stored with a GMT+0000 offset and are reported in UTC.")
    return notes


def order_crosscheck_section(df: pd.DataFrame, orders_path: str | Path) -> dict:
    """Does the log actually contain the orders the order book has? The answer
    changes what the log may be used for."""
    orders = pd.read_csv(orders_path)
    orders.columns = [str(c).strip().lstrip("\ufeff") for c in orders.columns]
    orders["datetime"] = parse_js_timestamps(orders[CREATED_FIELD])
    orders = orders[orders["datetime"].notna()]

    in_range = orders[orders["datetime"] >= df["datetime"].min()]
    logged = int((df[TYPE_FIELD] == "ORDER_ADDED").sum())
    id_overlap = (len(set(df["id"]) & set(orders["id"]))
                  if "id" in df.columns and "id" in orders.columns else None)
    coverage = round(logged / len(in_range) * 100, 1) if len(in_range) else None
    partial = coverage is not None and coverage < 95

    return {
        "orders_in_file": int(len(orders)),
        "orders_since_log_start": int(len(in_range)),
        "order_added_events_in_log": logged,
        "log_coverage_of_orders_pct": coverage,
        "id_overlap": id_overlap,
        "verdict": (f"The activity log captures only {coverage}% of orders created since it starts"
                    if partial else "The activity log tracks the order book closely"),
        "implication": ("Read order volume and revenue from the order book; treat order-creation "
                        "events in the log as a lower bound and use the log for behaviour (who "
                        "did what, when), not for totals." if partial else
                        "Order-creation counts in the log can be used as order volume."),
    }


# ---------------------------------------------------------------------------
# Human-facing tables -- rendered in code, so their numbers are always exact.
# ---------------------------------------------------------------------------

def _fmt_change(block: dict | None) -> str:
    """One place decides what a change block looks like in a table cell."""
    if not block:
        return "-"
    basis = block.get("basis")
    if basis == "no_baseline":
        return "from 0"
    if basis == "no_activity":
        return "-"
    pct = block.get("pct")
    if pct is None:
        return "n/a"
    flag = "*" if block.get("reliability") == "low_sample_low_confidence" else ""
    return f"{pct:+.1f}%{flag}"


def _fmt_change_legacy(block: dict | None) -> str:
    """Same figures, formatted the way the salesperson-orders export always has:
    0.0% when neither period had any orders, +100.0% when this period is the
    first with any. `basis` and `reliability` stay in the metrics for the agent
    either way -- this only changes what the table cell reads."""
    if not block:
        return "+0.0%"
    basis = block.get("basis")
    if basis == "no_activity":
        return "+0.0%"
    if basis == "no_baseline":
        return "+100.0%"
    pct = block.get("pct")
    if pct is None:
        return "n/a"
    flag = "*" if block.get("reliability") == "low_sample_low_confidence" else ""
    return f"{pct:+.1f}%{flag}"


def render_tables(metrics: dict) -> dict[str, dict]:
    """{key: {title, intro, markdown, footnote}} -- assembled into the report by
    the caller, never rewritten by a model."""
    tables: dict[str, dict] = {}
    thin_flag = False

    # A missing salesperson table is the most common "where did my data go"
    # question this report gets, so it becomes a visible callout rather than an
    # absence the reader has to notice.
    for n in metrics.get("data_quality") or []:
        if n.get("code") in ("no_orders_file", "orders_file_failed"):
            tables["_notice"] = {"text": n["message"]}
            break

    sp = metrics.get("salesperson_orders") or {}
    rows = sp.get("by_salesperson") or []
    if rows:
        summary_rows, summary_totals = _salesperson_summary_rows(sp)
        shown = summary_rows[:SALESPERSON_DISPLAY_LIMIT]
        thin_flag |= any(r["revenue_change_mom"].get("reliability") == "low_sample_low_confidence"
                         for r in shown)
        body = [[r["salesperson"], r["orders_entire_period"], money(r["revenue_entire_period"]),
                 r["orders_last_3_months"], money(r["revenue_last_3_months"]),
                 r["orders_last_month"], money(r["revenue_last_month"]),
                 _fmt_change_legacy(r["revenue_change_mom"])]
                for r in shown]
        body.append(["**Total**", summary_totals.get("orders_entire_period"),
                     money(summary_totals.get("revenue_entire_period")),
                     summary_totals.get("orders_last_3_months"),
                     money(summary_totals.get("revenue_last_3_months")),
                     summary_totals.get("orders_last_month"),
                     money(summary_totals.get("revenue_last_month")),
                     _fmt_change_legacy(summary_totals.get("revenue_change_mom"))])
        coverage = (metrics.get("order_crosscheck") or {}).get("log_coverage_of_orders_pct")
        coverage_note = (f"the log records only {coverage}% of orders and leaves the salesperson "
                         f"blank on order-creation rows" if coverage is not None else
                         "the log leaves the salesperson blank on order-creation rows")
        excluded = sp.get("excluded_orders") or 0
        exclusion_note = (f" {excluded} third-party order(s) worth "
                          f"{money(sp.get('excluded_revenue'))} are excluded, since they were "
                          f"placed through an outside channel rather than the team itself."
                          if excluded else "")
        tables["salesperson_orders"] = {
            "title": "Orders and revenue by salesperson",
            "intro": (f"From the order book rather than the activity log - {coverage_note}."
                      f"{exclusion_note} Sorted by total revenue. **Last 3 Months** and "
                      f"**Last Month** are the trailing 90- and 30-day windows; "
                      f"**Revenue % Change (MoM)** compares the last 30 days with the 30 before "
                      f"them, where 0.0% means neither period had any orders and 100.0% means the "
                      f"period went from none to some."),
            "markdown": md_table(
                ["Salesperson", "Orders (Entire Period)", "Revenue (Entire Period)",
                 "Orders (Last 3 Months)", "Revenue (Last 3 Months)", "Orders (Last Month)",
                 "Revenue (Last Month)", "Revenue % Change (MoM)"], body),
            "footnote": (f"Showing the top {SALESPERSON_DISPLAY_LIMIT} of {len(rows)} salespeople "
                         f"by revenue; the total row covers all of them."
                         if len(rows) > SALESPERSON_DISPLAY_LIMIT else None),
        }

    types = (metrics.get("activity_types") or {}).get("by_type", [])
    if types:
        thin_flag |= any(t["qoq"].get("reliability") == "low_sample_low_confidence" for t in types)
        tables["activity_types"] = {
            "title": "What is being done in the system",
            "intro": ("Every action recorded in the log, most frequent first. **QoQ** compares "
                      "the last 90 days against the 90 days before them. **Mix shift** shows how "
                      "much each activity's share of recent work moved against its share of all "
                      "time, in percentage points - a positive number means it makes up more of "
                      "the work now than it used to, even when the raw count fell."),
            "markdown": md_table(
                ["Activity", "Category", "Total", "Share of all", "Last 90 days", "QoQ",
                 "Mix shift"],
                [[t["label"], t["category_label"], t["count"], f"{t['share_pct']}%",
                  t["count_last_90d"], _fmt_change(t["qoq"]),
                  f"{t['share_shift_pp_vs_alltime']:+.1f} pp"
                  + ("*" if t.get("share_shift_reliability") != "adequate" else "")]
                 for t in types]),
            "footnote": None,
        }

    reps = (metrics.get("representatives") or {}).get("top_representatives", [])
    if reps:
        tables["representatives"] = {
            "title": "Who is generating the activity",
            "intro": ("**Type** separates real people from channel buckets and integrations - a "
                      "channel row is many people or a machine, not an individual. **Status** is "
                      "measured from the file's newest timestamp: active within 30 days, cooling "
                      "within 90, dormant within 180, churned beyond that."),
            "markdown": md_table(
                ["Representative", "Type", "Events", "Share", "Last 90 days", "Status",
                 "Days idle", "Main activity"],
                [[r["label"], r["kind"], r["events"], f"{r['share_of_all_events_pct']}%",
                  r["events_last_90d"], r["status"].capitalize(), r["days_since_last_event"],
                  f"{r['main_activity']} ({r['main_activity_share_pct']}%)"] for r in reps]),
            "footnote": None,
        }

    if thin_flag:
        tables["_legend"] = {"text": f"\\* comparison based on fewer than {MIN_SAMPLE} events on "
                                     f"one side - directional only."}
    return tables


def write_salesperson_summary_csv(salesperson_orders: dict, path: str | Path) -> None:
    """Writes the salesperson table to disk with the exact column names and
    layout of the business's existing export, Total row included, so it can
    drop into whatever already consumes that file."""
    headers = ["Salesperson", "Orders (Entire Period)", "Revenue (Entire Period)",
              "Orders (Last 3 Months)", "Revenue (Last 3 Months)", "Orders (Last Month)",
              "Revenue (Last Month)", "Revenue % Change (MoM)"]

    def row(r: dict) -> list[str]:
        return [r["salesperson"], r["orders_entire_period"], money(r["revenue_entire_period"]),
                r["orders_last_3_months"], money(r["revenue_last_3_months"]),
                r["orders_last_month"], money(r["revenue_last_month"]),
                _fmt_change_legacy(r["revenue_change_mom"]).rstrip("*")]

    rows, totals = _salesperson_summary_rows(salesperson_orders)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(row(r))
        w.writerow(row(totals))


def _norm_title(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(title or "").lower()).strip()


def render_tables_markdown(tables: dict[str, dict], notes: dict[str, str] | None = None,
                           keys: list[str] | None = None,
                           hide_title: str | None = None) -> str:
    """Render some or all tables, each with its intro and the Statistics
    Analyst's reading of it underneath.

    `keys` selects which tables to render, which is what lets the report be split
    into sections without rendering anything twice. `hide_title` drops a table's
    own heading when the section around it already carries the same words. The
    missing-orders notice travels with the salesperson table it explains, and the
    low-sample legend is emitted only when the rendered tables carry a flag.
    """
    notes = notes or {}
    hidden = _norm_title(hide_title)
    keys = [k for k in (keys if keys is not None else tables) if not k.startswith("_")]
    parts: list[str] = []

    notice = (tables.get("_notice") or {}).get("text")
    if notice and ("salesperson_orders" in keys or "salesperson_orders" not in tables):
        parts += [f"> **Note:** {notice}", ""]

    for key in keys:
        table = tables.get(key)
        if not table:
            continue
        if _norm_title(table["title"]) != hidden:
            parts += [f"**{table['title']}**", ""]
        parts += [table["intro"], "", table["markdown"]]
        if table.get("footnote"):
            parts += ["", table["footnote"]]
        note = (notes.get(key) or "").strip()
        if note:
            parts += ["", note]
        parts.append("")

    legend = (tables.get("_legend") or {}).get("text")
    if legend and any("*" in (tables.get(k) or {}).get("markdown", "") for k in keys):
        parts += [legend, ""]
    return "\n".join(parts).rstrip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze_activities_file(csv_path: str | Path,
                            orders_path: str | Path | None = None,
                            output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run every section. Only an unreadable activity file raises; anything else
    is downgraded to an entry in `analytics_errors`."""
    started = time.perf_counter()
    errors = ErrorLog()

    df = load_activities(csv_path, errors)
    raw_columns = list(df.columns)
    ref = df["datetime"].max()
    df = attach_windows(df, ref)

    metrics: dict[str, Any] = {
        "metadata": {
            "source_file": str(csv_path),
            "orders_file": str(orders_path) if orders_path else None,
            "rows_analysed": int(len(df)),
            "columns": [c for c in raw_columns if not c.startswith(("w_", "is_"))],
            "reference_date_utc": _iso(ref),
            "generated_in_seconds": None,
        }
    }
    profiles: list[dict] = []
    automation: dict = {}

    with errors.guard("overview"):
        metrics["overview"] = overview_section(df, ref)
    with errors.guard("activity_types"):
        metrics["activity_types"] = type_section(df)
    with errors.guard("representatives"):
        metrics["representatives"], profiles = representative_section(df, ref)
    with errors.guard("workflow"):
        metrics["workflow"] = workflow_section(df)
    with errors.guard("temporal"):
        metrics["temporal"] = temporal_section(df)
    with errors.guard("automation"):
        automation = automation_section(df)
        metrics["automation"] = automation

    # Before data_quality, so that section reports the real outcome (no file
    # given vs. a file that failed) instead of guessing.
    if orders_path:
        with errors.guard("order_crosscheck"):
            metrics["order_crosscheck"] = order_crosscheck_section(df, orders_path)
        with errors.guard("salesperson_orders"):
            metrics["salesperson_orders"] = salesperson_orders_section(orders_path, ref)

    orders_available = "salesperson_orders" in metrics
    orders_error = next((e["error"] for e in errors.errors
                         if e["section"] in ("salesperson_orders", "order_crosscheck")), None)
    with errors.guard("data_quality"):
        metrics["data_quality"] = data_quality_section(df, raw_columns, ref, automation,
                                                        orders_available, orders_error)

    tables: dict[str, dict] = {}
    with errors.guard("render_tables"):
        tables = render_tables(metrics)

    metrics["metadata"]["generated_in_seconds"] = round(time.perf_counter() - started, 3)
    result = {
        "metrics": metrics,
        "representative_profiles": profiles,
        "display_tables": tables,
        "analytics_errors": errors.errors,
    }

    if output_dir:
        try:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "activity_metrics.json").write_text(
                json.dumps(metrics, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
            (out / "representative_profiles.json").write_text(
                json.dumps(profiles, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
            if metrics.get("salesperson_orders"):
                write_salesperson_summary_csv(metrics["salesperson_orders"],
                                              out / "salesperson_orders_summary.csv")
        except Exception as exc:  # noqa: BLE001
            errors.errors.append({"section": "write_output", "error": str(exc)})
    return result
