"""The time axis.

The brief: *whatever* period the user picks - one day, a week, a year, all time -
the analyser must cut it into comparable equal intervals and compare them. This
module is the only place that decides how.

The approach
------------
1. **The period is resolved first** (explicit `period_from`/`period_to`, else the
   min/max of the filtered data). Statistics are never computed over a window the
   user did not ask for.

2. **Granularity is chosen from a ladder**, not from the calendar name of the
   period. The engine walks hour -> 3h -> 6h -> 12h -> day -> 2d -> week -> 2w ->
   month -> quarter -> half -> year and keeps the candidate whose bucket count is
   closest to `FORMS_TARGET_BUCKETS` while staying inside
   [`FORMS_MIN_BUCKETS`, `FORMS_MAX_BUCKETS`]. A one-day period therefore comes
   out as 24 hourly buckets and three years as 12 quarterly ones, from the same
   code path and with the same downstream statistics.

3. **A granularity finer than the data is refused.** `time_resolution` comes from
   the loader; asking for hourly buckets on day-precision timestamps would
   manufacture a diurnal pattern out of nothing, so the engine steps up to the
   finest honest unit and says so in `notes`.

4. **Buckets are calendar-snapped by default** (`mode="calendar"`): a monthly
   bucket starts on the 1st, a weekly one on Monday. This is what makes labels
   meaningful and makes two runs of the report line up. The cost is that
   calendar units are not equal in raw duration (28-31 days), so every bucket
   carries `days` and every volume metric is also reported per day. When exact
   equality matters more than readable labels, `mode="uniform"` divides the span
   into N identical timedeltas.

5. **Empty buckets are emitted.** A month with no submissions is a finding, and
   dropping it would flatten the trend line and shift every comparison.

6. **A previous, equal-length window is planned alongside** so "this period vs
   the one before" is available without a second call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from ..config import (FORMS_MAX_BUCKETS, FORMS_MIN_ANSWERS_PER_PERIOD,
                      FORMS_MIN_BUCKETS, FORMS_TARGET_BUCKETS)
from ..core.logging_setup import get_log

log = get_log("forms.intervals")

_RESOLUTION_RANK: Final[dict[str, int]] = {"second": 0, "minute": 1, "hour": 2, "day": 3}


@dataclass(frozen=True)
class Granularity:
    key: str
    label: str
    freq: str            # pandas offset alias
    step: int            # how many of `freq` per bucket
    approx_days: float   # for candidate selection only
    min_resolution: str  # finest data resolution this unit is honest about

    @property
    def offset(self) -> pd.DateOffset:
        """The calendar-anchored step, used when buckets snap to boundaries."""
        return pd.tseries.frequencies.to_offset(self.freq) * self.step

    @property
    def anchored_offset(self) -> Any:
        """The step to use when the grid starts at an arbitrary date.

        `to_offset("W-MON")` rolls forward to the next Monday, so adding it to a
        Wednesday start produces a 12-day first period instead of 14. A fixed
        timedelta keeps every period the same length; months and longer keep a
        calendar offset so the day-of-month is preserved.
        """
        if self.freq == "h":
            return pd.Timedelta(hours=self.step)
        if self.freq == "D":
            return pd.Timedelta(days=self.step)
        if self.freq.startswith("W"):
            return pd.Timedelta(days=7 * self.step)
        if self.freq == "MS":
            return pd.DateOffset(months=self.step)
        if self.key == "half_year":
            return pd.DateOffset(months=6)
        if self.freq == "QS":
            return pd.DateOffset(months=3 * self.step)
        if self.freq == "YS":
            return pd.DateOffset(years=self.step)
        return pd.Timedelta(days=self.approx_days * self.step)

    def snap(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Floor a timestamp to this unit's natural boundary."""
        if self.freq == "h":
            floored = ts.floor("h")
            return floored - pd.Timedelta(hours=floored.hour % self.step) \
                if self.step > 1 else floored
        if self.freq == "D":
            return ts.normalize()
        if self.freq.startswith("W"):
            return ts.normalize() - pd.Timedelta(days=ts.weekday())
        if self.freq == "MS":
            return ts.normalize().replace(day=1)
        if self.key == "half_year":
            return ts.normalize().replace(month=1 if ts.month <= 6 else 7, day=1)
        if self.freq == "QS":
            first_month = 3 * ((ts.month - 1) // 3) + 1
            return ts.normalize().replace(month=first_month, day=1)
        if self.freq == "YS":
            return ts.normalize().replace(month=1, day=1)
        return ts.normalize()


LADDER: Final[tuple[Granularity, ...]] = (
    Granularity("hour",      "hour",      "h",  1,  1 / 24,  "hour"),
    Granularity("3h",        "3 hours",   "h",  3,  3 / 24,  "hour"),
    Granularity("6h",        "6 hours",   "h",  6,  6 / 24,  "hour"),
    Granularity("12h",       "12 hours",  "h", 12, 12 / 24,  "hour"),
    Granularity("day",       "day",       "D",  1,   1.0,    "day"),
    Granularity("2day",      "2 days",    "D",  2,   2.0,    "day"),
    # "W-MON" (not "W") so adding the offset to a Monday lands on the next
    # Monday. Bare "W" is week-ending-Sunday and silently shifts every edge.
    Granularity("week",      "week",      "W-MON",  1,   7.0,    "day"),
    Granularity("2week",     "2 weeks",   "W-MON",  2,  14.0,    "day"),
    Granularity("month",     "month",     "MS", 1,  30.44,   "day"),
    Granularity("quarter",   "quarter",   "QS", 1,  91.31,   "day"),
    Granularity("half_year", "half-year", "QS", 2, 182.62,   "day"),
    Granularity("year",      "year",      "YS", 1, 365.25,   "day"),
)

BY_KEY: Final[dict[str, Granularity]] = {g.key: g for g in LADDER}
# Friendly aliases a caller or a UI is likely to send.
BY_KEY.update({"hourly": BY_KEY["hour"], "daily": BY_KEY["day"],
               "weekly": BY_KEY["week"], "monthly": BY_KEY["month"],
               "quarterly": BY_KEY["quarter"], "yearly": BY_KEY["year"],
               "annual": BY_KEY["year"]})


@dataclass(frozen=True)
class Bucket:
    index: int
    key: str                 # stable machine key, e.g. "2025-03"
    label: str               # human label, e.g. "Mar 2025"
    start: pd.Timestamp
    end: pd.Timestamp        # exclusive
    days: float
    # True when the grid was anchored to the requested start rather than to a
    # calendar boundary, so labels must be date ranges: a bucket running 15 Jan to
    # 14 Feb is not "Jan 2025".
    anchored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "key": self.key, "label": self.label,
                "start": self.start.isoformat(), "end": self.end.isoformat(),
                "days": round(self.days, 3), "anchored": self.anchored}


@dataclass
class IntervalPlan:
    granularity: Granularity
    mode: str
    buckets: list[Bucket]
    period_start: pd.Timestamp
    period_end: pd.Timestamp          # exclusive
    previous_start: pd.Timestamp | None
    previous_end: pd.Timestamp | None
    time_resolution: str
    requested_granularity: str
    # The range the caller actually asked for, before buckets were snapped out to
    # calendar boundaries. Buckets that stick out past it are only partly covered
    # by data, so a per-day rate computed over their full width understates
    # reality - `coverage_of` is what lets that be detected instead of shipped.
    covered_start: pd.Timestamp | None = None
    covered_end: pd.Timestamp | None = None
    anchor: str = "calendar"
    notes: list[str] = field(default_factory=list)

    def coverage_of(self, bucket: Bucket) -> float:
        """Share of this bucket that lies inside the requested range (0.0-1.0)."""
        if self.covered_start is None or self.covered_end is None:
            return 1.0
        overlap = (min(bucket.end, self.covered_end)
                   - max(bucket.start, self.covered_start))
        width = bucket.end - bucket.start
        if width <= pd.Timedelta(0):
            return 0.0
        return max(0.0, min(1.0, overlap / width))

    def covered_days(self, bucket: Bucket) -> float:
        """Days of this bucket that lie inside the requested range."""
        return bucket.days * self.coverage_of(bucket)

    @property
    def modal_days(self) -> float:
        """The width most periods have; the yardstick for calling one short."""
        widths = [round(b.days, 3) for b in self.buckets]
        return max(set(widths), key=widths.count) if widths else 0.0

    def is_short(self, bucket: Bucket) -> bool:
        """A period narrower than most - the clipped tail of an anchored grid.

        Its rates are still comparable; its raw counts are not, which is why volume
        is always also reported per day.
        """
        modal = self.modal_days
        return bool(modal and bucket.days < modal * 0.8)

    @property
    def bucket_count(self) -> int:
        return len(self.buckets)

    @property
    def edges(self) -> np.ndarray:
        """Bucket boundaries as datetime64[ns]. Kept in datetime space rather
        than raw ints: pandas datetime columns can carry second, millisecond or
        microsecond units, and casting one side to int64 while the other is in
        nanoseconds silently puts every row outside the period."""
        return np.array([b.start for b in self.buckets] + [self.buckets[-1].end],
                        dtype="datetime64[ns]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "covered_start": (self.covered_start.isoformat()
                              if self.covered_start is not None else None),
            "covered_end": (self.covered_end.isoformat()
                            if self.covered_end is not None else None),
            "anchor": self.anchor,
            "granularity": self.granularity.key,
            "granularity_label": self.granularity.label,
            "requested_granularity": self.requested_granularity,
            "mode": self.mode,
            "bucket_count": self.bucket_count,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "period_days": round((self.period_end - self.period_start)
                                 / pd.Timedelta(days=1), 2),
            "previous_period": (
                {"start": self.previous_start.isoformat(),
                 "end": self.previous_end.isoformat()}
                if self.previous_start is not None else None),
            "time_resolution": self.time_resolution,
            "buckets": [{**b.to_dict(),
                         "coverage": round(self.coverage_of(b), 4),
                         "covered_days": round(self.covered_days(b), 3),
                         "partial": bool(self.coverage_of(b) < 0.9),
                         "short": self.is_short(b)}
                        for b in self.buckets],
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _range_key_and_label(start: pd.Timestamp, end: pd.Timestamp
                         ) -> tuple[str, str]:
    """Key and label for a bucket that does not sit on a calendar boundary."""
    last = end - pd.Timedelta(days=1) if (end - start) >= pd.Timedelta(days=1) \
        else end
    key = f"{start.strftime('%Y-%m-%d')}_{last.strftime('%Y-%m-%d')}"
    if start.date() == last.date():
        return key, start.strftime("%d %b %Y")
    if start.month == last.month and start.year == last.year:
        return key, f"{start.day}-{last.day} {last.strftime('%b %Y')}"
    return key, (f"{start.day} {start.strftime('%b')} - "
                 f"{last.day} {last.strftime('%b %Y')}")


def _key_and_label(granularity: Granularity, start: pd.Timestamp,
                   end: pd.Timestamp) -> tuple[str, str]:
    g = granularity.key
    if granularity.freq == "h":
        key = start.strftime("%Y-%m-%dT%H")
        label = (start.strftime("%d %b %H:00") if granularity.step == 1
                 else f"{start.strftime('%d %b %H:00')}-{end.strftime('%H:00')}")
    elif granularity.freq == "D":
        key = start.strftime("%Y-%m-%d")
        label = (start.strftime("%d %b %Y") if granularity.step == 1
                 else f"{start.strftime('%d %b')}-{(end - pd.Timedelta(days=1)).strftime('%d %b %Y')}")
    elif granularity.freq.startswith("W"):
        key = start.strftime("%G-W%V")
        label = (f"W{start.strftime('%V %G')}" if granularity.step == 1
                 else f"W{start.strftime('%V')}-{(end - pd.Timedelta(days=1)).strftime('W%V %G')}")
    elif granularity.freq == "MS":
        key, label = start.strftime("%Y-%m"), start.strftime("%b %Y")
    elif granularity.freq == "QS":
        quarter = (start.month - 1) // 3 + 1
        key = f"{start.year}-Q{quarter}"
        label = (f"Q{quarter} {start.year}" if granularity.step == 1
                 else f"H{1 if quarter <= 2 else 2} {start.year}")
    elif granularity.freq == "YS":
        key = label = str(start.year)
    else:                                             # uniform mode
        key = start.strftime("%Y-%m-%dT%H:%M")
        label = f"{start.strftime('%d %b %Y')} +"
    if g == "half_year":
        label = f"H{1 if start.month <= 6 else 2} {start.year}"
    return key, label


# ---------------------------------------------------------------------------
# Granularity choice
# ---------------------------------------------------------------------------

def _allowed(granularity: Granularity, time_resolution: str) -> bool:
    """Is this unit honest given how precise the underlying timestamps are?"""
    return _RESOLUTION_RANK[granularity.min_resolution] >= _RESOLUTION_RANK[time_resolution]


def choose_granularity(span_days: float, time_resolution: str, *,
                       target: int = FORMS_TARGET_BUCKETS,
                       min_buckets: int = FORMS_MIN_BUCKETS,
                       max_buckets: int = FORMS_MAX_BUCKETS,
                       submissions: int | None = None,
                       min_answers_per_period: int = FORMS_MIN_ANSWERS_PER_PERIOD
                       ) -> tuple[Granularity, list[str]]:
    """Pick the unit to split the period by.

    Two things decide it, not one.

    **Span**, as before: the count should land near `target`.

    **Volume**, which is new and matters more. Splitting 172 forms into 14 weeks
    leaves 12 answers a week. At that size a yes-rate wanders across a 50-point
    range by chance alone, so all fourteen numbers are noise and any "highest week"
    read out of them is invented. Splitting the same forms into 7 fortnights leaves
    25 each, which can at least show a large move. So a candidate whose periods
    would hold fewer than `min_answers_per_period` answers is penalised in
    proportion to how far short it falls.

    Preference on a tie: the finer unit, because more periods means more chance of
    seeing a real change - but only once each period holds enough to be read.
    """
    notes: list[str] = []
    candidates = [g for g in LADDER if _allowed(g, time_resolution)]
    if not candidates:
        candidates = [BY_KEY["day"]]

    scored: list[tuple[float, int, Granularity, int, float]] = []
    for index, granularity in enumerate(candidates):
        count = max(1, int(np.ceil(span_days / granularity.approx_days)))
        penalty = float(abs(count - target))
        if count < min_buckets or count > max_buckets:
            penalty += 1000            # out of range: only used if nothing fits
        per_period = (submissions / count) if submissions else None
        if per_period is not None and per_period < min_answers_per_period:
            # Weighted to outrank the bucket-count preference: an extra period is
            # worthless if none of them holds enough answers to interpret.
            penalty += 40.0 * (min_answers_per_period / max(per_period, 0.5) - 1)
        scored.append((penalty, index, granularity, count, per_period or 0.0))

    penalty, _, best, best_count, best_per_period = min(
        scored, key=lambda row: (row[0], row[1]))

    if penalty >= 1000:
        notes.append(
            f"the period is too {'short' if span_days < BY_KEY['day'].approx_days * min_buckets else 'long'} "
            f"to hit {min_buckets}-{max_buckets} buckets at any available unit; "
            f"using {best.label} buckets")

    if submissions:
        # What the date range alone would have chosen, ignoring volume. Naming that
        # unit is what makes the note true: "weekly would leave 13 answers a week".
        span_only = min(
            ((float(abs(row[3] - target))
              + (1000 if (row[3] < min_buckets or row[3] > max_buckets) else 0),
              row[1], row[2], row[3]) for row in scored),
            key=lambda row: (row[0], row[1]))
        span_choice, span_count = span_only[2], span_only[3]
        span_per_period = submissions / span_count if span_count else 0.0
        if span_choice.key != best.key and span_per_period < min_answers_per_period:
            notes.append(
                f"splitting by {span_choice.label} would leave about "
                f"{span_per_period:.0f} "
                f"{'answer' if round(span_per_period) == 1 else 'answers'} in each "
                f"one - too few to tell a real change from ordinary variation - so "
                f"{best.label} periods (about {best_per_period:.0f} "
                f"{'answer' if round(best_per_period) == 1 else 'answers'} each) "
                f"were used instead")
        elif best_per_period and best_per_period < min_answers_per_period:
            notes.append(
                f"each period holds only about {best_per_period:.0f} answers, "
                f"which is below the {min_answers_per_period} needed to separate a "
                f"real change from ordinary variation; treat period-to-period "
                f"differences as provisional")
    return best, notes


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _calendar_buckets(granularity: Granularity, start: pd.Timestamp,
                      end: pd.Timestamp) -> list[Bucket]:
    snapped = granularity.snap(start)
    offset = granularity.offset
    edges: list[pd.Timestamp] = [snapped]
    guard = 0
    while edges[-1] < end:
        edges.append(edges[-1] + offset)
        guard += 1
        if guard > 10_000:                            # cannot happen; cheap seatbelt
            raise RuntimeError("interval edge generation runaway")
    if len(edges) < 2:
        edges.append(snapped + offset)

    buckets: list[Bucket] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        key, label = _key_and_label(granularity, left, right)
        buckets.append(Bucket(index, key, label, left, right,
                              (right - left) / pd.Timedelta(days=1)))
    return buckets


def _anchored_buckets(granularity: Granularity, start: pd.Timestamp,
                      end: pd.Timestamp) -> list[Bucket]:
    """Buckets that begin at the requested start date and step by the unit.

    The alternative - snapping outwards to calendar boundaries - creates a leading
    and a trailing period only partly covered by the requested dates. Those two
    periods then carry a handful of answers each and get quoted as the highest and
    lowest of the whole range. Anchoring removes them: the first period starts
    exactly where the caller asked, and the last is clipped to the end, so every
    period is fully inside the range.
    """
    offset = granularity.anchored_offset
    edges: list[pd.Timestamp] = [start]
    guard = 0
    while edges[-1] < end:
        nxt = edges[-1] + offset
        edges.append(min(nxt, end) if nxt > end else nxt)
        guard += 1
        if guard > 10_000:
            raise RuntimeError("interval edge generation runaway")
    if len(edges) < 2:
        edges.append(end)

    buckets: list[Bucket] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        key, label = _range_key_and_label(left, right)
        buckets.append(Bucket(index, key, label, left, right,
                              (right - left) / pd.Timedelta(days=1), anchored=True))
    # A clipped tail shorter than a third of the unit holds too little to compare;
    # fold it into the period before it rather than reporting a stub.
    if len(buckets) > 1 and buckets[-1].days < buckets[0].days / 3:
        tail = buckets.pop()
        merged = buckets[-1]
        key, label = _range_key_and_label(merged.start, tail.end)
        buckets[-1] = Bucket(merged.index, key, label, merged.start, tail.end,
                             (tail.end - merged.start) / pd.Timedelta(days=1),
                             anchored=True)
    return buckets


def _uniform_buckets(start: pd.Timestamp, end: pd.Timestamp, count: int) -> list[Bucket]:
    total = end - start
    width = total / count
    sub_day = width < pd.Timedelta(days=1)
    fmt = "%d %b %H:%M" if sub_day else "%d %b %Y"
    buckets: list[Bucket] = []
    for index in range(count):
        left = start + width * index
        right = end if index == count - 1 else start + width * (index + 1)
        last = right - pd.Timedelta(seconds=1)
        buckets.append(Bucket(
            index,
            f"{index + 1:02d}|{left.strftime('%Y-%m-%dT%H:%M')}",
            f"P{index + 1}: {left.strftime(fmt)} - {last.strftime(fmt)}",
            left, right, (right - left) / pd.Timedelta(days=1)))
    return buckets


def build_plan(event_times: pd.Series | None = None, *,
               period_from: Any = None, period_to: Any = None,
               granularity: str = "auto",
               bucket_count: int | None = None,
               mode: str = "calendar",
               anchor: str = "auto",
               time_resolution: str = "day",
               target_buckets: int = FORMS_TARGET_BUCKETS,
               submissions: int | None = None,
               min_answers_per_period: int = FORMS_MIN_ANSWERS_PER_PERIOD
               ) -> IntervalPlan:
    """Resolve the period and cut it into buckets.

    `period_from`/`period_to` are inclusive dates as the user thinks of them; the
    plan stores `period_end` as an exclusive bound one unit past `period_to`.
    """
    notes: list[str] = []

    observed_min = observed_max = None
    if event_times is not None and len(event_times):
        values = pd.to_datetime(event_times, errors="coerce").dropna()
        if len(values):
            observed_min, observed_max = values.min(), values.max()

    start = pd.Timestamp(period_from) if period_from is not None else observed_min
    end_inclusive = pd.Timestamp(period_to) if period_to is not None else observed_max
    if start is None or end_inclusive is None:
        raise ValueError("cannot build an interval plan: no dates supplied and no "
                         "timestamped responses in the filtered data")
    if start > end_inclusive:
        start, end_inclusive = end_inclusive, start
        notes.append("period_from was after period_to; the two were swapped")

    # The upper bound is exclusive internally, so it has to sit strictly after the
    # last thing being counted. An inclusive date with no time means "the whole of
    # that day"; a bound taken from the data itself is nudged past its own last
    # observation, or that observation falls outside the grid it defined.
    end = end_inclusive
    if end == end.normalize():
        end = end + pd.Timedelta(days=1)
    elif period_to is None:
        end = end + pd.Timedelta(seconds=1)
    span_days = max((end - start) / pd.Timedelta(days=1), 1e-9)

    requested = str(granularity or "auto").lower()
    if requested in ("auto", "", "none"):
        chosen, auto_notes = choose_granularity(
            span_days, time_resolution, target=bucket_count or target_buckets,
            min_buckets=1 if bucket_count else FORMS_MIN_BUCKETS,
            max_buckets=bucket_count if bucket_count else FORMS_MAX_BUCKETS,
            submissions=submissions,
            min_answers_per_period=min_answers_per_period)
        notes += auto_notes
    else:
        if requested not in BY_KEY:
            raise ValueError(f"unknown granularity {granularity!r}; "
                             f"expected 'auto' or one of {sorted(set(BY_KEY))}")
        chosen = BY_KEY[requested]
        if not _allowed(chosen, time_resolution):
            finer = next(g for g in LADDER if _allowed(g, time_resolution))
            notes.append(
                f"{chosen.label} buckets were requested but the timestamps are "
                f"{time_resolution}-precision; using {finer.label} instead so the "
                f"report does not invent sub-{time_resolution} detail")
            chosen = finer

    # Anchoring: "calendar" snaps outwards to natural boundaries, which creates a
    # part-covered period at each end; "period" starts exactly where the caller
    # asked. "auto" uses the calendar only when the requested start already falls
    # on a boundary of the chosen unit, so the readable case keeps its readable
    # labels and every other case avoids the stub periods.
    resolved_anchor = str(anchor or "auto").lower()
    if resolved_anchor == "auto":
        # Snapping only carries meaning for units with named boundaries (weeks,
        # months, quarters, years) and only when the caller's start is already on
        # one. Every hour is an hour boundary and every day a day boundary, so
        # snapping those achieves nothing and only produces a clipped tail period.
        nameable = chosen.freq in ("W-MON", "MS", "QS", "YS")
        # An explicit request for "week" or "month" is a request for *named* weeks
        # and months, so it snaps even at the cost of part-covered edges (which are
        # flagged and excluded from comparisons). Automatic selection prefers whole
        # periods inside the requested dates over pretty labels.
        explicit = requested not in ("auto", "", "none")
        resolved_anchor = (
            "calendar" if nameable and (explicit or chosen.snap(start) == start)
            else "period")

    if mode == "uniform":
        count = bucket_count or int(np.clip(round(span_days / chosen.approx_days),
                                            FORMS_MIN_BUCKETS, FORMS_MAX_BUCKETS))
        buckets = _uniform_buckets(start, end, max(int(count), 1))
        period_start, period_end = start, end
    elif resolved_anchor == "period":
        buckets = _anchored_buckets(chosen, start, end)
        if bucket_count and len(buckets) > bucket_count:
            buckets = [Bucket(i, b.key, b.label, b.start, b.end, b.days, True)
                       for i, b in enumerate(buckets[-bucket_count:])]
            notes.append(f"trimmed to the most recent {bucket_count} periods")
        period_start, period_end = buckets[0].start, buckets[-1].end
    else:
        buckets = _calendar_buckets(chosen, start, end)
        if bucket_count and len(buckets) > bucket_count:
            buckets = buckets[-bucket_count:]          # keep the most recent N
            buckets = [Bucket(i, b.key, b.label, b.start, b.end, b.days)
                       for i, b in enumerate(buckets)]
            notes.append(f"trimmed to the most recent {bucket_count} buckets")
        period_start, period_end = buckets[0].start, buckets[-1].end
        if period_start < start:
            notes.append(f"period widened to {period_start.date()} so buckets align "
                         f"to {chosen.label} boundaries")

    window = period_end - period_start
    previous_start, previous_end = period_start - window, period_start

    plan = IntervalPlan(granularity=chosen, mode=mode, buckets=buckets,
                        period_start=period_start, period_end=period_end,
                        previous_start=previous_start, previous_end=previous_end,
                        time_resolution=time_resolution,
                        requested_granularity=requested,
                        covered_start=start, covered_end=end, notes=notes)
    plan.anchor = resolved_anchor
    partial = [b.label for b in buckets if plan.coverage_of(b) < 0.9]
    if partial:
        notes.append(
            f"{len(partial)} period(s) at the edges are only partly inside the "
            f"requested dates ({', '.join(partial[:4])}); their per-day figures are "
            f"calculated over the days actually covered, and they are left out of "
            f"the comparison between periods")
    short = [b.label for b in buckets if plan.is_short(b)]
    if short:
        notes.append(
            f"the last period ({short[-1]}) is shorter than the others because the "
            f"chosen dates do not divide evenly; shares and rates are still "
            f"comparable, and counts are also given per day")
    log.info("interval plan: %s x %s (%s, %s-anchored) over %s..%s",
             plan.bucket_count, chosen.label, mode, resolved_anchor,
             period_start.date(), period_end.date())
    return plan


def assign_buckets(event_times: pd.Series, plan: IntervalPlan) -> pd.Series:
    """Bucket index per row; -1 for rows outside the plan or with no timestamp."""
    values = pd.to_datetime(event_times, errors="coerce").to_numpy(
        dtype="datetime64[ns]")
    edges = plan.edges
    index = np.searchsorted(edges, values, side="right") - 1
    outside = np.isnat(values) | (values < edges[0]) | (values >= edges[-1])
    index = np.where(outside, -1, index)
    return pd.Series(index.astype(int), index=event_times.index, name="bucket")
