"""Plain language. The only module allowed to produce user-facing wording.

Nothing outside this file decides how a number is written, and nothing downstream
of it is permitted to emit statistical vocabulary. That means no `yes_rate`, no
`delta 0.5714`, no `tau`, no `p_value`, no `NUMERIC`/`MULTIPLE_ANSWER`, no
"median"/"standard deviation", no spark bars, and no "erratic"/"flat (low
confidence)". Those all still exist in `result.statistics` for programmatic
callers - they just never reach a reader or the model.

The rules:

* A proportion is written as a percentage: `0.5714` -> `"57%"`.
* A count is written with thousands separators: `9195352` -> `"9,195,352"`.
* A measured value keeps at most one decimal, and only when it needs one.
* A movement is a sentence, not a symbol: "rose from 12 to 19 by the end".
* Statistical confidence becomes ordinary hedging: a rank test that is not
  significant becomes "no clear change", not "flat (low confidence)".
* When the answers are too spread out for one number, the wording says so
  instead of quoting a value that averages 10 and 100 into 55.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Words that must never appear in user-facing output. `tests/test_plain_language.py`
# asserts this over the rendered report, the agent payload and the instructions.
BANNED_TERMS: tuple[str, ...] = (
    "yes_rate", "no_rate", "response_rate", "reference_share", "modal_share",
    "avg_selections", "median_selections", "primary_value", "primary_metric",
    "effective_type", "declared_type", "question_type", "low_n", "bucket",
    "p_value", "p-value", "tau", "mann", "kendall", "ols", "r_squared",
    "std", "stddev", "standard deviation", "median", "iqr", "qcd",
    # "mean" only in its statistical senses: the bare word is an ordinary verb
    # ("this could mean that ...") and banning it would fail every clean run.
    "mean of", "the mean", "arithmetic mean", "mean value",
    "coefficient of variation", "wilson", "sparkline", "erratic", "volatility",
    "delta", "NUMERIC", "YES_NO", "MULTIPLE_ANSWER", "SINGLE_ANSWER",
    "TEXT", "RATING", "granularity", "Filters:", "dispersion", "outlier",
    "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█",
)


def contains_banned(text: str) -> list[str]:
    """Which banned terms appear in `text`. Empty list is the goal.

    Two matching modes. Form-builder type names ("TEXT", "RATING") and the block
    characters are matched literally, so "a follow-up text/email" is not a hit.
    Everything else is matched case-insensitively but must not sit inside a longer
    word - "tau" does not fire on "Tauranga" - while still catching a longer field
    name built from it, since `_` is not a letter and "yes_rate_floor" must fail.

    Used both by the test sweep over our own output and at runtime over the
    model's, where a hit is folded into the one repair pass.
    """
    haystack = str(text or "")
    found = []
    for term in BANNED_TERMS:
        if term.isupper() or not term.isascii():
            if term in haystack:                        # literal, case-sensitive
                found.append(term)
            continue
        if re.search(rf"(?<![a-zA-Z]){re.escape(term)}(?![a-zA-Z])",
                     haystack, re.I):
            found.append(term)
    return found


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def number(value: Any, decimals: int | None = None) -> str:
    """A measured value: thousands separators, at most one decimal unless asked."""
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if not isinstance(value, (int, float)):
        return str(value)
    if decimals is not None:
        return f"{value:,.{decimals}f}"
    if float(value).is_integer():
        return f"{int(value):,}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.1f}" if abs(value) >= 1 else f"{value:.2f}"


def percent(value: Any, decimals: int = 0) -> str:
    """A proportion as a percentage. `0.5714` -> `"57%"`."""
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value * 100:.{decimals}f}%"


def count(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def value_of(raw: Any, kind: str) -> str:
    """Format by measure kind: `rate` -> percentage, `count` -> integer, else a
    measured value."""
    if raw is None:
        return "-"
    if kind == "rate":
        return percent(raw)
    if kind == "count":
        return count(raw)
    return number(raw)


def range_of(low: Any, high: Any, kind: str = "value",
             joiner: str = "to") -> str | None:
    """"131 to 111,001", or with joiner="and" for "between 131 and 111,001"."""
    if low is None or high is None:
        return None
    if low == high:
        return value_of(low, kind)
    return f"{value_of(low, kind)} {joiner} {value_of(high, kind)}"


def change_phrase(before: Any, after: Any, kind: str,
                  suffix: str | None = None) -> str | None:
    """"57% up to 71%" - the two values, never a raw difference.

    `suffix` names the unit once at the end ("1.4 down to 0.9 forms a day") so a
    bare rate does not arrive without saying what it counts.
    """
    if before is None or after is None:
        return None
    tail = f" {suffix}" if suffix else ""
    if before == after:
        return f"unchanged at {value_of(after, kind)}{tail}"
    direction = "up to" if after > before else "down to"
    return (f"{value_of(before, kind)} {direction} "
            f"{value_of(after, kind)}{tail}")


MAX_NAME_CHARS = 30


def person(name: Any, limit: int = MAX_NAME_CHARS) -> str:
    """A name fit for a table cell.

    The reference export contained a 63-character name that broke every table it
    appeared in and left the model calling someone "the very long-named person".
    Truncation happens here so the exact value stays in `statistics`.
    """
    text = str(name or "").strip() or "(unnamed)"
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "\u2026"


# ---------------------------------------------------------------------------
# Dates and periods
# ---------------------------------------------------------------------------

def _day(ts: Any) -> str:
    stamp = pd.Timestamp(ts)
    return f"{stamp.day} {stamp.strftime('%B %Y')}"


def period_phrase(start: Any, end_exclusive: Any) -> str:
    """"1 January to 31 March 2025", inclusive as a reader expects."""
    first = pd.Timestamp(start)
    last = pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)
    if first.date() == last.date():
        return _day(first)
    if first.year == last.year:
        if first.month == last.month:
            return f"{first.day}-{last.day} {last.strftime('%B %Y')}"
        return (f"{first.day} {first.strftime('%B')} to "
                f"{last.day} {last.strftime('%B %Y')}")
    return f"{_day(first)} to {_day(last)}"


#: Interval-unit key -> the word used when talking about one of them.
_UNIT_WORDS = {
    "hour": "hour", "3h": "3-hour block", "6h": "6-hour block",
    "12h": "half-day", "day": "day", "2day": "2-day block", "week": "week",
    "2week": "fortnight", "month": "month", "quarter": "quarter",
    "half_year": "half-year", "year": "year",
}


def unit_word(key: str, plural: bool = False) -> str:
    word = _UNIT_WORDS.get(key, "period")
    if not plural:
        return word
    return word + ("es" if word.endswith("s") else "s")


def period_label(bucket: dict[str, Any], unit_key: str) -> str:
    """A reader-friendly label for one interval.

    Week labels become date ranges: "W13 2025" means nothing to most people,
    "24-30 Mar" does.
    """
    start = pd.Timestamp(bucket["start"])
    if bucket.get("anchored") and unit_key not in ("hour", "3h", "6h", "12h"):
        # An anchored grid does not sit on named boundaries: a period running
        # 15 Jan to 14 Feb is not "Jan 2025", so it is labelled by its dates.
        last = start + pd.Timedelta(days=max(bucket.get("days", 1) - 1, 0))
        if start.date() == last.date():
            return f"{start.day} {start.strftime('%b')}"
        if start.month == last.month:
            return f"{start.day}-{last.day} {last.strftime('%b')}"
        return f"{start.day} {start.strftime('%b')} - {last.day} {last.strftime('%b')}"
    if unit_key in ("hour", "3h", "6h", "12h"):
        return start.strftime("%H:%M")
    if unit_key == "day":
        return f"{start.day} {start.strftime('%b')}"
    if unit_key in ("2day", "week", "2week"):
        end = pd.Timestamp(bucket["start"]) + pd.Timedelta(days=bucket["days"] - 1)
        if start.month == end.month:
            return f"{start.day}-{end.day} {end.strftime('%b')}"
        return f"{start.day} {start.strftime('%b')} - {end.day} {end.strftime('%b')}"
    if unit_key == "month":
        return start.strftime("%b %Y")
    if unit_key == "quarter":
        return f"Q{(start.month - 1) // 3 + 1} {start.year}"
    if unit_key == "half_year":
        return f"{'Jan' if start.month <= 6 else 'Jul'}-" \
               f"{'Jun' if start.month <= 6 else 'Dec'} {start.year}"
    return str(start.year)


# ---------------------------------------------------------------------------
# How a question was answered (never the internal type name)
# ---------------------------------------------------------------------------

_ANSWER_STYLE = {
    "NUMERIC": "a number",
    "RATING": "a rating",
    "YES_NO": "yes or no",
    "SINGLE_ANSWER": "one answer from a list",
    "MULTIPLE_ANSWER": "any number of options from a list",
    "TEXT": "written in freely",
    "DATE": "a date",
}


def answer_style(effective_type: str) -> str:
    return _ANSWER_STYLE.get(effective_type, "written in freely")


def retype_note(declared: str, effective: str, reason: str | None) -> str | None:
    """Explain a re-typed question without naming a single internal type."""
    if not reason:
        return None
    return (f"The form asks for this as {answer_style(declared)}, but the answers "
            f"are actually {answer_style(effective)}, so it is reported that way.")


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------
# `trend.direction` and `trend.confidence` come from a rank-based test. Here they
# become ordinary English, and a result that is not significant is written as
# "no clear change" rather than dressed up with a statistic.

def movement_phrase(trend: dict[str, Any], kind: str, unit_key: str,
                    suffix: str | None = None) -> str:
    """How this measure moved - and, first, whether it moved at all.

    The important branch is the second one. When every period sits inside the range
    its own sample size would produce anyway, the honest sentence is that the
    differences are ordinary, not that one period was the highest. That is the
    sentence the old report was missing, and its absence is why "peaked in
    mid-February" appeared in a summary describing pure noise.
    """
    direction = (trend or {}).get("direction") or "insufficient_data"
    confidence = (trend or {}).get("confidence") or "none"
    first, last = trend.get("first_bucket_value"), trend.get("last_bucket_value")
    change = change_phrase(first, last, kind, suffix)
    unit = unit_word(unit_key)

    if direction == "insufficient_data":
        return "Not enough answers spread over time to say whether this changed."

    if trend.get("variation_within_normal_range") and direction not in (
            "rising", "falling"):
        answers = trend.get("typical_period_answers")
        detectable = trend.get("smallest_detectable_change")
        sentence = (f"The figure moves from one {unit} to the next, but never by "
                    f"more than would happen anyway with this many answers, so "
                    f"there is no real change here")
        if answers:
            sentence += f" (about {count(answers)} answers per {unit}"
            if detectable and kind == "rate":
                sentence += (f"; only a swing of about {percent(detectable)} "
                             f"would show up as real")
            sentence += ")"
        return sentence + "."
    if direction in ("rising", "falling"):
        word = "higher" if direction == "rising" else "lower"
        strength = ("clearly and consistently" if confidence == "high"
                    else "consistently")
        sentence = f"Ended the period {strength} {word} than it started"
        return f"{sentence} ({change})." if change else f"{sentence}."
    if direction in ("possibly_rising", "possibly_falling"):
        word = "upwards" if direction == "possibly_rising" else "downwards"
        sentence = (f"Drifted {word}, but not consistently enough from one {unit} "
                    f"to the next to call it a real change")
        return f"{sentence} ({change})." if change else f"{sentence}."
    if change and first != last:
        return (f"No clear direction - it moved {change} but went up and down "
                f"along the way.")
    return "No clear change across the period."


def notable_periods_phrase(trend: dict[str, Any], kind: str) -> str | None:
    """Name only the periods that stand outside their expected range."""
    rows = (trend or {}).get("notable_periods") or []
    if not rows:
        return None
    above = [r for r in rows if r.get("direction") == "above"]
    below = [r for r in rows if r.get("direction") == "below"]
    parts = []
    if above:
        parts.append("higher than the rest in "
                     + ", ".join(f"{r['label']} ({value_of(r['value'], kind)})"
                                 for r in above[:3]))
    if below:
        parts.append("lower in "
                     + ", ".join(f"{r['label']} ({value_of(r['value'], kind)})"
                                 for r in below[:3]))
    return "Stands out as " + " and ".join(parts) + "."


def steadiness_phrase(volatility: dict[str, Any] | None, unit_key: str) -> str | None:
    """"Held steady from week to week" / "jumped around a lot" - no jargon."""
    label = (volatility or {}).get("label")
    unit = unit_word(unit_key)
    if (volatility or {}).get("within_normal_range"):
        return None            # already said, and saying it twice implies a finding
    if label == "stable":
        return f"Held steady from one {unit} to the next."
    if label == "moderate":
        return f"Moved around somewhat from one {unit} to the next."
    if label in ("erratic",):
        return f"Swung widely from one {unit} to the next, with no settled level."
    return None


def reliability_phrase(answers: int, thin_periods: int, unit_key: str) -> str:
    """State the evidence base plainly, and warn about thin intervals."""
    base = f"Based on {count(answers)} answer{'' if answers == 1 else 's'}."
    if thin_periods:
        base += (f" {count(thin_periods)} {unit_word(unit_key, thin_periods != 1)} "
                 f"had too few answers to read anything into.")
    return base


# ---------------------------------------------------------------------------
# Spread: the answer to the 10-and-100 problem
# ---------------------------------------------------------------------------

def spread_phrase(dispersion: dict[str, Any] | None, low: Any, high: Any,
                  kind: str = "value", trending: bool = False) -> str | None:
    """How the answers are spread, and whether one value may stand for them.

    `trending` says the spread is wide only *between* periods, not inside them -
    a question rising 100 to 600 has a six-month range as wide as a question
    mixing 30s and 90,000s, and describing the first as "too widely spread for one
    figure" hides the very thing that makes it interesting.
    """
    if not dispersion:
        return None
    verdict = dispersion.get("verdict")
    shape = dispersion.get("shape")
    if trending:
        full_range = range_of(low, high, kind)
        return (f"Answers cover a wide range over the whole period ({full_range}) "
                f"because the figure moved, not because they disagree: within any "
                f"one period they sit close together.") if full_range else None
    usual = range_of(dispersion.get("typical_low"), dispersion.get("typical_high"),
                     kind, joiner="and")
    full = range_of(low, high, kind)

    if verdict == "no_answers":
        return None
    if verdict == "too_few_answers":
        return f"Too few answers to describe a range (all answers: {full})." \
            if full else None
    if shape == "two_groups":
        return (f"The answers fall into two clearly separate groups rather than "
                f"around one typical value, so no single figure represents them. "
                f"They range from {full}.")
    if verdict == "wide":
        return (f"The answers are spread far too widely for one figure to stand "
                f"for them - from {full}, with the middle half between {usual}.")
    if shape == "long_tail":
        return (f"Most answers sit between {usual}, with a few much larger ones "
                f"stretching up to {value_of(high, kind)}.")
    if verdict == "moderate":
        return f"Most answers fall between {usual} (full range {full})."
    if verdict == "tight":
        return f"The answers cluster tightly, almost all between {usual}."
    return None


def headline_phrase(measure: str, value: Any, kind: str,
                    dispersion: dict[str, Any] | None,
                    trending: bool = False) -> str:
    """The one-line answer to "what did people say?".

    When the answers do not cluster, this deliberately does **not** quote a single
    value - that is the whole point of the change. A question whose spread comes
    from moving over time is not that case: each period has a perfectly good
    figure, so one is quoted and the movement sentence explains the range.
    """
    if not trending and dispersion \
            and dispersion.get("single_value_representative") is False \
            and dispersion.get("verdict") not in (None, "tight", "moderate"):
        usual = range_of(dispersion.get("typical_low"), dispersion.get("typical_high"),
                         kind, joiner="and")
        return (f"No single {measure} represents these answers"
                + (f"; the middle half sit between {usual}." if usual else "."))
    formatted = value_of(value, kind)
    if formatted == "-":
        return "No answers to summarise."
    return f"{measure.capitalize()}: {formatted}"


def bands_phrase(bands: list[dict[str, Any]] | None, kind: str = "value") -> str | None:
    """"1 in 4 answered 10 or less, 1 in 4 answered 77 or more" - a distribution
    a reader can picture, without a histogram."""
    if not bands or len(bands) < 2:
        return None
    parts = []
    for band in bands:
        share = band.get("share")
        if not share:
            continue
        parts.append(f"{percent(share)} between {value_of(band['from'], kind)} "
                     f"and {value_of(band['to'], kind)}")
    return "; ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Free-text content
# ---------------------------------------------------------------------------

def amount_phrase(summary: dict[str, Any]) -> str | None:
    """What the amounts written into a text field add up to.

    "Did you secure an order? If so, how much?" was previously reported as
    "answered this question: 100%". This is the sentence that replaces it.
    """
    share = summary.get("amount_share")
    if not summary.get("amount_count"):
        return None
    parts = [f"{percent(share)} of forms named an amount"]
    middle = summary.get("amount_median")
    if middle is not None:
        parts.append(f"typically {number(middle)}")
    total = summary.get("amount_total")
    if total is not None:
        parts.append(f"{number(total)} in total")
    spread = range_of(summary.get("amount_lowest"), summary.get("amount_highest"))
    if spread:
        parts.append(f"ranging {spread}")
    return ", ".join(parts) + "."


def compliance_phrase(summary: dict[str, Any]) -> str | None:
    """Whether answers met the minimum the question itself asks for."""
    minimum = summary.get("required_minimum")
    rate = summary.get("compliance_rate")
    if not minimum or rate is None:
        return None
    listed = (summary.get("items_listed") or {}).get("median")
    sentence = (f"The question asks for at least {count(minimum)}; "
                f"{percent(rate)} of answers listed that many")
    if listed is not None:
        sentence += f", with {number(listed)} listed in a typical answer"
    return sentence + "."


def templating_phrase(templating: dict[str, Any] | None) -> str | None:
    """Answers that share one shape once the numbers are stripped out.

    Distinct text with identical structure is the signature of boilerplate - or, as
    in the reference export, of generated data that a duplicate check reads as 0%
    duplicated.
    """
    if not templating or not templating.get("checked"):
        return None
    if not templating.get("looks_generated"):
        return None
    share = templating.get("template_share")
    example = templating.get("dominant_example")
    return (f"{percent(share)} of the answers follow one identical pattern with "
            f"only the numbers changing (for example \"{example}\"), so they look "
            f"filled in automatically rather than written by a person.")


def themes_phrase(themes: list[dict[str, Any]] | None) -> str | None:
    if not themes:
        return None
    return "; ".join(f"{row['theme']} ({count(row['answers'])} answers)"
                     for row in themes[:4])


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------

def people_phrase(segment: dict[str, Any] | None, kind: str) -> str:
    """Whether the people genuinely differ, in the terms a reader can act on.

    Three separate reasons not to name anybody, all of which the old report ignored:
    too few answers to say, differences no larger than sample size explains, and a
    difference that fails a test against the rest of the team.
    """
    if not segment or not segment.get("rows"):
        return "No answers to compare between people."

    comparable = segment.get("comparable_people") or 0
    excluded = segment.get("excluded_too_few") or 0
    tail = ""
    if excluded:
        tail = (f" {count(excluded)} "
                f"{'person' if excluded == 1 else 'people'} answered too few times "
                f"to be compared.")

    if comparable < 2:
        return ("Too few people answered enough times to compare them." + tail).strip()
    if not segment.get("differ"):
        return ("Everyone is within the normal range of each other on this - the "
                "differences are no larger than the number of answers each person "
                "gave would produce anyway." + tail).strip()

    standouts = [row for row in segment["rows"] if row.get("stands_out")]
    named = "; ".join(
        f"{person(row.get('representative_label') or row.get('customer_id'))} at "
        f"{value_of(row.get('raw_value', row.get('primary_value')), kind)}"
        for row in standouts[:3])
    return (f"{named} sits far enough from the rest of the team to be a real "
            f"difference rather than chance.{tail}").strip()


def dominance_phrase(dominance: dict[str, Any] | None) -> str | None:
    """One person carrying the team's figures is a fact about the figures."""
    if not dominance or not dominance.get("dominated"):
        return None
    return (f"{person(dominance.get('top_person'))} submitted "
            f"{percent(dominance.get('top_share'))} of all the forms, so the overall "
            f"figures largely describe their work rather than the team's. Each "
            f"question also shows a figure with every person counted once.")


def test_account_phrase(names: list[str] | None) -> str | None:
    """Test accounts in a live report are worth removing, not narrating."""
    if not names:
        return None
    shown = ", ".join(f'"{person(name)}"' for name in names[:4])
    return (f"{count(len(names))} of the accounts look like test or placeholder "
            f"entries ({shown}). They are included in these figures - exclude them "
            f"to see the real team's numbers.")


# ---------------------------------------------------------------------------
# How the figure has moved: the stronger readings
# ---------------------------------------------------------------------------

def level_phrase(progress: dict[str, Any], kind: str, unit_key: str,
                 representative: bool = True) -> str | None:
    """Whether the figure sits at one level or genuinely moves between periods.

    This is the distinction a manager needs and the old wording collapsed. A check
    steady at 52% every fortnight is a process running at 52%: change the process.
    A check swinging between fortnights means something differs between them: find
    out what. "No real change here" was true of both and useful for neither.
    """
    holds = (progress or {}).get("holds_one_level") or {}
    steady = holds.get("steady")
    if steady is None:
        return None
    swing = (progress or {}).get("swing") or {}
    unit = unit_word(unit_key)
    units = unit_word(unit_key, plural=True)
    low, high = swing.get("low"), swing.get("high")

    # Where no single figure represents the answers, the figure's own swing is an
    # artefact - the middle of two groups lands in the gap between them and jumps
    # from one to the other on a single extra answer. Quoting that swing as though
    # it were movement is the thing this whole branch exists to avoid.
    if not representative:
        if steady:
            return (f"The same mix of answers appears in every {unit}. Any middle "
                    "figure jumps about only because the middle of two separate "
                    "groups falls in the gap between them.")
        return f"The mix of answers genuinely differs between {units}."

    span = ""
    if low is not None and high is not None and low != high:
        span = f" it ran between {value_of(low, kind)} and {value_of(high, kind)},"

    if steady:
        if low == high and low is not None:
            return f"Identical in every {unit}, at {value_of(low, kind)}."
        return (f"Steady at one level all period: across the {units}{span} and "
                f"tested over all of them at once that is ordinary variation, not "
                f"the figure moving. Shifting it means changing how the work is "
                f"done rather than chasing one {unit}.")
    return (f"This genuinely moves between {units} by more than chance explains:"
            f"{span} so it is worth asking what differed at each end.")


def halves_phrase(progress: dict[str, Any], kind: str,
                  unit_key: str = "period") -> str | None:
    """First half of the window against the second.

    Period-by-period comparison is the weakest test available; pooling into two
    halves doubles the answers on each side and so detects a shift roughly half
    the size. Where the halves agree, saying so is a real finding - it rules out
    a drift that the per-period view is too thin to rule out.
    """
    halves = (progress or {}).get("halves") or {}
    first, second = halves.get("first") or {}, halves.get("second") or {}
    if first.get("value") is None or second.get("value") is None:
        return None
    before, after = value_of(first["value"], kind), value_of(second["value"], kind)
    differs = halves.get("differs")
    if first["value"] == second["value"]:
        return f"Both halves of the period are identical, at {before}."
    if differs:
        moved = "up" if second["value"] > first["value"] else "down"
        return (f"Across the two halves of the period it really has moved "
                f"{moved}: {before} then {after}.")
    return (f"The two halves of the period agree ({before} then {after}), which "
            f"rules out a drift too small to see one {unit_word(unit_key)} at a "
            f"time.")


def target_phrase(target: dict[str, Any] | None, kind: str,
                  unit_key: str = "period") -> str | None:
    """Whether the target is met, and what closing the gap costs.

    The shortfall is given in forms because that is the thing a manager can
    assign. "Twenty-eight percentage points below" is arithmetic; "48 of these
    172 forms would have had to go the other way" is a workload.
    """
    if not target:
        return None
    met, level = target.get("met"), target.get("value")
    if met is None or level is None:
        return None
    goal = percent(level) if target.get("as_share") else number(level)
    if met:
        return f"This meets the {goal} it is expected to reach."

    sentence = f"Short of the {goal} expected"
    forms = target.get("shortfall_forms")
    if forms:
        sentence += (f": {count(forms)} more answers would have had to go the "
                     f"other way")
    sentence += "."
    compared = target.get("periods_compared") or 0
    if target.get("never_reached") and compared >= 2:
        sentence += (f" Not reached in any of the {count(compared)} "
                     f"{unit_word(unit_key, plural=True)}.")
    elif compared >= 2 and target.get("periods_reaching_it"):
        sentence += (f" It was reached in {count(target['periods_reaching_it'])} "
                     f"of {count(compared)}.")
    return sentence


def band_row_label(band: dict[str, Any]) -> str:
    """A value band as a reader sees it: "1,459 to 1,886"."""
    return f"{number(band.get('from'))} to {number(band.get('to'))}"


def bands_intro(dispersion: dict[str, Any] | None) -> str:
    """Why a table of bands is being shown instead of one figure."""
    shape = (dispersion or {}).get("shape")
    if shape == "two_groups":
        return ("The answers fall into two separate groups, so no single figure "
                "stands for them. This is how they are spread instead, and how "
                "that spread has changed:")
    return ("No single figure represents these answers well, so this is how they "
            "are spread, and how that spread has changed:")


def generated_answers_phrase(templating: dict[str, Any] | None) -> str | None:
    """Say plainly that the answers cannot be read as what people reported."""
    if not isinstance(templating, dict) or not templating.get("looks_generated"):
        return None
    share = templating.get("template_share")
    example = templating.get("dominant_example")
    sentence = "The written answers here all follow one pattern"
    if share is not None:
        sentence += f" ({percent(share)} of them)"
    if example:
        sentence += f', for example "{example}"'
    return (sentence + ". They look filled in automatically rather than written "
            "by a person, so nothing here should be read as what the team "
            "actually reported.")
