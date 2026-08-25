"""The reader's view of the statistics.

One structure, built once, consumed by both the Markdown renderer and the agent
payload builder. That is deliberate: if the report and the model read the same
plain-language view, they cannot disagree, and a jargon term removed here is
removed from both at once.

Shape:

    {
      "period": "1 January to 31 March 2025",
      "scope": "8 team members, 172 forms submitted",
      "activity": {...},
      "questions": [ {one block per question, self-contained} ],
      "quality_notes": [...],
    }

Each question block carries everything that belongs to that question - its
headline, how it moved period by period, and which people differ on it - because
those are one story and were previously three sections apart.

Values are pre-formatted strings ("57%", "1,886"). Two consequences: nothing
downstream can render `0.5714`, and the model receives short tokens it can only
quote, which both cuts input size and tightens the grounding check.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_log
from . import metrics as MET
from . import phrasing as PH

log = get_log("forms.presentation")


# ---------------------------------------------------------------------------
# Accounts that are not real people
# ---------------------------------------------------------------------------
# The reference export carried "Qwe qwe", "Reps TnC" and a 63-character run of
# consonants through every table, and the reviewer ended up describing one of them
# as "the very long-named person". Flagging them is more useful than narrating
# them, and `exclude_representatives` removes them once the caller agrees.

_KEYBOARD_RUNS = ("qwe", "asdf", "zxc", "qaz", "wsx", "1234", "abcd")
_TEST_WORDS = ("test", "demo", "sample", "dummy", "example", "tnc", "temp",
               "delete", "asdf", "foo", "bar")
_VOWELS = set("aeiouyаеиоуяюієї")


def looks_like_test_account(name: Any) -> str | None:
    """Why this name looks like a test entry, or None if it looks like a person."""
    text = str(name or "").strip()
    if not text:
        return "no name recorded"
    lowered = text.lower()
    words = lowered.split()

    if any(word in _TEST_WORDS for word in words):
        return "contains a word normally used for test accounts"
    if any(run in lowered for run in _KEYBOARD_RUNS):
        return "contains a run of adjacent keyboard keys"
    if len(words) >= 2 and len(set(words)) == 1:
        return "the same word repeated"
    if len(text) > 45:
        return "far longer than a real name"
    long_words = [word for word in words if len(word) >= 4]
    if long_words:
        vowel_share = sum(
            1 for word in long_words for char in word if char in _VOWELS
        ) / sum(len(word) for word in long_words)
        if vowel_share < 0.2:
            return "almost no vowels, so probably typed at random"
    return None


def find_test_accounts(statistics: dict[str, Any]) -> list[dict[str, str]]:
    rows = (statistics.get("overview") or {}).get(
        "submissions_by_representative") or []
    out = []
    for row in rows:
        reason = looks_like_test_account(row.get("representative"))
        if reason:
            out.append({"name": row["representative"], "reason": reason})
    return out


# ---------------------------------------------------------------------------
# Scope, in words rather than a filter dump
# ---------------------------------------------------------------------------

def _scope_phrase(statistics: dict[str, Any], filters: dict[str, Any] | None) -> str:
    overview = statistics["overview"]
    applied = (filters or {}).get("applied") or {}
    people = overview.get("representatives") or 0
    parts = [f"{PH.count(overview.get('submissions'))} forms submitted",
             f"{PH.count(people)} team member{'' if people == 1 else 's'}"]

    named = applied.get("representative_id")
    if named:
        names = [row["representative"] for row
                 in overview.get("submissions_by_representative", [])]
        parts.insert(0, f"only {', '.join(names) if names else 'one team member'}")
    if applied.get("completed_only"):
        parts.append("completed forms only")
    if applied.get("include_autofilled") is False:
        parts.append("answers filled in by hand only")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------

def _label_map(statistics: dict[str, Any], unit_key: str) -> dict[str, str]:
    """Internal period label -> the label a reader sees.

    Trend fields such as the strongest and weakest period come back from the
    statistics carrying internal labels like "W13 2025". Everything a reader sees
    goes through this map, so "24-30 Mar" is what reaches the page.
    """
    return {bucket["label"]: PH.period_label(bucket, unit_key)
            for bucket in statistics["intervals"]["buckets"]}


def _activity(statistics: dict[str, Any], unit_key: str,
              labels: dict[str, str]) -> dict[str, Any]:
    overview = statistics["overview"]
    rows = overview.get("by_interval", [])
    per_period = [{
        "period": PH.period_label(row, unit_key),
        "forms": PH.count(row["submissions"]),
        "answers": PH.count(row["responses"]),
        "each_day": PH.number(row.get("submissions_per_day"), 1),
        "people": PH.count(row.get("representatives")),
        "too_few": bool(row.get("low_n")),
        "part_period": bool(row.get("partial")),
        "short_period": bool(row.get("short")),
        "stands_out": bool(row.get("notable")),
    } for row in rows]

    empty = overview.get("empty_intervals") or []
    unit_plural = PH.unit_word(unit_key, True)
    volume_trend = dict(overview.get("volume_trend") or {})
    volume_trend["variation_within_normal_range"] = overview.get(
        "volume_within_normal_range")
    volume_trend["typical_period_answers"] = None
    lines = [PH.movement_phrase(volume_trend, "value", unit_key,
                                suffix="forms a day")]
    steadiness = PH.steadiness_phrase(
        {**(overview.get("volume_volatility") or {}),
         "within_normal_range": overview.get("volume_within_normal_range")},
        unit_key)
    if steadiness:
        lines.append(steadiness)
    if empty:
        lines.append(f"Nothing was submitted at all in {PH.count(len(empty))} "
                     f"{unit_plural}.")
    partial = overview.get("partial_intervals") or []
    if partial:
        shown = ", ".join(labels.get(name, name) for name in partial)
        lines.append(f"{shown} {'falls' if len(partial) == 1 else 'fall'} only "
                     f"partly inside the dates chosen, so {'it is' if len(partial) == 1 else 'they are'} "
                     f"left out of the comparison.")

    dominance = PH.dominance_phrase(overview.get("dominance"))
    if dominance:
        lines.append(dominance)

    people_rows = [{
        "name": PH.person(row["representative"]),
        "forms": PH.count(row["submissions"]),
        "share": PH.percent(row.get("share")),
    } for row in overview.get("submissions_by_representative", [])]

    return {
        "headline": (f"{PH.count(overview.get('submissions'))} forms with "
                     f"{PH.count(overview.get('responses'))} answers, "
                     f"split into {len(rows)} {unit_plural}"),
        "summary": " ".join(lines),
        # Only named when they actually stand outside the expected range; a
        # "busiest week" that is one of fourteen ordinary weeks is not a fact.
        "busiest": labels.get(overview.get("busiest_interval"),
                              overview.get("busiest_interval")),
        "quietest": labels.get(overview.get("quietest_interval"),
                               overview.get("quietest_interval")),
        "by_period": per_period,
        "by_person": people_rows,
    }


# ---------------------------------------------------------------------------
# One question, end to end
# ---------------------------------------------------------------------------

def _people_for(statistics: dict[str, Any], question_id: str,
                balanced: float | None) -> dict[str, Any] | None:
    """Per-person rows, ranked on the sample-size-adjusted figure.

    The table shows the raw figure, the adjusted one, and how many answers it rests
    on, so a reader can see immediately that "17%" came from six answers. Ranking
    and any callout use the adjusted figure and a significance test - which is why
    nobody is singled out on this data any more.
    """
    segments = (statistics.get("segments") or {}).get("by_representative") or []
    entry = next((row for row in segments if row["question_id"] == question_id), None)
    if not entry or not entry.get("rows"):
        return None
    kind = entry["metric"]["kind"]
    rows = [{
        "name": PH.person(row["representative_label"]),
        "value": PH.value_of(row.get("primary_value"), kind),
        "adjusted": PH.value_of(row.get("adjusted_value"), kind),
        "raw": row.get("primary_value"),
        "answers": PH.count(row.get("answers") or row.get("responses")),
        "too_few": bool(row.get("too_few_to_compare")),
        "stands_out": bool(row.get("stands_out")),
    } for row in entry["rows"]]

    return {
        "differ": bool(entry.get("differ")),
        "note": PH.people_phrase(entry, kind),
        "rows": rows,
        "everyone_counted_once": PH.value_of(balanced, kind) if balanced is not None
        else None,
        "adjusted_explained": (
            "The adjusted column pulls each figure towards the team average in "
            "proportion to how few answers it rests on, so a person with a handful "
            "of answers is not ranked as though they had a hundred."),
    }


def _question(block: dict[str, Any], statistics: dict[str, Any], unit_key: str,
              labels: dict[str, str]) -> dict[str, Any]:
    metric = block["primary_metric"]
    kind, measure = metric["kind"], metric["label"]
    overall = block.get("overall", {})
    trend = block.get("trend") or {}
    dispersion = overall.get("dispersion")

    rows = block.get("by_interval", [])
    thin = sum(1 for row in rows if row.get("low_n"))
    target = block.get("target") or {}
    metric_key = metric.get("key")

    by_period = []
    for row in rows:
        entry = {
            "period": PH.period_label(row, unit_key),
            "value": PH.value_of(row.get("primary_value"), kind),
            "raw": row.get("primary_value"),
            "answers": PH.count(row.get("responses")),
            "too_few": bool(row.get("low_n")),
            "part_period": bool(row.get("partial")),
            "short_period": bool(row.get("short")),
            "stands_out": bool(row.get("notable")),
        }
        # A rate is far easier to act on next to the counts behind it: "54% (15 of
        # 28)" tells a manager both how bad it is and how much it rests on, and
        # makes a swing on tiny numbers obvious without any statistical wording.
        if kind == "rate":
            counts = MET.rate_counts(row.get("summary") or {}, metric_key)
            if counts:
                entry["of_counted"] = f"{PH.count(counts[0])} of {PH.count(counts[1])}"
                # Only worth its own column when it says something the answers
                # column does not - i.e. some answers were neither yes nor no.
                entry["counted_differs"] = counts[1] != row.get("responses")
        if target.get("value") is not None and isinstance(
                row.get("primary_value"), (int, float)):
            entry["meets_target"] = bool(row["primary_value"] >= target["value"])
        by_period.append(entry)

    progress = block.get("progress") or {}
    # Matches the rule in stats.py: a wide spread *across* periods is a trend and
    # keeps its per-period figure; a wide spread *within* each period is what
    # leaves a question with no middle value to report.
    representative = not progress.get("nothing_stands_out_because", "").startswith(
        "the answers have no middle value")
    # Wide overall but tight inside each period: the range is the movement, and
    # calling it "too spread out for one figure" would bury the actual finding.
    trending = bool(
        representative
        and (dispersion or {}).get("single_value_representative") is False
        and progress.get("middle_value_works_within_a_period") is True)
    level = PH.level_phrase(progress, kind, unit_key, representative)

    # The pooled reading answers the question the reader actually has - is this
    # figure sitting still or genuinely moving - and answers it with more power
    # than the period-by-period view. Where it exists it *replaces* the "no real
    # change here" sentence rather than following it: both said the same thing,
    # and the pooled one also says what to do about it.
    lines: list[str] = []
    if not (level and trend.get("variation_within_normal_range")):
        lines.append(PH.movement_phrase(trend, kind, unit_key))
    for phrase in (level,
                   PH.halves_phrase(progress, kind, unit_key) if representative else None):
        if phrase:
            lines.append(phrase)
    steadiness = PH.steadiness_phrase(
        {**(trend.get("volatility") or {}),
         "within_normal_range": trend.get("variation_within_normal_range")},
        unit_key)
    if steadiness:
        lines.append(steadiness)
    # Named only when a period sits outside the range its own sample size would
    # produce. On thin data this is silent, which is the correct outcome.
    notable = PH.notable_periods_phrase(
        {**trend, "notable_periods": [
            {**row, "label": labels.get(row["label"], row["label"])}
            for row in (trend.get("notable_periods") or [])]}, kind)
    if notable:
        lines.append(notable)

    previous = block.get("previous_period") or {}
    against_previous = None
    if previous.get("primary_value") is not None:
        change = PH.change_phrase(previous["primary_value"],
                                  previous.get("current_primary_value"), kind)
        if previous.get("comparable"):
            against_previous = f"Compared with the period before: {change}."
        else:
            against_previous = ("The period before had too few answers to compare "
                               "against.")

    details = _detail_lines(block, overall, kind)

    out: dict[str, Any] = {
        "number": block["order_index"] + 1,
        "question_id": block["question_id"],
        "question": block["question"],
        "answered_with": PH.answer_style(block["effective_type"]),
        "measure": measure,
        "headline": PH.headline_phrase(measure, metric.get("overall"), kind,
                                       dispersion, trending=trending),
        "spread": PH.spread_phrase(dispersion, overall.get("min"), overall.get("max"),
                                   kind, trending=trending),
        "movement": " ".join(lines),
        "against_target": PH.target_phrase(block.get("target"), kind, unit_key),
        "against_previous": against_previous,
        "reliability": PH.reliability_phrase(overall.get("answered") or 0, thin,
                                             unit_key),
        "by_period": by_period,
        "details": details,
        "notes": [note for note in
                  [PH.retype_note(block["declared_type"], block["effective_type"],
                                  block.get("retype_reason"))] if note],
    }
    spread_table = _spread_table(block, unit_key)
    if spread_table:
        out["spread_table"] = spread_table
        # The per-period middle value stays in the structure - it is a real
        # figure and a programmatic caller may want it - but nothing that
        # *narrates* should use it, because quoting it contradicts the sentence
        # above it. The report and the model both read the spread table instead.
        out["hide_by_period"] = True
        out["details"] = [row for row in out["details"]
                          if "distributed" not in row.get("label", "").lower()]

    generated = PH.generated_answers_phrase(overall.get("templating"))
    if generated:
        out["notes"].append(generated)

    balanced = ((statistics.get("overview") or {}).get("balanced_by_question")
                or {}).get(block["question_id"])
    people = _people_for(statistics, block["question_id"], balanced)
    if people:
        out["by_person"] = people
    return out


def _spread_table(block: dict[str, Any], unit_key: str) -> dict[str, Any] | None:
    """How the answers are spread across value bands, period by period.

    For a question whose answers do not cluster, a middle value is not a fact
    about anything - the middle of two groups sits in the empty gap between them,
    and which group it lands nearer to depends on which one happened to get an
    extra answer. What *is* a fact, and what does move meaningfully, is the share
    of answers in each band. The band edges are fixed once over the whole window
    so the per-period shares can be compared at all.
    """
    progress = block.get("progress") or {}
    bands = progress.get("bands_overall") or []
    periods = progress.get("bands_by_period") or []
    if len(bands) < 2 or not periods:
        return None
    if (progress.get("nothing_stands_out_because") or "") != \
            "the answers have no middle value that could be compared":
        # The middle value does represent these answers period by period, so the
        # bands would be noise and the per-period figure is the better reading.
        return None
    dispersion = (block.get("overall") or {}).get("dispersion") or {}

    return {
        "intro": PH.bands_intro(dispersion),
        "bands": [PH.band_row_label(band) for band in bands],
        "overall": [{"band": PH.band_row_label(band),
                     "answers": PH.count(band.get("count")),
                     "share": PH.percent(band.get("share"))}
                    for band in bands],
        "by_period": [
            {"period": PH.period_label(
                next((row for row in block.get("by_interval", [])
                      if row.get("bucket") == period.get("bucket")), {}),
                unit_key) or period.get("label"),
             "shares": [PH.percent(share.get("share"))
                        for share in period.get("shares") or []],
             "answers": PH.count(sum(share.get("count") or 0
                                     for share in period.get("shares") or []))}
            for period in periods],
    }


def _detail_lines(block: dict[str, Any], overall: dict[str, Any], kind: str
                  ) -> list[dict[str, str]]:
    """The type-specific extras, as label/value pairs a reader can scan."""
    effective = block["effective_type"]
    out: list[dict[str, str]] = []

    if effective == "YES_NO":
        yes, no = overall.get("yes"), overall.get("no")
        if yes is not None:
            out.append({"label": "Answered Yes", "value": PH.count(yes)})
            out.append({"label": "Answered No", "value": PH.count(no)})
        interval = overall.get("confidence_interval_95") or {}
        if interval.get("low") is not None:
            out.append({"label": "Realistic range for that share",
                        "value": f"{PH.percent(interval['low'])} to "
                                 f"{PH.percent(interval['high'])}"})
        if overall.get("unclear"):
            out.append({"label": "Answers that were neither yes nor no",
                        "value": PH.count(overall["unclear"])})

    elif effective in ("NUMERIC", "RATING"):
        bands = PH.bands_phrase(overall.get("bands"), kind)
        if bands:
            out.append({"label": "How the answers are distributed", "value": bands})
        flagged = (overall.get("outliers_mad") or {}).get("values") or []
        if flagged:
            shown = ", ".join(PH.value_of(v, kind) for v in flagged[:5])
            out.append({"label": "Answers far from the rest", "value": shown})
        if overall.get("unparsable"):
            out.append({"label": "Answers that were not a number",
                        "value": f"{PH.count(overall['unparsable'])} "
                                 f"(e.g. {overall.get('unparsable_examples')})"})

    elif effective in ("SINGLE_ANSWER", "MULTIPLE_ANSWER"):
        if overall.get("unused_options"):
            out.append({"label": "Options nobody ever picked",
                        "value": ", ".join(overall["unused_options"][:10])})
        pairs = overall.get("top_co_occurrences") or []
        if pairs:
            best = pairs[0]
            out.append({"label": "Most often picked together",
                        "value": f"{' + '.join(best['options'])} "
                                 f"({PH.count(best['count'])} forms)"})

    elif effective == "TEXT":
        amount = PH.amount_phrase(overall)
        if amount:
            out.append({"label": "Amounts written in", "value": amount})
        compliance = PH.compliance_phrase(overall)
        if compliance:
            out.append({"label": "Against what the question asks for",
                        "value": compliance})
            shortfalls = overall.get("shortfall_examples") or []
            if shortfalls:
                out.append({"label": "Examples that fell short",
                            "value": "; ".join(f'"{text}"' for text in shortfalls)})
        themes = PH.themes_phrase(overall.get("themes"))
        if themes:
            out.append({"label": "What the answers are about", "value": themes})
        elif overall.get("top_terms"):
            out.append({"label": "Words that come up most",
                        "value": ", ".join(f"{t['term']} ({t['count']})"
                                           for t in overall["top_terms"][:8])})
        if overall.get("duplicate_rate"):
            out.append({"label": "Answers repeated word for word",
                        "value": PH.percent(overall["duplicate_rate"])})

    for extra in block.get("secondary_metrics", []):
        if extra.get("overall") is None:
            continue
        # A key naming a rate or a share is a proportion, and must be written as a
        # percentage - otherwise "gave a real answer" prints as "1".
        key = extra["key"]
        kind_of = "rate" if key.endswith(("_rate", "_share")) else "value"
        out.append({"label": extra["label"].capitalize(),
                    "value": PH.value_of(extra["overall"], kind_of)})
    return out


def _options_table(block: dict[str, Any]) -> list[dict[str, str]]:
    overall = block.get("overall", {})
    rows = overall.get("top_values") or overall.get("selection_counts") or []
    return [{"option": str(row.get("value") or row.get("option")),
             "times": PH.count(row["count"]),
             "share": PH.percent(row.get("share")
                                 or row.get("share_of_respondents"))}
            for row in rows[:8]]


# ---------------------------------------------------------------------------
# Quality notes, in plain words
# ---------------------------------------------------------------------------

_QUALITY_REWRITES = (
    ("interval(s) have fewer than", "some periods had very few answers, so they "
                                    "are shown but nothing is read into them"),
    ("responses are autofilled", "a large share of answers were filled in "
                                 "automatically rather than by a person"),
)


def _quality_notes(statistics: dict[str, Any], warnings: list[str]) -> list[str]:
    quality = statistics.get("data_quality", {})
    notes: list[str] = []

    # Why the period was split the way it was, and what that means for reading it,
    # belong at the top of this list: they qualify everything else.
    for note in (statistics.get("intervals") or {}).get("notes", []):
        if not note:
            continue
        sentence = note[0].upper() + note[1:]
        notes.append(sentence if sentence.endswith((".", "!", "?"))
                     else sentence + ".")

    # Generated answers and placeholder accounts used to be described here as
    # caveats. They are stronger than caveats - they say the data is not a record
    # of what happened - so they now live in `data_warnings`, which the report
    # prints above everything and the model is told to lead with. Repeating them
    # here put the same paragraph in two places.

    thin = quality.get("low_n_intervals") or []
    if thin:
        notes.append(f"{PH.count(len(thin))} of the periods had very few answers. "
                     f"They are shown for completeness but no conclusion rests on "
                     f"them.")
    if quality.get("empty_questions"):
        notes.append(f"{PH.count(len(quality['empty_questions']))} question(s) were "
                     f"never answered in this period.")
    share = quality.get("autofilled_share") or 0
    if share > 0.25:
        notes.append(f"{PH.percent(share)} of answers were filled in automatically "
                     f"rather than typed by a person.")
    for entry in quality.get("unparsable_answers", []):
        notes.append(f"\"{entry['question']}\" got {PH.count(entry['count'])} "
                     f"answer(s) that were not numbers.")
    if quality.get("time_resolution") == "day":
        notes.append("Submission times are recorded to the day only, so nothing "
                     "can be broken down by hour.")

    for warning in warnings:
        lowered = warning.lower()
        if "import timestamp" in lowered or "disagree on the calendar day" in lowered:
            notes.append("Two of the stored dates on each form look like they were "
                         "written by an import job rather than by the person "
                         "submitting, so the date the form covers is used instead.")
        elif "day-precision" in lowered:
            continue
        elif "no model output" in lowered:
            notes.append("The written commentary could not be generated this run; "
                         "the figures below are unaffected.")
    seen: set[str] = set()
    unique: list[str] = []
    for note in notes:
        fingerprint = " ".join(str(note).lower().split()).rstrip(".")
        if fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(note)
    return unique


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _attention(statistics: dict[str, Any], unit_key: str) -> list[dict[str, Any]]:
    """The short list the report opens with, in reader language.

    Every question is still analysed in full below. This only fixes the order, so
    the one question where nothing met a stated minimum is not ranked tenth
    between the fridge and the follow-up email.
    """
    unit = PH.unit_word(unit_key)
    out: list[dict[str, Any]] = []
    for row in statistics.get("attention") or []:
        reason, detail = row.get("reason"), row.get("detail") or {}
        kind = row.get("kind") or "value"
        # Quoting a middle value in the same row as "there is no middle value"
        # is the contradiction this whole change set exists to remove.
        what = ("no single figure" if reason == "no_typical_value"
                else PH.value_of(row.get("value"), kind))

        if reason == "requirement_never_met":
            asked, listed = detail.get("asked_for"), detail.get("typically_listed")
            issue = (f"The question asks for at least {PH.count(asked)}, and not "
                     f"one answer listed that many")
            if listed is not None:
                issue += f" - a usual answer listed {PH.number(listed)}"
            issue += "."
            action = "Check whether the requirement is understood at all."
        elif reason == "fails_target":
            issue = PH.target_phrase(row.get("target"), kind, unit_key) or ""
            action = ("Raising this is a change to how the work is done, not to "
                      "one period.")
        elif reason == "no_typical_value":
            issue = ("Answers do not settle around one figure, so this cannot be "
                     "summarised as a single number.")
            action = ("Agree what should be counted before reading anything into "
                      "the values.")
        elif reason == "period_stands_out":
            named = ", ".join(str(p) for p in (detail.get("periods") or [])[:3])
            issue = (f"One {unit} sits outside what the number of answers would "
                     f"produce by chance: {named}.")
            action = f"Ask what was different in that {unit}."
        elif reason == "person_stands_out":
            who = ", ".join(PH.person(name) for name in (detail.get("who") or [])[:3])
            issue = f"{who} differs from the rest of the team by more than chance."
            action = "Worth a conversation before it is read as a problem."
        elif reason == "answers_look_generated":
            issue = (PH.generated_answers_phrase(
                {"looks_generated": True, "template_share": detail.get("share"),
                 "dominant_example": detail.get("example")}) or "")
            action = "Nothing here reflects what the team reported."
        else:
            continue

        out.append({
            "number": (row.get("order_index") or 0) + 1,
            "question": row.get("question"),
            "question_id": row.get("question_id"),
            # Naming the measure alongside "no single figure" reads as a
            # contradiction, so it is dropped for that row.
            "measure": None if reason == "no_typical_value" else row.get("measure"),
            "figure": what,
            "issue": issue,
            "what_to_do": action,
            "answers": PH.count(row.get("answers")),
        })
    return out


def _data_warnings(statistics: dict[str, Any],
                   test_accounts: list[dict[str, str]]) -> list[str]:
    """Signs that parts of this data are not a record of what happened.

    Deliberately *not* a gate. The analysis runs on whatever the caller has, test
    data included - refusing would make the tool useless on exactly the datasets
    people try first. What matters is that these get stated at the top, in the
    write-up, rather than appearing as footnote six where the reference report put
    them while eleven sections above it drew earnest conclusions.
    """
    out: list[str] = []
    generated = [block["question"] for block in statistics.get("questions") or []
                 if ((block.get("overall") or {}).get("templating") or {})
                 .get("looks_generated")]
    total = len(statistics.get("questions") or [])
    if generated:
        listed = "; ".join(f'"{name}"' for name in generated[:3])
        more = f" and {len(generated) - 3} more" if len(generated) > 3 else ""
        out.append(
            f"The written answers to {PH.count(len(generated))} of "
            f"{PH.count(total)} questions all follow one pattern with only the "
            f"numbers changing ({listed}{more}). They look filled in "
            f"automatically, so nothing in those questions reflects what the "
            f"team actually reported.")

    if test_accounts:
        named = "; ".join(f'"{PH.person(row["name"])}"' for row in test_accounts[:3])
        out.append(
            f"{PH.count(len(test_accounts))} of the accounts look like test or "
            f"placeholder entries ({named}). Their answers are included in every "
            f"figure here.")

    # Values that are suspiciously regular are the clearest tell that a generator
    # produced them, and the clearest thing to say to a reader wondering whether
    # to trust the numbers at all.
    for block in statistics.get("questions") or []:
        outliers = ((block.get("overall") or {}).get("outliers_iqr") or {}).get("values")
        if not outliers or len(outliers) < 4:
            continue
        gaps = {round(b - a, 6) for a, b in zip(outliers, outliers[1:])}
        if len(gaps) == 1 and gaps != {0.0}:
            step = next(iter(gaps))
            out.append(
                f'In "{block["question"]}" the extreme answers are spaced exactly '
                f"{PH.number(step)} apart, which does not happen when people are "
                f"counting real stock.")
            break
    return out


def build_presentation(statistics: dict[str, Any], *,
                       filters: dict[str, Any] | None = None,
                       warnings: list[str] | None = None) -> dict[str, Any]:
    """Turn the statistics payload into the reader's view."""
    intervals = statistics["intervals"]
    unit_key = intervals["granularity"]
    labels = _label_map(statistics, unit_key)

    questions = []
    for block in statistics.get("questions", []):
        entry = _question(block, statistics, unit_key, labels)
        options = _options_table(block)
        if options:
            entry["options"] = options
        questions.append(entry)

    test_accounts = find_test_accounts(statistics)
    presentation = {
        # The dates the reader asked for, not the widened bucket boundaries.
        "period": PH.period_phrase(
            intervals.get("covered_start") or intervals["period_start"],
            intervals.get("covered_end") or intervals["period_end"]),
        "period_split": (f"{intervals['bucket_count']} "
                         f"{PH.unit_word(unit_key, intervals['bucket_count'] != 1)} "
                         f"of equal length, compared against each other"),
        "scope": _scope_phrase(statistics, filters),
        "attention": _attention(statistics, unit_key),
        "activity": _activity(statistics, unit_key, labels),
        "questions": questions,
        "quality_notes": _quality_notes(statistics, warnings or []),
        "data_warnings": _data_warnings(statistics, test_accounts),
        "test_accounts": test_accounts,
    }
    log.info("presentation: %s question block(s), %s period(s)",
             len(questions), intervals["bucket_count"])
    return presentation
