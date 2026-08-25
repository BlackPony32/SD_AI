"""Assembly: filtered facts + an interval plan -> the statistics payload.

This is the "prepared statistics" the agent later reads. Two rules govern what
goes in it:

* **Every number the report might state is computed here.** Deltas, percent
  changes, per-day rates, shares - all of it. The model is then never required to
  do arithmetic, which is both the main source of hallucinated figures and the
  thing the grounding check would reject.
* **Nothing is silently dropped.** Empty buckets, unanswered questions,
  unparseable values and low-n warnings all appear explicitly. A gap that is
  visible is a finding; a gap that is removed is a bug.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import (FORMS_BAND_Z, FORMS_DOMINANCE_SHARE, FORMS_MIN_BUCKET_N,
                      FORMS_MIN_PERSON_ANSWERS, FORMS_PEOPLE_DIFFER_QCD,
                      FORMS_PERSON_ALPHA, FORMS_ROUND)
from ..core.logging_setup import get_log
from . import metrics as MET
from . import schema as S
from . import statmath as M
from . import targets as TG
from . import textual as TX
from .intervals import IntervalPlan, assign_buckets

log = get_log("forms.stats")


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

def round_payload(node: Any, places: int = FORMS_ROUND) -> Any:
    """Recursively round every float. Applied once, at the boundary, so the
    payload the model sees and the payload the grounding check validates against
    are byte-identical."""
    if isinstance(node, dict):
        return {k: round_payload(v, places) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [round_payload(v, places) for v in node]
    if isinstance(node, bool) or node is None:
        return node
    if isinstance(node, float):
        if node != node or node in (float("inf"), float("-inf")):
            return None
        return round(node, places)
    if isinstance(node, (int,)):
        return node
    if hasattr(node, "item"):                       # numpy scalar
        try:
            return round_payload(node.item(), places)
        except Exception:
            return str(node)
    if isinstance(node, pd.Timestamp):
        return node.isoformat()
    return node


# ---------------------------------------------------------------------------
# Question contexts
# ---------------------------------------------------------------------------

def build_contexts(questions: pd.DataFrame, facts: pd.DataFrame
                   ) -> list[MET.QuestionContext]:
    """One context per question, with the effective type resolved on the whole
    period and the categorical reference option pinned."""
    contexts: list[MET.QuestionContext] = []
    by_question = dict(tuple(facts.groupby("question_id", sort=False))) if len(facts) else {}

    for row in questions.sort_values("order_index").itertuples():
        frame = by_question.get(row.question_id)
        if frame is None:
            frame = facts.iloc[0:0]
        declared = getattr(row, "question_type", S.UNKNOWN)
        effective, reason = MET.effective_type(
            declared, frame["answer_text"] if len(frame) else pd.Series(dtype="string"),
            frame["answer_multi"] if len(frame) else None)

        ctx = MET.QuestionContext(
            question_id=row.question_id,
            text=str(getattr(row, "question_text", "") or ""),
            declared_type=declared, effective_type=effective,
            order_index=int(getattr(row, "order_index", 0) or 0),
            options=list(getattr(row, "options", []) or []),
            required=getattr(row, "required", None),
            retype_reason=reason)

        # Free text: decide what these answers are measured on once, over the
        # whole period, so every period is measured the same way.
        if effective == S.TEXT and len(frame):
            answers = [a for a in frame["answer_text"].astype("string").tolist()
                       if a and str(a).strip()]
            text_profile = TX.profile(ctx.text, answers, required=ctx.required)
            ctx.reference.update({
                "content_kind": text_profile["content_kind"],
                "required_minimum": text_profile["required_minimum"]})

        # Pin the reference option for categorical series before any bucketing.
        if effective in (S.SINGLE_ANSWER, S.MULTIPLE_ANSWER) and len(frame):
            probe = MET.analyse(frame, ctx)
            reference = probe.get("modal_value")
            if reference is None:
                counts = probe.get("selection_counts") or []
                reference = counts[0]["option"] if counts else None
            ctx.reference = {"modal_value": reference}
        contexts.append(ctx)
    return contexts


# ---------------------------------------------------------------------------
# Form-level overview
# ---------------------------------------------------------------------------

def _band_for(kind: str, overall: dict[str, Any], overall_value: float | None,
              n: int, days: float) -> dict[str, Any]:
    """The range this period's figure would fall in by chance, given its size.

    This is the guard against the report's worst habit. With ~12 answers a period
    and a yes-rate near half, chance alone produces anything from 26% to 78%; every
    "highest period" read out of that is invented. A period is only called high or
    low if it lands outside its own band.
    """
    if kind == "rate":
        return M.expected_rate_band(overall_value, n, z=FORMS_BAND_Z)
    if kind == "count":
        return M.expected_count_band(overall_value, max(days, 1e-9), z=FORMS_BAND_Z)
    return M.expected_median_band(overall_value, overall.get("iqr"), n, z=FORMS_BAND_Z)


def _comparable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The periods a comparison may legitimately rest on: fully inside the
    requested dates and holding enough answers to mean something."""
    return [row for row in rows if not row.get("partial") and not row.get("low_n")]


def _overview(facts: pd.DataFrame, plan: IntervalPlan, contexts: list[MET.QuestionContext]
              ) -> dict[str, Any]:
    buckets = plan.buckets
    per_bucket: list[dict[str, Any]] = []
    for bucket in buckets:
        slice_ = facts.loc[facts["bucket"] == bucket.index]
        submissions = int(slice_["progress_id"].nunique())
        # Rate over the days actually inside the requested range, not the bucket's
        # full width: an edge bucket holding one day of data is not a quiet week.
        covered = plan.covered_days(bucket)
        per_bucket.append({
            **bucket.to_dict(),
            "coverage": round(plan.coverage_of(bucket), 4),
            "covered_days": round(covered, 3),
            "partial": bool(plan.coverage_of(bucket) < 0.9),
            "submissions": submissions,
            "responses": int(len(slice_)),
            "short": plan.is_short(bucket),
            "submissions_per_day": M.safe_div(submissions, covered or bucket.days),
            "representatives": int(slice_["representative_label"].nunique()),
            "low_n": bool(len(slice_) < FORMS_MIN_BUCKET_N),
        })

    # Volume gets the same treatment as everything else: a busiest period is only
    # busiest if it is outside what arrival variation alone would produce.
    overall_rate = M.safe_div(
        sum(b["submissions"] for b in per_bucket),
        sum((b["covered_days"] or b["days"]) for b in per_bucket))
    for row in per_bucket:
        band = M.expected_count_band(overall_rate,
                                     max(row["covered_days"] or row["days"], 1e-9),
                                     z=FORMS_BAND_Z)
        row["expected_band"] = band
        row["notable"] = bool(not row["partial"]
                              and M.is_notable(row["submissions_per_day"], band))

    # Partial and thin periods are shown but never shape the comparison.
    volume = [b["submissions_per_day"] or 0.0 for b in per_bucket]
    trend_input = [b["submissions_per_day"] or 0.0 for b in per_bucket
                   if not b["partial"] and not b["low_n"]]
    mk = M.mann_kendall(trend_input)
    ols = M.ols_trend(trend_input)

    submission_dates = facts.drop_duplicates("progress_id")["event_time"].dropna()
    cadence: dict[str, Any] = {"active_days": int(submission_dates.dt.normalize().nunique())}
    if len(submission_dates) > 2:
        gaps = (submission_dates.sort_values().diff().dropna()
                / pd.Timedelta(days=1))
        cadence.update({
            "median_gap_days": float(gaps.median()),
            "max_gap_days": float(gaps.max()),
            "longest_silence_ended": (
                submission_dates.sort_values().iloc[int(gaps.to_numpy().argmax()) + 1]
                .isoformat() if len(gaps) else None),
        })

    reps = (facts.drop_duplicates("progress_id")
            .groupby("representative_label", sort=False)["progress_id"]
            .count().sort_values(ascending=False))

    return {
        "responses": int(len(facts)),
        "submissions": int(facts["progress_id"].nunique()),
        "questions": len(contexts),
        "questions_with_answers": int(facts["question_id"].nunique()),
        "representatives": int(facts["representative_label"].nunique()),
        "period": {"start": plan.period_start.isoformat(),
                   "end": plan.period_end.isoformat(),
                   "days": round((plan.period_end - plan.period_start)
                                 / pd.Timedelta(days=1), 2)},
        "responses_outside_period": int((facts["bucket"] == -1).sum()),
        "by_interval": per_bucket,
        "volume_trend": M.classify_trend(
            mk, ols, trend_input[0] if trend_input else None,
            trend_input[-1] if trend_input else None),
        "volume_volatility": M.volatility(trend_input),
        "volume_largest_shift": M.largest_shift(
            [b["submissions_per_day"] or 0.0 for b in _comparable(per_bucket)],
            [b["label"] for b in _comparable(per_bucket)]),
        "partial_intervals": [b["label"] for b in per_bucket if b["partial"]],
        "short_intervals": [b["label"] for b in per_bucket if b["short"]],
        "notable_intervals": [b["label"] for b in per_bucket if b["notable"]],
        "volume_within_normal_range": bool(
            per_bucket and not any(b["notable"] for b in per_bucket)),
        "busiest_interval": (
            max([b for b in _comparable(per_bucket) if b["notable"]],
                key=lambda b: b["submissions_per_day"] or 0)["label"]
            if any(b["notable"] for b in _comparable(per_bucket)) else None),
        "quietest_interval": (
            min([b for b in _comparable(per_bucket) if b["notable"]],
                key=lambda b: b["submissions_per_day"] or 0)["label"]
            if any(b["notable"] for b in _comparable(per_bucket)) else None),
        "empty_intervals": [b["label"] for b in per_bucket if b["responses"] == 0],
        "cadence": cadence,
        "submissions_by_representative": [
            {"representative": label, "submissions": int(count),
             "share": M.safe_div(int(count), int(reps.sum()))}
            for label, count in reps.items()],
    }


# ---------------------------------------------------------------------------
# Per-question statistics
# ---------------------------------------------------------------------------

def _varies(rows: list[dict[str, Any]]) -> bool:
    """True when the per-interval values are not all identical."""
    values = {row["primary_value"] for row in rows}
    return len(values) > 1


def _progress(frame: pd.DataFrame, ctx: MET.QuestionContext, plan: IntervalPlan,
              rows: list[dict[str, Any]], key: str, kind: str,
              overall: dict[str, Any]) -> dict[str, Any]:
    """How the figure has actually moved, tested three ways.

    Period-by-period comparison is the weakest reading available: with 25 answers
    a fortnight almost nothing clears the noise, which is why the honest answer is
    so often "no real change here". Three stronger readings are added.

    **Halves.** The first half of the window against the second. Doubling the
    answers on each side roughly halves the difference that can be detected, so a
    shift invisible fortnight-to-fortnight can be visible here.

    **Homogeneity.** One pooled test across every period, which answers a
    different and more useful question than any single period does: is this figure
    sitting at one level, or does it genuinely move between periods? A steady 52%
    and a 39%-59% swing need opposite responses from a manager.

    **Bands.** For a question whose answers do not cluster, the trendable figure
    is not a middle value that represents nothing - it is what share of answers
    falls in each band, with the band edges fixed once over the whole window so
    the per-period shares are comparable.
    """
    usable = [row for row in rows if not row.get("short") and not row.get("partial")]
    out: dict[str, Any] = {"periods_compared": len(usable)}

    # --- first half against second half ----------------------------------
    if len(usable) >= 4:
        middle = len(usable) // 2
        halves = []
        for part in (usable[:middle], usable[len(usable) - middle:]):
            buckets = [row["bucket"] for row in part]
            slice_ = frame.loc[frame["bucket"].isin(buckets)]
            summary = MET.analyse(slice_, ctx)
            value = MET.dig(summary, key)
            halves.append({
                "from": part[0]["label"], "to": part[-1]["label"],
                "value": float(value) if isinstance(value, (int, float))
                         and not isinstance(value, bool) else None,
                "answers": int(len(slice_)),
                "counts": MET.rate_counts(summary, key) if kind == "rate" else None,
                "values": MET.numbers_in(slice_) if kind == "value" else None,
            })
        first, second = halves
        comparison: dict[str, Any] = {
            "first": {k: v for k, v in first.items() if k != "values"},
            "second": {k: v for k, v in second.items() if k != "values"},
            "p": None, "differs": None,
        }
        if kind == "rate" and first["counts"] and second["counts"]:
            comparison["p"] = M.two_proportion_p(first["counts"][0], first["counts"][1],
                                                 second["counts"][0], second["counts"][1])
        elif kind == "value" and first["values"] and second["values"]:
            comparison["p"] = M.values_homogeneous(
                [first["values"], second["values"]]).get("p")
        if comparison["p"] is not None:
            comparison["differs"] = bool(comparison["p"] < FORMS_PERSON_ALPHA)
        out["halves"] = comparison

    # --- one level, or genuinely moving? ---------------------------------
    if kind == "rate":
        counted = [MET.rate_counts(row.get("summary") or {}, key) for row in usable]
        counted = [pair for pair in counted if pair]
        if len(counted) >= 2:
            out["holds_one_level"] = M.rates_homogeneous(
                [s for s, _ in counted], [n for _, n in counted],
                alpha=FORMS_PERSON_ALPHA)
    elif kind == "value":
        groups = [MET.numbers_in(frame.loc[frame["bucket"] == row["bucket"]])
                  for row in usable]
        groups = [group for group in groups if group]
        if len(groups) >= 2:
            out["holds_one_level"] = M.values_homogeneous(
                groups, alpha=FORMS_PERSON_ALPHA)

    # --- the observed swing, stated plainly ------------------------------
    seen = [row["primary_value"] for row in usable
            if isinstance(row.get("primary_value"), (int, float))]
    if len(seen) >= 2:
        out["swing"] = {"low": min(seen), "high": max(seen),
                        "periods": len(seen)}

    # --- is the spread inside each period, or only between them? ----------
    # A question rising 100 -> 600 over six months has a very wide spread overall
    # and a perfectly good middle value in every single month. A question where
    # every month mixes 130s and 110,000s is wide *inside* each month and has no
    # middle value anywhere. Judging that on the whole-period spread alone
    # confuses the two and suppresses real trends, so the within-period spread is
    # what decides it.
    if kind == "value":
        verdicts = []
        for row in usable:
            values = MET.numbers_in(frame.loc[frame["bucket"] == row["bucket"]])
            if len(values) >= 4:
                verdicts.append(bool(M.dispersion(values).get(
                    "single_value_representative")))
        if verdicts:
            representative = sum(verdicts)
            out["middle_value_works_within_a_period"] = bool(
                representative > len(verdicts) / 2)

    # --- band shares, for answers that do not cluster --------------------
    bands = overall.get("bands") or []
    if kind == "value" and bands:
        out["bands_overall"] = bands
        out["bands_by_period"] = [
            {"label": row["label"], "bucket": row["bucket"],
             "shares": M.band_shares(
                 MET.numbers_in(frame.loc[frame["bucket"] == row["bucket"]]), bands)}
            for row in rows]
    return out


def _question_block(frame: pd.DataFrame, ctx: MET.QuestionContext, plan: IntervalPlan,
                    previous: pd.DataFrame | None) -> dict[str, Any]:
    overall = MET.analyse(frame, ctx)
    key, label, kind = MET.primary_metric(ctx)
    overall_value = MET.dig(overall, key)
    overall_value = float(overall_value) if isinstance(overall_value, (int, float)) \
        and not isinstance(overall_value, bool) else None

    by_interval: list[dict[str, Any]] = []
    series: list[float | None] = []
    for bucket in plan.buckets:
        slice_ = frame.loc[frame["bucket"] == bucket.index]
        summary = MET.analyse(slice_, ctx) if len(slice_) else MET.analyse(
            frame.iloc[0:0], ctx)
        value = MET.dig(summary, key)
        value = float(value) if isinstance(value, (int, float)) and not isinstance(
            value, bool) else None
        series.append(value)
        covered = plan.covered_days(bucket)
        band = _band_for(kind, overall, overall_value, len(slice_),
                         covered or bucket.days)
        by_interval.append({
            "bucket": bucket.index, "key": bucket.key, "label": bucket.label,
            "start": bucket.start.isoformat(), "days": round(bucket.days, 3),
            "coverage": round(plan.coverage_of(bucket), 4),
            "partial": bool(plan.coverage_of(bucket) < 0.9),
            "short": plan.is_short(bucket),
            "responses": int(len(slice_)),
            "low_n": bool(len(slice_) < FORMS_MIN_BUCKET_N),
            "primary_value": value,
            "responses_per_day": M.safe_div(len(slice_), covered or bucket.days),
            "expected_band": band,
            "notable": bool(M.is_notable(value, band)),
            "summary": summary,
        })

    progress = _progress(frame, ctx, plan, by_interval, key, kind, overall)

    # The pooled test is the gatekeeper for naming any period at all.
    #
    # Two failures this closes. First, multiple comparisons: 7 periods x 12
    # questions is 84 band checks, so at the 5% level four of them come back
    # "notable" on data where nothing is happening. Second, and worse, the median
    # band assumes a single cluster - on a question whose answers fall into two
    # groups the median flips between them depending on which group got one extra
    # answer, and every flip is flagged as remarkable. Both vanish once a period
    # must also survive a single test across all periods, which has more power
    # than any one of them and cannot be gamed by repetition.
    holds_one_level = progress.get("holds_one_level") or {}
    # Only a question with no middle value *inside its own periods* loses the
    # per-period figure. Where each period is internally tight, the spread across
    # periods is the finding, not a reason to stop reporting one.
    no_middle_value = (
        kind == "value"
        and (overall.get("dispersion") or {}).get("single_value_representative")
        is False
        and progress.get("middle_value_works_within_a_period") is not True)
    if no_middle_value:
        # Unconditional, and independent of the pooled result. The figure being
        # compared is a middle value that represents nothing: it sits in the gap
        # between two groups and moves to whichever one gained an answer. Even
        # where the periods *do* genuinely differ, this is not the statistic that
        # shows it - the band shares are, and they are reported instead.
        reason = "the answers have no middle value that could be compared"
    elif holds_one_level.get("steady") is True:
        reason = "every period tests as one level"
    else:
        reason = None
    if reason:
        for row in by_interval:
            row["notable"] = False
        progress["nothing_stands_out_because"] = reason

    # Thin and partly-covered periods are shown but excluded from the fit: a
    # 2-answer week, or a week with one day of data in it, must not decide whether
    # something is rising.
    fit_rows = _comparable(by_interval)
    fit_values = [row["primary_value"] for row in fit_rows
                  if row["primary_value"] is not None]
    populated = [row for row in fit_rows if row["primary_value"] is not None]
    notable = [row for row in populated if row["notable"]]
    first = populated[0]["primary_value"] if populated else None
    last = populated[-1]["primary_value"] if populated else None

    mk = M.mann_kendall(fit_values)
    ols = M.ols_trend(fit_values)
    trend = M.classify_trend(mk, ols, first, last)
    trend.update({
        "buckets_used": len(fit_values),
        "buckets_total": len(by_interval),
        "buckets_excluded_low_n": sum(1 for r in by_interval if r["low_n"]),
        "buckets_excluded_partial": sum(1 for r in by_interval if r["partial"]),
        "mann_kendall": mk,
        "ols": ols,
        "volatility": M.volatility(fit_values),
        "largest_shift": M.largest_shift(
            [r["primary_value"] for r in populated],
            [r["label"] for r in populated]),
        # Only periods outside their own expected range may be named. A flat
        # series has no best or worst; neither does a series whose spread is
        # entirely explained by how few answers each period holds.
        "best_interval": (max(notable, key=lambda r: r["primary_value"])["label"]
                          if notable and _varies(populated) else None),
        "worst_interval": (min(notable, key=lambda r: r["primary_value"])["label"]
                           if notable and _varies(populated) else None),
        "notable_periods": [
            {"label": row["label"], "value": row["primary_value"],
             "answers": row["responses"],
             "direction": ("above" if row["primary_value"] > (overall_value or 0)
                           else "below")}
            for row in notable],
        # True when every period sits inside the range chance would produce: the
        # variation is real but it is not evidence of anything.
        "variation_within_normal_range": bool(populated and not notable),
        "typical_period_answers": (
            int(sum(r["responses"] for r in fit_rows) / len(fit_rows))
            if fit_rows else 0),
        "smallest_detectable_change": (
            M.detectable_difference(
                int(sum(r["responses"] for r in fit_rows) / len(fit_rows)),
                overall_value if kind == "rate" and overall_value else 0.5)
            if fit_rows and kind == "rate" else None),
        "moving_average_3": M.moving_average(
            [r["primary_value"] for r in populated], 3),
    })

    block: dict[str, Any] = {
        "question_id": ctx.question_id,
        "order_index": ctx.order_index,
        "question": ctx.text,
        "declared_type": ctx.declared_type,
        "effective_type": ctx.effective_type,
        "retype_reason": ctx.retype_reason,
        "options": ctx.options,
        "required": ctx.required,
        "primary_metric": {
            "key": key, "label": label, "kind": kind,
            "overall": MET.dig(overall, key),
            "series": series,
        },
        "secondary_metrics": [
            {"key": secondary_key, "label": secondary_label,
             "overall": MET.dig(overall, secondary_key)}
            for secondary_key, secondary_label in MET.secondary_metrics(ctx)],
        "overall": overall,
        "by_interval": by_interval,
        "trend": trend,
        "progress": progress,
    }

    if previous is not None:
        previous_slice = previous.loc[previous["question_id"] == ctx.question_id]
        previous_summary = MET.analyse(previous_slice, ctx)
        previous_value = MET.dig(previous_summary, key)
        current_value = MET.dig(overall, key)
        block["previous_period"] = {
            "start": plan.previous_start.isoformat() if plan.previous_start else None,
            "end": plan.previous_end.isoformat() if plan.previous_end else None,
            "responses": int(len(previous_slice)),
            "primary_value": previous_value,
            "current_primary_value": current_value,
            "change_absolute": (current_value - previous_value
                                if isinstance(previous_value, (int, float))
                                and isinstance(current_value, (int, float))
                                and not isinstance(previous_value, bool) else None),
            "change_pct": M.pct_change(
                previous_value if isinstance(previous_value, (int, float)) else None,
                current_value if isinstance(current_value, (int, float)) else None),
            "comparable": bool(len(previous_slice) >= FORMS_MIN_BUCKET_N),
        }
    return block


def _segments(frame: pd.DataFrame, contexts: list[MET.QuestionContext],
              dimension: str) -> list[dict[str, Any]]:
    """Per-person view of each question, with the small-sample trap closed.

    Three things happen here that did not before.

    **Shrinkage.** A person with 1 yes out of 6 answers is not "17%" in any useful
    sense - the honest reading is "somewhere between 3% and 56%, probably close to
    the team". Empirical Bayes pulls each figure towards the team in proportion to
    how little data it rests on, and the ranking uses the adjusted figure. Raw
    numbers are kept alongside so nothing is hidden.

    **A floor on being named.** Below `FORMS_MIN_PERSON_ANSWERS` a person is marked
    `too_few_to_compare` and is not eligible to be called highest or lowest.

    **A test before a claim.** `stands_out` now requires a two-proportion test
    against the rest of the team to clear `FORMS_PERSON_ALPHA`, not merely sitting
    at the end of a sorted list.
    """
    if dimension not in frame.columns or frame[dimension].isna().all():
        return []

    out: list[dict[str, Any]] = []
    for ctx in contexts:
        question_frame = frame.loc[frame["question_id"] == ctx.question_id]
        if question_frame.empty:
            continue
        key, label, kind = MET.primary_metric(ctx)

        rows: list[dict[str, Any]] = []
        for name, group in question_frame.groupby(dimension, sort=False):
            summary = MET.analyse(group, ctx)
            value = MET.dig(summary, key)
            answered = int(summary.get("answered") or 0)
            rows.append({
                dimension: str(name),
                "responses": int(len(group)),
                "answers": answered,
                "submissions": int(group["progress_id"].nunique()),
                "primary_value": (float(value) if isinstance(value, (int, float))
                                  and not isinstance(value, bool) else None),
                # For rate measures, the numerator behind the rate - needed for a
                # proper test rather than a comparison of percentages.
                "successes": (int(round(float(value) * answered))
                              if kind == "rate" and isinstance(value, (int, float))
                              and not isinstance(value, bool) else None),
                "too_few_to_compare": bool(answered < FORMS_MIN_PERSON_ANSWERS),
            })

        adjusted: dict[str, dict[str, Any]] = {}
        if kind == "rate" and all(r["successes"] is not None for r in rows):
            for entry in M.shrink_rates([(r[dimension], r["successes"], r["answers"])
                                         for r in rows if r["answers"] > 0]):
                adjusted[entry["key"]] = entry
        else:
            for entry in M.shrink_values([(r[dimension], r["primary_value"],
                                           r["answers"]) for r in rows]):
                adjusted[entry["key"]] = entry

        for row in rows:
            entry = adjusted.get(row[dimension], {})
            row["adjusted_value"] = entry.get("adjusted_rate",
                                              entry.get("adjusted_value"))
            row["low"] = entry.get("low")
            row["high"] = entry.get("high")
            row["p_vs_rest"] = entry.get("p_vs_rest")
            row["stands_out"] = bool(
                not row["too_few_to_compare"]
                and row["p_vs_rest"] is not None
                and row["p_vs_rest"] < FORMS_PERSON_ALPHA)

        # Ranked on the adjusted figure: sorting on the raw one is what put a
        # 6-answer person at the bottom of the table and into the commentary.
        rows.sort(key=lambda r: (r["adjusted_value"] is None,
                                 -(r["adjusted_value"] or 0)))
        eligible = [r for r in rows if not r["too_few_to_compare"]
                    and r["adjusted_value"] is not None]
        raw_values = [r["primary_value"] for r in rows
                      if r["primary_value"] is not None]
        adjusted_values = [r["adjusted_value"] for r in eligible]

        spread = M.dispersion(adjusted_values)
        out.append({
            "question_id": ctx.question_id,
            "question": ctx.text,
            "metric": {"key": key, "label": label, "kind": kind},
            "rows": rows,
            "spread": M.describe(raw_values),
            "adjusted_spread": M.describe(adjusted_values),
            "dispersion": spread,
            "comparable_people": len(eligible),
            "excluded_too_few": sum(1 for r in rows if r["too_few_to_compare"]),
            # Real difference between people, after allowing for sample size and
            # requiring at least one person to survive the test.
            "differ": bool(len(eligible) >= 2
                           and (spread.get("qcd") or 0) >= FORMS_PEOPLE_DIFFER_QCD
                           and any(r["stands_out"] for r in rows)),
            "highest": eligible[0][dimension] if eligible else None,
            "lowest": eligible[-1][dimension] if eligible else None,
        })
    return out


def _dominance(facts: pd.DataFrame, dimension: str = "representative_label"
               ) -> dict[str, Any]:
    """Is one person's work standing in for the whole team's figures?

    Mariana submitted 58% of the forms in the reference export, so every "team"
    number was mostly hers. Reporting that, and a person-balanced figure beside it,
    is the difference between a team average and one person's average wearing a
    team's name.
    """
    submissions = facts.drop_duplicates("progress_id")
    if submissions.empty or dimension not in submissions.columns:
        return {"dominated": False}
    counts = submissions.groupby(dimension, sort=False)["progress_id"].count()
    total = int(counts.sum())
    top_share = float(counts.max() / total) if total else 0.0
    people = int(len(counts))
    return {
        # Two conditions, because a large share is only remarkable relative to how
        # many people there are: half the forms from one of two people is expected,
        # half from one of eight is not. "More than twice a fair share, and at least
        # FORMS_DOMINANCE_SHARE of everything."
        "dominated": bool(people >= 3 and top_share >= FORMS_DOMINANCE_SHARE
                          and top_share >= 2.0 / people),
        "top_person": str(counts.idxmax()),
        "top_share": top_share,
        "fair_share": (1.0 / people) if people else None,
        "people": people,
        "median_share": float(counts.median() / total) if total else None,
    }


def _balanced_values(segments: list[dict[str, Any]]) -> dict[str, float | None]:
    """Each question's figure with every person counted once.

    The plain figure is answer-weighted, so it follows whoever submits most. This
    one gives each person equal say, and a gap between the two is itself the
    finding.
    """
    out: dict[str, float | None] = {}
    for entry in segments:
        values = [row["primary_value"] for row in entry["rows"]
                  if row["primary_value"] is not None
                  and not row["too_few_to_compare"]]
        out[entry["question_id"]] = (sum(values) / len(values)) if values else None
    return out


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def _data_quality(facts: pd.DataFrame, plan: IntervalPlan,
                  question_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    flags: list[str] = []
    low_n = [b["label"] for b in
             ({**bucket.to_dict(),
               "responses": int((facts["bucket"] == bucket.index).sum())}
              for bucket in plan.buckets)
             if b["responses"] < FORMS_MIN_BUCKET_N]
    if low_n:
        flags.append(f"{len(low_n)} interval(s) have fewer than {FORMS_MIN_BUCKET_N} "
                     f"responses and are excluded from trend fitting: "
                     f"{', '.join(low_n[:6])}")

    unanswered = [b["question"] for b in question_blocks
                  if (b["overall"].get("answered") or 0) == 0]
    if unanswered:
        flags.append(f"{len(unanswered)} question(s) received no answers at all")

    retyped = [{"question": b["question"], "declared": b["declared_type"],
                "effective": b["effective_type"], "reason": b["retype_reason"]}
               for b in question_blocks if b.get("retype_reason")]

    unparsable = [{"question": b["question"],
                   "count": b["overall"]["unparsable"],
                   "examples": b["overall"].get("unparsable_examples", [])}
                  for b in question_blocks
                  if b["overall"].get("unparsable")]

    autofilled = (float(facts["autofilled"].fillna(False).mean())
                  if "autofilled" in facts.columns and len(facts) else 0.0)
    if autofilled > 0.25:
        flags.append(f"{autofilled:.0%} of answers are autofilled rather than "
                     f"entered by a person")

    return {
        "flags": flags,
        "low_n_intervals": low_n,
        "empty_questions": unanswered,
        "retyped_questions": retyped,
        "unparsable_answers": unparsable,
        "autofilled_share": autofilled,
        "time_resolution": plan.time_resolution,
        "interval_notes": list(plan.notes),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_statistics(facts: pd.DataFrame, questions: pd.DataFrame,
                       plan: IntervalPlan, *,
                       previous_facts: pd.DataFrame | None = None,
                       include_segments: bool = True,
                       rules: Any = None) -> dict[str, Any]:
    """Build the full statistics payload. Pure and synchronous - the async
    wrapper in analysis.py runs it in a worker thread."""
    facts = facts.copy()
    facts["bucket"] = assign_buckets(facts["event_time"], plan)

    contexts = build_contexts(questions, facts)
    in_period = facts.loc[facts["bucket"] >= 0]

    question_blocks = [
        _question_block(in_period.loc[in_period["question_id"] == ctx.question_id],
                        ctx, plan, previous_facts)
        for ctx in contexts]

    payload: dict[str, Any] = {
        "intervals": plan.to_dict(),
        "overview": _overview(in_period, plan, contexts),
        "questions": question_blocks,
        "data_quality": _data_quality(in_period, plan, question_blocks),
    }
    if include_segments:
        by_person = _segments(in_period, contexts, "representative_label")
        payload["segments"] = {
            "by_representative": by_person,
            "by_customer": _segments(in_period, contexts, "customer_id"),
        }
        payload["overview"]["dominance"] = _dominance(in_period)
        payload["overview"]["balanced_by_question"] = _balanced_values(by_person)

    # Targets are checked, and the short list of things worth a decision is ranked,
    # after the segments exist: "this person stands out" is one of the reasons a
    # question earns a place on that list.
    TG.apply_targets(payload, rules)
    payload["attention"] = TG.attention(payload, rules)

    log.info("statistics: %s question(s) x %s interval(s), %s needing attention",
             len(question_blocks), plan.bucket_count, len(payload["attention"]))
    return round_payload(payload)
