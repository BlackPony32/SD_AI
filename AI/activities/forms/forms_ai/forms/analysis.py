"""The public entry point: `analyze_form`.

One async call takes a user id and a set of filters and returns the finished
analysis. The shape of the return value is fixed and documented in
`FormAnalysisResult` so callers can depend on it.

The pipeline, in order:

    load  ->  filter  ->  plan intervals  ->  compute statistics
          ->  LLM (interval comparison, analysis, summary)  ->  render

Failure policy, inherited from the rest of the codebase: the statistical half
raises on real problems (no export on disk, an impossible filter) because a
caller must know; the model half never raises. If any agent fails, times out or
cannot be grounded, the result still comes back with complete statistics and a
code-rendered report, and `ai.status` explains what happened.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..config import (AGENT_TIMEOUT, FORMS_ANALYST_DETAIL_LIMIT,
                      FORMS_LLM_CONCURRENCY, FORMS_MIN_ANSWERS_PER_PERIOD,
                      FORMS_QUESTION_BATCH, FORMS_TARGET_BUCKETS)
from ..core import prompts as P
from ..core.llm import UsageTracker, write_with_grounding
from ..core.logging_setup import get_log
from . import intervals as IV
from . import presentation as PR
from . import render as R
from . import stats as ST
from . import phrasing as PH
from .filters import FilterSpec, apply_filters
from .loader import FormDataset, load_dataset

log = get_log("forms.analysis")


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

@dataclass
class FormAnalysisResult:
    """What `analyze_form` returns.

    `statistics` is always populated. `ai` is populated when a model was
    available and produced grounded output; `report_markdown` is always a
    complete report either way.
    """

    ok: bool
    user_id: str
    generated_at: str
    duration_seconds: float
    meta: dict[str, Any]
    filters: dict[str, Any]
    intervals: dict[str, Any]
    statistics: dict[str, Any]
    presentation: dict[str, Any]
    ai: dict[str, Any]
    report_markdown: str
    usage: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# LLM stage
# ---------------------------------------------------------------------------

def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


async def _run_question_insights(presentation: dict[str, Any],
                                 rules: P.AnalysisRules, usage: UsageTracker,
                                 concurrency: int) -> tuple[list[dict], list[str]]:
    """One agent per batch of questions, bounded by a semaphore.

    Batching rather than one call per question: a 40-question form would otherwise
    be 40 round trips, and the shared instructions would be re-sent 40 times.
    """
    question_ids = [q["question_id"] for q in presentation.get("questions", [])]
    if not question_ids:
        return [], []

    semaphore = asyncio.Semaphore(max(1, concurrency))
    unsupported: list[str] = []

    async def one(index: int, batch: list[str]) -> list[dict]:
        instructions, user_input, allowed = await P.prompt_question_insights(
            presentation, batch, rules=rules)
        async with semaphore:
            parsed, bad = await write_with_grounding(
                f"QuestionInsight_{index + 1}", instructions, user_input, allowed,
                usage, json_mode=True, fallback=None, wording=PH.contains_banned)
        unsupported.extend(bad)
        if not isinstance(parsed, dict):
            return []
        rows = parsed.get("insights")
        return rows if isinstance(rows, list) else []

    batches = _batches(question_ids, max(1, FORMS_QUESTION_BATCH))
    results = await asyncio.gather(*(one(i, b) for i, b in enumerate(batches)),
                                   return_exceptions=True)
    insights: list[dict] = []
    for result in results:
        if isinstance(result, BaseException):
            log.error("  question insight batch failed: %s", result)
            continue
        insights.extend(result)
    return insights, unsupported


async def _run_llm_stage(presentation: dict[str, Any], rules: P.AnalysisRules,
                         usage: UsageTracker, *, want_summary: bool,
                         concurrency: int, detail_limit: int) -> dict[str, Any]:
    """Per-question insights and the whole-form review run in parallel; the written
    summary depends on both and runs last, reading the review rather than the data
    so the cheapest stage stays cheap."""
    started = time.perf_counter()
    analyst_instructions, analyst_input, analyst_allowed = \
        await P.prompt_form_analysis(presentation, rules=rules,
                                     detail_limit=detail_limit)

    insights_task = asyncio.create_task(_run_question_insights(
        presentation, rules, usage, concurrency))
    analysis_task = asyncio.create_task(write_with_grounding(
        "FormAnalyst", analyst_instructions, analyst_input, analyst_allowed,
        usage, json_mode=True, fallback=None, wording=PH.contains_banned))

    (insights, insight_bad), (analysis, analysis_bad) = await asyncio.gather(
        insights_task, analysis_task)

    summary: str | None = None
    summary_bad: list[str] = []
    if want_summary and isinstance(analysis, dict):
        instructions, user_input, allowed = await P.prompt_written_summary(
            analysis, insights, presentation=presentation, rules=rules)
        summary, summary_bad = await write_with_grounding(
            "WrittenSummary", instructions, user_input, allowed, usage,
            json_mode=False, fallback=None, wording=PH.contains_banned)

    by_question = {row["question_id"]: row["insight"]
                   for row in insights
                   if isinstance(row, dict) and row.get("question_id")
                   and row.get("insight")}

    produced = analysis is not None or bool(insights) or bool(summary)
    status = "ok" if produced else "unavailable"
    if produced and (analysis_bad or insight_bad or summary_bad):
        status = "ok_with_ungrounded_figures"

    return {
        "enabled": True,
        "status": status,
        "analysis": analysis if isinstance(analysis, dict) else None,
        "question_insights": insights,
        "insight_by_question": by_question,
        "summary_text": summary,
        "seconds": round(time.perf_counter() - started, 2),
        "grounding": {
            "unsupported_figures": {
                "form_analyst": analysis_bad,
                "question_insight": insight_bad,
                "written_summary": summary_bad,
            },
            "total_unsupported": len(analysis_bad) + len(insight_bad)
            + len(summary_bad),
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def analyze_form(
    user_id: str,
    *,
    # --- what to include -------------------------------------------------
    form_id: str | None = None,
    period_from: Any = None,
    period_to: Any = None,
    representative_id: str | list[str] | None = None,
    exclude_representative_id: str | list[str] | None = None,
    customer_id: str | list[str] | None = None,
    completed_only: bool = False,
    include_autofilled: bool = True,
    # --- how to cut the time axis ----------------------------------------
    granularity: str = "auto",
    bucket_count: int | None = None,
    interval_mode: str = "calendar",
    interval_anchor: str = "auto",
    target_buckets: int = FORMS_TARGET_BUCKETS,
    min_answers_per_period: int = FORMS_MIN_ANSWERS_PER_PERIOD,
    compare_previous_period: bool = True,
    # --- how to analyse ---------------------------------------------------
    analysis_rules: Any = None,
    use_llm: bool = True,
    include_segments: bool = True,
    include_summary: bool = True,
    llm_concurrency: int = FORMS_LLM_CONCURRENCY,
    analyst_detail_limit: int = FORMS_ANALYST_DETAIL_LIMIT,
    # --- plumbing ---------------------------------------------------------
    data_root: Any = None,
    event_time_field: str | None = None,
    use_cache: bool = True,
) -> FormAnalysisResult:
    """Analyse one form for one user and return the finished result.

    Parameters
    ----------
    user_id
        Whose already-downloaded export to read. The files are located on disk;
        nothing is fetched.
    period_from, period_to
        Inclusive bounds, anything `pandas.Timestamp` accepts. Omit both to
        analyse all time.
    representative_id
        One id or a list; a representative's display name is also accepted.
    exclude_representative_id
        People to leave out - ids or display names. Use it to drop test and
        placeholder accounts, which the report flags for you under
        `presentation["test_accounts"]`.
    customer_id
        Reserved. Accepted end to end, but the current export has no customer
        column, so passing one raises `FilterError` rather than silently
        returning nothing.
    granularity
        `"auto"` (default) picks the unit that yields roughly `target_buckets`
        equal intervals over whatever period was selected - hours for a single
        day, months for a year, quarters for several years. Pass `"hour"`,
        `"day"`, `"week"`, `"month"`, `"quarter"`, `"year"` (or `"3h"`, `"2week"`,
        `"half_year"`) to force one. A unit finer than the timestamps in the data
        is refused and stepped up, with a note in the result.
    interval_mode
        `"calendar"` snaps intervals to natural boundaries (readable labels,
        slightly unequal lengths - use the `_per_day` figures for volume).
        `"uniform"` cuts the span into exactly equal timedeltas.
    interval_anchor
        `"auto"` (default) starts the grid on the requested date unless that date
        already falls on a boundary of the chosen unit, so no period is left only
        partly inside the range. `"calendar"` always snaps outwards (named months
        and weeks, at the cost of two part-covered edge periods); `"period"` always
        starts on the requested date.
    min_answers_per_period
        The engine will choose a coarser split rather than leave periods holding
        fewer answers than this, because below roughly 30 a rate wanders across a
        wide range by chance and period-to-period comparison means nothing.
    analysis_rules
        Extra instructions for the agent: an `AnalysisRules`, a dict, a string or
        a list of strings. Adds focus, thresholds, audience, language and tone.
        Cannot relax the grounding or uncertainty rules.
    use_llm
        False computes the statistics and renders the report without any model
        call - the fast, free, fully deterministic path.
    analyst_detail_limit
        How many questions get their full period-by-period detail in the reviewer's
        payload. The rest are sent as one line each. Lower means cheaper; the
        questions that moved or look unusual are always the ones chosen.

    Returns
    -------
    FormAnalysisResult
        `.statistics` (technical, for programmatic callers), `.presentation` (the
        plain-language reader's view, also what the model reads),
        `.report_markdown` (always a full report) and `.ai` (populated when a model
        ran). Call `.to_dict()` for a JSON-safe dict.

    Raises
    ------
    DatasetNotFound
        No export on disk for this user.
    SchemaError
        The export exists but lacks the columns needed to analyse anything.
    FilterError
        The requested filter cannot be applied (unknown form, unknown
        representative, customer filter on an export without customers).
    """
    started = time.perf_counter()
    rules = P.AnalysisRules.coerce(analysis_rules)
    log.info("analyze_form: user=%s form=%s period=%s..%s rep=%s granularity=%s "
             "llm=%s", user_id, form_id, period_from, period_to, representative_id,
             granularity, use_llm)

    # 1. Load ---------------------------------------------------------------
    dataset: FormDataset = await load_dataset(
        user_id, root=data_root, event_time_field=event_time_field,
        use_cache=use_cache)
    warnings = list(dataset.warnings)

    # 2. Filter -------------------------------------------------------------
    spec = FilterSpec(form_id=form_id, representative_id=representative_id,
                      exclude_representative_id=exclude_representative_id,
                      customer_id=customer_id, period_from=period_from,
                      period_to=period_to, completed_only=completed_only,
                      include_autofilled=include_autofilled)
    filtered = await asyncio.to_thread(
        apply_filters, dataset.facts, dataset.questions, spec)
    warnings += filtered.warnings

    if filtered.facts.empty:
        return _empty_result(user_id, dataset, spec, filtered, rules, warnings,
                             started)

    # 3. Plan the intervals -------------------------------------------------
    # The submission count goes in so the engine can refuse a split too fine to
    # interpret: 172 forms over 14 weeks is 12 answers a week, which cannot
    # separate a real change from ordinary variation at any confidence.
    plan = IV.build_plan(
        filtered.facts["event_time"], period_from=period_from, period_to=period_to,
        granularity=granularity, bucket_count=bucket_count, mode=interval_mode,
        anchor=interval_anchor,
        time_resolution=dataset.time_resolution, target_buckets=target_buckets,
        submissions=int(filtered.facts["progress_id"].nunique()),
        min_answers_per_period=min_answers_per_period)

    # 4. The previous, equal-length window ----------------------------------
    # Filters are re-applied without the period so the comparison window is not
    # cut off by the user's date range.
    previous_facts = None
    if compare_previous_period and plan.previous_start is not None:
        base_spec = FilterSpec(
            form_id=form_id, representative_id=representative_id,
            exclude_representative_id=exclude_representative_id,
            customer_id=customer_id, completed_only=completed_only,
            include_autofilled=include_autofilled)
        base = await asyncio.to_thread(
            apply_filters, dataset.facts, dataset.questions, base_spec)
        times = base.facts["event_time"]
        previous_facts = base.facts.loc[(times >= plan.previous_start)
                                        & (times < plan.previous_end)]
        log.info("  previous window %s..%s: %s response(s)",
                 plan.previous_start.date(), plan.previous_end.date(),
                 len(previous_facts))

    # 5. Statistics ---------------------------------------------------------
    statistics = await asyncio.to_thread(
        ST.compute_statistics, filtered.facts, filtered.questions, plan,
        previous_facts=previous_facts, include_segments=include_segments,
        rules=rules)

    # 6. The reader's view --------------------------------------------------
    # Built once and used by both the report and the model, so the wording and
    # the figures they see cannot diverge.
    filters_payload = filtered.to_dict()
    presentation = await asyncio.to_thread(
        PR.build_presentation, statistics, filters=filters_payload,
        warnings=warnings)

    # 7. The model ----------------------------------------------------------
    usage = UsageTracker()
    if use_llm:
        ai = await _run_llm_stage(presentation, rules, usage,
                                  want_summary=include_summary,
                                  concurrency=llm_concurrency,
                                  detail_limit=analyst_detail_limit)
    else:
        ai = {"enabled": False, "status": "disabled", "analysis": None,
              "question_insights": [], "insight_by_question": {},
              "summary_text": None,
              "grounding": {"unsupported_figures": {}, "total_unsupported": 0}}
    if ai["status"] == "unavailable":
        warnings.append("no model output; the report was assembled from the figures "
                        "alone")
    elif ai["status"] == "ok_with_ungrounded_figures":
        warnings.append(
            f"{ai['grounding']['total_unsupported']} figure(s) in the written "
            f"commentary could not be traced back to the data even after a repair "
            f"pass")

    # 8. Render -------------------------------------------------------------
    report = await asyncio.to_thread(
        R.render_report, presentation, ai_summary=ai.get("summary_text"),
        ai_analysis=ai.get("analysis"),
        insights=ai.get("insight_by_question") or {})

    result = FormAnalysisResult(
        ok=True, user_id=str(user_id),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duration_seconds=round(time.perf_counter() - started, 2),
        meta=_meta(dataset, plan, rules),
        filters=filters_payload,
        intervals=statistics["intervals"],
        statistics=statistics,
        presentation=presentation,
        ai=ai,
        report_markdown=report,
        usage=usage.totals(),
        warnings=warnings)
    log.info("analyze_form done in %.2fs (%s question(s), %s period(s), "
             "%s llm call(s), %s input tokens, $%s)", result.duration_seconds,
             len(statistics["questions"]), plan.bucket_count,
             result.usage["llm_calls"], result.usage["input_tokens"],
             result.usage["estimated_cost_usd"])
    return result


def _meta(dataset: FormDataset, plan: IV.IntervalPlan | None,
          rules: P.AnalysisRules) -> dict[str, Any]:
    return {
        "dataset": {
            "root": str(dataset.paths.root),
            "files": {"questions": dataset.paths.questions.name,
                      "progresses": dataset.paths.progresses.name,
                      "responses": dataset.paths.responses.name},
            "forms_available": dataset.form_ids,
            "questions": int(len(dataset.questions)),
            "submissions": int(len(dataset.progresses)),
            "responses": int(len(dataset.responses)),
            "has_customer_dimension": dataset.has_customer_dimension(),
        },
        "time_axis": {
            "event_time_field": dataset.event_time_field,
            "resolution": dataset.time_resolution,
            "candidates": dataset.time_column_report,
        },
        "analysis_rules": asdict(rules),
        "agent_timeout_seconds": AGENT_TIMEOUT,
    }


def _empty_result(user_id: str, dataset: FormDataset, spec: FilterSpec,
                  filtered: Any, rules: P.AnalysisRules, warnings: list[str],
                  started: float) -> FormAnalysisResult:
    """Filters matched nothing. That is an answer, not an error - return the
    reason and the filter trace so the caller can see which step emptied it."""
    warnings.append("nothing matched the chosen dates and people, so there is "
                    "nothing to report")
    scope = "Nothing was submitted that matches the dates and people chosen."
    return FormAnalysisResult(
        ok=False, user_id=str(user_id),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duration_seconds=round(time.perf_counter() - started, 2),
        meta=_meta(dataset, None, rules),
        filters=filtered.to_dict(),
        intervals={}, statistics={}, presentation={},
        ai={"enabled": False, "status": "skipped_no_data", "analysis": None,
            "question_insights": [], "insight_by_question": {},
            "summary_text": None,
            "grounding": {"unsupported_figures": {}, "total_unsupported": 0}},
        report_markdown=R.render_no_data(scope, filtered.steps),
        usage=UsageTracker().totals(),
        warnings=warnings)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

async def analyze_form_json(user_id: str, **kwargs: Any) -> dict[str, Any]:
    """`analyze_form` as a plain JSON-safe dict, for an HTTP handler."""
    result = await analyze_form(user_id, **kwargs)
    return result.to_dict()


async def list_representatives(user_id: str, *, data_root: Any = None
                               ) -> list[dict[str, Any]]:
    """Who can be filtered on, with their submission counts and date ranges -
    what a filter dropdown needs."""
    dataset = await load_dataset(user_id, root=data_root)
    submissions = dataset.facts.drop_duplicates("progress_id")
    grouped = submissions.groupby(["representative_id", "representative_label"],
                                  dropna=False)
    out = [{"representative_id": (None if pd.isna(rep_id) else str(rep_id)),
            "representative_name": str(label),
            "submissions": int(len(group)),
            "first_submission": (group["event_time"].min().isoformat()
                                 if group["event_time"].notna().any() else None),
            "last_submission": (group["event_time"].max().isoformat()
                                if group["event_time"].notna().any() else None)}
           for (rep_id, label), group in grouped]
    return sorted(out, key=lambda row: -row["submissions"])
