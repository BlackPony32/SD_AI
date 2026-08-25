"""Pipeline for turning raw, messy CRM notes into a prioritised report.

    notes CSV -> cleanup -> statistics -> prompt -> agent -> report

Unlike activities and tasks, the model writes this report whole, so the sections
are carved back out of it by heading rather than assembled from parts. A heading
the model failed to produce is recorded in `missing_sections`, never faked.

Failure policy: an agent that produces nothing gives a code-rendered report from
the statistics alone, and `status` says which happened.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import MAX_NOTES, MODEL
from ..core.grounding import grounding_check
from ..core.llm import UsageTracker, run_agent, strip_fences
from ..core.logging_setup import get_log
from ..core.report import ReportResult
from .analytics import (Note, build_notes_context, compute_notes_statistics,
                        filter_important_notes, load_notes_from_csv, stats_to_text_block)
from .prompts import prompt_notes_report_agent

log = get_log("notes_agent")

TITLE = "# Notes Analysis Report"

# (section key, heading as written in the prompt). Renaming a heading here
# without renaming it in the prompt is what makes a section come back missing.
NOTE_SECTIONS: list[tuple[str, str]] = [
    ("executive_summary", "Executive Summary"),
    ("action_items", "Action Items"),
]


def render_fallback_report(stats: dict, reason: str) -> str:
    """Written in code from the statistics alone. Keeps the heading structure so
    the sections can still be carved out of it."""
    dr = stats.get("date_range") or {}
    top_distributors = list((stats.get("notes_per_distributor") or {}).items())[:5]
    top_terms = list((stats.get("top_keyword_hits") or {}).items())[:5]

    lines = [TITLE, "",
             "## Executive Summary", "",
             f"Automated commentary was unavailable ({reason}), so this report is generated "
             f"from the verified figures only. {stats.get('total_kept_notes')} of "
             f"{stats.get('total_raw_notes')} notes carried usable content, covering "
             f"{dr.get('earliest')} to {dr.get('latest')}.", "",
             "## Recurring Themes", ""]
    lines += [f"- {term}: mentioned in {n} notes" for term, n in top_terms] or ["- None flagged."]
    lines += ["", "## Action Items", "",
              "| Action | Owner (account/rep) | Due / Mentioned Date |",
              "|---|---|---|"]
    lines += [f"| Review the {n} open notes on this account | {name} | - |"
              for name, n in top_distributors] or ["| Review the kept notes by hand | - | - |"]
    lines += ["", "## Notes Reviewed", "",
              f"{stats.get('total_kept_notes')} notes analysed out of "
              f"{stats.get('total_raw_notes')} received, covering {dr.get('earliest')} to "
              f"{dr.get('latest')}. {stats.get('money_mentions')} mention a monetary figure."]
    return "\n".join(lines)


def _with_generated_line(report: str, stats: dict) -> str:
    """Title first, then a one-line provenance stamp under it."""
    generated = (f"*Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from "
                 f"{stats['total_kept_notes']} of {stats['total_raw_notes']} notes.*")
    if not report.startswith("# "):
        return f"{TITLE}\n\n{generated}\n\n{report}"
    head, _, rest = report.partition("\n")
    return f"{head}\n\n{generated}\n{rest}"


async def build_report(notes: list[Note] | str | Path, output_dir: str | Path | None = None,
                       *, min_score: int = 0, max_notes: int | None = MAX_NOTES) -> ReportResult:
    """Full pipeline. Always returns a ReportResult carrying a report string.

    `status` is one of:
      full     -- the agent wrote the report
      fallback -- no usable LLM output; the report is rendered from statistics
      empty    -- no note survived filtering
      failed   -- the file itself could not be read
    """
    started = time.perf_counter()
    usage = UsageTracker(model=MODEL)
    log.info("=== notes report started ===")

    if isinstance(notes, (str, Path)):
        try:
            notes = load_notes_from_csv(str(notes))
        except Exception as exc:
            log.exception("FATAL: could not read %s: %s", notes, exc)
            return ReportResult(
                topic="notes", status="failed",
                report="The notes file could not be read, so no analysis could be produced.",
                usage=usage.totals(), seconds=round(time.perf_counter() - started, 1),
                extras={"error": f"{type(exc).__name__}: {exc}"})

    kept, filter_stats = filter_important_notes(notes, min_score=min_score, max_notes=max_notes)
    stats = compute_notes_statistics(notes, kept, filter_stats)

    if not kept:
        log.warning("no notes survived filtering; writing an empty report")
        result = ReportResult(
            topic="notes", status="empty",
            report=f"{TITLE}\n\nNo notes with meaningful content were found for this period.",
            metrics=stats, usage=usage.totals(),
            seconds=round(time.perf_counter() - started, 1))
        if output_dir:
            result.save(output_dir)
        return result

    context = build_notes_context(kept)
    instructions = await prompt_notes_report_agent(stats, context)
    raw = await run_agent("Notes_Analysis_Agent", instructions, "Generate the report now.", usage)

    if raw:
        report, status = strip_fences(raw), "full"
        # The statistics block is the only place a figure may come from; the
        # note text itself is quoted, so its own numbers are legitimate too.
        ungrounded = grounding_check(report, stats_to_text_block(stats) + context)
        if ungrounded:
            log.warning("grounding: %s unsupported figure(s): %s",
                        len(ungrounded), sorted(set(ungrounded))[:10])
    else:
        report, status, ungrounded = render_fallback_report(stats, "the agent was unavailable"), \
            "fallback", []

    result = ReportResult.from_markdown(
        "notes", _with_generated_line(report, stats), NOTE_SECTIONS,
        status=status,
        metrics=stats,
        usage=usage.totals(),
        ungrounded_figures=ungrounded,
        seconds=round(time.perf_counter() - started, 1),
        extras={"kept_notes": len(kept)},
    )

    if output_dir:
        result.save(output_dir)

    totals = result.usage
    log.info("=== notes report finished: status=%s, %s calls, %s tokens, ~$%s, %.1fs ===",
             status, totals["llm_calls"], totals["total_tokens"],
             totals["estimated_cost_usd"], result.seconds)
    return result
