"""Two-agent pipeline over the task-backlog deterministic layer.

    metrics payload ──► [A] Backlog Analyst ─┐
                                             ├─► render_body() ──► final report
    pair payload    ──► [B] Task Text Reader ┘        (in code)

A and B run concurrently and never wait on each other, so wall clock is one LLM
call rather than a chain of them.

The report is returned as a single section: it is one continuous piece of
reading with no part a caller would want on its own. `ReportResult.sections`
still carries it, so every topic has the same shape.

Failure policy: no stage can take the whole report down.
  * Analyst fails     -> the report is written from the text findings alone.
  * Text reader fails -> the report is written from the metrics alone.
  * Both fail         -> a deterministic report is rendered in code.
The user always receives something, and `status` says what it is.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from ..config import MAX_TASK_PAIRS, MODEL
from ..core.grounding import grounding_check
from ..core.llm import UsageTracker, parse_json_output, run_agent
from ..core.logging_setup import get_log
from ..core.markdown import bullets, clean_bullet, dedupe
from ..core.report import ReportResult, Section
from .analytics import analyze_tasks_file, build_metrics_payload, build_pairs_payload
from .prompts import prompt_backlog_analyst, prompt_text_reader

log = get_log("task_agents")

REPORT_SECTION = ("task_backlog", "Task Backlog")

EMPTY_ANALYST = {"bottom_line": "", "findings": [], "actions": [], "takeaway": ""}
EMPTY_READER = {"themes": [], "text_quality": [], "actions": [], "takeaway": ""}


# ---------------------------------------------------------------------------
# 1. The two agents
# ---------------------------------------------------------------------------

async def run_backlog_analyst(metrics: dict, usage: UsageTracker) -> dict:
    """Agent A. Metrics in, findings JSON out. Returns the empty shape on any
    failure so the pipeline can continue with the text side alone."""
    try:
        instructions = await prompt_backlog_analyst(build_metrics_payload(metrics))
    except Exception as exc:
        log.exception("  analyst prompt build failed: %s", exc)
        return dict(EMPTY_ANALYST)
    raw = await run_agent("Backlog_Analyst", instructions, "Produce the JSON.", usage)
    parsed = parse_json_output(raw, dict(EMPTY_ANALYST))
    log.info("  analyst: %s finding(s)", len(parsed.get("findings", [])))
    return parsed


async def run_text_reader(metrics: dict, corpus: list[dict], usage: UsageTracker) -> dict:
    """Agent B. Title+description pairs in, themes JSON out. One call: the corpus
    is already deduplicated to distinct pairs, so there is nothing to chunk."""
    if not corpus:
        log.warning("  text reader skipped: no task text to read")
        return dict(EMPTY_READER)
    try:
        instructions = await prompt_text_reader(build_pairs_payload(metrics, corpus))
    except Exception as exc:
        log.exception("  reader prompt build failed: %s", exc)
        return dict(EMPTY_READER)
    raw = await run_agent("Task_Text_Reader", instructions, "Produce the JSON.", usage)
    parsed = parse_json_output(raw, dict(EMPTY_READER))
    log.info("  text reader: %s theme(s)", len(parsed.get("themes", [])))
    return parsed


# ---------------------------------------------------------------------------
# 2. Report body (no LLM -- this is where the editor agent used to be)
# ---------------------------------------------------------------------------

def render_body(analyst: dict, reader: dict, metrics: dict) -> str:
    """Assemble the markdown from both agents' JSON.

    Written in code on purpose: merging two structured blobs into fixed parts
    needs no judgement, so a third model call bought latency and a chance to
    hallucinate, and nothing else. Parts whose agent produced nothing are
    omitted."""
    dq = metrics.get("text_pairs") or {}
    ov = metrics.get("overview") or {}
    parts: list[str] = []

    bottom = (analyst.get("bottom_line") or "").strip()
    if not bottom and ov:
        bottom = (f"{ov.get('open_tasks')} of {ov.get('total_tasks')} tasks are still open "
                  f"and {ov.get('open_overdue')} are overdue.")
    if bottom:
        parts.append(f"**Overview** — {clean_bullet(bottom)}")

    findings = bullets(analyst.get("findings"))
    if findings:
        parts.append("**What's happening**\n" + "\n".join(f"- {b}" for b in findings))

    themes = bullets(reader.get("themes"))
    if themes:
        parts.append("**What the tasks actually say**\n" + "\n".join(f"- {b}" for b in themes))

    quality = bullets(reader.get("text_quality"), limit=3)
    if not quality and dq.get("duplicate_group_count"):
        examples = ", ".join(f'"{t}"' for t in (dq.get("unactionable_examples") or [])[:2])
        quality = [f"{dq.get('no_description_pct')}% of tasks have no description, and "
                   f"{dq.get('duplicate_group_count')} duplicate groups add "
                   f"{dq.get('duplicate_extra_rows')} redundant rows"
                   + (f", including {examples}." if examples else ".")]
    if quality:
        parts.append("**Data you can't trust yet**\n" + "\n".join(f"- {b}" for b in quality))

    actions = dedupe(bullets(analyst.get("actions"), limit=3)
                     + bullets(reader.get("actions"), limit=2))[:3]
    if actions:
        parts.append("**Do this week**\n"
                     + "\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1)))

    # The last thing the reader sees: one sentence from each half of the
    # analysis, in the agents' own words.
    takeaways = dedupe([t for t in ((analyst.get("takeaway") or "").strip(),
                                    (reader.get("takeaway") or "").strip()) if t])
    if takeaways:
        parts.append("**Key takeaways**\n" + "\n".join(f"- {t}" for t in takeaways))

    return "\n\n".join(parts)


def render_fallback_body(metrics: dict, reason: str) -> str:
    """Written in code, from metrics only. This is what the user gets when every
    LLM call fails: blunter, but true, and infinitely better than an error page."""
    ov = metrics.get("overview") or {}
    ag = metrics.get("aging") or {}
    dq = metrics.get("text_pairs") or {}
    lines = [f"**Overview** — automated commentary was unavailable ({reason}); "
             "here are the verified numbers.", "", "**What's happening**"]

    if ov:
        lines.append(f"- {ov.get('open_tasks')} open tasks out of {ov.get('total_tasks')}; "
                     f"{ov.get('open_overdue')} are overdue "
                     f"({ov.get('overdue_share_of_open_pct')}% of open work).")
    if ag.get("overdue_buckets"):
        lines.append(f"- {ag['overdue_buckets'].get('90d+', 0)} tasks are more than 90 days "
                     f"late; the worst is {ag.get('overdue_days_max')} days overdue.")
    for ex in (ag.get("oldest_overdue") or [])[:2]:
        lines.append(f"- \"{ex.get('title')}\" — {ex.get('days_overdue')} days late, "
                     f"owner {ex.get('owner')}.")
    for name, v in list((metrics.get("by_owner") or {}).items())[:3]:
        lines.append(f"- {name}: {v.get('open')} open, {v.get('overdue')} overdue.")

    if dq:
        lines += ["", "**Data you can't trust yet**",
                  f"- {dq.get('duplicate_group_count')} duplicate groups "
                  f"({dq.get('duplicate_extra_rows')} redundant rows) and "
                  f"{dq.get('unactionable_pair_count')} tasks whose title and description "
                  f"together say nothing actionable.",
                  f"- {ov.get('open_unowned')} open tasks have no owner and "
                  f"{ov.get('open_without_due_date')} have no due date."]

    lines += ["", "**Do this week**",
              "1. Close or re-date every task overdue by more than 90 days.",
              "2. Assign an owner to every unowned open task.",
              "3. Merge the duplicate tasks and fill in the missing descriptions."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Orchestration
# ---------------------------------------------------------------------------

async def build_report(csv_path: str | Path,
                       output_dir: str | Path | None = None) -> ReportResult:
    """Full pipeline. Always returns a ReportResult carrying a report string.

    `status` is one of:
      full          -- both agents succeeded
      numbers_only  -- the text reader produced nothing
      text_only     -- the analyst produced nothing
      fallback      -- no usable LLM output; report rendered from metrics
      failed        -- the file itself could not be read
    """
    started = time.perf_counter()
    usage = UsageTracker(model=MODEL)
    log.info("=== task report started: %s ===", csv_path)

    try:
        data = analyze_tasks_file(csv_path, output_dir=output_dir, corpus_size=MAX_TASK_PAIRS)
    except Exception as exc:
        log.exception("FATAL: could not analyse %s: %s", csv_path, exc)
        return ReportResult(
            topic="tasks", status="failed",
            report="The task file could not be read, so no analysis could be produced.",
            usage=usage.totals(), seconds=round(time.perf_counter() - started, 1),
            extras={"error": f"{type(exc).__name__}: {exc}"})

    metrics, corpus = data.get("metrics", {}), data.get("pair_corpus", [])

    log.info("agents: launching Backlog_Analyst and Task_Text_Reader concurrently")
    t_agents = time.perf_counter()
    analyst, reader = await asyncio.gather(
        run_backlog_analyst(metrics, usage),
        run_text_reader(metrics, corpus, usage),
    )
    log.info("agents: both finished in %.2fs", time.perf_counter() - t_agents)

    has_numbers = bool(analyst.get("findings") or analyst.get("bottom_line"))
    has_text = bool(reader.get("themes") or reader.get("text_quality"))

    if has_numbers or has_text:
        body = render_body(analyst, reader, metrics)
        status = "full" if (has_numbers and has_text) else (
            "numbers_only" if has_numbers else "text_only")
    else:
        log.error("both agents produced nothing usable")
        body, status = render_fallback_body(metrics, "no agent output"), "fallback"

    # The agents' whole world was these two payloads, so they are the only place
    # a figure in the report could legitimately have come from.
    allowed = (json.dumps(build_metrics_payload(metrics), default=str)
               + json.dumps(build_pairs_payload(metrics, corpus), default=str))
    ungrounded = [] if status == "fallback" else grounding_check(body, allowed)
    if ungrounded:
        log.warning("grounding: %s unsupported figure(s): %s",
                    len(ungrounded), sorted(set(ungrounded))[:10])

    result = ReportResult.from_sections(
        "tasks",
        [Section(key=REPORT_SECTION[0], title=REPORT_SECTION[1], body=body,
                 include_heading=False)],
        status=status,
        metrics=metrics,
        usage=usage.totals(),
        ungrounded_figures=ungrounded,
        analytics_errors=data.get("analytics_errors", []),
        seconds=round(time.perf_counter() - started, 1),
        extras={"analyst_findings": analyst, "text_findings": reader},
    )

    if output_dir:
        result.save(output_dir)
        try:
            (Path(output_dir) / "findings.json").write_text(
                json.dumps({"analyst": analyst, "text_reader": reader},
                           indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            log.warning("could not write findings.json: %s", exc)

    totals = result.usage
    log.info("=== finished: status=%s, %s calls, %s tokens, ~$%s, %.1fs ===",
             status, totals["llm_calls"], totals["total_tokens"],
             totals["estimated_cost_usd"], result.seconds)
    return result
