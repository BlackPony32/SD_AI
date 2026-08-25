"""Two-agent pipeline over the activity-log statistics.

    metrics ─┬─► [A] Statistics Analyst ──► key points + a note per table
             │                                        │
             └─► [B] Situation Writer ────► narrative │
                        (A and B run concurrently)    │
                                                      ▼
                             Orders and revenue by salesperson
                                    + Activities Distribution
                                    + Key Analysis

The report is built as sections and the full report is the sections joined, so
the two views cannot disagree.

Failure policy -- no stage can take the report down:
  * Statistics Analyst fails   -> tables keep code-written notes and key points.
  * Situation Writer fails     -> a deterministic narrative is rendered in code.
  * Both fail                  -> the whole report is rendered in code.
  * An analytics section fails -> that section is missing, the rest still runs.
Only an unreadable file produces no report, and `status` says which case it is.

The tables are never sent through a model: they come out of the deterministic
layer already formatted, so the figures a reader is most likely to copy into a
spreadsheet cannot drift.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ..config import KEY_POINT_TARGET, MODEL
from ..core.llm import UsageTracker, write_with_grounding
from ..core.logging_setup import get_log
from ..core.report import ReportResult, Section
from .analytics import analyze_activities_file, render_tables_markdown
from .prompts import prompt_situation_writer, prompt_statistics_analyst
from ..core.response_format import to_payload
log = get_log("activity_agents")

# Which rendered tables belong to which section of the report. A table key that
# is not listed here still reaches the reader: it falls into TABLE_SECTIONS's
# last group via `_section_keys`.
TABLE_SECTIONS: list[tuple[str, str, list[str]]] = [
    ("orders_and_revenue_by_salesperson", "Orders and revenue by salesperson",
     ["salesperson_orders"]),
    ("activities_distribution", "Activities Distribution",
     ["activity_types", "representatives"]),
]
ANALYSIS_SECTION = ("key_analysis", "Key Analysis")


# ---------------------------------------------------------------------------
# 1. Fact sheet
# ---------------------------------------------------------------------------

def _verbalise_change(block: dict | None, label: str) -> str | None:
    """Turn a {pct, basis, reliability} block into English once, correctly, here
    -- so the model never has to decide what `basis: no_baseline` means."""
    if not block:
        return None
    basis = block.get("basis")
    if basis == "no_activity":
        return None
    if basis == "no_baseline":
        return f"{label}: up from zero in the previous period"
    pct = block.get("pct")
    if pct is None:
        return None
    hedge = (" (few events either side - directional only)"
             if block.get("reliability") == "low_sample_low_confidence" else "")
    return f"{label}: {pct:+.1f}%{hedge}"


def build_key_facts(metrics: dict) -> str:
    """Pre-verbalise the figures the writer is allowed to use.

    Handing over ready-made sentences is far more reliable than saying "do not
    hallucinate": the fact sheet is also the allow-list the grounding check
    scores against, so anything absent here is flagged automatically. Every line
    is individually guarded -- a missing section costs one line."""
    lines: list[str] = []

    def add(fmt: str, *args) -> None:
        try:
            if any(a is None for a in args):
                return
            lines.append(fmt.format(*args))
        except Exception:
            pass

    o = metrics.get("overview") or {}
    t = metrics.get("activity_types") or {}
    rep = metrics.get("representatives") or {}
    sp = metrics.get("salesperson_orders") or {}
    wf = (metrics.get("workflow") or {}).get("pairs") or {}
    tm = metrics.get("temporal") or {}
    au = metrics.get("automation") or {}
    oc = metrics.get("order_crosscheck") or {}

    if o:
        add("{} events in the log, {} distinct activities, {} distinct representatives",
            o.get("total_events"), o.get("distinct_types"), o.get("distinct_representatives"))
        add("covers {} to {} ({} days, active on {} of them = {}% of days)",
            str(o.get("first_event"))[:10], str(o.get("last_event"))[:10],
            o.get("span_days"), o.get("active_days"), o.get("active_day_coverage_pct"))
        w = o.get("window_counts") or {}
        add("events last 30 days: {} (previous 30: {}); last 90: {} (previous 90: {})",
            w.get("last_30d"), w.get("prev_30d"), w.get("last_90d"), w.get("prev_90d"))
        for key, label in (("mom", "activity month-over-month"),
                           ("qoq", "activity quarter-over-quarter"),
                           ("yoy", "activity year-over-year")):
            line = _verbalise_change((o.get("volume_change") or {}).get(key), label)
            if line:
                lines.append(line)
        add("busiest single day: {} with {} events",
            (o.get("busiest_day") or {}).get("date"), (o.get("busiest_day") or {}).get("events"))
        add("{}% of events fall on a weekend", o.get("weekend_share_pct"))

    for row in (t.get("by_type") or [])[:6]:
        add("{}: {} events all time ({}% of all activity), {} in the last 90 days",
            row.get("label"), row.get("count"), row.get("share_pct"), row.get("count_last_90d"))
    add("activity in the last 90 days totals {} events", t.get("events_in_last_90d"))
    idle = t.get("activities_with_no_recent_use") or []
    if idle:
        add("{} activities were not used at all in the last 90 days: {}",
            len(idle), ", ".join(idle[:6]))
    for cat, v in list((t.get("by_category") or {}).items())[:5]:
        add("{} activity ({}): {} events, {}% of all activity",
            v.get("label"), v.get("description"), v.get("count"), v.get("share_pct"))

    if rep:
        add("{} representatives in the log: {} are named people, the rest are channel buckets "
            "or integrations", rep.get("representative_count"), rep.get("named_person_count"))
        add("only {} events are attributed to a named person",
            rep.get("named_person_event_count"))
        add("the largest single source accounts for {}% of all events (concentration index {})",
            rep.get("top_share_pct"), rep.get("concentration_hhi"))
        for status, n in (rep.get("status_counts") or {}).items():
            add("{} representative(s) are {}", n, status)
        for a in (rep.get("top_representatives") or [])[:5]:
            add("{} ({}): {} events, {}% of the total, {} in the last 90 days, last active {} "
                "days before the reference date, mostly {} ({}% of their activity)",
                a.get("label"), a.get("kind"), a.get("events"), a.get("share_of_all_events_pct"),
                a.get("events_last_90d"), a.get("days_since_last_event"), a.get("main_activity"),
                a.get("main_activity_share_pct"))

    if sp:
        totals = sp.get("totals") or {}
        add("order book: {} orders worth ${:,.0f} from {} salespeople",
            totals.get("orders"), totals.get("revenue"), sp.get("salesperson_count"))
        add("orders in the last 30 days: {} worth ${:,.0f}; last 90 days: {} worth ${:,.0f}",
            totals.get("orders_last_30d"), totals.get("revenue_last_30d"),
            totals.get("orders_last_90d"), totals.get("revenue_last_90d"))
        line = _verbalise_change(totals.get("orders_mom"), "orders month-over-month")
        if line:
            lines.append(line)
        skew = sp.get("revenue_skew") or {}
        add("average order value is ${:,.0f} against a median of ${:,.0f} ({}x) - revenue is "
            "concentrated in a few large orders, the largest being ${:,.0f}",
            skew.get("mean_order_value"), skew.get("median_order_value"),
            skew.get("mean_to_median_ratio"), skew.get("largest_order"))
        conc = sp.get("concentration") or {}
        add("the top 3 salespeople account for {}% of all revenue",
            conc.get("top3_revenue_share_pct"))
        add("{} orders ({}% of all orders) have no salesperson attached",
            conc.get("unassigned_orders"), conc.get("unassigned_share_pct"))
        for r in (sp.get("by_salesperson") or [])[:5]:
            add("salesperson {}: {} orders worth ${:,.0f} ({}% of revenue), {} orders in the last "
                "90 days, average order ${:,.0f}, last order {} days ago",
                r.get("salesperson"), r.get("orders_total"), r.get("revenue_total"),
                r.get("share_of_revenue_pct"), r.get("orders_last_90d"), r.get("avg_order_value"),
                r.get("days_since_last_order"))

    for label, v in wf.items():
        add("{}: {} opened ({}) vs {} closed ({}) all time, a {} rate of {}% - {}",
            label, v.get("opened"), v.get("opening_activity"), v.get("closed"),
            v.get("closing_activity"), v.get("relation"), v.get("rate_pct"), v.get("rate_means"))
        if v.get("net_open_implied") is not None:
            add("{}: {} more opened than closed over the whole period",
                label, v.get("net_open_implied"))

    if tm:
        tr = tm.get("trend_last_12m") or {}
        add("last 12 months: {} events in the first half vs {} in the second half",
            tr.get("first_half_events"), tr.get("second_half_events"))
        add("12-month direction: {} ({} events per month)",
            tr.get("direction"), tr.get("slope_events_per_month"))
        add("median month has {} events; {} of {} months had none",
            tm.get("monthly_median"), tm.get("months_with_zero_events"), tm.get("months_covered"))
        for a in (tm.get("anomalous_months") or [])[:3]:
            add("unusual month {}: {} events ({})", a.get("month"), a.get("events"), a.get("kind"))
        peak = (tm.get("best_contiguous_windows") or {}).get("3h") or {}
        add("busiest 3-hour block is {}:00-{}:00 UTC, holding {}% of all events",
            peak.get("start_hour_utc"), peak.get("end_hour_utc"), peak.get("share_pct"))

    for a in (au.get("automation_suspected_hours") or [])[:3]:
        add("hour {}:00 UTC is {}% '{}' against its usual {}% ({}x) across {} events - consistent "
            "with an automated job rather than people working",
            a.get("hour_utc"), a.get("dominant_share_pct"), a.get("dominant_activity"),
            a.get("baseline_share_pct"), a.get("lift_vs_baseline"), a.get("events_in_hour"))
    if au.get("bulk_import_row_count"):
        add("{} rows ({}% of the file) were created in shared exact seconds, i.e. imported in bulk",
            au.get("bulk_import_row_count"), au.get("bulk_import_share_pct"))

    if oc:
        add("the log records {} order-creation events against {} real orders placed since it "
            "starts - {}% coverage", oc.get("order_added_events_in_log"),
            oc.get("orders_since_log_start"), oc.get("log_coverage_of_orders_pct"))

    for n in metrics.get("data_quality") or []:
        if n.get("severity") == "high":
            lines.append(f"DATA LIMIT: {n.get('message')}")

    log.info("  key facts sheet: %s lines", len(lines))
    return "\n".join(f"- {line}" for line in lines)


# ---------------------------------------------------------------------------
# 2. Deterministic fallbacks (no LLM)
# ---------------------------------------------------------------------------

def fallback_table_notes(metrics: dict) -> dict[str, str]:
    """One sentence per table, written in code, so a table is never left without
    a reading when the Statistics Analyst is unavailable."""
    notes: dict[str, str] = {}
    sp = metrics.get("salesperson_orders") or {}
    rows = sp.get("by_salesperson") or []
    if rows:
        top = rows[0]
        conc = sp.get("concentration") or {}
        notes["salesperson_orders"] = (
            f"{top['salesperson']} leads on revenue with {top['orders_total']} orders worth "
            f"${top['revenue_total']:,.0f} ({top['share_of_revenue_pct']}% of the total); the top "
            f"three together hold {conc.get('top3_revenue_share_pct')}% of revenue, and "
            f"{conc.get('unassigned_orders')} orders ({conc.get('unassigned_share_pct')}%) have "
            f"no salesperson attached.")

    t = metrics.get("activity_types") or {}
    types = t.get("by_type") or []
    if types:
        idle = t.get("activities_with_no_recent_use") or []
        notes["activity_types"] = (
            f"{types[0]['label']} is the most common action at {types[0]['share_pct']}% of all "
            f"activity. {len(idle)} of {len(types)} activities were not used at all in the last "
            f"90 days, so the recent mix rests on {t.get('events_in_last_90d')} events in total.")

    rep = metrics.get("representatives") or {}
    reps = rep.get("top_representatives") or []
    if reps:
        notes["representatives"] = (
            f"{reps[0]['label']} generates {reps[0]['share_of_all_events_pct']}% of all events, "
            f"and only {rep.get('named_person_event_count')} events across the whole file are "
            f"attributed to a named person, so this table describes channels more than people.")
    return notes


def fallback_key_points(metrics: dict) -> list[dict]:
    """Key findings derived in code, ordered the way the Statistics Analyst is
    asked to order its own: what changed, who it depends on, what limits it."""
    points: list[dict] = []

    def add(point: str, why: str, confidence: str = "high") -> None:
        points.append({"point": point, "why_it_matters": why, "confidence": confidence})

    o = metrics.get("overview") or {}
    w = o.get("window_counts") or {}
    if o:
        add(f"Activity is running at {w.get('last_30d')} events in the last 30 days against "
            f"{w.get('prev_30d')} in the 30 before, and {w.get('last_90d')} against "
            f"{w.get('prev_90d')} over 90 days.",
            "The system is barely being used compared with its own history.",
            "medium" if (w.get("last_30d") or 0) < 30 else "high")

    sp = metrics.get("salesperson_orders") or {}
    rows = sp.get("by_salesperson") or []
    if rows:
        conc = sp.get("concentration") or {}
        add(f"The top three salespeople hold {conc.get('top3_revenue_share_pct')}% of all revenue "
            f"across {sp.get('salesperson_count')} salespeople.",
            "Revenue depends on very few people; losing one is a material risk.")
        add(f"{conc.get('unassigned_orders')} orders ({conc.get('unassigned_share_pct')}%) carry "
            f"no salesperson.",
            "Commission, coverage and territory questions cannot be answered for those orders.")
        skew = sp.get("revenue_skew") or {}
        add(f"Average order value is ${skew.get('mean_order_value'):,.0f} against a median of "
            f"${skew.get('median_order_value'):,.0f}, with a largest order of "
            f"${skew.get('largest_order'):,.0f}.",
            "Averages are pulled by a few outsized orders; plan against the median.")

    for label, v in ((metrics.get("workflow") or {}).get("pairs") or {}).items():
        if v.get("relation") == "completion" and v.get("rate_pct") is not None:
            add(f"{v['opened']} {label} were opened and {v['closed']} closed "
                f"({v['rate_pct']}% completion).",
                "Work is being created faster than it is being finished.")
            break

    au = metrics.get("automation") or {}
    if au.get("automation_suspected_hours"):
        a = au["automation_suspected_hours"][0]
        add(f"Hour {a['hour_utc']}:00 UTC is {a['dominant_share_pct']}% '{a['dominant_activity']}' "
            f"against its usual {a['baseline_share_pct']}%.",
            "Part of the activity is a scheduled job, not people working.", "medium")

    for n in metrics.get("data_quality") or []:
        if n.get("severity") == "high" and len(points) < KEY_POINT_TARGET:
            add(n["message"], "This limits what the rest of the numbers can be used for.")

    return points[:KEY_POINT_TARGET]


def render_fallback_narrative(metrics: dict, reason: str) -> str:
    """The analysis half, written in code from metrics only. Blunter than the
    real thing, but true, and infinitely better than an error page."""
    o = metrics.get("overview") or {}
    t = metrics.get("activity_types") or {}
    rep = metrics.get("representatives") or {}
    sp = metrics.get("salesperson_orders") or {}
    wf = (metrics.get("workflow") or {}).get("pairs") or {}
    w = o.get("window_counts") or {}

    lines = ["**Executive summary**", "",
             f"Automated commentary was unavailable ({reason}), so this section is generated "
             f"from the verified figures only.", ""]
    if o:
        lines += ["**What the activity shows**", "",
                  f"- {o.get('total_events')} events across {o.get('span_days')} days, with "
                  f"activity on {o.get('active_days')} of them.",
                  f"- {w.get('last_30d')} events in the trailing 30 days against "
                  f"{w.get('prev_30d')} in the 30 before that."]
        for row in (t.get("by_type") or [])[:3]:
            lines.append(f"- {row.get('label')}: {row.get('count')} all time "
                         f"({row.get('share_pct')}%), {row.get('count_last_90d')} in the last "
                         f"90 days.")
        lines.append("")

    lines += ["**Who is doing the work**", ""]
    for r in (rep.get("top_representatives") or [])[:2]:
        lines.append(f"- {r.get('label')} ({r.get('kind')}): {r.get('events')} events, "
                     f"{r.get('share_of_all_events_pct')}% of the total, status {r.get('status')}.")
    for r in (sp.get("by_salesperson") or [])[:2]:
        lines.append(f"- {r.get('salesperson')}: {r.get('orders_total')} orders worth "
                     f"${r.get('revenue_total'):,.0f} ({r.get('share_of_revenue_pct')}% of "
                     f"revenue).")
    lines.append("")

    problems = [f"- {n.get('message')}" for n in (metrics.get("data_quality") or [])
                if n.get("severity") in ("high", "medium")]
    for label, v in wf.items():
        if v.get("relation") == "completion" and v.get("rate_pct") is not None:
            problems.append(f"- {label}: {v.get('opened')} opened vs {v.get('closed')} closed "
                            f"({v.get('rate_pct')}% completion).")
    if problems:
        lines += ["**What looks wrong**", ""] + problems[:4] + [""]

    lines += ["**Do this week**", "",
              "1. Confirm whether the drop in recent activity is real or an export cut-off.",
              "2. Fix salesperson attribution so every order can be traced to a person.",
              "3. Review the work that was opened but never closed with whoever owns that queue."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. The two agent stages
# ---------------------------------------------------------------------------

async def run_statistics_analyst(metrics: dict, table_keys: list[str],
                                 usage: UsageTracker) -> tuple[dict, list[str]]:
    """Agent A. Falls back to code-written notes and points on any failure."""
    fallback = {"key_points": fallback_key_points(metrics),
                "table_notes": fallback_table_notes(metrics)}
    try:
        instructions = await prompt_statistics_analyst(metrics, table_keys, KEY_POINT_TARGET)
    except Exception as exc:
        log.exception("  statistics prompt build failed: %s", exc)
        return fallback, []

    parsed, bad = await write_with_grounding(
        "Statistics_Analyst", instructions, "Produce the JSON.", instructions, usage,
        json_mode=True, fallback=fallback)
    if not isinstance(parsed, dict) or not parsed.get("key_points"):
        log.warning("  statistics analyst produced nothing usable; using code-written findings")
        return fallback, bad
    # A missing note costs a note, not the section.
    notes = fallback["table_notes"] | {k: v for k, v in (parsed.get("table_notes") or {}).items()
                                       if isinstance(v, str) and v.strip()}
    log.info("  statistics analyst returned %s key point(s), %s table note(s)",
             len(parsed.get("key_points", [])), len(parsed.get("table_notes") or {}))
    return {"key_points": parsed["key_points"][:KEY_POINT_TARGET], "table_notes": notes}, bad


async def run_situation_writer(metrics: dict, key_facts: str, profiles: list[dict],
                               usage: UsageTracker) -> tuple[str | None, list[str]]:
    """Agent B. Verified fact sheet in; the current-situation narrative out."""
    try:
        instructions = await prompt_situation_writer(key_facts, profiles,
                                                     metrics.get("data_quality", []))
    except Exception as exc:
        log.exception("  situation prompt build failed: %s", exc)
        return None, []

    text, bad = await write_with_grounding(
        "Situation_Writer", instructions, "Write the analysis.", instructions, usage,
        json_mode=False, fallback=None)
    return (text or None), bad


# ---------------------------------------------------------------------------
# 4. Assembly
# ---------------------------------------------------------------------------

def render_key_points(points: list[dict]) -> str:
    lines = []
    for i, p in enumerate(points or [], 1):
        why = (p.get("why_it_matters") or "").strip()
        low = " *(low confidence)*" if p.get("confidence") == "low" else ""
        lines.append(f"{i}. {p.get('point', '').strip()}{low}" + (f" {why}" if why else ""))
    return "\n".join(lines)


def build_sections(tables: dict[str, dict], stats: dict, narrative: str) -> list[Section]:
    """One Section per part of the report. The tables come first, each with its
    reading, then the key findings and the analysis of them."""
    listed = {k for _, _, keys in TABLE_SECTIONS for k in keys}
    leftovers = [k for k in tables if not k.startswith("_") and k not in listed]

    sections: list[Section] = []
    for index, (key, title, table_keys) in enumerate(TABLE_SECTIONS):
        keys = list(table_keys)
        # Anything the deterministic layer starts rendering that has no home yet
        # joins the last table section rather than vanishing from the report.
        if index == len(TABLE_SECTIONS) - 1:
            keys += leftovers
        body = render_tables_markdown(tables, stats.get("table_notes"), keys, hide_title=title)
        sections.append(Section(key=key, title=title, body=body, bold=True))

    analysis: list[str] = []
    points = render_key_points(stats.get("key_points") or [])
    if points:
        analysis += ["**Key findings**", "", points, ""]
    if (narrative or "").strip():
        analysis.append(narrative.strip())
    sections.append(Section(key=ANALYSIS_SECTION[0], title=ANALYSIS_SECTION[1],
                            body="\n".join(analysis).strip()))
    return sections


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------

async def build_report(csv_path: str | Path, orders_path: str | Path | None = None,
                       output_dir: str | Path | None = None) -> ReportResult:
    """Full pipeline. Always returns a ReportResult carrying a report string.

    `status` is one of:
      full            -- both agents succeeded
      statistics_only -- the narrative was written in code
      analysis_only   -- the key findings and table notes were written in code
      fallback        -- no usable LLM output; the whole report is code-rendered
      failed          -- the file itself could not be read
    """
    started = time.perf_counter()
    usage = UsageTracker(model=MODEL)
    log.info("=== activity report started: %s ===", csv_path)

    try:
        data = analyze_activities_file(csv_path, orders_path=orders_path, output_dir=output_dir)
    except Exception as exc:
        log.exception("FATAL: could not analyse %s: %s", csv_path, exc)
        return ReportResult(
            topic="activities", status="failed",
            report="The activity file could not be read, so no analysis could be produced.",
            usage=usage.totals(), seconds=round(time.perf_counter() - started, 1),
            extras={"error": f"{type(exc).__name__}: {exc}"})

    metrics = data.get("metrics", {})
    profiles = data.get("representative_profiles", [])
    tables = data.get("display_tables", {})
    analytics_errors = data.get("analytics_errors", [])
    table_keys = [k for k in tables if not k.startswith("_")]
    log.info("STAGE analytics: %s section(s) in %.3fs, %s table(s), %s section error(s)",
             len(metrics), metrics.get("metadata", {}).get("generated_in_seconds", 0),
             len(table_keys), len(analytics_errors))

    key_facts = build_key_facts(metrics)

    t_agents = time.perf_counter()
    (stats, stats_bad), (narrative, narrative_bad) = await asyncio.gather(
        run_statistics_analyst(metrics, table_keys, usage),
        run_situation_writer(metrics, key_facts, profiles, usage),
    )
    log.info("STAGE agents: both finished in %.2fs", time.perf_counter() - t_agents)

    llm_stats = any(c["stage"].startswith("Statistics_Analyst") for c in usage.calls)
    llm_narrative = narrative is not None
    if not narrative:
        narrative = render_fallback_narrative(metrics, "the writer stage was unavailable")

    status = ("full" if (llm_stats and llm_narrative) else
              "statistics_only" if llm_stats else
              "analysis_only" if llm_narrative else "fallback")

    result = ReportResult.from_sections(
        "activities", build_sections(tables, stats, narrative),
        status=status,
        metrics=metrics,
        usage=usage.totals(),
        # Figures a model wrote that could not be traced back to its input; the
        # code-rendered parts are grounded by construction and never listed.
        ungrounded_figures=sorted(set(stats_bad) | set(narrative_bad)),
        analytics_errors=analytics_errors,
        seconds=round(time.perf_counter() - started, 1),
        extras={"statistics_findings": stats, "narrative": narrative,
                "key_facts": key_facts, "representative_profiles": profiles},
    )

    if output_dir:
        result.save(output_dir)
        try:
            (Path(output_dir) / "key_facts.md").write_text(key_facts, encoding="utf-8")
        except Exception as exc:
            log.warning("could not write key_facts.md: %s", exc)

    totals = result.usage
    log.info("=== activity report finished: status=%s, %s calls, %s tokens, ~$%s, %.1fs ===",
             status, totals["llm_calls"], totals["total_tokens"],
             totals["estimated_cost_usd"], result.seconds)
    return result
