"""One entry point for every topic.

    run_report(uuid, "activities") -> ReportResult

Every topic resolves its inputs the same way -- `data/<uuid>/work_data_folder/`
-- writes to `data/<uuid>/agent_input/`, and returns the same ReportResult, so a
caller can add a topic without learning a new shape.

Each result carries both views of the report:

    result.report                       # the whole thing
    result.sections                     # the same content, addressable
    result.section("action_items")      # one part of it

The legacy `process_*_topic` helpers still return the report string alone.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from AI.activities.core.response_format import to_payload

from . import config
from .activities import build_report as build_activity_report
from .core.logging_setup import get_log
from .core.report import ReportResult
from .notes import build_report as build_notes_report
from .tasks import build_report as build_task_report

log = get_log("report_runner")

TOPICS = ("activities", "tasks", "notes")



async def run_report(uuid: str, topic: str, *, data_root: str | Path | None = None,
                     filenames: dict[str, str] | None = None,
                     write_output: bool = True) -> dict:
    """Run one topic for one tenant and return the wire payload.
 
    `filenames` overrides the defaults per role, e.g. {"orders": "july.csv"}.
    """
    if topic not in TOPICS:
        raise ValueError(f"unknown topic {topic!r}; expected one of {', '.join(TOPICS)}")
 
    names = {**config.DEFAULT_FILES[topic], **(filenames or {})}
    path = lambda role: config.input_file(uuid, topic, role, names.get(role), data_root)
    out = str(config.output_dir(uuid, topic, data_root)) if write_output else None
 
    if topic == "activities":
        orders = path("orders")
        if not orders.exists():
            # The salesperson tables are optional; the analytics layer records
            # their absence in data_quality and the report says so itself.
            log.warning("orders file not found at %s -- proceeding without salesperson "
                        "statistics", orders)
            orders = None
        result = await build_activity_report(path("activities"), orders_path=orders,
                                             output_dir=out)
    elif topic == "tasks":
        result = await build_task_report(path("tasks"), output_dir=out)
    else:
        result = await build_notes_report(path("notes"), output_dir=out)
 
    # Logging stays on the ReportResult: these are attributes, and the payload
    # keeps the same values under `metadata` for whoever consumes the response.
    if result.status not in ("full",):
        log.warning("report for %s/%s degraded: status=%s, analytics_errors=%s",
                    uuid, topic, result.status, len(result.analytics_errors))
    if result.ungrounded_figures:
        log.warning("report for %s/%s contains unverified figures: %s",
                    uuid, topic, result.ungrounded_figures)
    if result.missing_sections:
        log.warning("report for %s/%s is missing section(s): %s",
                    uuid, topic, result.missing_sections)
 
    return to_payload(result, uuid=uuid)
 
 
async def run_all(uuid: str, **kwargs) -> dict[str, dict]:
    """Every topic for one tenant, concurrently. A topic that raises is reported
    as a failed payload rather than taking the others down."""
    async def guarded(topic: str) -> dict:
        try:
            return await run_report(uuid, topic, **kwargs)
        except Exception as exc:
            log.exception("topic %s failed outright: %s", topic, exc)
            failed = ReportResult(topic=topic, status="failed",
                                  report=f"The {topic} report could not be produced.",
                                  extras={"error": f"{type(exc).__name__}: {exc}"})
            # Same envelope as a success, so callers never branch on shape.
            return to_payload(failed, uuid=uuid)
 
    topics = list(TOPICS)
    payloads = await asyncio.gather(*(guarded(t) for t in topics))
    # Keyed by the topic we asked for: a payload is a dict and has no `.topic`,
    # and a failed one may not carry a usable topic in metadata either.
    return dict(zip(topics, payloads))


# ---------------------------------------------------------------------------
# Legacy entry points -- same signatures and return type as before
# ---------------------------------------------------------------------------

async def process_activity_topic(uuid: str, filename: str = "raw_file_activities.csv",
                                 orders_filename: str | None = "raw_file_orders.csv") -> str:
    names = {"activities": filename}
    if orders_filename:
        names["orders"] = orders_filename
    result = await run_report(uuid, "activities", filenames=names)
    return result.report


async def process_standard_topic(uuid: str, filename: str = "tasks_synthetic_3.csv") -> str:
    result = await run_report(uuid, "tasks", filenames={"tasks": filename})
    return result.report


async def process_notes_topic(uuid: str, filename: str = "file_notes.csv") -> str:
    result = await run_report(uuid, "notes", filenames={"notes": filename})
    return result.report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate a report for one tenant")
    ap.add_argument("uuid", help="tenant uuid, i.e. the folder under the data root")
    ap.add_argument("topic", nargs="?", default="all", choices=(*TOPICS, "all"))
    ap.add_argument("--data-root", default=None, help=f"default: {config.DATA_ROOT}")
    ap.add_argument("--section", default=None, help="print only this section key")
    args = ap.parse_args()

    if args.topic == "all":
        results = asyncio.run(run_all(args.uuid, data_root=args.data_root))
    else:
        results = {args.topic: asyncio.run(
            run_report(args.uuid, args.topic, data_root=args.data_root))}

    for topic, res in results.items():
        print(f"\n{'=' * 70}\n{topic.upper()}  [status={res.status} | {res.seconds}s | "
              f"{res.usage.get('total_tokens', 0)} tokens | "
              f"sections={', '.join(res.section_keys) or 'none'}]\n{'=' * 70}\n")
        print(res.section_markdown(args.section) if args.section else res.report)


if __name__ == "__main__":
    main()
