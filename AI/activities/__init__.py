"""Report generation for the activity log, the task backlog and CRM notes.

    from AI.acti import run_report

    result = await run_report(uuid, "activities")
    result.report                      # the full markdown report
    result.section("key_analysis")     # one section of it
    result.section_keys                # what this topic produced

Layout:
    config.py            model, pricing, paths, per-topic filenames
    core/                logging, LLM calls, grounding, markdown, report shape
    activities/ tasks/ notes/
                         analytics.py  deterministic layer (no LLM)
                         prompts.py    the agents' instructions
                         pipeline.py   agents + assembly -> ReportResult
    runner.py            one entry point for every topic
"""

from .config import DEFAULT_FILES, MODEL
from .core.report import ReportResult, Section
from .runner import (TOPICS, process_activity_topic, process_notes_topic,
                     process_standard_topic, run_all, run_report)

__all__ = ["run_report", "run_all", "ReportResult", "Section", "TOPICS", "MODEL",
           "DEFAULT_FILES", "process_activity_topic", "process_standard_topic",
           "process_notes_topic"]
