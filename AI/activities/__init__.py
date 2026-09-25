"""Report generation for the activity log, the task backlog and CRM notes."""

from .config import DEFAULT_FILES, MODEL
from .core.report import ReportResult, Section
from .runner import (TOPICS, process_activity_topic, process_notes_topic,
                     process_standard_topic, run_all, run_report)

__all__ = ["run_report", "run_all", "ReportResult", "Section", "TOPICS", "MODEL",
           "DEFAULT_FILES", "process_activity_topic", "process_standard_topic",
           "process_notes_topic"]
