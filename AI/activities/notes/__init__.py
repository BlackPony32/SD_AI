"""CRM notes report: cleanup and statistics plus one writing agent."""

from .analytics import (Note, compute_notes_statistics, filter_important_notes,
                        load_notes_from_csv)
from .pipeline import NOTE_SECTIONS, build_report

__all__ = ["Note", "load_notes_from_csv", "filter_important_notes",
           "compute_notes_statistics", "build_report", "NOTE_SECTIONS"]
