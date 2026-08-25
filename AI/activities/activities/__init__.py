"""Activity-log report: statistics tables plus two concurrent agents."""

from .analytics import analyze_activities_file, render_tables_markdown
from .pipeline import ANALYSIS_SECTION, TABLE_SECTIONS, build_key_facts, build_report

__all__ = ["analyze_activities_file", "render_tables_markdown", "build_report",
           "build_key_facts", "TABLE_SECTIONS", "ANALYSIS_SECTION"]
