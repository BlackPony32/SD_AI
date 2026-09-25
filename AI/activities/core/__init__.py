"""Shared plumbing: logging, LLM calls, grounding, markdown, report shape."""

from .grounding import grounding_check
from .llm import (UsageTracker, parse_json_output, run_agent, strip_fences,
                  write_with_grounding)
from .logging_setup import get_log
from .markdown import (bullets, clean_bullet, dedupe, find_heading_body, md_table, money,
                       normalise_title, split_headings)
from .prompts import RULES, SHARED_RULES, build_rules, json_payload, prompt_repair_input
from .report import ReportResult, Section

__all__ = [
    "UsageTracker", "ReportResult", "Section",
    "RULES", "SHARED_RULES", "build_rules",
    "bullets", "clean_bullet", "dedupe", "find_heading_body", "get_log", "grounding_check",
    "json_payload", "md_table", "money", "normalise_title", "parse_json_output",
    "prompt_repair_input", "run_agent", "split_headings", "strip_fences", "write_with_grounding",
]
