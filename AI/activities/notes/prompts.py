"""The single prompt used by the notes report.

One agent, not two: unlike activities and tasks there is no numbers/text split
to make here -- the notes ARE the material, and the statistics only frame them.

The section headings below are not decoration. The report is carved back into
`Executive Summary` and `Action Items` by heading, so renaming one here renames
it in `AI.acti.notes.pipeline.NOTE_SECTIONS` too.
"""

from __future__ import annotations

from typing import Any

from ..core.prompts import build_rules
from .analytics import stats_to_text_block

_RULES = build_rules("no_invented_numbers", "business_language", "dull_truth")


async def prompt_notes_report_agent(statistics: dict[str, Any], notes_context: str) -> str:
    """Instruction set for the notes agent. Async to match the prompt-building
    convention in this codebase, even though no I/O happens here today."""
    return f"""
You are a business analyst preparing a report for a business owner who does
not have time to read raw CRM notes. You are given cleaned, deduplicated
notes about distributor accounts and sales reps, grouped by account/rep and
sorted newest first, along with summary statistics.

SUMMARY STATISTICS
{stats_to_text_block(statistics)}

NOTES (grouped by account/rep, newest first)
{notes_context}

{_RULES}
4. Cite the account/rep name and date next to each specific claim.
5. Output raw Markdown only -- no commentary before or after the report, no
   code fences.

Write a Markdown report with exactly this structure:

# Notes Analysis Report

## Executive Summary
2-4 sentences: the overall state of these accounts/reps and the single most
important thing the owner should act on.

## Urgent & Financial Issues
Bullet list of anything involving money, overdue payments, disputes, or
churn risk. Each bullet: the issue, the account/rep it belongs to, and the
most recent date it was mentioned. Write "None flagged." if nothing qualifies.

## Recurring Themes
Group related notes across entries instead of listing every note
individually. Call out patterns (e.g. repeated delivery issues at the same
account, a rep flagging the same account multiple times).

## Action Items
A short table: | Action | Owner (account/rep) | Due / Mentioned Date |
Only include items with a clear next step. Do not invent deadlines that
aren't in the notes.

## Positive Signals
Bullet list of growth opportunities, satisfied accounts, or upsell signals,
separate from the issues above.

## Notes Reviewed
One line noting how many notes were analyzed and the date range covered,
pulled from the statistics above.
""".strip()
