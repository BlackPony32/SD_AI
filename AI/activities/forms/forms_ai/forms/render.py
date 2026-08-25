"""The report a person reads.

Built from `presentation.build_presentation`, never from the raw statistics, so
every number arrives already formatted and no statistical vocabulary can leak into
it. It works with or without model output: without, the blocks are the figures and
the plain-language descriptions; with, each question block ends with that
question's written insight.

Layout - one self-contained block per question, which is the point:

    # Form report
    Scope / period
    ## Overall activity          (volume, by period, by person)
    ## Summary                   (written, only when a model ran)
    ## 1. <question text>
        headline, spread, movement, comparison with the period before
        table: period by period
        who differs
        > Insight: <written commentary for this question>
    ## 2. ...
    ## Things to keep in mind
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_log

log = get_log("forms.render")


def _flag(row: dict) -> str:
    """The right-hand marker column: how to read this period, if at all."""
    if row.get("part_period"):
        return "only part of this period is inside the dates chosen"
    if row.get("short_period"):
        return "shorter period"
    if row.get("too_few"):
        return "very few answers"
    return "stands out" if row.get("stands_out") else ""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def _attention_section(rows: list[dict[str, Any]]) -> str:
    """The short list, first.

    A reader who stops after this table should still have seen everything that
    warrants a decision. Everything below it is the evidence.
    """
    if not rows:
        return ("## What needs attention\n\nNothing in this form crosses the line "
                "into needing a decision: every question either holds steady or "
                "has too few answers to say otherwise.\n")
    parts = ["## What needs attention\n",
             f"{len(rows)} of the questions below warrant a decision. "
             f"The rest hold steady.\n"]
    parts.append(_table(["#", "Question", "Where it stands", "What it means",
                         "What to do"],
                        [[str(row["number"]), row["question"],
                          (f"{row['measure']}: {row['figure']}" if row.get("measure")
                           else row["figure"]),
                          row["issue"], row["what_to_do"]]
                         for row in rows]))
    return "\n".join(parts)


def _activity_section(activity: dict[str, Any]) -> str:
    parts = ["## Overall activity\n",
             f"**{activity['headline']}.** {activity['summary']}\n"]
    if activity.get("busiest") and activity.get("quietest"):
        parts.append(f"Busiest: **{activity['busiest']}**. "
                     f"Quietest: **{activity['quietest']}**.\n")
    parts.append(_table(["Period", "Forms", "Answers", "Each day", "People", ""],
                        [[row["period"], row["forms"], row["answers"],
                          row["each_day"], row["people"], _flag(row)]
                         for row in activity["by_period"]]))
    if activity.get("by_person"):
        parts.append("\n**Who submitted them**\n")
        parts.append(_table(["Team member", "Forms", "Share of all forms"],
                            [[row["name"], row["forms"], row["share"]]
                             for row in activity["by_person"]]))
    return "\n".join(parts)


def _period_table(question: dict[str, Any]) -> str:
    """The period-by-period table, with the columns that question type earns.

    A rate gets the counts behind it, because "54% (15 of 28)" is actable where
    "54%" alone hides whether it rests on 28 answers or 3. Where a target exists
    it gets a met/missed column, so the reader can see the run of misses without
    reading seven numbers against a number held in their head.
    """
    rows = question.get("by_period") or []
    if not rows or question.get("hide_by_period"):
        return ""
    measure = question["measure"].capitalize()
    headers = ["Period", measure]
    has_counts = any(row.get("of_counted") for row in rows)
    has_target = any("meets_target" in row for row in rows)
    # "44% (12 of 27)" already contains the answer count, so a separate column
    # for it is noise unless the two denominators genuinely differ.
    show_answers = not has_counts or any(row.get("counted_differs") for row in rows)
    if has_counts:
        headers.append("Behind that share")
    if show_answers:
        headers.append("Answers")
    if has_target:
        headers.append("Target")
    headers.append("")

    body = []
    for row in rows:
        cells = [row["period"], row["value"]]
        if has_counts:
            cells.append(row.get("of_counted") or "-")
        if show_answers:
            cells.append(row["answers"])
        if has_target:
            cells.append("met" if row.get("meets_target") else
                         ("missed" if "meets_target" in row else "-"))
        cells.append(_flag(row))
        body.append(cells)
    return f"\n**{measure}, period by period**\n\n" + _table(headers, body)


def _spread_section(question: dict[str, Any]) -> str:
    """Bands instead of a middle value, when a middle value means nothing.

    Two tables: how the answers are spread over the whole period, and how that
    spread shifts period by period. The second is what makes a split question
    trendable at all - the share in each band moves for real reasons, where the
    middle value only flips between the groups.
    """
    table = question.get("spread_table")
    if not table:
        return ""
    parts = [f"\n{table['intro']}\n",
             _table(["Answers between", "How many", "Share of answers"],
                    [[row["band"], row["answers"], row["share"]]
                     for row in table["overall"]])]
    by_period = table.get("by_period") or []
    if by_period:
        parts.append("\n**How that spread shifted, period by period**\n")
        parts.append(_table(["Period"] + table["bands"] + ["Answers"],
                            [[row["period"]] + row["shares"] + [row["answers"]]
                             for row in by_period]))
    return "\n".join(parts)


def _question_section(question: dict[str, Any], insight: str | None) -> str:
    parts = [f"## {question['number']}. {question['question']}\n",
             f"_Answered with: {question['answered_with']}._\n",
             f"**{question['headline']}**\n"]

    for key in ("spread", "movement", "against_target", "against_previous"):
        if question.get(key):
            parts.append(question[key] + "\n")
    parts.append(f"_{question['reliability']}_\n")

    for note in question.get("notes", []):
        parts.append(f"> Note: {note}\n")

    if question.get("details"):
        parts.append("")
        for detail in question["details"]:
            parts.append(f"- {detail['label']}: {detail['value']}")
        parts.append("")

    if question.get("options"):
        parts.append("\n**Answers given**\n")
        parts.append(_table(["Option", "Times chosen", "Share of forms"],
                            [[row["option"], row["times"], row["share"]]
                             for row in question["options"]]))

    parts.append(_period_table(question))
    parts.append(_spread_section(question))

    people = question.get("by_person")
    if people:
        parts.append(f"\n**By team member**\n\n{people['note']}\n")
        if people.get("everyone_counted_once"):
            parts.append(f"With every person counted once rather than by how many "
                         f"forms they submitted: **{people['everyone_counted_once']}**.\n")
        parts.append(_table(
            ["Team member", question["measure"].capitalize(),
             "Adjusted for how many answers", "Answers", ""],
            [[row["name"], row["value"], row["adjusted"], row["answers"],
              "stands out" if row["stands_out"]
              else ("too few to compare" if row["too_few"] else "")]
             for row in people["rows"]]))
        parts.append(f"_{people['adjusted_explained']}_\n")

    if insight:
        parts.append(f"\n> **Insight.** {insight.strip()}\n")
    return "\n".join(parts)


def _written_summary(analysis: dict[str, Any] | None, summary: str | None) -> str:
    if summary:
        return "## Summary\n\n" + summary.strip() + "\n"
    if not analysis:
        return ""
    parts = []
    if analysis.get("headline"):
        parts.append("## Summary\n\n" + str(analysis["headline"]) + "\n")
    if analysis.get("activity"):
        parts.append(str(analysis["activity"]) + "\n")
    if analysis.get("vs_previous_period"):
        parts.append(str(analysis["vs_previous_period"]) + "\n")
    # Keys match the reviewer's output schema in core/prompts.py.
    for heading, key, fields in (
            ("What stands out", "stands_out", ("point", "evidence", "why_it_matters")),
            ("Who differs", "people", ("who", "what", "evidence")),
            ("Worth watching", "worth_watching", ("issue", "evidence")),
            ("What to do", "do_next", ("action", "because"))):
        entries = analysis.get(key) or []
        if not entries:
            continue
        parts.append(f"\n### {heading}\n")
        for entry in entries:
            if isinstance(entry, dict):
                head = entry.get(fields[0]) or ""
                tail = " - ".join(str(entry[f]) for f in fields[1:] if entry.get(f))
                parts.append(f"- **{head}**" + (f" {tail}" if tail else ""))
            else:
                parts.append(f"- {entry}")
        parts.append("")
    return "\n".join(parts)


def render_report(presentation: dict[str, Any], *,
                  ai_summary: str | None = None,
                  ai_analysis: dict[str, Any] | None = None,
                  insights: dict[str, str] | None = None,
                  title: str = "Form report") -> str:
    """The full Markdown report.

    `insights` maps question id -> that question's written commentary, so each
    block closes with the commentary for the question it belongs to rather than
    everything being collected in a separate section far from the figures.
    """
    insights = insights or {}
    parts: list[str] = [
        f"# {title}\n",
        f"**{presentation['period']}** - {presentation['scope']}.  ",
        f"Split into {presentation['period_split']}.\n",
    ]

    written = _written_summary(ai_analysis, ai_summary)
    if written:
        parts.append(written)

    # Whether the data is a real record comes before anything computed from it.
    # The analysis still runs - the caller asked for statistics on what they have -
    # but a reader who learns this in footnote six has already believed the report.
    warnings = presentation.get("data_warnings") or []
    if warnings:
        parts.append("## Before reading this\n")
        parts.append("\n".join(f"- {note}" for note in warnings) + "\n")
        parts.append("The figures below are still exactly what the answers say. "
                     "Treat them as a description of the data, not of the work.\n")

    # Then the short list of what warrants a decision. Twelve questions given equal
    # weight is twelve questions with no priority, and the findings that matter get
    # lost among the ones that say "nothing changed".
    parts.append(_attention_section(presentation.get("attention") or []))
    parts.append("\n---\n")

    parts.append(_activity_section(presentation["activity"]))
    parts.append("\n---\n")

    for question in presentation["questions"]:
        parts.append(_question_section(question,
                                       insights.get(question["question_id"])))
        parts.append("\n---\n")

    # The reviewer is told to carry the important caveats into its own output, so
    # its list overlaps this one almost exactly. Printing both is what produced the
    # duplicated section; they are merged on a normalised fingerprint instead.
    notes = list(presentation.get("quality_notes") or [])
    notes += [str(caveat) for caveat in
              ((ai_analysis or {}).get("keep_in_mind") or [])]
    merged: list[str] = []
    seen: set[str] = set()
    for note in notes:
        fingerprint = " ".join(str(note).lower().split()).rstrip(".")
        if not fingerprint or fingerprint in seen:
            continue
        # Also drop a caveat that merely restates one already listed.
        if any(fingerprint in existing or existing in fingerprint
               for existing in seen):
            continue
        seen.add(fingerprint)
        merged.append(note)
    if merged:
        parts.append("## Things to keep in mind\n")
        parts += [f"- {note}" for note in merged]

    return "\n".join(parts).strip() + "\n"


def render_no_data(scope_note: str, trace: list[dict[str, Any]]) -> str:
    """The report when nothing matched, explaining which step emptied the set -
    in plain words, not as a filter dump."""
    lines = ["# Form report\n",
             f"**No matching forms.** {scope_note}\n",
             "Where the data went:\n"]
    human = {"form_id": "picking one form", "customer_id": "the chosen customer",
             "representative_id": "the chosen team member",
             "completed_only": "keeping only completed forms",
             "include_autofilled": "excluding automatically filled answers",
             "period": "the chosen dates"}
    for step in trace:
        label = human.get(step["filter"], step["filter"])
        lines.append(f"- After {label}: "
                     f"{step['rows_before']:,} answers became "
                     f"{step['rows_after']:,}")
    if not trace:
        lines.append("- No narrowing was applied; the form has no answers at all.")
    return "\n".join(lines) + "\n"
