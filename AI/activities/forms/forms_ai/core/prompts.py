"""Agent instructions and the payloads they read.

Three agents:

* **QuestionInsight** - one short written insight per question, from that
  question's period-by-period figures. Runs in batches, concurrently.
* **FormAnalyst** - the whole-form verdict: what stands out, what to do.
* **WrittenSummary** - the opening paragraph a reader sees first.

Two things drive every decision in this file.

**1. Plain language.** The model reads the reader's view of the data
(`forms/presentation.py`), not the statistics. Values arrive pre-formatted as
`"57%"` and `"1,886"`; measures arrive named as `"answered Yes"`, not `yes_rate`.
The instructions then forbid the vocabulary the payload no longer contains -
metric keys, question type names, test statistics, spark bars, "erratic",
"flat (low confidence)". A term the model never sees and is told not to invent is
a term that cannot reach a user.

**2. A small input.** Input tokens are the whole cost of this pipeline, and the
statistics payload is large. Four measures, in order of how much they save:

* *Shared period labels.* Period names are listed once and each question sends
  parallel arrays against them, so a label is never repeated per question.
* *Digest plus selective detail.* The analyst gets one line per question, and full
  period detail only for the questions that actually moved or look unusual
  (`FORMS_ANALYST_DETAIL_LIMIT`). A form where nothing changed is nearly free.
* *Pre-formatted strings.* `"57%"` is shorter than `0.5714`, and it is also the
  only form the model can quote - which tightens grounding at the same time.
* *Compact serialisation and hard omission.* No indentation, no null fields, no
  people table for a question where everybody answers alike, no examples of
  written answers beyond a handful of short ones.

`serialise` produces the one string used both as the model's input and as the
grounding allow-list. They must be identical or the check means nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..config import (FORMS_ANALYST_DETAIL_LIMIT, FORMS_MODEL_MAX_PEOPLE,
                      FORMS_MODEL_MAX_PERIODS, FORMS_PAYLOAD_WARN_CHARS)
from .logging_setup import get_log

log = get_log("prompts")


# ---------------------------------------------------------------------------
# Caller-supplied analysis rules
# ---------------------------------------------------------------------------

@dataclass
class AnalysisRules:
    """How this particular analysis should be performed. Everything is optional."""

    audience: str | None = None            # "regional sales manager"
    goal: str | None = None                # "decide where to send field support"
    language: str = "English"
    tone: str | None = None                # "direct, no hedging padding"
    focus_questions: list[str] = field(default_factory=list)
    ignore_questions: list[str] = field(default_factory=list)
    domain_notes: list[str] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    # Per-question targets, keyed by question id or by any fragment of the
    # question text. `thresholds["yes_rate_floor"]` is the blanket fallback for
    # every question answered yes or no. Both are resolved and *checked* in code
    # (forms/targets.py); the model is told the outcome, never asked to judge it.
    targets: dict[str, Any] = field(default_factory=dict)
    max_findings: int = 6
    max_recommendations: int = 5
    extra_rules: list[str] = field(default_factory=list)

    @classmethod
    def coerce(cls, value: Any) -> "AnalysisRules":
        if value is None:
            return cls()
        if isinstance(value, AnalysisRules):
            return value
        if isinstance(value, dict):
            known = {f for f in cls.__dataclass_fields__}
            extra = [f"{k}: {v}" for k, v in value.items() if k not in known]
            return cls(**{k: v for k, v in value.items() if k in known},
                       **({"extra_rules": extra} if extra and "extra_rules"
                          not in value else {}))
        if isinstance(value, str):
            return cls(extra_rules=[value])
        if isinstance(value, Iterable):
            return cls(extra_rules=[str(v) for v in value])
        return cls(extra_rules=[str(value)])

    @staticmethod
    def _target_line(name: str, value: Any) -> str:
        """A target expressed the way the reader should see it.

        The caller's key is an internal field name ("yes_rate_floor"). Echoing it
        into the report is exactly the notation the writing rules forbid, so the
        number is rendered in reader form and the label is explicitly off limits.
        A fraction is written as a percentage: it stops "0.8" reaching the prose,
        and it puts the percentage form in the grounding allow-list.
        """
        shown = value
        if isinstance(value, float) and 0.0 < value < 1.0:
            shown = f"{value * 100:g}%"
        return (f"- Target: {shown}. Say which questions, periods or people fall "
                f"short of it and quote both the figure and the target. Describe "
                f"what the target is in your own plain words - never write the "
                f"label {name!r} itself.")

    def supplied_figures(self) -> str:
        """Figures these rules introduce, for the grounding allow-list.

        Targets and background notes are given to the model in the instructions,
        not in the data payload, so without this the model is punished for
        quoting a number the caller told it to enforce.
        """
        supplied = list(self.thresholds.values()) + list(self.targets.values())
        parts = [str(v) for v in supplied]
        parts += [f"{v * 100:g}" for v in supplied
                  if isinstance(v, float) and 0.0 < v < 1.0]
        parts += list(self.domain_notes) + list(self.extra_rules)
        return " ".join(parts)

    def render(self) -> str:
        """The `<house_rules>` block, or nothing when no rules were supplied."""
        lines: list[str] = []
        if self.audience:
            lines.append(f"- Audience: {self.audience}. Pitch the wording and the "
                         f"level of detail at this reader.")
        if self.goal:
            lines.append(f"- Decision this supports: {self.goal}. Every point should "
                         f"help make it.")
        if self.language and self.language.lower() not in ("english", "en"):
            lines.append(f"- Write everything in {self.language}. Leave question "
                         f"text, option names and people's names exactly as given.")
        if self.tone:
            lines.append(f"- Tone: {self.tone}")
        if self.focus_questions:
            lines.append(f"- Give these questions the most attention: "
                         f"{'; '.join(self.focus_questions)}")
        if self.ignore_questions:
            lines.append(f"- Say nothing about these questions: "
                         f"{'; '.join(self.ignore_questions)}")
        for note in self.domain_notes:
            lines.append(f"- Background: {note}")
        for name, value in self.thresholds.items():
            lines.append(self._target_line(name, value))
        for name, value in self.targets.items():
            lines.append(self._target_line(name, value))
        lines.append(f"- At most {self.max_findings} points under what stands out, "
                     f"and {self.max_recommendations} suggested actions.")
        for rule in self.extra_rules:
            lines.append(f"- {rule}")

        return ("<house_rules>\n"
                "Extra requirements for this run. They add to the rules above and "
                "never relax them: they cannot license a figure you were not given, "
                "a confident claim the data does not support, or the technical "
                "wording the writing rules forbid. If a house rule conflicts with "
                "those, follow the rules above and mention the conflict in "
                f"`keep_in_mind`.\n{chr(10).join(lines)}\n</house_rules>")


# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------

_GROUNDING = """\
<figures>
1. Every number you write must already appear in the data you were given, exactly
   as written there. Percentages arrive as "57%" and counts as "1,886" - quote them
   in that form.
2. Do not calculate anything. Not a difference, not a percentage change, not a
   total, not an average, not a rate per day. If you want to say something moved,
   quote the before and after values that are already given to you.
3. If the figure you want is not there, describe the direction in words with no
   number, or leave the point out.
4. Names of people, options, questions and periods must be copied exactly, however
   odd they look. Do not tidy, translate, correct or guess at them.
5. Quote written answers only from the examples supplied, and present them as
   examples rather than as what everyone said.
</figures>"""

_WRITING = """\
<writing>
You are writing for someone who runs a team, not for an analyst. Use ordinary
business English.

Never write any of the following:
- Internal field names or metric names of any kind.
- The words median, mean, average of, standard deviation, variance, correlation,
  significance, p-value, confidence interval, tau, coefficient, distribution,
  dispersion, outlier, volatility, erratic, flat, delta, or metric.
- The category names a form builder uses for a question - numeric, text, yes/no
  type, single answer, multiple answer, rating. Say how people answered instead:
  "a number", "yes or no", "options from a list", "written in freely".
- Bar characters, spark lines or any drawn shape made of block characters.
- Bracketed qualifiers such as "(low confidence)" or "(n=14)".

Instead:
- "57% of forms said yes", not "yes_rate 0.5714".
- "ended the quarter higher than it started, 62% up to 71%", not "rising trend,
  delta 0.09".
- "moved up and down all quarter with no settled level", not "erratic".
- "there is no clear change", not "flat (low confidence)".
- "too few answers in that week to read anything into it", not "low n".

Every claim must be a sentence a manager could repeat out loud in a meeting.
</writing>"""

_UNCERTAINTY = """\
<being_careful>
1. The data already tells you, in words, how each question moved and how reliable
   that is. Use its wording; do not upgrade it. If it says there is no clear
   change, there is no clear change. If it says something drifted but not
   consistently, do not call it a trend.
2. **Only name a period as highest, lowest, best, worst, a peak or a dip if it
   appears in `periods_that_stand_out`.** Where `no_period_stands_out` is true, the
   differences between periods are no larger than the number of answers in each one
   would produce by chance. Reading a pattern out of them - "peaked in February",
   "dipped in early March" - is inventing a finding. Say that the figure held
   steady within normal variation, and move on.
3. **Only name a person if they appear in `stands_out`.** The `verdict` sentence
   under `people` has already applied a floor on how many answers are needed and a
   test against the rest of the team. Do not rank people yourself, do not compare
   their percentages, and do not describe anyone as behind or ahead unless the data
   flagged them.
4. **Do not invent a standard.** If no target is given for a question, you may not
   describe a figure as low, poor, below expectations or short of what is
   reasonable. Report what it is and how it moved. Where a target is supplied, the
   comparison has already been made for you: quote its wording rather than judging
   it yourself.
5. Where a period is marked as having very few answers, is only partly inside the
   dates, or is shorter than the others, you may mention the figure but must not
   build a point on it.
6. Where the data says no single figure represents the answers, do not invent one,
   and do not describe the answers as rising or falling. Such a question carries
   `how_answers_are_spread` instead: the shares in each band are what moved, and
   they are what to write about. That there is no single figure is itself a
   finding - people are answering the same question about very different things.
7. Figures show that two things moved together, never that one caused the other.
   Write "at the same time as", not "because of" or "driven by".
8. Anything under `keep_in_mind` affects how much the rest can be trusted. Carry
   across at most the two that change how the figures should be read, in your own
   words - the report already prints the full list, so repeating it verbatim just
   duplicates a section.
</being_careful>"""

_WHAT_MATTERS = """\
<what_matters_here>
`needs_attention` is the list of questions that warrant a decision, already ranked,
already checked against the targets and the sample sizes. It is not a hint - it is
the answer to "what should this report be about".

1. Open with what is in `needs_attention`, in that order, and give it most of your
   words. Its `issue` wording has been computed and is safe to quote or paraphrase.
2. A question that is not in the list held steady. Say so briefly and collectively -
   "the remaining checks held steady" - rather than writing a paragraph each. If the
   list is empty, the honest write-up is short: nothing here needs a decision yet.
3. Do not promote a question into the list yourself, and do not demote one out of
   it. If you think something else matters, say why in `keep_in_mind`.
4. `data_may_not_be_real`, when present, comes before every other finding, in the
   opening paragraph. It means parts of this data are not a record of what people
   did - answers generated by a machine, placeholder accounts, values spaced too
   regularly to be real counts. Say so plainly and say which questions it affects.
   Then continue with the analysis anyway: the figures are still what was asked
   for, and the reader decides what to do about the source. Never write the analysis
   as though the data were sound while leaving this in a footnote.
</what_matters_here>"""

_HOW_PERIODS_WORK = """\
<how_the_periods_work>
The date range was split into equal periods of the same length so they can be
compared against each other. `periods` lists them oldest first, and every
question's `values` and `answers` arrays line up with that list position by
position. `activity` covers how much came in overall; each question then has its
own figures for the same periods.

How many answers each period holds decides what can be said about it. A yes-rate
built on a dozen answers swings across a 50-point range by chance alone, so the
figures have already been checked against the range their own sample size would
produce: `periods_that_stand_out` lists the periods that fall outside it, and
`no_period_stands_out` means none do. Comparisons are only worth making on the
periods that stand out, and against `vs_previous` where it is given.

Periods with no answers are included on purpose - a gap is a finding, not a missing
row.
</how_the_periods_work>"""


# ---------------------------------------------------------------------------
# Agent 1: one insight per question
# ---------------------------------------------------------------------------

def question_insight_instructions(rules: AnalysisRules | None = None) -> str:
    rules = rules or AnalysisRules()
    return f"""\
You write one short insight for each form question you are given. It is printed
directly underneath that question's figures, so it must add something a reader
cannot see for themselves in the table - what the pattern means, what it suggests
doing, or what makes it worth a second look. No cross-question comparison and no
overall recommendations; another writer handles those.

{_HOW_PERIODS_WORK}

For each question:
- Lead with what the answers actually say, in one sentence.
- Then the pattern over the periods: steady, a step up or down, a spike, a slide,
  or moving about with no settled level. Name the periods involved.
- If the people differ from each other, say who is unusual and by how much.
- If the answers do not settle around one value, say that plainly and say what it
  probably means about how the question is being understood.
- Two to four sentences. Stop when you have said the useful thing.

{_GROUNDING}

{_WRITING}

{_UNCERTAINTY}

{_WHAT_MATTERS}

{rules.render()}

<output_format>
Return one JSON object and nothing else - no prose around it, no code fence:

{{"insights": [{{"question_id": "copied exactly from the data",
                "insight": "two to four sentences",
                "watch_out": "one short caution, or null"}}]}}
</output_format>"""


# ---------------------------------------------------------------------------
# Agent 2: the whole-form verdict
# ---------------------------------------------------------------------------

_ANALYST_SCHEMA = """\
{
  "headline": "one sentence a manager would read first",
  "activity": "how much came in and whether that is holding up, in one or two sentences",
  "vs_previous_period": "one sentence, or null if no comparison was given",
  "stands_out": [
    {"point": "one specific thing",
     "evidence": "the figures behind it, quoted from the data",
     "how_sure": "certain | likely | worth checking",
     "questions": ["question ids it comes from"],
     "why_it_matters": "what the reader should take from it"}
  ],
  "people": [
    {"who": "name copied exactly",
     "what": "how they differ",
     "evidence": "figures quoted from the data"}
  ],
  "worth_watching": [
    {"issue": "...", "evidence": "...", "how_urgent": "high | medium | low"}
  ],
  "do_next": [
    {"action": "concrete and doable this month",
     "because": "points back to a figure above",
     "priority": "high | medium | low"}
  ],
  "keep_in_mind": ["what limits how far these figures can be pushed"]
}"""


def form_analyst_instructions(rules: AnalysisRules | None = None) -> str:
    rules = rules or AnalysisRules()
    return f"""\
You review a set of completed forms and report what a manager needs to know: what
the answers say, what changed over the period, who differs from whom, and what to
do about it. The figures are already worked out for you and described in plain
words - your job is judgement, not calculation.

{_HOW_PERIODS_WORK}

<what_you_are_given>
- `period`, `scope`, `split` - what this covers and how it was divided.
- `periods` - the period names, oldest first.
- `activity` - how many forms came in per period, and who submitted them.
- `questions` - one entry per question. Every entry has the question text, what is
  being measured, the figure for the whole period, and a plain description of how
  it moved. Entries for the questions that moved or look unusual also carry
  `by_period` (values and answer counts, lining up with `periods`), the people who
  differ, and extra detail.
- `keep_in_mind` - read this before trusting anything else.
</what_you_are_given>

<method>
1. Read `keep_in_mind` first. If the data cannot support a review, say so in the
   headline and keep the rest short.
2. Establish how much came in and whether that is holding up.
3. Go through the questions. Give weight to the ones with `by_period` detail -
   those are the ones that moved or look unusual. Mention a steady question only
   if steadiness is itself worth saying.
4. Compare against the previous period where a comparison is given.
5. Look for questions that moved together, and for people who sit apart from the
   rest.
6. Only then write what stands out and what to do. Every point must trace to a
   figure you already quoted.
</method>

{_GROUNDING}

{_WRITING}

{_UNCERTAINTY}

{_WHAT_MATTERS}

{rules.render()}

<output_format>
Return one JSON object and nothing else. Use this shape exactly; use null or [] for
anything the data does not support:

{_ANALYST_SCHEMA}
</output_format>"""


# ---------------------------------------------------------------------------
# Agent 3: the opening summary
# ---------------------------------------------------------------------------

def written_summary_instructions(rules: AnalysisRules | None = None) -> str:
    rules = rules or AnalysisRules()
    return f"""\
You write the opening summary of a form report - the part someone reads before the
tables. The review has already been done; you are writing it up.

Write two or three short paragraphs of flowing prose:
1. What this covers, how much came in, and the single most important thing in it.
2. What changed, with the figures.
3. What to do about it.

Rules:
- 150 to 300 words. Shorter beats padded.
- No headings, no bullet points, no lists.
- Every figure must be copied from the review you were given. Work nothing out,
  including totals or averages of figures already there.
- Keep the review's level of certainty. If it says something is worth checking, do
  not present it as established.
- If the review says any part of the data may not be a real record of what people
  did - answers that look machine-generated, placeholder accounts - that goes in
  the first paragraph, before any finding. A reader who is told this at the end has
  already believed the rest.
- Where the review says a figure holds steady rather than moving, do not turn it
  into a trend, and do not name a period as high or low unless the review does.

{_GROUNDING}

{_WRITING}

{rules.render()}

Return the prose only - no JSON, no code fence, no title."""


# ---------------------------------------------------------------------------
# Repair pass (required by llm.write_with_grounding)
# ---------------------------------------------------------------------------

async def prompt_repair_input(bad_figures: list[str], previous_output: str,
                              bad_wording: list[str] | None = None) -> str:
    """The second-chance prompt. Names the offending numbers and closes the two easy
    escapes: inventing a different number, or deleting the sentence.

    Wording slips ride along in the same pass: a run that has to be repaired
    anyway should not pay for a second call to fix its vocabulary.
    """
    unique = sorted(set(bad_figures), key=lambda token: (-len(token), token))
    listed = ", ".join(f"`{token}`" for token in unique[:40])
    more = f" (and {len(unique) - 40} more)" if len(unique) > 40 else ""

    sections = []
    if unique:
        sections.append(f"""\
Some numbers were not in the data you were given. Most often this happens because a
difference or a percentage change was worked out rather than quoted.

Numbers with no source: {listed}{more}

Fix each one by doing exactly one of:
- replace it with the figure that is actually in the data;
- rewrite the sentence so it makes the point in words with no number;
- remove the point.

Do not swap in a different invented number.""")
    if bad_wording:
        terms = ", ".join(f"`{term}`" for term in sorted(set(bad_wording))[:20])
        sections.append(f"""\
Wording a manager would not use: {terms}

These are internal names and statistical jargon. Rewrite each sentence so it says
the same thing in ordinary business English, with no field name and no technical
term. Do not simply delete the sentence.""")

    body = "\n\n".join(sections)
    return f"""\
Your previous answer needs one correction pass.

{body}

Do not delete surrounding content that was already correct. Keep the same format,
structure and length.

Your previous answer:
---
{previous_output}
---

Return the corrected answer only."""


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def _thin(values: list[Any], labels: list[str], limit: int
          ) -> tuple[list[Any], list[str], bool]:
    """Keep a series readable within `limit` points.

    The first, the last and the extremes are always kept - they are what any claim
    about movement rests on - and the remainder are sampled evenly. The full series
    stays in `statistics` and in the report; only the model's copy is thinned.
    """
    if len(values) <= limit:
        return values, labels, False

    numeric = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    must_keep = {0, len(values) - 1}
    if numeric:
        must_keep.add(max(numeric, key=lambda pair: pair[1])[0])
        must_keep.add(min(numeric, key=lambda pair: pair[1])[0])
    remaining = limit - len(must_keep)
    if remaining > 0:
        step = max(1, len(values) // (remaining + 1))
        must_keep.update(range(0, len(values), step))
    keep = sorted(must_keep)[:limit]
    return ([values[i] for i in keep], [labels[i] for i in keep], True)


def _numeric_series(question: dict[str, Any]) -> list[float | None]:
    return [row.get("raw") for row in question.get("by_period", [])]


def interest_score(question: dict[str, Any]) -> float:
    """(see below) - scored on what can actually be said about the question."""
    """How much this question deserves full period detail in the analyst's payload.

    Movement is worth the most, then answers that do not settle around one value,
    then people disagreeing, then a change against the previous period. A question
    that is steady, consistent and unanimous scores zero and costs one line.
    """
    score = 0.0
    movement = (question.get("movement") or "").lower()
    if "no real change here" in movement:
        # Variation entirely explained by sample size: there is nothing to review,
        # and sending the period detail would only invite a story about it.
        return 0.0
    if "consistently" in movement and "not consistently" not in movement:
        score += 3.0
    elif "drifted" in movement:
        score += 1.5
    elif "no clear direction" in movement:
        score += 1.0

    spread = (question.get("spread") or "").lower()
    if "two clearly separate groups" in spread:
        score += 2.5
    elif "far too widely" in spread:
        score += 2.0

    people = question.get("by_person") or {}
    if people.get("differ"):
        score += 1.5
    if question.get("against_previous") and "unchanged" not in \
            (question["against_previous"] or ""):
        score += 0.5
    if any(row.get("stands_out") for row in question.get("by_period", [])):
        score += 1.0
    return score


def _question_line(question: dict[str, Any]) -> dict[str, Any]:
    """The one-line form: what it asks, what the answer is, how it moved."""
    line = {
        "question_id": question["question_id"],
        "n": question["number"],
        "q": question["question"],
        "measure": question["measure"],
        "overall": question["headline"],
        "movement": question["movement"],
    }
    if question.get("spread"):
        line["spread"] = question["spread"]
    if question.get("against_previous"):
        line["vs_previous"] = question["against_previous"]
    return line


def _question_detail(question: dict[str, Any], labels: list[str],
                     max_periods: int, max_people: int) -> dict[str, Any]:
    """The one-line form plus the period arrays, people and extras."""
    detail = _question_line(question)
    rows = question.get("by_period", [])
    values = [row["value"] for row in rows]
    answers = [int(str(row["answers"]).replace(",", "") or 0) for row in rows]

    if question.get("hide_by_period"):
        # These answers have no middle value, so the per-period middle is an
        # artefact of which group got one extra answer. Handing it over is
        # handing over a story to tell about nothing; the band shares are the
        # figures that actually move here.
        spread = question.get("spread_table") or {}
        detail["by_period"] = {
            "note": "this question has no single typical answer, so no figure "
                    "per period is given - use how_answers_are_spread instead",
        }
        # Shares only, positionally aligned with the shared `periods` array at the
        # top of the payload - repeating the period labels once per band table
        # would undo the saving that array exists for.
        detail["how_answers_are_spread"] = {
            "bands": spread.get("bands"),
            "share_of_all_answers": [row["share"] for row in
                                     (spread.get("overall") or [])],
            "share_per_period_in_order": [
                row["shares"] for row in
                (spread.get("by_period") or [])][:max_periods],
        }
    else:
        kept_values, kept_labels, thinned = _thin(values, labels, max_periods)
        if thinned:
            kept_answers, _, _ = _thin(answers, labels, max_periods)
            detail["by_period"] = {"periods": kept_labels, "values": kept_values,
                                   "answers": kept_answers,
                                   "note": "a sample of the periods, oldest first"}
        else:
            detail["by_period"] = {"values": kept_values, "answers": answers}
    unreadable = [row["period"] for row in rows
                  if row.get("too_few") or row.get("part_period")
                  or row.get("short_period")]
    if unreadable:
        detail["periods_not_to_read_into"] = unreadable[:6]
    standout_periods = [row["period"] for row in rows if row.get("stands_out")]
    detail["periods_that_stand_out"] = standout_periods or None
    if not standout_periods:
        # Stated as a fact rather than left to inference: no period here differs
        # from the others by more than its sample size explains.
        detail["no_period_stands_out"] = True

    people = question.get("by_person") or {}
    if people:
        # The sentence, not the table: it already encodes whether the people
        # genuinely differ, who cleared a significance test, and how many answered
        # too few times to be compared. Handing over raw percentages instead is what
        # let a 1-in-6 figure be quoted as a finding.
        detail["people"] = {"verdict": people["note"]}
        if people.get("everyone_counted_once"):
            detail["people"]["everyone_counted_once"] = \
                people["everyone_counted_once"]
        if people.get("differ") and people.get("rows"):
            standouts = [row for row in people["rows"] if row.get("stands_out")]
            detail["people"]["stands_out"] = [
                {"name": row["name"], "figure": row["value"],
                 "adjusted": row["adjusted"], "answers": row["answers"]}
                for row in standouts[:max_people]]
    if question.get("details"):
        detail["extra"] = {row["label"]: row["value"]
                           for row in question["details"][:4]}
    if question.get("options"):
        detail["answers_given"] = [f"{row['option']}: {row['share']}"
                                   for row in question["options"][:6]]
    if question.get("notes"):
        detail["notes"] = question["notes"]
    return detail


def build_reader_payload(presentation: dict[str, Any], *,
                         question_ids: list[str] | None = None,
                         detail_ids: list[str] | None = None,
                         include_activity: bool = True,
                         max_periods: int = FORMS_MODEL_MAX_PERIODS,
                         max_people: int = FORMS_MODEL_MAX_PEOPLE
                         ) -> dict[str, Any]:
    """What an agent reads.

    `question_ids` limits which questions appear at all (used for batching);
    `detail_ids` selects which of those get full period arrays. Pass
    `detail_ids=[]` for a pure digest, or None to give every included question
    full detail.
    """
    labels = [row["period"] for row in presentation["activity"]["by_period"]]
    questions = presentation["questions"]
    if question_ids is not None:
        wanted = set(question_ids)
        questions = [q for q in questions if q["question_id"] in wanted]
    detail = set(q["question_id"] for q in questions) if detail_ids is None \
        else set(detail_ids)

    payload: dict[str, Any] = {
        "period": presentation["period"],
        "scope": presentation["scope"],
        "split": presentation["period_split"],
        "periods": labels,
        # First in the payload, because it is what the write-up should open with.
        # Computed in code, not asked of the model: a model invited to decide what
        # matters will always find six things, and will rank the question that
        # nothing complied with tenth.
        # Keyed by question number only: the question text and its headline figure
        # are already in `questions`, and repeating them here was the single
        # largest avoidable cost in this payload.
        "needs_attention": [
            {"n": row["number"], "issue": row["issue"],
             "what_to_do": row["what_to_do"]}
            for row in (presentation.get("attention") or [])],
        "questions": [
            _question_detail(q, labels, max_periods, max_people)
            if q["question_id"] in detail else _question_line(q)
            for q in questions],
    }
    if include_activity:
        activity = presentation["activity"]
        forms, _, _ = _thin([row["forms"] for row in activity["by_period"]],
                            labels, max_periods)
        payload["activity"] = {
            "summary": f"{activity['headline']}. {activity['summary']}",
            "forms_per_period": forms,
            "busiest": activity.get("busiest"),
            "quietest": activity.get("quietest"),
            "by_person": [f"{row['name']}: {row['forms']} ({row['share']})"
                          for row in activity.get("by_person", [])[:10]],
        }
    if presentation.get("quality_notes"):
        payload["keep_in_mind"] = presentation["quality_notes"]
    if presentation.get("data_warnings"):
        # Kept separate from `keep_in_mind`, which is a list of caveats about how
        # far figures can be pushed. These are stronger than a caveat: they say
        # parts of the data are not a record of what happened. Every question is
        # still analysed - the reader asked for statistics on whatever they have -
        # but the write-up has to lead with this rather than bury it.
        payload["data_may_not_be_real"] = presentation["data_warnings"]
    return payload


def choose_detail_questions(presentation: dict[str, Any],
                            limit: int = FORMS_ANALYST_DETAIL_LIMIT) -> list[str]:
    """Which questions earn full period detail in the analyst's payload."""
    scored = sorted(((interest_score(q), q["number"], q["question_id"])
                     for q in presentation["questions"]),
                    key=lambda row: (-row[0], row[1]))
    chosen = [question_id for score, _, question_id in scored
              if score > 0][:max(0, limit)]
    if not chosen and scored:                 # nothing moved: send the first few
        chosen = [question_id for _, _, question_id in scored[:min(2, limit)]]
    return chosen


def serialise(payload: dict[str, Any]) -> str:
    """One canonical serialisation, used as the model's input *and* as the
    grounding allow-list. Compact on purpose: separators without spaces and no
    indentation save roughly a quarter of the characters of a pretty-printed
    payload, for no loss of meaning."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                      default=str)


def _sized(name: str, text: str) -> str:
    if len(text) > FORMS_PAYLOAD_WARN_CHARS:
        log.warning("  payload[%s] is %s chars (~%s tokens), above the %s budget",
                    name, len(text), len(text) // 4, FORMS_PAYLOAD_WARN_CHARS)
    else:
        log.info("  payload[%s]: %s chars (~%s tokens)", name, len(text),
                 len(text) // 4)
    return text


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _allow(data: str, rules: AnalysisRules | None) -> str:
    """The grounding allow-list: the data payload plus any figure the caller's
    rules told the model to enforce.

    Without the second part, a run configured with a target loses a repair pass
    on every agent to a number it was instructed to quote.
    """
    supplied = AnalysisRules.coerce(rules).supplied_figures() if rules else ""
    return f"{data}\n{supplied}" if supplied.strip() else data


async def prompt_question_insights(presentation: dict[str, Any],
                                   question_ids: list[str], *,
                                   rules: AnalysisRules | None = None
                                   ) -> tuple[str, str, str]:
    """(instructions, user_input, allowed) for one batch of questions."""
    payload = build_reader_payload(presentation, question_ids=question_ids,
                                  include_activity=False)
    data = _sized(f"insights[{len(question_ids)}q]", serialise(payload))
    header = (f"Write an insight for each of these {len(question_ids)} question(s)."
              f"\n\n<data>\n{data}\n</data>")
    return question_insight_instructions(rules), header, _allow(data, rules)


async def prompt_form_analysis(presentation: dict[str, Any], *,
                               rules: AnalysisRules | None = None,
                               detail_limit: int = FORMS_ANALYST_DETAIL_LIMIT
                               ) -> tuple[str, str, str]:
    """(instructions, user_input, allowed) for the whole-form review."""
    detail_ids = choose_detail_questions(presentation, detail_limit)
    payload = build_reader_payload(presentation, detail_ids=detail_ids)
    data = _sized("analyst", serialise(payload))
    header = f"Review these forms.\n\n<data>\n{data}\n</data>"
    return form_analyst_instructions(rules), header, _allow(data, rules)


async def prompt_written_summary(analysis: dict[str, Any],
                                 insights: list[dict[str, Any]], *,
                                 presentation: dict[str, Any] | None = None,
                                 rules: AnalysisRules | None = None
                                 ) -> tuple[str, str, str]:
    """(instructions, user_input, allowed) for the opening summary.

    The summary reads the review, not the raw data - it is a write-up, and sending
    the figures again would double the cost of the cheapest stage. The allow-list
    still includes the digest so any figure the review quoted is accepted.
    """
    body = {"review": analysis,
            "question_insights": [
                {"question_id": row.get("question_id"), "insight": row.get("insight")}
                for row in insights if isinstance(row, dict)]}
    user_input = f"Write the summary from this review.\n\n<review>\n{serialise(body)}\n</review>"
    allowed = serialise(body)
    if presentation:
        allowed += "\n" + serialise(
            build_reader_payload(presentation, detail_ids=[]))
    _sized("summary", user_input)
    return written_summary_instructions(rules), user_input, _allow(allowed, rules)
