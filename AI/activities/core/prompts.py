"""Prompt material shared by every topic.

Two ideas run through all of them:
  * The model is a WRITER, not a CALCULATOR. It receives figures already
    computed and is told, in as many words, that inventing or re-deriving a
    number is the one unrecoverable error.
  * Each agent may only speak about what it was given. The split is enforced by
    the payloads, not by good intentions.

The rules are named rather than copied. Each topic composes the ones that apply
to it with `build_rules`, which numbers them in order, so the sentences every
agent shares are written once and the topic-specific ones stay local.
"""

from __future__ import annotations

import json
from typing import Any

RULES: dict[str, str] = {
    "no_invented_numbers": """\
Never state a number that is not in your input. Copy figures exactly; do not
   re-derive, sum, re-round or convert them. If the number you want does not
   exist, describe the pattern in words instead.""",

    "business_language": """\
Write for a business owner, not an analyst. No field names, no JSON keys, no
   "the data shows". Short sentences.""",

    "dull_truth": """\
Report what is there, including when it is dull. A dull true finding beats an
   exciting invented one, and saying nothing is a valid answer.""",

    "json_only": """\
Return ONLY valid JSON matching the schema. No markdown fences, no preamble.""",

    "respect_reliability": """\
Respect the `reliability` field on every comparison. When it reads
   `low_sample_low_confidence`, say the comparison rests on very few events, or
   drop the comparison. Never present it as a trend.""",

    "respect_basis": """\
`basis` qualifies a percentage: `no_baseline` means the previous period had
   none (say "up from none", not a percentage), `no_activity` means both periods
   were empty (say nothing), `dropped_to_zero` means the activity stopped.""",

    "human_labels": """\
Use the human labels exactly as given -- "Task Added", never TASK_ADDED. The
   reader does not know the database's vocabulary and should not have to.""",

    "channel_is_not_a_person": """\
A representative whose `kind` is "Channel / integration" is a bucket of many
   people or a machine. Never write about one as if it were a person.""",
}


def build_rules(*names: str, header: str = "Hard rules, in priority order:") -> str:
    """Numbered rule block, in the order given."""
    return "\n".join([header] + [f"{i}. {RULES[n]}" for i, n in enumerate(names, 1)])


# The four every JSON-returning agent uses, in the order they have always been in.
SHARED_RULES = build_rules("no_invented_numbers", "business_language", "dull_truth", "json_only")


def json_payload(payload: Any, label: str = "payload") -> str:
    """Compact separators, no indent: models parse it just as reliably as pretty
    JSON and it is roughly a third fewer input tokens on every single run.
    Never raises -- a degraded prompt beats an exception that kills the report."""
    try:
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception as exc:
        return f'{{"error":"could not serialise {label}: {type(exc).__name__}"}}'


async def prompt_repair_input(unsupported: list[str], previous: str) -> str:
    """Sent as the user turn on a second pass; the instructions stay unchanged so
    the model rewrites within the same brief instead of drifting."""
    listed = ", ".join(sorted(set(unsupported))[:20])
    return f"""Your previous output contained figures that do not appear in the inputs \
you were given: {listed}

Produce it again. Every one of those figures must be either replaced with the correct \
figure from the input, or removed and the sentence rewritten in words with no number at \
all. Change nothing else: same structure, same format, same ordering, same length.

PREVIOUS OUTPUT:
{previous}"""
