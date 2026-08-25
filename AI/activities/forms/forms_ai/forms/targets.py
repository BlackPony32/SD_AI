"""Targets, and the short list of things that actually need attention.

Two jobs, deliberately kept together because the second depends on the first.

**Targets.** A caller can say "at least 80% of forms should answer yes". Until now
that reached the model as a sentence in its instructions and the model decided,
in prose, whether a question fell short. That is the wrong place for the decision:
it is arithmetic, it should be computed once, and the answer should be the same
every run. `resolve()` finds the target that applies to a question and `status()`
says whether it is met and - the part a manager can act on - how many forms would
have to change to meet it.

**Attention.** A report with twelve questions where eight of them say "no real
change, everyone normal" buries its own findings. `attention()` ranks the
questions by how much they warrant a decision and returns only those that do, so
the report can open with them. The ranking is computed from the statistics, not
asked of the model, for the same reason: a model asked "what matters here?" will
always find six things.

Nothing here suppresses a question. Every question is still analysed and still
appears in full further down; this only decides what goes first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_log

log = get_log("forms.targets")

# How many rows the attention table may hold. Beyond this it stops being a short
# list and becomes the report again.
MAX_ATTENTION = 6

# Severity weights. Ordering between categories is what matters, not the numbers.
_SEVERITY = {
    "requirement_never_met": 100,   # the question states a minimum; nothing met it
    "fails_target":           80,   # a target the caller set is missed
    "no_typical_value":       55,   # answers split in two - a definition problem
    "period_stands_out":      45,   # a period outside what chance would produce
    "person_stands_out":      40,   # someone significantly different from the team
    "answers_look_generated": 35,   # the answers cannot be read as what was reported
}


@dataclass(frozen=True)
class Target:
    """A level a question is expected to reach."""

    value: float
    scope: str          # "this question" | "every yes or no question"
    as_share: bool      # True when the value is a share of forms

    def reached_by(self, value: float | None) -> bool | None:
        if value is None:
            return None
        return float(value) >= self.value


# ---------------------------------------------------------------------------
# Resolving which target applies to which question
# ---------------------------------------------------------------------------

def _as_target(value: Any, scope: str) -> Target | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:                                    # NaN
        return None
    # A share may be given either way round: 0.8 and 80 both mean 80% of forms.
    if 0.0 < number <= 1.0:
        return Target(number, scope, as_share=True)
    if 1.0 < number <= 100.0:
        return Target(number / 100.0, scope, as_share=True)
    return Target(number, scope, as_share=False)


def resolve(block: dict[str, Any], rules: Any = None) -> Target | None:
    """The target for one question, or None.

    Precedence, most specific first: an entry keyed by this question's id, then
    the longest matching fragment of the question text, then a blanket floor for
    every question answered yes or no.
    """
    if rules is None:
        return None
    per_question = dict(getattr(rules, "targets", None) or {})
    kind = (block.get("primary_metric") or {}).get("kind")

    question_id = str(block.get("question_id") or "")
    if question_id and question_id in per_question:
        return _as_target(per_question[question_id], "this question")

    text = str(block.get("question") or "").lower()
    matches = [(key, value) for key, value in per_question.items()
               if key and str(key).lower() in text]
    if matches:
        key, value = max(matches, key=lambda pair: len(str(pair[0])))
        return _as_target(value, "this question")

    thresholds = dict(getattr(rules, "thresholds", None) or {})
    floor = thresholds.get("yes_rate_floor")
    if floor is not None and kind == "rate":
        return _as_target(floor, "every yes or no question")
    return None


def status(block: dict[str, Any], target: Target | None) -> dict[str, Any] | None:
    """Whether the question meets its target, and what closing the gap means.

    The shortfall is expressed in forms rather than percentage points because
    "48 more forms need to say yes" is a workload and "28 percentage points" is
    not.
    """
    if target is None:
        return None
    primary = block.get("primary_metric") or {}
    value = primary.get("overall")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)

    answered = int((block.get("overall") or {}).get("answered") or 0)
    shortfall_forms = None
    if target.as_share and answered and value < target.value:
        # How many of the answers already given would have had to go the other way.
        shortfall_forms = int(math.ceil((target.value - value) * answered))

    periods = [row for row in (block.get("by_interval") or [])
               if isinstance(row.get("primary_value"), (int, float))
               and not row.get("short")]
    reached = [row for row in periods if row["primary_value"] >= target.value]

    return {
        "value": target.value,
        "as_share": target.as_share,
        "scope": target.scope,
        "met": target.reached_by(value),
        "shortfall": round(target.value - value, 6) if value < target.value else 0.0,
        "shortfall_forms": shortfall_forms,
        "periods_reaching_it": len(reached),
        "periods_compared": len(periods),
        "never_reached": bool(periods) and not reached,
    }


# ---------------------------------------------------------------------------
# What needs attention
# ---------------------------------------------------------------------------

def _requirement_never_met(block: dict[str, Any]) -> dict[str, Any] | None:
    """A question that states a minimum which no answer met.

    This is the strongest finding a form can produce and it needs no target from
    the caller: the question itself says what it wants. It is also the finding
    most easily lost in a long report, because 0% looks like an empty row.
    """
    overall = block.get("overall") or {}
    minimum = overall.get("required_minimum")
    if not minimum:
        return None
    meeting = overall.get("meeting_minimum")
    answered = int(overall.get("answered") or 0)
    if meeting is None or not answered:
        return None
    if int(meeting) > 0:
        return None
    listed = overall.get("items_listed") or {}
    return {"reason": "requirement_never_met", "answers": answered,
            "asked_for": int(minimum),
            "typically_listed": listed.get("median")}


def _people_standing_out(statistics: dict[str, Any], question_id: str) -> list[str]:
    """Who was flagged as significantly different on this question.

    Read from the segment rows rather than the question block, because that is
    where shrinkage and the significance test were applied.
    """
    for segment in ((statistics.get("segments") or {}).get("by_representative") or []):
        if segment.get("question_id") != question_id:
            continue
        return [row.get("representative_label")
                for row in (segment.get("rows") or [])
                if row.get("stands_out") and not row.get("too_few_to_compare")]
    return []


def _finding(block: dict[str, Any], statistics: dict[str, Any],
             rules: Any) -> dict[str, Any] | None:
    """The single most serious thing about one question, or None if it is quiet."""
    trend = block.get("trend") or {}
    overall = block.get("overall") or {}
    primary = block.get("primary_metric") or {}
    target = resolve(block, rules)
    target_status = status(block, target)

    reasons: list[dict[str, Any]] = []

    never_met = _requirement_never_met(block)
    if never_met:
        reasons.append(never_met)

    if target_status and target_status.get("met") is False:
        reasons.append({"reason": "fails_target",
                        "target": target_status["value"],
                        "value": primary.get("overall"),
                        "shortfall_forms": target_status.get("shortfall_forms"),
                        "never_reached": target_status.get("never_reached")})

    dispersion = overall.get("dispersion") or {}
    if isinstance(dispersion, dict) and dispersion.get(
            "single_value_representative") is False:
        reasons.append({"reason": "no_typical_value",
                        "shape": dispersion.get("shape"),
                        "typical_low": dispersion.get("typical_low"),
                        "typical_high": dispersion.get("typical_high")})

    notable = trend.get("notable_periods") or []
    if notable:
        reasons.append({"reason": "period_stands_out",
                        "periods": [row.get("label") for row in notable][:3],
                        "direction": notable[0].get("direction")})

    people = _people_standing_out(statistics, block.get("question_id"))
    if people:
        reasons.append({"reason": "person_stands_out", "who": people[:3]})

    templating = overall.get("templating") or {}
    if isinstance(templating, dict) and templating.get("looks_generated"):
        reasons.append({"reason": "answers_look_generated",
                        "share": templating.get("template_share"),
                        "example": templating.get("dominant_example")})

    if not reasons:
        return None

    reasons.sort(key=lambda row: -_SEVERITY.get(row["reason"], 0))
    top = reasons[0]
    return {
        "question_id": block.get("question_id"),
        "order_index": block.get("order_index"),
        "question": block.get("question"),
        "reason": top["reason"],
        "severity": _SEVERITY.get(top["reason"], 0),
        "detail": top,
        "also": [row["reason"] for row in reasons[1:]],
        "target": target_status,
        "measure": primary.get("label"),
        "value": primary.get("overall"),
        "kind": primary.get("kind"),
        "answers": int(overall.get("answered") or 0),
    }


def attention(statistics: dict[str, Any], rules: Any = None,
              limit: int = MAX_ATTENTION) -> list[dict[str, Any]]:
    """The questions worth a decision, most serious first.

    Ties break on how many answers the finding rests on, then on the question's
    position in the form, so the order is stable across runs.
    """
    found = [_finding(block, statistics, rules)
             for block in (statistics.get("questions") or [])]
    ranked = sorted((row for row in found if row),
                    key=lambda row: (-row["severity"], -row["answers"],
                                     row.get("order_index") or 0))
    if len(ranked) > limit:
        log.info("attention: %s question(s) flagged, showing the %s most serious",
                 len(ranked), limit)
    return ranked[:limit]


def apply_targets(statistics: dict[str, Any], rules: Any = None) -> None:
    """Attach the resolved target to every question block, in place."""
    for block in statistics.get("questions") or []:
        resolved = status(block, resolve(block, rules))
        if resolved is not None:
            block["target"] = resolved
