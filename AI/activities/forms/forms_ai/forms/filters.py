"""Filtering, applied before anything is counted.

Order matters and is fixed: form -> customer -> representative -> completion ->
period. Every step records how many rows it removed, because "the numbers look
low" is almost always a filter question, and a report that cannot explain its own
denominator is not auditable.

v1 filters on date range and `representative_id`. `customer_id` is wired through
end to end but optional: the current export has no customer column, so passing
one raises a clear error instead of silently returning zero rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from ..core.logging_setup import get_log

log = get_log("forms.filters")


class FilterError(ValueError):
    """The requested filter cannot be applied to this dataset."""


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        return [text] if text.strip() else None
    if isinstance(value, Iterable):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out or None
    return [str(value)]


@dataclass
class FilterSpec:
    form_id: str | None = None
    representative_id: str | list[str] | None = None
    exclude_representative_id: str | list[str] | None = None
    customer_id: str | list[str] | None = None
    period_from: Any = None
    period_to: Any = None
    completed_only: bool = False
    include_autofilled: bool = True

    def normalised(self) -> dict[str, Any]:
        return {
            "form_id": self.form_id,
            "representative_id": _as_list(self.representative_id),
            "exclude_representative_id": _as_list(self.exclude_representative_id),
            "customer_id": _as_list(self.customer_id),
            "period_from": (pd.Timestamp(self.period_from).isoformat()
                            if self.period_from is not None else None),
            "period_to": (pd.Timestamp(self.period_to).isoformat()
                          if self.period_to is not None else None),
            "completed_only": self.completed_only,
            "include_autofilled": self.include_autofilled,
        }


@dataclass
class FilterResult:
    facts: pd.DataFrame
    questions: pd.DataFrame
    spec: FilterSpec
    steps: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def rows_in(self) -> int:
        return self.steps[0]["rows_before"] if self.steps else len(self.facts)

    def to_dict(self) -> dict[str, Any]:
        return {"applied": self.spec.normalised(),
                "steps": self.steps,
                "rows_in": self.rows_in,
                "rows_out": len(self.facts),
                "warnings": list(self.warnings)}


def apply_filters(facts: pd.DataFrame, questions: pd.DataFrame,
                  spec: FilterSpec) -> FilterResult:
    """Narrow the fact table. The period filter is applied on `event_time`, the
    column the loader chose as the real answer time."""
    steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    current = facts

    def step(name: str, mask: pd.Series, detail: Any = None) -> None:
        nonlocal current
        before = len(current)
        current = current.loc[mask]
        steps.append({"filter": name, "value": detail, "rows_before": before,
                      "rows_after": len(current), "rows_removed": before - len(current)})

    # --- form ------------------------------------------------------------
    available_forms = (sorted(facts["form_id"].dropna().unique().tolist())
                       if "form_id" in facts.columns else [])
    if spec.form_id:
        if spec.form_id not in available_forms:
            raise FilterError(f"form_id {spec.form_id!r} is not in this export; "
                              f"available: {available_forms}")
        step("form_id", current["form_id"] == spec.form_id, spec.form_id)
    elif len(available_forms) > 1:
        warnings.append(f"no form_id given and the export contains "
                        f"{len(available_forms)} forms; statistics mix them. "
                        f"Pass form_id to analyse one form.")

    # --- customer (optional dimension, reserved for the next export) ------
    customers = _as_list(spec.customer_id)
    if customers:
        if "customer_id" not in current.columns or current["customer_id"].isna().all():
            raise FilterError(
                "customer_id filtering was requested but this export has no "
                "customer column. It is supported as soon as the export includes "
                "one - no code change needed.")
        step("customer_id", current["customer_id"].astype("string").isin(customers),
             customers)

    # --- representative ---------------------------------------------------
    representatives = _as_list(spec.representative_id)
    if representatives:
        by_id = current["representative_id"].astype("string")
        by_label = current["representative_label"].astype("string")
        mask = by_id.isin(representatives) | by_label.isin(representatives)
        if not mask.any():
            known = sorted(set(by_id.dropna()) | set(by_label.dropna()))[:20]
            raise FilterError(f"no responses for representative(s) {representatives}. "
                              f"Known values include: {known}")
        step("representative_id", mask, representatives)

    # --- excluded people --------------------------------------------------
    # For dropping test and placeholder accounts. The report flags likely ones
    # under `presentation["test_accounts"]`; this is how a caller acts on that.
    excluded = _as_list(spec.exclude_representative_id)
    if excluded:
        by_id = current["representative_id"].astype("string")
        by_label = current["representative_label"].astype("string")
        mask = ~(by_id.isin(excluded) | by_label.isin(excluded))
        if mask.all():
            warnings.append(f"none of the excluded people {excluded} appear in "
                            f"this data, so nothing was removed")
        step("exclude_representative_id", mask, excluded)

    # --- completion -------------------------------------------------------
    if spec.completed_only and "is_completed" in current.columns:
        step("completed_only", current["is_completed"].fillna(False), True)

    if not spec.include_autofilled and "autofilled" in current.columns:
        step("include_autofilled", ~current["autofilled"].fillna(False), False)

    # --- period -----------------------------------------------------------
    if spec.period_from is not None or spec.period_to is not None:
        times = current["event_time"]
        mask = pd.Series(True, index=current.index)
        if spec.period_from is not None:
            mask &= times >= pd.Timestamp(spec.period_from)
        if spec.period_to is not None:
            upper = pd.Timestamp(spec.period_to)
            if upper == upper.normalize():
                upper = upper + pd.Timedelta(days=1)   # inclusive day
            mask &= times < upper
        step("period", mask.fillna(False),
             {"from": str(spec.period_from), "to": str(spec.period_to)})

    if current.empty:
        warnings.append("the filters matched no responses")

    kept_questions = questions
    if not current.empty:
        kept_questions = questions.loc[
            questions["question_id"].isin(current["question_id"].unique())]
        silent = len(questions) - len(kept_questions)
        if silent:
            warnings.append(f"{silent} question(s) received no answers under these "
                            f"filters and are reported as unanswered")

    log.info("filters: %s -> %s rows (%s step(s))", len(facts), len(current), len(steps))
    return FilterResult(facts=current, questions=kept_questions, spec=spec,
                        steps=steps, warnings=warnings)
