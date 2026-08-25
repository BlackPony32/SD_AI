"""Per-question-type analysers.

A form is not a fixed schema - the same code has to handle a 12-question stock
check and a 40-question audit whose builder has since grown three new field
types. Two mechanisms make that work:

1. **A registry keyed by question type.** Supporting a new type means adding one
   function and one entry, and nothing else changes.
2. **Effective typing.** The declared type is a hint, not the truth. A question
   declared TEXT whose answers are all "Yes"/"No" is a rate question and should
   be reported as one; a TEXT field holding numbers is a numeric question. The
   sniffer promotes those cases, records why, and reports both the declared and
   the effective type so nothing is hidden.

Every analyser returns a flat-ish dict of JSON-safe values, computed over one
slice of the fact table. The same function computes the whole-period summary and
each bucket's summary - identical logic on both sides of a comparison is the only
way the comparison means anything.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

import pandas as pd

from ..config import (FORMS_DISPERSION_TIGHT, FORMS_DISPERSION_WIDE,
                      FORMS_MAX_VERBATIMS, FORMS_OUTLIER_IQR_K,
                      FORMS_OUTLIER_MAD_Z, FORMS_SPREAD_RATIO_WIDE,
                      FORMS_TOP_K_CATEGORIES, FORMS_TOP_K_TOKENS,
                      FORMS_VALUE_BANDS, FORMS_VERBATIM_CHARS)
from ..core.logging_setup import get_log
from . import schema as S
from . import statmath as M
from . import textual as TX

log = get_log("forms.metrics")


# ---------------------------------------------------------------------------
# Question context
# ---------------------------------------------------------------------------

@dataclass
class QuestionContext:
    """Everything an analyser needs beyond the rows themselves.

    `reference` carries decisions taken on the full period so that per-bucket
    numbers stay comparable - most importantly the option a categorical series is
    tracked against. Without a fixed reference, "top option share" would silently
    change meaning between buckets.
    """

    question_id: str
    text: str
    declared_type: str
    effective_type: str
    order_index: int = 0
    options: list[str] = field(default_factory=list)
    required: bool | None = None
    retype_reason: str | None = None
    # Decisions taken once over the whole period so every period is measured the
    # same way: the option a categorical series tracks, and - for free text - what
    # the answers are measured on at all (`content_kind`, `required_minimum`).
    reference: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Effective typing
# ---------------------------------------------------------------------------

_MIN_SNIFF_N = 20
_SNIFF_CONFIDENCE = 0.95


def _non_empty(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("")
    return text[text.str.strip() != ""]


def effective_type(declared: str, answers: pd.Series, multi: pd.Series | None = None
                   ) -> tuple[str, str | None]:
    """Resolve the type actually present in the data.

    Promotion only happens above `_SNIFF_CONFIDENCE` on at least `_MIN_SNIFF_N`
    answers, so one stray "Yes" in a comment box cannot re-type a question.
    """
    if multi is not None and multi.map(bool).any():
        if declared in (S.UNKNOWN, S.TEXT):
            return S.MULTIPLE_ANSWER, "answers arrive as a multi-select list"
        return declared, None

    values = _non_empty(answers)
    n = len(values)
    if declared not in (S.TEXT, S.UNKNOWN, S.SINGLE_ANSWER) or n < _MIN_SNIFF_N:
        return (declared if declared != S.UNKNOWN else S.TEXT), (
            None if declared != S.UNKNOWN else "type missing from the export; "
            "treated as free text")

    boolean_share = values.map(S.to_bool).notna().mean()
    if boolean_share >= _SNIFF_CONFIDENCE:
        return S.YES_NO, (f"declared {declared} but {boolean_share:.0%} of answers "
                          f"are yes/no values")

    numeric_share = values.map(S.to_number).notna().mean()
    if numeric_share >= _SNIFF_CONFIDENCE:
        return S.NUMERIC, (f"declared {declared} but {numeric_share:.0%} of answers "
                           f"parse as numbers")

    distinct = values.nunique()
    mean_length = float(values.str.len().mean())
    if declared == S.SINGLE_ANSWER:
        return S.SINGLE_ANSWER, None
    if distinct <= 12 and n >= 30 and mean_length <= 40 and distinct / n <= 0.5:
        return S.SINGLE_ANSWER, (f"declared {declared} but answers form {distinct} "
                                 f"repeating short values")
    return S.TEXT, (None if declared == S.TEXT else
                    "type missing from the export; treated as free text")


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

def _coverage(frame: pd.DataFrame, ctx: QuestionContext) -> dict[str, Any]:
    """Volume and answer coverage - meaningful for every type."""
    total = len(frame)
    if ctx.effective_type == S.MULTIPLE_ANSWER:
        answered = int(frame["answer_multi"].map(bool).sum())
    else:
        answered = int(_non_empty(frame["answer_text"]).shape[0])
    autofilled = (int(frame["autofilled"].fillna(False).sum())
                  if "autofilled" in frame.columns else 0)
    return {
        "rows": total,
        "answered": answered,
        "blank": total - answered,
        "response_rate": M.safe_div(answered, total),
        "autofilled": autofilled,
        "autofilled_share": M.safe_div(autofilled, total),
        "submissions": int(frame["progress_id"].nunique()) if total else 0,
    }


# ---------------------------------------------------------------------------
# Analysers
# ---------------------------------------------------------------------------

def analyse_numeric(frame: pd.DataFrame, ctx: QuestionContext) -> dict[str, Any]:
    """Robust-first.

    The headline is the **middle value** (median) with the usual range around it
    (p25-p75), not the arithmetic mean - and when the answers are too spread out
    or fall into more than one group, `dispersion.single_value_representative` is
    False and no single value should be reported at all. Answers of 10 and 100 do
    not summarise to 55; they summarise to "10 to 100, in two groups".

    The mean and standard deviation are still computed for completeness, but no
    presentation layer reads them as the headline.
    """
    raw = _non_empty(frame["answer_text"])
    parsed = raw.map(S.to_number)
    values = [v for v in parsed.tolist() if v is not None]

    out = _coverage(frame, ctx)
    out.update(M.describe(values))
    out["dispersion"] = M.dispersion(
        values, tight=FORMS_DISPERSION_TIGHT, wide=FORMS_DISPERSION_WIDE,
        spread_ratio_wide=FORMS_SPREAD_RATIO_WIDE)
    out["bands"] = M.value_bands(values, FORMS_VALUE_BANDS)
    out["unparsable"] = int(parsed.isna().sum())
    out["unparsable_examples"] = sorted(set(raw[parsed.isna()].tolist()))[:5]
    out["outliers_iqr"] = M.outliers_iqr(values, FORMS_OUTLIER_IQR_K)
    out["outliers_mad"] = M.outliers_mad(values, FORMS_OUTLIER_MAD_Z)
    out["zero_share"] = M.safe_div(sum(1 for v in values if v == 0), len(values))
    return out


def analyse_rating(frame: pd.DataFrame, ctx: QuestionContext) -> dict[str, Any]:
    out = analyse_numeric(frame, ctx)
    values = [v for v in _non_empty(frame["answer_text"]).map(S.to_number).tolist()
              if v is not None]
    if values:
        counter = Counter(values)
        out["scale_counts"] = [{"value": value, "count": count,
                                "share": count / len(values)}
                               for value, count in sorted(counter.items())]
        low, high = min(values), max(values)
        if low >= 0 and high <= 10 and len(values) >= 10:
            promoters = sum(1 for v in values if v >= 9)
            detractors = sum(1 for v in values if v <= 6)
            out["nps"] = round((promoters - detractors) / len(values) * 100, 2)
    return out


def analyse_yes_no(frame: pd.DataFrame, ctx: QuestionContext) -> dict[str, Any]:
    parsed = _non_empty(frame["answer_text"]).map(S.to_bool)
    yes = int((parsed == True).sum())      # noqa: E712 - None must stay separate
    no = int((parsed == False).sum())      # noqa: E712
    unclear = int(parsed.isna().sum())
    out = _coverage(frame, ctx)
    out.update({
        "yes": yes, "no": no, "unclear": unclear,
        "yes_rate": M.safe_div(yes, yes + no),
        "no_rate": M.safe_div(no, yes + no),
        "confidence_interval_95": M.wilson_interval(yes, yes + no),
    })
    if unclear:
        out["unclear_examples"] = sorted(
            set(_non_empty(frame["answer_text"])[parsed.isna()].tolist()))[:5]
    return out


def _category_counts(values: pd.Series) -> Counter:
    return Counter(values.str.strip().tolist())


def analyse_single_answer(frame: pd.DataFrame, ctx: QuestionContext) -> dict[str, Any]:
    values = _non_empty(frame["answer_text"])
    counter = _category_counts(values)
    total = sum(counter.values())
    ranked = counter.most_common()
    top = [{"value": value, "count": count, "share": M.safe_div(count, total)}
           for value, count in ranked[:FORMS_TOP_K_CATEGORIES]]

    out = _coverage(frame, ctx)
    out.update({
        "distinct_values": len(counter),
        "top_values": top,
        "tail_count": sum(count for _, count in ranked[FORMS_TOP_K_CATEGORIES:]),
        "diversity": M.normalised_entropy(counter.values()),
        "concentration_gini": M.gini(counter.values()),
        "modal_value": ranked[0][0] if ranked else None,
        "modal_share": M.safe_div(ranked[0][1], total) if ranked else None,
    })
    if ctx.options:
        declared = {o.strip() for o in ctx.options}
        seen = set(counter)
        out["unused_options"] = sorted(declared - seen)
        out["off_list_values"] = sorted(seen - declared)[:10]
    # Fixed reference so the per-bucket series tracks one option, not "whatever
    # happened to win this week".
    reference = ctx.reference.get("modal_value")
    if reference is not None:
        out["reference_value"] = reference
        out["reference_share"] = M.safe_div(counter.get(reference, 0), total)
    return out


def analyse_multiple_answer(frame: pd.DataFrame, ctx: QuestionContext) -> dict[str, Any]:
    selections = frame["answer_multi"].tolist()
    non_empty = [s for s in selections if s]
    counter = Counter(option for group in non_empty for option in group)
    respondents = len(non_empty)
    total_selections = sum(len(group) for group in non_empty)

    pairs = Counter()
    for group in non_empty:
        for a, b in combinations(sorted(set(group)), 2):
            pairs[(a, b)] += 1

    per_form = [len(group) for group in non_empty]
    out = _coverage(frame, ctx)
    out.update({
        "respondents_with_selection": respondents,
        "none_selected": len(selections) - respondents,
        "total_selections": total_selections,
        "avg_selections": M.safe_div(total_selections, respondents),
        # The median is the reported figure: one form listing every option should
        # not drag the typical count upwards.
        "median_selections": (M.describe(per_form)["median"] if per_form else None),
        "selections_dispersion": M.dispersion(per_form) if per_form else None,
        "selection_counts": [
            {"option": option, "count": count,
             "share_of_respondents": M.safe_div(count, respondents)}
            for option, count in counter.most_common(FORMS_TOP_K_CATEGORIES)],
        "distinct_options_used": len(counter),
        "diversity": M.normalised_entropy(counter.values()),
        "top_co_occurrences": [
            {"options": list(pair), "count": count,
             "share_of_respondents": M.safe_div(count, respondents)}
            for pair, count in pairs.most_common(5)],
    })
    if ctx.options:
        declared = {o.strip() for o in ctx.options}
        out["unused_options"] = sorted(declared - set(counter))
        out["off_list_values"] = sorted(set(counter) - declared)[:10]
        out["option_coverage"] = M.safe_div(len(set(counter) & declared), len(declared))
    reference = ctx.reference.get("modal_value")
    if reference is not None:
        out["reference_value"] = reference
        out["reference_share"] = M.safe_div(counter.get(reference, 0), respondents)
    return out


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "was", "were", "are", "have",
    "has", "had", "not", "but", "you", "your", "they", "them", "their", "our",
    "from", "all", "any", "can", "will", "would", "there", "then", "than",
    "into", "out", "about", "just", "some", "more", "very", "too", "also",
    "did", "does", "done", "get", "got", "one", "two", "who", "how", "what",
    "when", "where", "which", "been", "being", "over", "only", "his", "her",
    "she", "him", "its", "his", "yes", "no", "answer", "text",
}
_WORD = re.compile(r"[A-Za-zЀ-ӿ][A-Za-zЀ-ӿ'\-]{2,}")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)
            if w.lower() not in _STOPWORDS]


def analyse_text(frame: pd.DataFrame, ctx: QuestionContext) -> dict[str, Any]:
    """Free-text answers, measured on their content.

    The response rate is still computed but is no longer the headline: for a
    required field it is 100% by construction and tells a reader nothing. What is
    reported instead depends on what the answers actually contain - an amount, a
    list checked against the minimum the question asks for, or whether the answer
    says anything at all. `content_kind` records which, and is fixed for the whole
    period in `build_contexts` so every period is measured the same way.
    """
    values = _non_empty(frame["answer_text"])
    answers = values.tolist()
    lengths = values.str.len().tolist()
    word_counts = values.map(lambda t: len(t.split())).tolist()

    tokens = Counter()
    for text in answers:
        tokens.update(_tokens(text))

    duplicates = Counter(values.str.strip().str.lower().tolist())
    repeated = {value: count for value, count in duplicates.items() if count > 1}

    out = _coverage(frame, ctx)
    kind = ctx.reference.get("content_kind") or "response_rate"
    minimum = ctx.reference.get("required_minimum")
    out.update({
        "content_kind": kind,
        "length_chars": M.describe(lengths),
        "word_count": M.describe(word_counts),
        "distinct_answers": int(values.nunique()),
        "duplicate_rate": M.safe_div(sum(repeated.values()), len(values)),
        "most_repeated": [{"value": value[:120], "count": count}
                          for value, count in Counter(repeated).most_common(3)],
        "top_terms": [{"term": term, "count": count}
                      for term, count in tokens.most_common(FORMS_TOP_K_TOKENS)],
        "themes": TX.themes(answers),
        "templating": TX.templating(answers),
        # Verbatims are the one place raw user text reaches the model, and the
        # most expensive thing in the payload per unit of insight - so: few,
        # short, longest first (short answers carry the least signal).
        "sample_verbatims": [t[:FORMS_VERBATIM_CHARS] for t in
                             sorted(set(answers), key=len, reverse=True
                                    )[:FORMS_MAX_VERBATIMS]],
    })

    # --- substance ------------------------------------------------------
    substantive = sum(1 for text in answers if TX.is_substantive(text))
    out["substantive"] = substantive
    out["substantive_rate"] = M.safe_div(substantive, len(frame) or None)

    # --- amounts --------------------------------------------------------
    if kind == "amount":
        rows = [TX.extract_amounts(text) for text in answers]
        amounts = [row["largest_amount"] for row in rows
                   if row["largest_amount"] is not None]
        said_yes = sum(1 for row in rows if row["says_yes"])
        units = Counter(q["unit"] for row in rows for q in row["quantities"])
        described = M.describe(amounts)
        out.update({
            "amount_count": len(amounts),
            "amount_share": M.safe_div(len(amounts), len(frame) or None),
            "amount_median": described["median"],
            "amount_total": described["sum"],
            "amount_lowest": described["min"],
            "amount_highest": described["max"],
            "amount_dispersion": M.dispersion(
                amounts, tight=FORMS_DISPERSION_TIGHT, wide=FORMS_DISPERSION_WIDE,
                spread_ratio_wide=FORMS_SPREAD_RATIO_WIDE),
            "positive_rate": M.safe_div(said_yes, len(frame) or None),
            "positive_count": said_yes,
            "quantity_units": [{"unit": unit, "count": count}
                               for unit, count in units.most_common(3)],
        })

    # --- compliance with a minimum the question states -------------------
    if kind == "compliance" and minimum:
        counts = [TX.count_items(text) for text in answers]
        meeting = sum(1 for count in counts if count >= minimum)
        out.update({
            "required_minimum": minimum,
            "meeting_minimum": meeting,
            "compliance_rate": M.safe_div(meeting, len(frame) or None),
            "items_listed": M.describe(counts),
            "shortfall_examples": [text[:120] for text, count
                                   in zip(answers, counts)
                                   if count < minimum][:3],
        })
    return out


def analyse_date(frame: pd.DataFrame, ctx: QuestionContext) -> dict[str, Any]:
    parsed = S.parse_datetimes(_non_empty(frame["answer_text"]))
    valid = parsed.dropna()
    out = _coverage(frame, ctx)
    out["unparsable"] = int(parsed.isna().sum())
    if len(valid):
        out.update({
            "earliest": valid.min().isoformat(),
            "latest": valid.max().isoformat(),
            "span_days": round((valid.max() - valid.min()) / pd.Timedelta(days=1), 2),
            "distinct_days": int(valid.dt.normalize().nunique()),
            "weekday_counts": {str(k): int(v) for k, v in
                               valid.dt.day_name().value_counts().items()},
        })
    return out


ANALYSERS: dict[str, Callable[[pd.DataFrame, QuestionContext], dict[str, Any]]] = {
    S.NUMERIC: analyse_numeric,
    S.RATING: analyse_rating,
    S.YES_NO: analyse_yes_no,
    S.SINGLE_ANSWER: analyse_single_answer,
    S.MULTIPLE_ANSWER: analyse_multiple_answer,
    S.TEXT: analyse_text,
    S.DATE: analyse_date,
}


def analyse(frame: pd.DataFrame, ctx: QuestionContext) -> dict[str, Any]:
    analyser = ANALYSERS.get(ctx.effective_type, analyse_text)
    try:
        return analyser(frame, ctx)
    except Exception as exc:                       # one bad question, not a bad run
        log.exception("  analyser %s failed for question %s: %s",
                      ctx.effective_type, ctx.question_id, exc)
        out = _coverage(frame, ctx)
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


# ---------------------------------------------------------------------------
# Which single number represents this question over time
# ---------------------------------------------------------------------------
# The trend engine needs one comparable series per question. The choice per type
# is the metric that a human would actually argue about in a review meeting.

# Free text has no single right measure - it depends what the answers contain.
# Chosen once per question by `textual.profile`, then fixed for every period.
TEXT_PRIMARY: dict[str, tuple[str, str, str]] = {
    "amount":        ("amount_median",     "typical amount given",        "value"),
    "compliance":    ("compliance_rate",   "met the minimum asked for",   "rate"),
    "substance":     ("substantive_rate",  "gave a real answer",          "rate"),
    "response_rate": ("response_rate",     "answered this question",      "rate"),
}

PRIMARY_METRIC: dict[str, tuple[str, str, str]] = {
    #  type            -> (key,                human-facing label,          kind)
    # NUMERIC and RATING use the middle value, never the average: one absurd
    # answer must not move the headline, and the average of 10 and 100 is not a
    # fact about either of them.
    S.NUMERIC:         ("median",             "typical answer",             "value"),
    S.RATING:          ("median",             "typical rating",             "value"),
    S.YES_NO:          ("yes_rate",           "answered Yes",               "rate"),
    S.SINGLE_ANSWER:   ("reference_share",    "chose the most common answer", "rate"),
    # Median rather than average for the same reason.
    S.MULTIPLE_ANSWER: ("median_selections",  "options ticked per form",     "value"),
    S.TEXT:            ("response_rate",      "answered this question",      "rate"),
    S.DATE:            ("answered",           "answers given",               "count"),
}

# Extras worth showing. Labels here are read by people, so they carry no
# statistical vocabulary.
SECONDARY_METRICS: dict[str, list[tuple[str, str]]] = {
    S.NUMERIC: [("min", "lowest answer"), ("max", "highest answer"),
                ("sum", "everything added together")],
    S.RATING: [("min", "lowest rating"), ("max", "highest rating"),
               ("nps", "net promoter score")],
    S.YES_NO: [("answered", "answers counted")],
    S.SINGLE_ANSWER: [("distinct_values", "different answers given")],
    S.MULTIPLE_ANSWER: [("total_selections", "options ticked in total")],
    S.TEXT: [("word_count.mean", "average length in words")],
    S.DATE: [("distinct_days", "different dates given")],
}


def dig(summary: dict[str, Any], key: str) -> Any:
    """Read "a.b.c" out of a nested summary dict."""
    node: Any = summary
    for part in key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def primary_metric(ctx: "QuestionContext | str") -> tuple[str, str, str]:
    """(key, human label, kind) for the one number that represents a question.

    Accepts a context so free text can be routed by its content, or a bare type
    string for the simple cases.
    """
    if isinstance(ctx, str):
        return PRIMARY_METRIC.get(ctx, ("answered", "answers given", "count"))
    if ctx.effective_type == S.TEXT:
        kind = ctx.reference.get("content_kind") or "response_rate"
        return TEXT_PRIMARY.get(kind, TEXT_PRIMARY["response_rate"])
    return PRIMARY_METRIC.get(ctx.effective_type,
                              ("answered", "answers given", "count"))


# Which pair of counts a rate is actually built from. Needed because a rate on
# its own cannot be tested: "52%" carries no weight, "90 of 172" does. Deriving
# the counts by multiplying the rate back out would round, and would use the
# wrong denominator wherever unclear answers are excluded from it.
RATE_COUNTS: dict[str, tuple[str, str | None]] = {
    #  rate key         -> (successes key,    total key or None for "successes+rest")
    "yes_rate":           ("yes",             None),          # yes + no
    "no_rate":            ("no",              None),          # no + yes
    "compliance_rate":    ("meeting_minimum", "answered"),
    "substantive_rate":   ("substantive",     "answered"),
    "response_rate":      ("answered",        "rows"),
}


def numbers_in(frame: pd.DataFrame) -> list[float]:
    """The parsed numeric answers in `frame`, for tests that need the raw sample
    rather than a summary. Same parsing path as `analyse_numeric`, so a value
    counted here is a value counted there."""
    if frame.empty or "answer_text" not in frame.columns:
        return []
    parsed = _non_empty(frame["answer_text"]).map(S.to_number)
    return [float(v) for v in parsed.tolist() if v is not None]


def rate_counts(summary: dict[str, Any], key: str) -> tuple[int, int] | None:
    """(successes, total) behind a rate, or None when it is not a countable rate."""
    mapping = RATE_COUNTS.get(key)
    if not mapping or not isinstance(summary, dict):
        return None
    success_key, total_key = mapping
    successes = summary.get(success_key)
    if successes is None:
        return None
    if total_key is None:                       # yes/no share one denominator
        other = summary.get("no" if success_key == "yes" else "yes") or 0
        total = int(successes) + int(other)
    else:
        total = summary.get(total_key)
        if total is None:
            return None
        total = int(total)
    if total <= 0 or int(successes) > total:
        return None
    return int(successes), total


def secondary_metrics(ctx: "QuestionContext") -> list[tuple[str, str]]:
    """Extras worth showing, including the ones only some text questions have."""
    if ctx.effective_type == S.TEXT:
        kind = ctx.reference.get("content_kind") or "response_rate"
        if kind == "amount":
            return [("amount_share", "answers naming an amount"),
                    ("amount_total", "everything added together"),
                    ("positive_rate", "answers reporting something secured")]
        if kind == "compliance":
            return [("items_listed.median", "typically listed"),
                    ("substantive_rate", "gave a real answer")]
        return [("word_count.mean", "average length in words"),
                ("distinct_answers", "different answers given")]
    return SECONDARY_METRICS.get(ctx.effective_type, [])
