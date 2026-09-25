"""Tunables; each one can be overridden by an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

from .core import cost as _cost


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# --- Model ---

MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
AGENT_TIMEOUT = _int("AGENT_TIMEOUT", 180)

# Pricing lives in core/cost.py (calculate_cost); re-exported under the name the codebase uses.
PRICING = _cost.PRICING
PRICING_FALLBACK_MODEL = _cost.FALLBACK_MODEL

# `calculate_cost` prints its totals. True routes that output through the logger
# so a library call does not write to stdout; False lets it print as written.
FORMS_COST_PRINT_TO_LOG = os.environ.get("FORMS_COST_PRINT_TO_LOG", "1") != "0"

# --- Where a user's downloaded exports live ---
# {FORMS_DATA_ROOT}/{user_id}/raw_file_form_{questions,progresses,responses}.csv; fallbacks in loader.py.

FORMS_DATA_ROOT = Path(os.environ.get("FORMS_DATA_ROOT", "./data")).expanduser()

# --- Interval engine ---

# The engine aims for this many buckets and will pick the finest calendar unit
# that keeps the count inside [MIN, MAX]. See forms/intervals.py.
FORMS_TARGET_BUCKETS = _int("FORMS_TARGET_BUCKETS", 12)
FORMS_MIN_BUCKETS = _int("FORMS_MIN_BUCKETS", 4)
FORMS_MAX_BUCKETS = _int("FORMS_MAX_BUCKETS", 24)

# A bucket with fewer than this many responses is still reported, but is flagged
# `low_n` and excluded from trend fitting so a 3-answer week cannot swing a slope.
FORMS_MIN_BUCKET_N = _int("FORMS_MIN_BUCKET_N", 5)

# --- Statistics ---

FORMS_OUTLIER_IQR_K = _float("FORMS_OUTLIER_IQR_K", 1.5)
FORMS_OUTLIER_MAD_Z = _float("FORMS_OUTLIER_MAD_Z", 3.5)
FORMS_TOP_K_CATEGORIES = _int("FORMS_TOP_K_CATEGORIES", 10)
FORMS_TOP_K_TOKENS = _int("FORMS_TOP_K_TOKENS", 15)
FORMS_TREND_ALPHA = _float("FORMS_TREND_ALPHA", 0.05)

# Dispersion, qcd = (p75 - p25) / (p75 + p25), decides whether one typical value is fair:
# below TIGHT one value, below WIDE the value with its usual range, otherwise only ranges and bands.
FORMS_DISPERSION_TIGHT = _float("FORMS_DISPERSION_TIGHT", 0.15)
FORMS_DISPERSION_WIDE = _float("FORMS_DISPERSION_WIDE", 0.40)
# A top-to-bottom quartile ratio at or above this is "wide" whatever qcd says.
FORMS_SPREAD_RATIO_WIDE = _float("FORMS_SPREAD_RATIO_WIDE", 10.0)
# Two clusters: the largest gap between consecutive sorted answers covers this
# much of the range, with at least MIN_SIDE of the answers on each side of it.
FORMS_CLUSTER_GAP_SHARE = _float("FORMS_CLUSTER_GAP_SHARE", 0.35)
FORMS_CLUSTER_MIN_SIDE = _float("FORMS_CLUSTER_MIN_SIDE", 0.25)
FORMS_VALUE_BANDS = _int("FORMS_VALUE_BANDS", 4)

# Decimal places for numbers sent to the model (grounding checks the rounded values).
# 4, not 2: rates are 0..1, and 0.686 -> 0.69 would render as a wrong 69%.
FORMS_ROUND = _int("FORMS_ROUND", 4)

# --- LLM stage ---

# How many per-question comparison agents may run at once.
FORMS_LLM_CONCURRENCY = _int("FORMS_LLM_CONCURRENCY", 4)

# Questions are analysed by the model in batches of this size; one call per
# batch keeps cost sane on a 40-question form.
FORMS_QUESTION_BATCH = _int("FORMS_QUESTION_BATCH", 6)

# Input-token budget: one line per question, full period detail only where something moved.
FORMS_ANALYST_DETAIL_LIMIT = _int("FORMS_ANALYST_DETAIL_LIMIT", 6)

# Above this many periods the series sent to the model is thinned (first, last, extremes).
FORMS_MODEL_MAX_PERIODS = _int("FORMS_MODEL_MAX_PERIODS", 14)

# People are only sent for a question when they actually differ from each other,
# and then only the top and bottom few.
FORMS_MODEL_MAX_PEOPLE = _int("FORMS_MODEL_MAX_PEOPLE", 4)
FORMS_PEOPLE_DIFFER_QCD = _float("FORMS_PEOPLE_DIFFER_QCD", 0.15)

# Written answers: how many examples reach the model, and how long each may be.
FORMS_MAX_VERBATIMS = _int("FORMS_MAX_VERBATIMS", 3)
FORMS_VERBATIM_CHARS = _int("FORMS_VERBATIM_CHARS", 180)

# Hard ceiling on the characters of one agent input. Exceeding it logs a warning
# with the size, so a runaway payload is visible instead of just expensive.
FORMS_PAYLOAD_WARN_CHARS = _int("FORMS_PAYLOAD_WARN_CHARS", 24_000)

# Minimum answers per period; coarser splits are chosen below it (30 shows a ~25-point move at 80% power).
FORMS_MIN_ANSWERS_PER_PERIOD = _int("FORMS_MIN_ANSWERS_PER_PERIOD", 30)

# Confidence level for the bands that separate real highs and lows from ordinary variation.
FORMS_BAND_Z = _float("FORMS_BAND_Z", 1.959963985)

# Minimum answers before a person is named, and the p-value their two-proportion test must clear.
FORMS_MIN_PERSON_ANSWERS = _int("FORMS_MIN_PERSON_ANSWERS", 10)
FORMS_PERSON_ALPHA = _float("FORMS_PERSON_ALPHA", 0.05)

# One person submitting at least this share of all forms means the overall figures
# are mostly that person's figures, which is worth saying out loud.
FORMS_DOMINANCE_SHARE = _float("FORMS_DOMINANCE_SHARE", 0.40)

# Answers whose shape is identical once numbers are stripped look generated rather
# than written. Above this share the question is flagged.
FORMS_TEMPLATE_SHARE = _float("FORMS_TEMPLATE_SHARE", 0.60)
