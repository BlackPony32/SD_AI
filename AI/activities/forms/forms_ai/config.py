"""Single source of truth for tunables. Everything here is overridable by an
environment variable so the pipeline can be retuned without a code change.

The MODEL / PRICING / AGENT_TIMEOUT names already existed for the activity and
task pipelines; the FORMS_* block is new and belongs to the form analyser.
"""

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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
AGENT_TIMEOUT = _int("AGENT_TIMEOUT", 180)

# Pricing lives in core/cost.py, inside `calculate_cost`. It is re-exported here
# under the name the rest of the codebase already used, so there is one table and
# not two that can drift apart.
PRICING = _cost.PRICING
PRICING_FALLBACK_MODEL = _cost.FALLBACK_MODEL

# `calculate_cost` prints its totals. True routes that output through the logger
# so a library call does not write to stdout; False lets it print as written.
FORMS_COST_PRINT_TO_LOG = os.environ.get("FORMS_COST_PRINT_TO_LOG", "1") != "0"

# ---------------------------------------------------------------------------
# Where a user's already-downloaded exports live
# ---------------------------------------------------------------------------
# Layout expected by loader.resolve_dataset():
#     {FORMS_DATA_ROOT}/{user_id}/raw_file_form_questions.csv
#     {FORMS_DATA_ROOT}/{user_id}/raw_file_form_progresses.csv
#     {FORMS_DATA_ROOT}/{user_id}/raw_file_form_responses.csv
# Flat layout ({FORMS_DATA_ROOT}/{user_id}_raw_file_form_questions.csv) and a
# recursive search are both accepted fallbacks - see loader.py.

FORMS_DATA_ROOT = Path(os.environ.get("FORMS_DATA_ROOT", "./data")).expanduser()

# ---------------------------------------------------------------------------
# Interval engine
# ---------------------------------------------------------------------------

# The engine aims for this many buckets and will pick the finest calendar unit
# that keeps the count inside [MIN, MAX]. See forms/intervals.py.
FORMS_TARGET_BUCKETS = _int("FORMS_TARGET_BUCKETS", 12)
FORMS_MIN_BUCKETS = _int("FORMS_MIN_BUCKETS", 4)
FORMS_MAX_BUCKETS = _int("FORMS_MAX_BUCKETS", 24)

# A bucket with fewer than this many responses is still reported, but is flagged
# `low_n` and excluded from trend fitting so a 3-answer week cannot swing a slope.
FORMS_MIN_BUCKET_N = _int("FORMS_MIN_BUCKET_N", 5)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

FORMS_OUTLIER_IQR_K = _float("FORMS_OUTLIER_IQR_K", 1.5)
FORMS_OUTLIER_MAD_Z = _float("FORMS_OUTLIER_MAD_Z", 3.5)
FORMS_TOP_K_CATEGORIES = _int("FORMS_TOP_K_CATEGORIES", 10)
FORMS_TOP_K_TOKENS = _int("FORMS_TOP_K_TOKENS", 15)
FORMS_TREND_ALPHA = _float("FORMS_TREND_ALPHA", 0.05)

# --- when a single number must not be reported -------------------------------
# Numeric answers to field forms are frequently heterogeneous: 10 and 100 in the
# same question do not average to 55, they mean the question is being answered
# about different things. Dispersion is measured with the quartile coefficient of
# dispersion, qcd = (p75 - p25) / (p75 + p25), which is scale-free and robust.
#   qcd < TIGHT      -> one typical value is a fair summary
#   qcd < WIDE       -> report the typical value together with the usual range
#   otherwise        -> refuse a single value, report the range and the bands
FORMS_DISPERSION_TIGHT = _float("FORMS_DISPERSION_TIGHT", 0.15)
FORMS_DISPERSION_WIDE = _float("FORMS_DISPERSION_WIDE", 0.40)
# A top-to-bottom quartile ratio at or above this is "wide" whatever qcd says.
FORMS_SPREAD_RATIO_WIDE = _float("FORMS_SPREAD_RATIO_WIDE", 10.0)
# Two clusters: the largest gap between consecutive sorted answers covers this
# much of the range, with at least MIN_SIDE of the answers on each side of it.
FORMS_CLUSTER_GAP_SHARE = _float("FORMS_CLUSTER_GAP_SHARE", 0.35)
FORMS_CLUSTER_MIN_SIDE = _float("FORMS_CLUSTER_MIN_SIDE", 0.25)
FORMS_VALUE_BANDS = _int("FORMS_VALUE_BANDS", 4)

# Numbers handed to the model are rounded to this many places. The grounding
# check compares against the *rounded* payload, so a tighter number here
# directly limits how much precision the model can invent.
# 4 rather than 2: rates live in 0..1, and rounding 0.686 to 0.69 turns into a
# visibly wrong "69%" once it is rendered as a percentage.
FORMS_ROUND = _int("FORMS_ROUND", 4)

# ---------------------------------------------------------------------------
# LLM stage
# ---------------------------------------------------------------------------

# How many per-question comparison agents may run at once.
FORMS_LLM_CONCURRENCY = _int("FORMS_LLM_CONCURRENCY", 4)

# Questions are analysed by the model in batches of this size; one call per
# batch keeps cost sane on a 40-question form.
FORMS_QUESTION_BATCH = _int("FORMS_QUESTION_BATCH", 6)

# --- input-token budget -------------------------------------------------------
# The analyst reads a digest: one line per question, plus the full period detail
# only for the questions that actually moved or look unusual. A form where
# nothing changed costs almost nothing to analyse.
FORMS_ANALYST_DETAIL_LIMIT = _int("FORMS_ANALYST_DETAIL_LIMIT", 6)

# Per-question period values are sent as parallel arrays against one shared list
# of period labels, so a label is never repeated per question. Above this many
# periods the series is thinned for the model (the full series stays in
# `statistics` and in the report) by keeping the first, the last and the extremes.
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

# --- how fine a split the data can actually support ---------------------------
# Splitting 172 forms into 14 weeks leaves 12 answers a week; at that size a
# yes-rate swings across a 50-point range by chance alone, so every weekly figure
# is noise. `choose_granularity` penalises any split that leaves fewer than this
# many answers per period, and says in the notes why it went coarser.
# 30 is the point at which a ~25-point move becomes visible at 80% power.
FORMS_MIN_ANSWERS_PER_PERIOD = _int("FORMS_MIN_ANSWERS_PER_PERIOD", 30)

# A period whose value sits inside the range its own sample size would produce
# anyway is not "the highest" - it is ordinary variation. Bands use this
# confidence level.
FORMS_BAND_Z = _float("FORMS_BAND_Z", 1.959963985)

# Nobody is named as high or low on fewer than this many answers, and a person is
# only called out when a two-proportion test against the rest of the team clears
# this p-value.
FORMS_MIN_PERSON_ANSWERS = _int("FORMS_MIN_PERSON_ANSWERS", 10)
FORMS_PERSON_ALPHA = _float("FORMS_PERSON_ALPHA", 0.05)

# One person submitting at least this share of all forms means the overall figures
# are mostly that person's figures, which is worth saying out loud.
FORMS_DOMINANCE_SHARE = _float("FORMS_DOMINANCE_SHARE", 0.40)

# Answers whose shape is identical once numbers are stripped look generated rather
# than written. Above this share the question is flagged.
FORMS_TEMPLATE_SHARE = _float("FORMS_TEMPLATE_SHARE", 0.60)
