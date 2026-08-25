"""Schema normalisation: turn whatever the export happens to look like into a
predictable set of columns, dtypes and question types.

Everything downstream (intervals, metrics, prompts) talks to the normalised
names defined here, so a rename in the source export is a one-file change.
"""

from __future__ import annotations

import re
from typing import Any, Final

import pandas as pd

from ..core.logging_setup import get_log

log = get_log("forms.schema")

# ---------------------------------------------------------------------------
# Question types
# ---------------------------------------------------------------------------
# The five that exist in the current export plus the ones the builder can emit.
# Anything unrecognised becomes UNKNOWN and is resolved by sniffing the answers
# (see metrics.effective_type) - that is what makes the engine form-agnostic.

NUMERIC: Final = "NUMERIC"
YES_NO: Final = "YES_NO"
SINGLE_ANSWER: Final = "SINGLE_ANSWER"
MULTIPLE_ANSWER: Final = "MULTIPLE_ANSWER"
TEXT: Final = "TEXT"
DATE: Final = "DATE"
RATING: Final = "RATING"
UNKNOWN: Final = "UNKNOWN"

ALL_TYPES: Final = (NUMERIC, YES_NO, SINGLE_ANSWER, MULTIPLE_ANSWER, TEXT, DATE, RATING, UNKNOWN)

_TYPE_ALIASES: Final[dict[str, str]] = {
    "numeric": NUMERIC, "number": NUMERIC, "int": NUMERIC, "integer": NUMERIC,
    "float": NUMERIC, "decimal": NUMERIC, "quantity": NUMERIC, "count": NUMERIC,
    "yes_no": YES_NO, "yesno": YES_NO, "boolean": YES_NO, "bool": YES_NO,
    "checkbox": YES_NO, "toggle": YES_NO,
    "single_answer": SINGLE_ANSWER, "singleanswer": SINGLE_ANSWER,
    "single_choice": SINGLE_ANSWER, "radio": SINGLE_ANSWER, "select": SINGLE_ANSWER,
    "dropdown": SINGLE_ANSWER, "choice": SINGLE_ANSWER,
    "multiple_answer": MULTIPLE_ANSWER, "multipleanswer": MULTIPLE_ANSWER,
    "multi_select": MULTIPLE_ANSWER, "multiselect": MULTIPLE_ANSWER,
    "checkbox_group": MULTIPLE_ANSWER, "multiple_choice": MULTIPLE_ANSWER,
    "text": TEXT, "string": TEXT, "textarea": TEXT, "long_text": TEXT,
    "short_text": TEXT, "paragraph": TEXT, "open": TEXT,
    "date": DATE, "datetime": DATE, "time": DATE,
    "rating": RATING, "scale": RATING, "nps": RATING, "stars": RATING, "likert": RATING,
    # Types that exist in the builder but carry no analysable answer.
    "photo": TEXT, "image": TEXT, "gallery": TEXT, "signature": TEXT, "file": TEXT,
}


def normalise_type(raw: Any) -> str:
    key = re.sub(r"[\s\-]+", "_", str(raw or "").strip().lower())
    return _TYPE_ALIASES.get(key, UNKNOWN)


# ---------------------------------------------------------------------------
# Column aliases
# ---------------------------------------------------------------------------
# camelCase from the export -> snake_case used internally. Unlisted columns are
# snake_cased and kept, so extra columns (customer_id, store_id, ...) survive
# untouched and become available to the filter layer automatically.

_QUESTION_ALIASES = {
    "id": "question_id", "formid": "form_id", "orderindex": "order_index",
    "text": "question_text", "type": "question_type", "options": "options_raw",
    "autofill": "autofill", "required": "required", "allowgallery": "allow_gallery",
    "createdat": "created_at", "updatedat": "updated_at", "deleted": "deleted",
}

_PROGRESS_ALIASES = {
    "id": "progress_id", "formid": "form_id",
    "representativeid": "representative_id", "representative_name": "representative_name",
    "representativename": "representative_name",
    "startdate": "start_date", "iscompleted": "is_completed",
    "completedat": "completed_at",
    # Reserved for the next export revision - optional everywhere in v1.
    "customerid": "customer_id", "customer_id": "customer_id",
    "customer_name": "customer_name", "customername": "customer_name",
    "storeid": "customer_id", "store_name": "customer_name",
}

_RESPONSE_ALIASES = {
    "id": "response_id", "questionid": "question_id", "progressid": "progress_id",
    "searchhash": "search_hash", "answer": "answer", "answerjson": "answer_json",
    "answerdate": "answer_date", "createdat": "created_at", "updatedat": "updated_at",
    "autofilled": "autofilled",
}

_ALIAS_SETS = {"questions": _QUESTION_ALIASES,
               "progresses": _PROGRESS_ALIASES,
               "responses": _RESPONSE_ALIASES}


def _snake(name: str) -> str:
    name = str(name).replace("﻿", "").strip()
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def normalise_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Rename to internal names. Unknown columns are snake_cased and kept."""
    aliases = _ALIAS_SETS[kind]
    mapping: dict[str, str] = {}
    for column in df.columns:
        raw = str(column).replace("﻿", "").strip()
        mapping[column] = aliases.get(raw.lower(), aliases.get(_snake(raw), _snake(raw)))
    out = df.rename(columns=mapping)
    return out.loc[:, ~out.columns.duplicated()]


def require_columns(df: pd.DataFrame, kind: str, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{kind}: missing required column(s) {missing}; "
                          f"got {sorted(df.columns)}")


class SchemaError(ValueError):
    """The export does not contain what the analyser needs to do anything."""


REQUIRED = {
    "questions": ("question_id", "question_text", "question_type"),
    "progresses": ("progress_id",),
    "responses": ("response_id", "question_id", "progress_id"),
}


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
# Three formats show up in this export and they all have to work:
#   1. JS Date.toString():  "Wed Apr 02 2025 08:25:19 GMT+0000 (Coordinated ...)"
#   2. dd/mm/yyyy:          "02/04/2025"          <- day-first, NOT US order
#   3. ISO 8601:            "2025-04-02T08:25:19Z"

_JS_TAIL = re.compile(r"\s*GMT[+-]\d{4}\s*\(.*\)\s*$")
_DDMMYYYY = re.compile(r"^\s*\d{1,2}[/.]\d{1,2}[/.]\d{4}\s*$")


def parse_datetimes(series: pd.Series) -> pd.Series:
    """Parse a mixed date column to tz-naive UTC datetimes. Unparseable values
    become NaT rather than raising - a broken row costs one row, not the run."""
    if series is None or len(series) == 0:
        return pd.Series(pd.to_datetime([]), dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series, errors="coerce")
        return out.dt.tz_localize(None) if getattr(out.dt, "tz", None) else out

    text = series.astype("string").str.strip()
    cleaned = text.str.replace(_JS_TAIL, "", regex=True)

    # Day-first block handled separately so 02/04/2025 is 2 April, not 4 Feb.
    dayfirst_mask = cleaned.str.match(_DDMMYYYY).fillna(False)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if dayfirst_mask.any():
        out.loc[dayfirst_mask] = pd.to_datetime(
            cleaned[dayfirst_mask], format="%d/%m/%Y", errors="coerce")
    rest = ~dayfirst_mask & cleaned.notna()
    if rest.any():
        parsed = pd.to_datetime(cleaned[rest], errors="coerce", utc=True, format="mixed")
        out.loc[rest] = parsed.dt.tz_localize(None)

    failed = int(out.isna().sum() - series.isna().sum())
    if failed > 0:
        log.warning("  %s/%s date values could not be parsed in %r",
                    failed, len(series), series.name)
    return out


# Ordered coarsest-last so `min()` picks the finest resolution present.
_RESOLUTIONS = ("second", "minute", "hour", "day")


def detect_resolution(series: pd.Series) -> str:
    """How precise is this timestamp column *in practice*?

    A column typed as datetime can still be day-precision data (every value at
    00:00:00) or, worse, a seeded ingest timestamp with two distinct values
    across 25k rows. Bucketing hourly on either is a lie, so the interval engine
    asks this before it picks a unit.
    """
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return "day"
    if (values.dt.hour.nunique() <= 1 and values.dt.minute.nunique() <= 1
            and values.dt.second.nunique() <= 1):
        return "day"
    if values.dt.second.nunique() > 1:
        return "second"
    if values.dt.minute.nunique() > 1:
        return "minute"
    return "hour"


def distinct_time_of_day(series: pd.Series) -> int:
    """Number of distinct times-of-day. A handful across thousands of rows means
    the timestamp was assigned by an import job, not by a human answering."""
    values = pd.to_datetime(series, errors="coerce").dropna()
    return 0 if values.empty else int(values.dt.time.nunique())


# ---------------------------------------------------------------------------
# Answer value coercion
# ---------------------------------------------------------------------------

_TRUE = {"yes", "y", "true", "1", "1.0", "да", "так", "ok", "done", "checked"}
_FALSE = {"no", "n", "false", "0", "0.0", "нет", "ні", "none", "unchecked"}


def to_bool(value: Any) -> bool | None:
    key = str(value).strip().lower()
    if key in _TRUE:
        return True
    if key in _FALSE:
        return False
    return None


def is_missing(value: Any) -> bool:
    """True for None, NaN, pandas.NA and friends. `pd.isna` on a list returns an
    array, so the container case is excluded first."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_number(value: Any) -> float | None:
    """Tolerant numeric parse: strips currency, thousands separators, units and
    a trailing percent sign, and accepts a comma decimal separator."""
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"[^\d,.\-+eE]", "", text)
    if not text or text in {"-", "+", ".", ","}:
        return None
    if "," in text and "." in text:                 # 1,234.56
        text = text.replace(",", "")
    elif text.count(",") == 1 and re.search(r",\d{1,2}$", text):   # 12,5
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


_OPTION_SPLIT = re.compile(r"\s*[,;|]\s*|\s*\n\s*")


def split_options(raw: Any) -> list[str]:
    """Split an option list or a multi-answer value. Handles the CSV form
    ("a,b,c"), JSON arrays, and real lists."""
    if isinstance(raw, (list, tuple, set)):
        return [str(v).strip() for v in raw if str(v).strip()]
    if is_missing(raw):
        return []
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        import json
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
    return [part.strip() for part in _OPTION_SPLIT.split(text) if part.strip()]
