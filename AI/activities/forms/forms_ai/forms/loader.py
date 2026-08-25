"""Turn a `user_id` into three already-downloaded CSVs and then into one flat
fact table.

The public entry point is `load_dataset(user_id)`. It is async because every
other public function in the pipeline is, and because the pandas work runs in a
worker thread (`asyncio.to_thread`) so a 25k-row parse cannot block the event
loop of the surrounding web app.

Files are located, never fetched. If they are not on disk, that is an error the
caller should surface, not something this layer papers over.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import FORMS_DATA_ROOT
from ..core.logging_setup import get_log
from . import schema as S

log = get_log("forms.loader")

# kind -> filename stems that count as a match, most specific first.
_FILE_STEMS: dict[str, tuple[str, ...]] = {
    "questions": ("generated_form_questions", "form_questions", "questions"),
    "progresses": ("generated_form_progresses", "form_progresses", "progresses", "progress"),
    "responses": ("generated_form_responses", "form_responses", "responses", "answers"),
}

_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".parquet")


class DatasetNotFound(FileNotFoundError):
    """No export on disk for this user."""


@dataclass(frozen=True)
class DatasetPaths:
    user_id: str
    root: Path
    questions: Path
    progresses: Path
    responses: Path

    def fingerprint(self) -> tuple:
        """(path, mtime, size) triples - the cache key. Re-download the export
        and the fingerprint changes, so the cache invalidates itself."""
        out = []
        for path in (self.questions, self.progresses, self.responses):
            stat = path.stat()
            out.append((str(path), stat.st_mtime_ns, stat.st_size))
        return tuple(out)


# ---------------------------------------------------------------------------
# Locating the files
# ---------------------------------------------------------------------------

def _find(directory: Path, stems: tuple[str, ...], user_id: str) -> Path | None:
    """Look for <stem><suffix>, then <user_id>_<stem><suffix>, then any file
    whose name contains the stem. Case-insensitive throughout."""
    if not directory.is_dir():
        return None
    entries = {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
    for stem in stems:
        for suffix in _SUFFIXES:
            for candidate in (f"{stem}{suffix}", f"{user_id}_{stem}{suffix}",
                              f"{stem}_{user_id}{suffix}"):
                hit = entries.get(candidate.lower())
                if hit:
                    return hit
    for stem in stems:
        for name, path in sorted(entries.items()):
            if stem in name and name.endswith(_SUFFIXES):
                return path
    return None


def resolve_dataset(user_id: str, root: str | os.PathLike | None = None) -> DatasetPaths:
    """Find this user's three exports.

    Search order:
      1. {root}/{user_id}/            - the expected per-user directory
      2. {root}/                      - flat layout, files prefixed by user_id
      3. {root}/**/{user_id}/         - one recursive sweep, for sharded roots
    """
    if not str(user_id or "").strip():
        raise ValueError("user_id is required")
    base = Path(root or FORMS_DATA_ROOT).expanduser()

    candidates: list[Path] = [base / str(user_id), base]
    if base.is_dir():
        candidates.extend(p for p in base.glob(f"**/{user_id}") if p.is_dir())

    tried: list[str] = []
    for directory in candidates:
        tried.append(str(directory))
        found = {kind: _find(directory, stems, str(user_id))
                 for kind, stems in _FILE_STEMS.items()}
        if all(found.values()):
            log.info("dataset for user %s resolved in %s", user_id, directory)
            return DatasetPaths(user_id=str(user_id), root=directory,
                                questions=found["questions"],       # type: ignore[arg-type]
                                progresses=found["progresses"],     # type: ignore[arg-type]
                                responses=found["responses"])       # type: ignore[arg-type]
        if any(found.values()):
            missing = [k for k, v in found.items() if not v]
            log.warning("  %s: partial export, missing %s", directory, missing)

    raise DatasetNotFound(
        f"no complete form export for user_id={user_id!r}. Looked in: {tried}. "
        f"Expected files named like {[s[0] for s in _FILE_STEMS.values()]}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    separator = "\t" if ".tsv" in path.suffixes else ","
    return pd.read_csv(path, sep=separator, dtype="string",
                       keep_default_na=True, encoding="utf-8-sig",
                       on_bad_lines="warn", low_memory=False)


# ---------------------------------------------------------------------------
# Choosing which timestamp represents "when the answer was given"
# ---------------------------------------------------------------------------
# Exports carry several date columns and they are not equally trustworthy. In
# the current one, `answerDate` is 100% null and `completedAt` is - for half the
# rows - the *import* timestamp, not a submission time. Hard-coding a column
# would silently produce a wrong time axis, so candidates are scored instead
# and the decision is written into the result metadata.

EVENT_TIME_CANDIDATES: tuple[str, ...] = (
    "answer_date",        # response-level: when this answer was recorded
    "completed_at",       # progress-level: when the form was submitted
    "start_date",         # progress-level: the business date being reported on
    "created_at",         # last resort: row insert time
)

# One timestamp accounting for more than this share of rows means the column was
# stamped by a job, not by people filling in a form.
_INGEST_STAMP_SHARE = 0.20


def score_time_column(series: pd.Series) -> dict[str, Any]:
    """Judge a candidate time column. Higher `score` is better; `usable` is
    False when the column cannot carry the time axis at all."""
    total = len(series)
    values = pd.to_datetime(series, errors="coerce").dropna()
    n = len(values)
    coverage = (n / total) if total else 0.0
    if n == 0:
        return {"coverage": 0.0, "resolution": "day", "distinct_days": 0,
                "top_timestamp_share": 1.0, "looks_like_ingest_stamp": False,
                "usable": False, "score": -1.0}

    top_share = float(values.value_counts(normalize=True).iloc[0])
    distinct_days = int(values.dt.normalize().nunique())
    resolution = S.detect_resolution(values)
    # A genuine event column spreads across many days. An ingest stamp piles up
    # on one instant and covers few days relative to its row count.
    ingest = top_share >= _INGEST_STAMP_SHARE and n >= 50

    score = coverage
    score += min(distinct_days / max(n, 1), 1.0)      # spread across the calendar
    score -= 2.0 if ingest else 0.0
    score += {"second": 0.3, "minute": 0.25, "hour": 0.2, "day": 0.0}[resolution]

    return {"coverage": round(coverage, 4), "resolution": resolution,
            "distinct_days": distinct_days,
            "top_timestamp_share": round(top_share, 4),
            "looks_like_ingest_stamp": ingest,
            "usable": coverage >= 0.5 and not ingest,
            "score": round(score, 4)}


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------

@dataclass
class FormDataset:
    """Normalised frames plus the flat fact table everything else reads."""

    user_id: str
    paths: DatasetPaths
    questions: pd.DataFrame
    progresses: pd.DataFrame
    responses: pd.DataFrame
    facts: pd.DataFrame
    event_time_field: str
    time_resolution: str
    time_column_report: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def form_ids(self) -> list[str]:
        return sorted(self.progresses["form_id"].dropna().unique().tolist()) \
            if "form_id" in self.progresses else []

    def has_customer_dimension(self) -> bool:
        return "customer_id" in self.progresses.columns and \
            bool(self.progresses["customer_id"].notna().any())


def _build(paths: DatasetPaths, event_time_field: str | None) -> FormDataset:
    warnings: list[str] = []

    questions = S.normalise_columns(_read_table(paths.questions), "questions")
    progresses = S.normalise_columns(_read_table(paths.progresses), "progresses")
    responses = S.normalise_columns(_read_table(paths.responses), "responses")
    for kind, frame in (("questions", questions), ("progresses", progresses),
                        ("responses", responses)):
        S.require_columns(frame, kind, S.REQUIRED[kind])

    # --- questions -------------------------------------------------------
    questions["question_type_raw"] = questions["question_type"]
    questions["question_type"] = questions["question_type_raw"].map(S.normalise_type)
    questions["options"] = questions.get("options_raw", pd.Series(dtype="string")) \
        .map(S.split_options)
    questions["order_index"] = pd.to_numeric(
        questions.get("order_index"), errors="coerce").fillna(0).astype(int)
    if "deleted" in questions:
        deleted = questions["deleted"].map(S.to_bool).fillna(False)
        if deleted.any():
            warnings.append(f"{int(deleted.sum())} deleted question(s) excluded")
            questions = questions.loc[~deleted].copy()
    unknown = questions["question_type"] == S.UNKNOWN
    if unknown.any():
        warnings.append(
            f"{int(unknown.sum())} question(s) have an unrecognised type "
            f"({sorted(set(questions.loc[unknown, 'question_type_raw'].dropna()))}); "
            "their answers will be typed by sniffing")

    # --- progresses ------------------------------------------------------
    for column in ("start_date", "completed_at", "created_at", "updated_at"):
        if column in progresses:
            progresses[column] = S.parse_datetimes(progresses[column])
    if "is_completed" in progresses:
        progresses["is_completed"] = progresses["is_completed"].map(S.to_bool)
    if "customer_id" not in progresses:
        progresses["customer_id"] = pd.NA      # reserved; optional in v1
    if "representative_id" not in progresses:
        progresses["representative_id"] = pd.NA
    if "representative_name" not in progresses:
        progresses["representative_name"] = pd.NA
    progresses["representative_label"] = (
        progresses["representative_name"].fillna(progresses["representative_id"])
        .fillna("(unattributed)"))

    if {"start_date", "completed_at"} <= set(progresses.columns):
        both = progresses[["start_date", "completed_at"]].dropna()
        if len(both) > 0:
            mismatch = (both["start_date"].dt.normalize()
                        != both["completed_at"].dt.normalize()).mean()
            if mismatch > 0.05:
                warnings.append(
                    f"start_date and completed_at disagree on the calendar day for "
                    f"{mismatch:.0%} of submissions - one of them is not a real "
                    f"submission time")

    # --- responses -------------------------------------------------------
    for column in ("answer_date", "created_at", "updated_at"):
        if column in responses:
            responses[column] = S.parse_datetimes(responses[column])
    if "autofilled" in responses:
        responses["autofilled"] = responses["autofilled"].map(S.to_bool).fillna(False)

    # --- join ------------------------------------------------------------
    progress_cols = [c for c in ("progress_id", "form_id", "representative_id",
                                 "representative_label", "customer_id", "customer_name",
                                 "is_completed", "start_date", "completed_at")
                     if c in progresses.columns]
    question_cols = [c for c in ("question_id", "question_text", "question_type",
                                 "question_type_raw", "order_index", "options",
                                 "required")
                     if c in questions.columns]

    facts = (responses
             .merge(progresses[progress_cols], on="progress_id", how="inner",
                    suffixes=("", "_progress"))
             .merge(questions[question_cols], on="question_id", how="inner",
                    suffixes=("", "_question")))

    dropped = len(responses) - len(facts)
    if dropped:
        warnings.append(f"{dropped} response(s) dropped: no matching progress or "
                        f"question row")

    # --- time axis -------------------------------------------------------
    report = {name: score_time_column(facts[name])
              for name in EVENT_TIME_CANDIDATES if name in facts.columns}

    if event_time_field and event_time_field in facts.columns:
        chosen = event_time_field
    else:
        usable = {k: v for k, v in report.items() if v["usable"]}
        pool = usable or {k: v for k, v in report.items() if v["coverage"] > 0}
        if not pool:
            raise S.SchemaError(
                "no usable timestamp column: cannot place responses on a time axis")
        chosen = max(pool, key=lambda k: pool[k]["score"])
        if not usable:
            warnings.append(f"no fully trustworthy time column; falling back to "
                            f"{chosen!r}")

    for name, info in report.items():
        if info["looks_like_ingest_stamp"]:
            warnings.append(
                f"{name!r} looks like an import timestamp "
                f"({info['top_timestamp_share']:.0%} of rows share one instant), "
                f"not a submission time")

    facts["event_time"] = pd.to_datetime(facts[chosen], errors="coerce")
    # Fill the gaps from the remaining candidates, best-scoring first.
    for name in sorted(report, key=lambda k: -report[k]["score"]):
        if name != chosen and facts["event_time"].isna().any():
            facts["event_time"] = facts["event_time"].fillna(
                pd.to_datetime(facts[name], errors="coerce"))
    missing_time = int(facts["event_time"].isna().sum())
    if missing_time:
        warnings.append(f"{missing_time} response(s) have no usable timestamp and are "
                        f"excluded from interval statistics")

    resolution = S.detect_resolution(facts["event_time"])
    if resolution == "day":
        warnings.append("event timestamps are day-precision: sub-daily (hourly) "
                        "buckets are not available for this export")

    facts["answer_text"] = facts["answer"].astype("string")
    if "answer_json" in facts:
        facts["answer_multi"] = facts["answer_json"].map(S.split_options)
    else:
        facts["answer_multi"] = [[] for _ in range(len(facts))]

    log.info("loaded user=%s: %s questions, %s progresses, %s facts; "
             "event_time=%s (%s precision)", paths.user_id, len(questions),
             len(progresses), len(facts), chosen, resolution)

    return FormDataset(user_id=paths.user_id, paths=paths, questions=questions,
                       progresses=progresses, responses=responses, facts=facts,
                       event_time_field=chosen, time_resolution=resolution,
                       time_column_report=report, warnings=warnings)


# ---------------------------------------------------------------------------
# Cache + public API
# ---------------------------------------------------------------------------

_CACHE: dict[tuple, FormDataset] = {}
_CACHE_LOCK = asyncio.Lock()


async def load_dataset(user_id: str, *, root: str | os.PathLike | None = None,
                       event_time_field: str | None = None,
                       use_cache: bool = True) -> FormDataset:
    """Locate, parse and normalise this user's export.

    Raises DatasetNotFound if the files are not on disk and SchemaError if they
    are there but unusable.
    """
    paths = await asyncio.to_thread(resolve_dataset, user_id, root)
    key = (paths.fingerprint(), event_time_field)

    if use_cache:
        async with _CACHE_LOCK:
            hit = _CACHE.get(key)
        if hit is not None:
            log.info("dataset cache hit for user=%s", user_id)
            return hit

    dataset = await asyncio.to_thread(_build, paths, event_time_field)

    if use_cache:
        async with _CACHE_LOCK:
            _CACHE[key] = dataset
            if len(_CACHE) > 16:                      # keep the cache bounded
                _CACHE.pop(next(iter(_CACHE)))
    return dataset


def clear_cache() -> None:
    _CACHE.clear()
