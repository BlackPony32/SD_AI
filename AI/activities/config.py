"""Single place for model, pricing, timeouts and on-disk layout.

Data lives at ``<DATA_ROOT>/<uuid>/<WORK_DIR_NAME>/<file>`` and every report
writes to ``<DATA_ROOT>/<uuid>/<OUTPUT_DIR_NAME>/``.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass

# --- models -----------------------------------------------------------------

MODEL = os.getenv("REPORT_AGENT_MODEL", "gpt-5.4-mini")

# USD per 1M tokens, for the local cost estimate only.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.25, 2.00),
    "default": (0.25, 2.00),
}

AGENT_TIMEOUT = int(os.getenv("REPORT_AGENT_TIMEOUT", "180"))

# --- per-topic knobs --------------------------------------------------------

MAX_TASK_PAIRS = 180        # distinct title+description pairs sent to the reader
MAX_NOTES = 200             # highest-signal notes kept for the notes agent
KEY_POINT_TARGET = 7        # key findings in the activity report

# --- filesystem -------------------------------------------------------------

DATA_ROOT = Path(os.getenv("REPORT_DATA_ROOT", "data"))
WORK_DIR_NAME = "work_data_folder"
OUTPUT_DIR_NAME = "agent_input"

DEFAULT_FILES: dict[str, dict[str, str]] = {
    "activities": {"activities": "raw_file_activities.csv", "orders": "raw_file_orders.csv"},
    "tasks": {"tasks": "raw_file_tasks.csv"},
    "notes": {"notes": "raw_file_notes.csv"},
}


def work_dir(uuid: str, data_root: str | Path | None = None) -> Path:
    return Path(data_root or DATA_ROOT) / uuid / WORK_DIR_NAME


def output_dir(uuid: str, topic: str | None = None,
               data_root: str | Path | None = None) -> Path:
    """Per-topic, so three reports for one tenant cannot overwrite each other's
    report.md the way they did when they all shared one folder."""
    base = Path(data_root or DATA_ROOT) / uuid / OUTPUT_DIR_NAME
    return base / topic if topic else base


def input_file(uuid: str, topic: str, role: str, filename: str | None = None,
               data_root: str | Path | None = None) -> Path:
    """Resolve one input file for a topic, e.g. ``input_file(u, "activities", "orders")``."""
    name = filename or DEFAULT_FILES[topic][role]
    return work_dir(uuid, data_root) / name
