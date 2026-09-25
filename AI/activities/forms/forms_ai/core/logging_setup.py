"""Minimal logging setup. Replace with the project's existing module if one is
already present - `get_log(name)` is the only contract llm.py depends on."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%H:%M:%S"))
    root = logging.getLogger("forms_ai")
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False
    _CONFIGURED = True


def get_log(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"forms_ai.{name}")
