"""One logger factory for every module (previously copy-pasted three times)."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def get_log(name: str, logfile: str = "project_log_many.log") -> logging.Logger:
    """Use the project's own logger when it is importable, else a stderr one."""
    try:
        from AI.utils import get_logger  # type: ignore

        return get_logger(name, logfile, False)
    except Exception:
        log = logging.getLogger(name)
        if not log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_FORMAT))
            log.addHandler(handler)
            log.setLevel(logging.INFO)
            log.propagate = False
        return log
