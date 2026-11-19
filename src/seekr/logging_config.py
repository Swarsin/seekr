"""Structured stdlib-only logging helpers for seekr.

Importing this module intentionally loads only Python's standard library so the
CLI keeps its zero-heavy-import startup guarantees.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

__all__ = ["LOGGER_NAME", "configure_logging", "log_event"]

LOGGER_NAME = "seekr"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or any(ch.isspace() or ch in ('"', "=", "\\") for ch in text):
        return json.dumps(text, ensure_ascii=False)
    return text


def configure_logging(*, verbose: bool = False, stream: Any = None) -> logging.Logger:
    """Configure the root ``seekr`` logger for stderr diagnostics or silence."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, "_seekr_managed", False):
            logger.removeHandler(handler)
            handler.close()

    if verbose:
        logger.setLevel(logging.DEBUG)
        handler: logging.Handler = logging.StreamHandler(
            stream if stream is not None else sys.stderr
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    else:
        logger.setLevel(logging.WARNING)
        handler = logging.NullHandler()

    handler._seekr_managed = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit one structured ``event=... key=value`` log line when enabled."""
    if not logger.isEnabledFor(level):
        return
    parts = [f"event={_format_value(event)}"]
    for key in sorted(fields):
        parts.append(f"{key}={_format_value(fields[key])}")
    logger.log(level, "%s", " ".join(parts))
