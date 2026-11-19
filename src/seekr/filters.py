"""Query filter parsing and validation helpers (stdlib-only)."""

from __future__ import annotations

import re
import time
from datetime import datetime

__all__ = ["VALID_KINDS", "normalise_kind", "parse_since"]

VALID_KINDS = frozenset({"pdf", "code", "markdown", "transcript", "text"})

_DURATION_MULTIPLIERS = {
    "m": 60,
    "h": 3600,
    "d": 86_400,
    "w": 604_800,
}
_DURATION_RE = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$", re.IGNORECASE)


def normalise_kind(kind: str | None) -> str | None:
    """Validate and normalise a ``--kind`` filter value."""
    if kind is None:
        return None
    if not isinstance(kind, str):
        raise TypeError("kind filter must be a string")
    value = kind.strip().lower()
    if not value:
        return None
    if value not in VALID_KINDS:
        choices = ", ".join(sorted(VALID_KINDS))
        raise ValueError(f"unknown --kind value {kind!r}; expected one of: {choices}")
    return value


def parse_since(value: str | None, *, now: float | None = None) -> int | None:
    """Parse ``--since`` (e.g. ``7d``, ``12h``, ``2026-01-15``) to ``mtime_ns``."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("since filter must be a string")
    raw = value.strip()
    if not raw:
        return None

    match = _DURATION_RE.match(raw)
    if match is not None:
        amount = int(match.group(1))
        if amount <= 0:
            raise ValueError(
                f"invalid --since duration {value!r}; duration count must be positive"
            )
        unit = match.group(2).lower()
        ref_now = time.time() if now is None else float(now)
        threshold_s = max(0.0, ref_now - amount * _DURATION_MULTIPLIERS[unit])
        return int(threshold_s * 1_000_000_000)

    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            f"invalid --since value {value!r}; expected a duration like 30m, 12h, "
            "7d, 2w or an ISO date like 2026-01-15"
        ) from exc

    return max(0, int(parsed.timestamp() * 1_000_000_000))
