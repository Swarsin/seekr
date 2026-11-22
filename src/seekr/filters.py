"""Query filter parsing and validation helpers (stdlib-only)."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from datetime import datetime

from .course import display_module

__all__ = ["VALID_KINDS", "normalise_kind", "parse_since", "parse_week", "resolve_module"]

VALID_KINDS = frozenset({"pdf", "code", "markdown", "transcript", "text"})

_DURATION_MULTIPLIERS = {
    "m": 60,
    "h": 3600,
    "d": 86_400,
    "w": 604_800,
}
_DURATION_RE = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$", re.IGNORECASE)
_WEEK_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


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


def parse_week(value: str | None) -> tuple[int, int] | None:
    """Parse ``--week`` (e.g. ``4`` or ``3-5``) into an inclusive ``(low, high)`` range."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("week filter must be a string")
    raw = value.strip()
    if not raw:
        return None

    match = _WEEK_RE.match(raw)
    if match is None:
        raise ValueError(
            f"invalid --week value {value!r}; expected N or N-M with 1 <= N <= M <= 99"
        )
    start = int(match.group(1))
    end = start if match.group(2) is None else int(match.group(2))
    if not (1 <= start <= end <= 99):
        raise ValueError(
            f"invalid --week value {value!r}; expected N or N-M with 1 <= N <= M <= 99"
        )
    return (start, end)


def resolve_module(value: str | None, known: Sequence[str]) -> str | None:
    """Resolve a module name or unique prefix against ``known`` raw labels."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("module filter must be a string")
    if not value.strip():
        return None

    known_items = list(dict.fromkeys(known))
    if not known_items:
        raise ValueError("no modules are labelled yet; run `seekr index` first")

    needle = display_module(value).casefold()
    if needle:
        candidates = [(item, display_module(item).casefold()) for item in known_items]
        for bucket in (
            [item for item, folded in candidates if folded == needle],
            [item for item, folded in candidates if folded.startswith(needle)],
        ):
            if len(bucket) == 1:
                return bucket[0]
            if len(bucket) > 1:
                matches = ", ".join(sorted(bucket, key=lambda name: (name.casefold(), name)))
                raise ValueError(f"module {value!r} is ambiguous; matches: {matches}")

    choices = ", ".join(sorted(known_items, key=lambda name: (name.casefold(), name)))
    raise ValueError(f"unknown module {value!r}; known modules: {choices}")
