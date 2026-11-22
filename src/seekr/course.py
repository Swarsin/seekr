"""Derive course module and week labels from file paths."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WEEK_PATTERN: str = r"^(?:week|wk|w)[\s_-]*0*(\d{1,2})\b"


@dataclass(frozen=True)
class CourseLabel:
    """Module and week labels derived from a document path."""

    module: str | None = None
    week: int | None = None


def compile_week_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a case-insensitive week regex containing exactly one capture group."""
    if not isinstance(pattern, str):
        raise TypeError("week_pattern must be a string")
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"invalid week_pattern {pattern!r}: {exc}") from exc
    if compiled.groups != 1:
        raise ValueError(
            f"week_pattern must contain exactly one capture group, got {compiled.groups}"
        )
    return compiled


def _extract_week(folder: str, week_re: re.Pattern[str]) -> int | None:
    match = week_re.search(folder)
    if match is None:
        return None
    raw = match.group(1)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def label_for(path: Path, roots: Sequence[Path], week_re: re.Pattern[str]) -> CourseLabel:
    """Infer module and week labels for ``path`` relative to the longest matching root."""
    matching_roots = [root for root in roots if path.is_relative_to(root)]
    if not matching_roots:
        return CourseLabel()

    root = max(matching_roots, key=lambda item: len(item.parts))
    rel = path.relative_to(root).parts
    if not rel:
        return CourseLabel()

    week: int | None = None
    for folder in reversed(rel[:-1]):
        parsed = _extract_week(folder, week_re)
        if parsed is not None:
            week = parsed
            break

    if len(rel) >= 2:
        if _extract_week(rel[0], week_re) is not None:
            module: str | None = root.name or None
        else:
            module = rel[0]
    else:
        module = None

    return CourseLabel(module=module, week=week)


def display_module(name: str) -> str:
    """Format a raw folder module name for display without altering letter case."""
    return re.sub(r"[-_\s]+", " ", name).strip()


def labels_signature(roots: Sequence[Path], week_pattern: str, enabled: bool) -> str:
    """Return a SHA-256 hex digest representing label-affecting configuration."""
    payload = json.dumps(
        [sorted(str(root) for root in roots), week_pattern, bool(enabled)],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
