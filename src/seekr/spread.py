"""Cross-course 'Appears in' spread grouping and formatting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .course import display_module
from .models import Hit

__all__ = [
    "SPREAD_CANDIDATES",
    "SpreadGroup",
    "compute_spread",
    "format_spread",
    "spread_to_dict",
]

SPREAD_CANDIDATES = 30


@dataclass(frozen=True)
class SpreadGroup:
    """One ``(module, week)`` bucket of search hits for a topic."""

    module: str | None
    week: int | None
    first: bool
    chunk_ids: tuple[int, ...]


def compute_spread(hits: Sequence[Hit]) -> list[SpreadGroup]:
    """Group *hits* by ``(module, week)`` and mark the earliest week per module."""
    if not hits:
        return []

    grouped: dict[tuple[str | None, int | None], list[int]] = {}
    seen_ids: dict[tuple[str | None, int | None], set[int]] = {}
    min_week: dict[str, int] = {}

    for hit in hits:
        key = (hit.module, hit.week)
        chunk_id = int(hit.chunk_id)
        if key not in grouped:
            grouped[key] = [chunk_id]
            seen_ids[key] = {chunk_id}
        elif chunk_id not in seen_ids[key]:
            seen_ids[key].add(chunk_id)
            grouped[key].append(chunk_id)
        if hit.module is not None and hit.week is not None:
            previous = min_week.get(hit.module)
            if previous is None or hit.week < previous:
                min_week[hit.module] = hit.week

    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            key[0] is None,
            (key[0] or "").casefold(),
            key[0] or "",
            key[1] is None,
            key[1] if key[1] is not None else 0,
        ),
    )

    return [
        SpreadGroup(
            module=module,
            week=week,
            first=(
                module is not None
                and week is not None
                and week == min_week.get(module)
            ),
            chunk_ids=tuple(grouped[(module, week)]),
        )
        for module, week in ordered_keys
    ]


def spread_to_dict(group: SpreadGroup) -> dict[str, object]:
    """Serialise one :class:`SpreadGroup` to a JSON-friendly dictionary."""
    return {
        "module": group.module,
        "week": group.week,
        "first": group.first,
        "chunk_ids": list(group.chunk_ids),
    }


def format_spread(groups: Sequence[SpreadGroup]) -> str:
    """Format *groups* into one ``Appears in: ...`` line per module."""
    if not groups:
        return ""

    by_module: dict[str | None, list[SpreadGroup]] = {}
    for group in groups:
        by_module.setdefault(group.module, []).append(group)

    lines: list[str] = []
    for module, mod_groups in by_module.items():
        display = (display_module(module) or "other") if module is not None else "other"
        parts: list[str] = []
        for group in mod_groups:
            if group.week is None:
                parts.append("other")
            elif group.first:
                parts.append(f"Week {group.week} (first)")
            else:
                parts.append(f"Week {group.week}")
        lines.append(f"Appears in: {display}: {' · '.join(parts)}")
    return "\n".join(lines)
