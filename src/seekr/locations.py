"""Human-readable locator descriptions and Windows/POSIX location strings."""

from __future__ import annotations

import re

__all__ = ["describe_locator", "format_location"]

_PAGE_RE = re.compile(r"p\.(\d+(?:-\d+)?)\Z")
_LINE_RANGE_RE = re.compile(r"L(\d+-\d+)\Z")
_LINE_SINGLE_RE = re.compile(r"L(\d+)\Z")
_SLIDE_RE = re.compile(r"slide\s+(\d+)\Z")
_CELL_RE = re.compile(r"cell\s+(\d+)\Z")
_TIMESTAMP_RE = re.compile(r"(?:\d+:)?\d{1,2}:\d{2}\Z")
_PARAGRAPH_RE = re.compile(r"\u00b6(\d+)\Z")


def describe_locator(locator: str) -> str:
    """Turn an internal chunk *locator* into a human-readable phrase."""
    text = locator.strip()
    if not text:
        return ""
    if text.startswith("\u00a7"):
        title = text[1:].strip()
        return f"section {title}" if title else text

    patterns: tuple[tuple[re.Pattern[str], str], ...] = (
        (_PAGE_RE, "page "),
        (_LINE_RANGE_RE, "lines "),
        (_LINE_SINGLE_RE, "line "),
        (_SLIDE_RE, "slide "),
        (_CELL_RE, "cell "),
        (_PARAGRAPH_RE, "paragraph "),
    )
    for pattern, prefix in patterns:
        match = pattern.fullmatch(text)
        if match is not None:
            return f"{prefix}{match.group(1)}"

    if _TIMESTAMP_RE.fullmatch(text):
        return f"at {text}"
    return text


def format_location(path: str, locator: str, *, windows_path: str | None) -> str:
    """Format *path* and *locator* for display, preferring *windows_path* when set."""
    base = windows_path if windows_path is not None else path
    description = describe_locator(locator)
    return f"{base} ({description})" if description else base
