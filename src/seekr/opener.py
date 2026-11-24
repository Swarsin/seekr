"""Format, parse and open search hits at their source location.

Pick lines keep the user-facing ``<path>:<locator>\t<snippet>`` shape.  The
path is escaped reversibly (backslashes, colons, tabs, newlines and invisible
control characters), so a locator may itself contain colons.  Transcript
timestamps and Markdown headings with colons therefore stay readable, while a
path containing a colon is still unambiguous.  Snippets are display-only and
are sanitised and bounded.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .models import Hit

__all__ = [
    "OPEN_SUCCESS_CODES",
    "Argv",
    "OpenAction",
    "OpenFailed",
    "StartFile",
    "format_pick_line",
    "open_command",
    "open_target",
    "parse_pick_line",
    "run_open",
]

OPEN_SUCCESS_CODES: dict[str, frozenset[int]] = {"explorer.exe": frozenset({0, 1})}

_MAX_SNIPPET_CHARS = 240
_LINE_LOCATOR_RE = re.compile(r"L(\d+)(?:-\d+)?\Z")
_PAGE_LOCATOR_RE = re.compile(r"p\.(\d+)\Z")

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE_START = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_LIST_PREFIX = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_CLOSING_HASHES = re.compile(r"[ \t]+#+[ \t]*$")
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_ESCAPE_WIDTHS = {"x": 2, "u": 4, "U": 8}
_TEXT_KINDS = frozenset({"code", "markdown", "text"})


def _is_invisible(character: str) -> bool:
    return unicodedata.category(character) in _INVISIBLE_CATEGORIES


def _hex_escape(character: str) -> str:
    codepoint = ord(character)
    if codepoint <= 0xFF:
        return f"\\x{codepoint:02x}"
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04x}"
    return f"\\U{codepoint:08x}"


def _escape_value(value: str, *, escape_colons: bool) -> str:
    parts: list[str] = []
    for character in value:
        if character == "\\":
            parts.append("\\\\")
        elif character == ":" and escape_colons:
            parts.append("\\:")
        elif character == "\t":
            parts.append("\\t")
        elif character == "\n":
            parts.append("\\n")
        elif character == "\r":
            parts.append("\\r")
        elif _is_invisible(character):
            parts.append(_hex_escape(character))
        else:
            parts.append(character)
    return "".join(parts)


def _unescape_value(value: str) -> str:
    parts: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character != "\\" or index + 1 >= len(value):
            parts.append(character)
            index += 1
            continue

        marker = value[index + 1]
        if marker == "\\":
            parts.append("\\")
            index += 2
        elif marker == ":":
            parts.append(":")
            index += 2
        elif marker == "t":
            parts.append("\t")
            index += 2
        elif marker == "n":
            parts.append("\n")
            index += 2
        elif marker == "r":
            parts.append("\r")
            index += 2
        elif marker in _ESCAPE_WIDTHS:
            width = _ESCAPE_WIDTHS[marker]
            digits = value[index + 2 : index + 2 + width]
            if len(digits) == width and all(digit in _HEX_DIGITS for digit in digits):
                codepoint = int(digits, 16)
                try:
                    parts.append(chr(codepoint))
                except ValueError:
                    parts.append("\\")
                    index += 1
                else:
                    index += 2 + width
            else:
                parts.append("\\")
                index += 1
        else:
            # Preserve an unknown escape so a raw Unix path such as
            # ``C:\Users`` is not silently rewritten.
            parts.append("\\")
            index += 1
    return "".join(parts)


def _split_encoded_colon(value: str) -> tuple[str, str]:
    """Split at the first colon that is not part of a backslash escape."""
    index = 0
    while index < len(value):
        character = value[index]
        if character == "\\":
            index += 2
            continue
        if character == ":":
            return value[:index], value[index + 1 :]
        index += 1
    return value, ""


def _display_snippet(snippet: str) -> str:
    cleaned: list[str] = []
    for character in snippet:
        if character.isspace() or _is_invisible(character):
            cleaned.append(" ")
        else:
            cleaned.append(character)
    collapsed = " ".join("".join(cleaned).split())
    if len(collapsed) > _MAX_SNIPPET_CHARS:
        collapsed = collapsed[: _MAX_SNIPPET_CHARS - 1].rstrip() + "…"
    return collapsed


def format_pick_line(hit: Hit) -> str:
    """Render one dmenu-friendly pick line for *hit*.

    The left side is ``<path>:<locator>``.  Colons in the path are escaped so
    the parser can split there, while colons in the locator remain visible
    (for example transcript timestamps and Markdown heading titles).  The
    snippet is display-only and follows a single tab.
    """
    path = _escape_value(hit.path, escape_colons=True)
    locator = _escape_value(hit.locator, escape_colons=False)
    return f"{path}:{locator}\t{_display_snippet(hit.snippet)}"


def parse_pick_line(line: str) -> tuple[str, str]:
    """Return ``(path, locator)`` from a pick line or raw ``path:locator``.

    The first tab separates the optional snippet.  The path ends at the first
    colon that is not escaped, so raw unambiguous ``path:locator`` inputs and
    the escaped output of :func:`format_pick_line` both round-trip.
    """
    left = line.split("\t", 1)[0]
    encoded_path, encoded_locator = _split_encoded_colon(left)
    return _unescape_value(encoded_path), _unescape_value(encoded_locator)


def _safe_argv_path(path: str) -> str:
    """Keep a path from being parsed as an option or an editor command.

    Editors treat a leading ``+`` as an Ex command, even after ``+<line>``;
    option parsers treat a leading ``-`` as a flag.  ``./`` is harmless for
    every opener and keeps both cases a literal relative path.
    """
    return f"./{path}" if path.startswith(("-", "+")) else path


def _line_number(locator: str) -> int | None:
    match = _LINE_LOCATOR_RE.fullmatch(locator)
    if match is None:
        return None
    start = int(match.group(1))
    return start if start > 0 else None


def _page_number(locator: str) -> int | None:
    match = _PAGE_LOCATOR_RE.fullmatch(locator)
    if match is None:
        return None
    page = int(match.group(1))
    return page if page > 0 else None


def _editor_argv(env: Mapping[str, str]) -> list[str]:
    raw = env.get("EDITOR", "")
    try:
        argv = shlex.split(raw) if raw else []
    except ValueError:
        argv = []
    return argv or ["nvim"]


def _fence_marker(line: str) -> tuple[str, int] | None:
    match = _FENCE_START.match(line)
    if match is None:
        return None
    marker = match.group(1)
    return marker[0], len(marker)


def _setext_level(lines: list[str], index: int) -> int | None:
    if index + 1 >= len(lines):
        return None
    title = lines[index]
    stripped = title.lstrip(" ")
    if not stripped or len(title) - len(stripped) > 3:
        return None
    if stripped.startswith(">") or _LIST_PREFIX.match(stripped):
        return None
    match = _SETEXT_UNDERLINE.match(lines[index + 1])
    if match is None:
        return None
    return 1 if match.group(1).startswith("=") else 2


def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> str:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title or "(untitled)"))
    return "\u00a7" + " > ".join(item_title for _, item_title in stack)


def _markdown_heading_line(path: str, locator: str) -> int | None:
    """Return the 1-based source line of *locator*, when the file still has it."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None

    lines = text.splitlines()
    stack: list[tuple[int, str]] = []
    fence: tuple[str, int] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]

        if fence is not None:
            marker = _fence_marker(line)
            if marker is not None and marker[0] == fence[0] and marker[1] >= fence[1]:
                fence = None
            index += 1
            continue

        marker = _fence_marker(line)
        if marker is not None:
            fence = marker
            index += 1
            continue

        heading = _ATX_HEADING.match(line)
        if heading is not None:
            title = (heading.group(2) or "").strip()
            title = _CLOSING_HASHES.sub("", title).strip()
            current = _push_heading(stack, len(heading.group(1)), title)
            if current == locator:
                return index + 1
            index += 1
            continue

        setext_level = _setext_level(lines, index)
        if setext_level is not None:
            current = _push_heading(stack, setext_level, line.strip())
            if current == locator:
                return index + 1
            index += 2
            continue

        index += 1
    return None


def open_command(
    path: str,
    locator: str,
    kind: str,
    env: Mapping[str, str],
    which: Callable[[str], str | None],
    *,
    wsl: bool = False,
    to_windows: Callable[[str], str | None] | None = None,
) -> list[str]:
    """Build the argv that opens *path* at *locator*.

    The caller executes the returned argv directly; it is never passed through
    a shell.  On WSL (``wsl=True``), ``wslview`` is preferred and otherwise the
    path is converted for ``explorer.exe``.  On native Linux, PDFs use zathura
    when present, located text uses ``$EDITOR`` (split with :mod:`shlex`) at
    the line, Markdown heading locators use the exact heading line when it can
    still be found, and everything else falls back to ``xdg-open``.
    """
    safe_path = _safe_argv_path(path)

    if wsl:
        if which("wslview"):
            return ["wslview", safe_path]
        win = to_windows(path) if to_windows is not None else None
        if not win:
            raise ValueError(
                f"cannot convert {path} to a Windows path; install wslu for wslview"
            )
        return ["explorer.exe", win]

    if kind == "pdf":
        page = _page_number(locator)
        if page is not None and which("zathura"):
            return ["zathura", "-P", str(page), safe_path]
        return ["xdg-open", safe_path]

    if kind in _TEXT_KINDS:
        line = _line_number(locator)
        if line is None and kind == "markdown" and locator.startswith("\u00a7"):
            line = _markdown_heading_line(path, locator)
        if line is not None:
            return [*_editor_argv(env), f"+{line}", safe_path]

    return ["xdg-open", safe_path]


@dataclass(frozen=True)
class Argv:
    """Run *argv* directly (never through a shell)."""

    argv: list[str]


@dataclass(frozen=True)
class StartFile:
    """Open *path* with the Windows file association."""

    path: str


OpenAction = Argv | StartFile


class OpenFailed(Exception):
    """The opener ran but exited with an unsuccessful status."""

    def __init__(self, program: str, returncode: int) -> None:
        super().__init__(f"opener {program!r} exited with status {returncode}")
        self.program = program
        self.returncode = returncode


def open_target(
    path: str,
    locator: str,
    kind: str,
    *,
    env: Mapping[str, str],
    which: Callable[[str], str | None],
    platform: str = sys.platform,
    wsl: bool = False,
    to_windows: Callable[[str], str | None] | None = None,
) -> OpenAction:
    """Choose how to open *path* on the current platform.

    Windows and macOS use the native opener and ignore *locator* and *kind*;
    Linux and WSL keep the :func:`open_command` behaviour.
    """
    if platform == "win32":
        return StartFile(path)
    if platform == "darwin":
        return Argv(["open", _safe_argv_path(path)])
    return Argv(
        open_command(path, locator, kind, env, which, wsl=wsl, to_windows=to_windows)
    )


def run_open(action: OpenAction) -> None:
    """Execute *action*.

    Raises :class:`OpenFailed` on an unsuccessful exit status; ``OSError`` and
    ``ValueError`` from launching propagate to the caller.
    """
    if isinstance(action, StartFile):
        getattr(os, "startfile")(action.path)  # noqa: B009 - Windows only
        return
    result = subprocess.run(action.argv, check=False)
    success_codes = OPEN_SUCCESS_CODES.get(
        Path(action.argv[0]).name.lower(), frozenset({0})
    )
    if result.returncode not in success_codes:
        raise OpenFailed(action.argv[0], result.returncode)
