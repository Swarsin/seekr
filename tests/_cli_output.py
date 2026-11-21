"""Helpers for asserting on Rich/Typer CLI output in tests."""

from __future__ import annotations

import re

# ANSI control sequence introducer (CSI): ESC "[" followed by parameter bytes
# (0-?), intermediate bytes (space through "/"), and a final byte (@-~).
# Rich emits sequences such as "\x1b[1;36m" inside styled option names, so
# assertions must run against the plain text rather than raw terminal output.
_CSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """Return *text* with ANSI CSI escape sequences removed.

    Typer's test runner honours ``FORCE_COLOR`` (as CI does), and Rich may
    split a single option name across several styled spans. Stripping the
    escapes keeps help assertions stable without changing production output
    or disabling color globally.
    """
    return _CSI_ESCAPE_RE.sub("", text)
