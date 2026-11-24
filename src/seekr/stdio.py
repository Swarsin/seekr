"""Make redirected standard streams carry UTF-8 on Windows."""

from __future__ import annotations

import codecs
import sys
from typing import Any, TextIO


def needs_utf8(platform: str, encoding: str | None, isatty: bool) -> bool:
    """Return whether a stream should be switched to UTF-8.

    Only redirected Windows streams qualify: Windows pipes and files default to
    the ANSI code page (for example cp1252), which cannot encode all of
    Unicode, while consoles use the Unicode console API and POSIX streams are
    already UTF-8 in practice.
    """
    if platform != "win32" or isatty:
        return False
    try:
        return codecs.lookup(encoding or "").name != "utf-8"
    except LookupError:
        return True


def configure_redirected_streams(
    platform: str = sys.platform,
    streams: tuple[tuple[TextIO | Any, str], ...] | None = None,
) -> None:
    """Reconfigure redirected stdin, stdout and stderr to UTF-8 where needed.

    ``streams`` pairs each stream with its error handler. Diagnostics must
    never crash on an odd character, so stderr escapes what it cannot encode;
    stdin and stdout stay strict.
    """
    if streams is None:
        streams = (
            (sys.stdin, "strict"),
            (sys.stdout, "strict"),
            (sys.stderr, "backslashreplace"),
        )
    for stream, errors in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            isatty = bool(stream.isatty())
        except (ValueError, OSError):
            isatty = False
        if needs_utf8(platform, getattr(stream, "encoding", None), isatty):
            reconfigure(encoding="utf-8", errors=errors)
