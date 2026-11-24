"""Platform facts shared by tests that exercise OS-specific behaviour."""

from __future__ import annotations

import os
import tempfile
import unittest

from seekr import daemon

LEGACY_DAEMON = daemon.legacy_daemon_supported()
LEGACY_DAEMON_REASON = (
    "the legacy Unix-socket daemon needs SO_PEERCRED peer checks, which only Linux has"
)
requires_legacy_daemon = unittest.skipUnless(LEGACY_DAEMON, LEGACY_DAEMON_REASON)


def short_tempdir(prefix: str) -> tempfile.TemporaryDirectory[str]:
    """Return a temporary directory short enough to hold an AF_UNIX socket.

    macOS ``TMPDIR`` paths are long enough to overflow ``sun_path``, so POSIX
    hosts use ``/tmp``; elsewhere the default temporary directory is used.
    """
    if os.name == "posix" and os.path.isdir("/tmp"):
        return tempfile.TemporaryDirectory(dir="/tmp", prefix=prefix)
    return tempfile.TemporaryDirectory(prefix=prefix)
