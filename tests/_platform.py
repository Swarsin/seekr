"""Platform facts shared by tests that exercise OS-specific behaviour."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pytest

from seekr import daemon

LEGACY_DAEMON = daemon.legacy_daemon_supported()
LEGACY_DAEMON_REASON = (
    "the legacy Unix-socket daemon needs SO_PEERCRED peer checks, which only Linux has"
)
requires_legacy_daemon = unittest.skipUnless(LEGACY_DAEMON, LEGACY_DAEMON_REASON)
legacy_daemon_only = pytest.mark.skipif(not LEGACY_DAEMON, reason=LEGACY_DAEMON_REASON)


def short_tempdir(prefix: str) -> tempfile.TemporaryDirectory[str]:
    """Return a temporary directory short enough to hold an AF_UNIX socket.

    macOS ``TMPDIR`` paths are long enough to overflow ``sun_path``, so POSIX
    hosts use ``/tmp``; elsewhere the default temporary directory is used.
    """
    if os.name == "posix" and os.path.isdir("/tmp"):
        return tempfile.TemporaryDirectory(dir="/tmp", prefix=prefix)
    return tempfile.TemporaryDirectory(prefix=prefix)


def use_posix_stored_paths(test: unittest.TestCase) -> None:
    """Apply Linux stored-path rules for *test*, which stores literal POSIX paths.

    On Windows a store matches prefixes on ``\\`` boundaries, so paths such as
    ``/notes/a.md`` would never sit "below" ``/notes``.  Tests of the Windows
    and macOS rules patch ``seekr.pathnorm._PLATFORM`` themselves.
    """
    patcher = mock.patch("seekr.pathnorm._PLATFORM", "linux")
    patcher.start()
    test.addCleanup(patcher.stop)
