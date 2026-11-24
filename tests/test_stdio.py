"""Tests for UTF-8 handling of redirected Windows streams."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from seekr.stdio import configure_redirected_streams, needs_utf8


class NeedsUtf8Tests(unittest.TestCase):
    def test_redirected_windows_ansi_stream(self) -> None:
        self.assertTrue(needs_utf8("win32", "cp1252", False))
        self.assertTrue(needs_utf8("win32", None, False))
        self.assertTrue(needs_utf8("win32", "bogus-codec", False))

    def test_leaves_consoles_posix_and_utf8_alone(self) -> None:
        self.assertFalse(needs_utf8("win32", "cp1252", True))
        self.assertFalse(needs_utf8("linux", "cp1252", False))
        self.assertFalse(needs_utf8("darwin", "ascii", False))
        self.assertFalse(needs_utf8("win32", "utf-8", False))
        self.assertFalse(needs_utf8("win32", "UTF8", False))

    def test_reconfigures_only_streams_that_need_it(self) -> None:
        pipe = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        utf8 = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        configure_redirected_streams("win32", (pipe, utf8, io.StringIO()))
        self.assertEqual(pipe.encoding, "utf-8")
        pipe.write("ﬀ")
        configure_redirected_streams("linux", (io.TextIOWrapper(io.BytesIO(), encoding="cp1252"),))


class RedirectedOutputTests(unittest.TestCase):
    def test_cp1252_pipe_would_fail_without_reconfigure(self) -> None:
        code = "import sys; sys.stdout.write('\\ufb00')"
        env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, check=False)
        self.assertNotEqual(proc.returncode, 0)

    def test_cli_entry_writes_utf8_after_reconfigure(self) -> None:
        # Simulates the Windows pipe on any host: force the platform check true.
        code = (
            "import sys; from seekr.stdio import configure_redirected_streams as c; "
            "c('win32'); sys.stdout.write('\\ufb00'); sys.stdout.flush()"
        )
        env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.txt"
            with out.open("wb") as handle:
                proc = subprocess.run([sys.executable, "-c", code], env=env, stdout=handle, check=False)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(out.read_bytes(), "ﬀ".encode())


if __name__ == "__main__":
    unittest.main()
