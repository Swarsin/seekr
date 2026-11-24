"""Tests for platform-aware path normalisation (pure; run on every OS)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from seekr.pathnorm import (
    canonical_path,
    case_insensitive,
    has_prefix,
    is_absolute,
    normalize_unicode,
    os_path,
    path_separator,
)


class CaseInsensitiveTests(unittest.TestCase):
    def test_windows_and_macos_are_case_insensitive(self) -> None:
        self.assertTrue(case_insensitive("win32"))
        self.assertTrue(case_insensitive("darwin"))
        self.assertFalse(case_insensitive("linux"))

    def test_separator(self) -> None:
        self.assertEqual(path_separator("win32"), "\\")
        self.assertEqual(path_separator("darwin"), "/")
        self.assertEqual(path_separator("linux"), "/")


class HasPrefixTests(unittest.TestCase):
    def test_has_prefix_case_insensitive_on_windows(self) -> None:
        self.assertTrue(
            has_prefix(r"C:\Users\Me\notes\a.md", r"c:\users\me\notes", platform="win32")
        )
        self.assertFalse(has_prefix("/home/Me/a.md", "/home/me", platform="linux"))

    def test_prefix_matches_on_separator_boundary(self) -> None:
        self.assertFalse(has_prefix(r"C:\notes2\a.md", r"C:\notes", platform="win32"))
        self.assertFalse(has_prefix("/notes2/a.md", "/notes", platform="linux"))
        self.assertTrue(has_prefix(r"C:\notes", "C:\\notes\\", platform="win32"))
        self.assertTrue(has_prefix("/notes/a.md", "/notes/", platform="linux"))

    def test_windows_accepts_forward_slashes(self) -> None:
        self.assertTrue(has_prefix(r"C:\notes\a.md", "c:/notes", platform="win32"))

    def test_drive_and_filesystem_roots_match_everything_below(self) -> None:
        self.assertTrue(has_prefix(r"C:\a.md", "C:\\", platform="win32"))
        self.assertTrue(has_prefix("/a.md", "/", platform="linux"))

    def test_macos_folds_case_and_unicode_form(self) -> None:
        self.assertTrue(
            has_prefix("/Users/Me/Mo\u0301dulo/a.md", "/users/me/m\u00f3dulo", platform="darwin")
        )

    def test_linux_backslash_is_an_ordinary_character(self) -> None:
        self.assertFalse(has_prefix("/notes\\x/a.md", "/notes", platform="linux"))


class CanonicalPathTests(unittest.TestCase):
    def test_nfc_on_darwin(self) -> None:
        nfd = "Mo\u0301dulo"
        # Against the host's own resolution (Windows adds a drive letter).
        self.assertEqual(
            canonical_path(Path("/x") / nfd, platform="darwin"),
            str((Path("/x") / "M\u00f3dulo").resolve()),
        )

    def test_linux_keeps_bytes_and_equals_resolve(self) -> None:
        nfd = Path("/x") / "Mo\u0301dulo"
        self.assertEqual(canonical_path(nfd, platform="linux"), str(nfd.resolve()))

    def test_relative_path_is_resolved(self) -> None:
        self.assertEqual(
            canonical_path(Path("rel/a.md"), platform="linux"),
            str(Path("rel/a.md").resolve()),
        )

    def test_normalize_unicode_only_on_darwin(self) -> None:
        self.assertEqual(normalize_unicode("e\u0301", platform="darwin"), "\u00e9")
        self.assertEqual(normalize_unicode("e\u0301", platform="linux"), "e\u0301")
        self.assertEqual(normalize_unicode("e\u0301", platform="win32"), "e\u0301")


class OsPathTests(unittest.TestCase):
    def test_long_windows_path(self) -> None:
        long_path = "C:\\" + "\\".join(["d" * 50] * 6) + "\\a.md"
        self.assertGreater(len(long_path), 259)
        self.assertEqual(os_path(long_path, platform="win32"), "\\\\?\\" + long_path)

    def test_long_unc_path(self) -> None:
        long_path = "\\\\server\\share\\" + "\\".join(["d" * 50] * 6)
        self.assertEqual(
            os_path(long_path, platform="win32"),
            "\\\\?\\UNC\\server\\share\\" + "\\".join(["d" * 50] * 6),
        )

    def test_short_or_prefixed_or_non_windows_paths_are_unchanged(self) -> None:
        self.assertEqual(os_path(r"C:\a.md", platform="win32"), r"C:\a.md")
        prefixed = "\\\\?\\C:\\" + "d" * 300
        self.assertEqual(os_path(prefixed, platform="win32"), prefixed)
        long_posix = "/" + "d" * 300
        self.assertEqual(os_path(long_posix, platform="linux"), long_posix)
        self.assertEqual(os_path(long_posix, platform="darwin"), long_posix)

    def test_default_platform_is_current(self) -> None:
        if sys.platform != "win32":
            self.assertEqual(os_path("/" + "d" * 300), "/" + "d" * 300)


if __name__ == "__main__":
    unittest.main()


class IsAbsoluteTests(unittest.TestCase):
    def test_windows_needs_a_drive_or_unc_share(self) -> None:
        win = {"platform": "win32"}
        for path in (r"C:\x", "C:/x", r"\\srv\share\x", "//srv/share/x", r"\\?\C:\x"):
            with self.subTest(path=path):
                self.assertTrue(is_absolute(path, **win))
        # Rooted but drive-less, drive-relative and relative paths depend on
        # the current drive or directory.
        for path in (r"\x", "/tmp/a", "C:x", "x", ""):
            with self.subTest(path=path):
                self.assertFalse(is_absolute(path, **win))

    def test_posix_is_a_leading_slash(self) -> None:
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                self.assertTrue(is_absolute("/tmp/a", platform=platform))
                self.assertFalse(is_absolute("tmp/a", platform=platform))
                self.assertFalse(is_absolute(r"C:\x", platform=platform))
