"""Tests for human-readable locator descriptions and Windows/POSIX locations."""

from __future__ import annotations

import unittest

from seekr.locations import describe_locator, format_location


class LocationTests(unittest.TestCase):
    def test_describe_and_format_location(self) -> None:
        cases = [
            ("", ""),
            ("   ", ""),
            ("p.14", "page 14"),
            ("p.14-16", "page 14-16"),
            ("L10-20", "lines 10-20"),
            ("L10", "line 10"),
            ("§A > B", "section A > B"),
            ("§Intro", "section Intro"),
            ("slide 7", "slide 7"),
            ("cell 3", "cell 3"),
            ("00:12:30", "at 00:12:30"),
            ("12:30", "at 12:30"),
            ("¶12", "paragraph 12"),
            ("  custom locator  ", "custom locator"),
        ]
        for locator, expected in cases:
            with self.subTest(locator=locator):
                self.assertEqual(describe_locator(locator), expected)

        self.assertEqual(
            format_location("/mnt/c/Users/me/a.pdf", "p.14", windows_path=r"C:\Users\me\a.pdf"),
            r"C:\Users\me\a.pdf (page 14)",
        )
        self.assertEqual(
            format_location("/mnt/c/Users/me/a.pdf", "", windows_path=r"C:\Users\me\a.pdf"),
            r"C:\Users\me\a.pdf",
        )
        self.assertEqual(
            format_location("/home/me/a.py", "L10-20", windows_path=None),
            "/home/me/a.py (lines 10-20)",
        )
        self.assertEqual(
            format_location("/home/me/a.py", "   ", windows_path=None),
            "/home/me/a.py",
        )


if __name__ == "__main__":
    unittest.main()
