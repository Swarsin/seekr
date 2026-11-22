"""Tests for relative links and required install/media references in docs."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)#\s]+)")


class DocsLinkTests(unittest.TestCase):
    def test_relative_links_resolve(self) -> None:
        files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
        broken: list[str] = []
        for md in files:
            for target in LINK.findall(md.read_text(encoding="utf-8")):
                if not (md.parent / target).exists():
                    broken.append(f"{md.relative_to(ROOT)}: {target}")
        self.assertEqual(broken, [])

    def test_readme_mentions_install_name(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("pipx install seekr-search", readme)

    def test_readme_links_benchmarks_and_images(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue((ROOT / "docs" / "benchmarks.md").is_file())
        for needle in (
            "docs/benchmarks.md",
            "docs/img/demo.gif",
            "docs/img/web.png",
            "claude mcp add seekr -- seekr mcp",
        ):
            self.assertIn(needle, readme)


if __name__ == "__main__":
    unittest.main()
