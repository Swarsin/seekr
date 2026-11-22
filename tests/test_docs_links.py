"""Tests for relative links, fragment anchors, and required docs references."""

from __future__ import annotations

import re
import unittest
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)#\s]+)")
ANCHOR_LINK = re.compile(r"\]\((?!https?://|mailto:)([^)\s#]*)#([^)\s]+)\)")
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
ATX_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
REPO_FILE_URL = re.compile(
    r"https://(?:raw\.githubusercontent\.com/Swarsin/seekr/main|github\.com/Swarsin/seekr/(?:blob|tree)/main)/"
    r"([^)\s#\"']+)"
)


def _iter_non_fenced_lines(text: str) -> Iterator[str]:
    """Yield lines from markdown text outside fenced code blocks."""
    fence_char = ""
    fence_len = 0
    for line in text.splitlines():
        match = FENCE_RE.match(line)
        if match:
            token = match.group(1)
            if not fence_char:
                fence_char = token[0]
                fence_len = len(token)
                continue
            if token[0] == fence_char and len(token) >= fence_len and line[match.end() :].strip() == "":
                fence_char = ""
                fence_len = 0
                continue
        if not fence_char:
            yield line


def _github_slug(heading: str, counts: dict[str, int]) -> str:
    """Return a GitHub-style heading anchor slug, tracking duplicates."""
    cleaned = re.sub(r"\s+#+\s*$", "", heading)
    cleaned = INLINE_LINK_RE.sub(r"\1", cleaned).replace("`", "")
    normalized = re.sub(r"[^\w\s-]", "", cleaned.lower()).replace(" ", "-")
    seen = counts.get(normalized, 0)
    counts[normalized] = seen + 1
    return normalized if seen == 0 else f"{normalized}-{seen}"


def _heading_slugs(md_path: Path) -> set[str]:
    """Extract GitHub-style heading slugs from a markdown file."""
    counts: dict[str, int] = {}
    slugs: set[str] = set()
    for line in _iter_non_fenced_lines(md_path.read_text(encoding="utf-8")):
        match = ATX_HEADING_RE.match(line)
        if match:
            slugs.add(_github_slug(match.group(1), counts))
    return slugs


def _doc_markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


class DocsLinkTests(unittest.TestCase):
    def test_relative_links_resolve(self) -> None:
        broken: list[str] = []
        for md in _doc_markdown_files():
            body = "\n".join(_iter_non_fenced_lines(md.read_text(encoding="utf-8")))
            for target in LINK.findall(body):
                if not (md.parent / target).exists():
                    broken.append(f"{md.relative_to(ROOT)}: {target}")
        self.assertEqual(broken, [])

    def test_relative_fragment_anchors_resolve(self) -> None:
        slug_cache: dict[Path, set[str]] = {}
        broken: list[str] = []
        for md in _doc_markdown_files():
            body = "\n".join(_iter_non_fenced_lines(md.read_text(encoding="utf-8")))
            for raw_target, fragment in ANCHOR_LINK.findall(body):
                target = md.resolve() if not raw_target else (md.parent / raw_target).resolve()
                if not target.is_file():
                    broken.append(f"{md.relative_to(ROOT)}: {raw_target}#{fragment}")
                    continue
                slugs = slug_cache.get(target)
                if slugs is None:
                    slugs = _heading_slugs(target)
                    slug_cache[target] = slugs
                if fragment not in slugs:
                    broken.append(f"{md.relative_to(ROOT)}: {raw_target}#{fragment}")
        self.assertEqual(broken, [])

    def test_readme_has_only_absolute_links_for_pypi(self) -> None:
        # PyPI renders the README without the repository, so relative links and
        # images break there; the README must point at GitHub instead.
        body = "\n".join(_iter_non_fenced_lines((ROOT / "README.md").read_text(encoding="utf-8")))
        self.assertEqual(LINK.findall(body), [])

    def test_repository_urls_point_to_existing_files(self) -> None:
        broken: list[str] = []
        for md in _doc_markdown_files():
            body = "\n".join(_iter_non_fenced_lines(md.read_text(encoding="utf-8")))
            for target in REPO_FILE_URL.findall(body):
                if not (ROOT / target).exists():
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
            "https://raw.githubusercontent.com/Swarsin/seekr/main/docs/img/demo.gif",
            "https://raw.githubusercontent.com/Swarsin/seekr/main/docs/img/web.png",
            "claude mcp add seekr -- seekr mcp",
        ):
            self.assertIn(needle, readme)


if __name__ == "__main__":
    unittest.main()
