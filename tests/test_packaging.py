"""Tests for package metadata, version alignment, and repository cleanup."""

from __future__ import annotations

import importlib.util
import tomllib
import unittest
from pathlib import Path

import seekr

ROOT = Path(__file__).resolve().parents[1]
_HAS_TYPER = importlib.util.find_spec("typer") is not None


class PackagingMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]

    def test_distribution_name_is_seekr_search(self) -> None:
        self.assertEqual(self.project["name"], "seekr-search")

    def test_command_is_still_seekr(self) -> None:
        self.assertEqual(self.project["scripts"], {"seekr": "seekr.cli:app"})

    def test_urls_point_at_real_repo(self) -> None:
        expected_urls = {
            "Homepage": "https://github.com/Swarsin/seekr",
            "Repository": "https://github.com/Swarsin/seekr",
            "Issues": "https://github.com/Swarsin/seekr/issues",
            "Documentation": "https://github.com/Swarsin/seekr#readme",
        }
        self.assertEqual(self.project["urls"], expected_urls)

    def test_authors_and_dev_release_tools(self) -> None:
        self.assertEqual(self.project["authors"], [{"name": "Swarsin"}])
        dev_deps = self.project["optional-dependencies"]["dev"]
        self.assertIn("build>=1.2", dev_deps)
        self.assertIn("twine>=5", dev_deps)

    def test_version_matches_package(self) -> None:
        self.assertEqual(self.project["version"], seekr.__version__)
        self.assertEqual(seekr.__version__, "0.1.1")

    @unittest.skipUnless(_HAS_TYPER, "typer is not installed")
    def test_cli_version_prints_seekr_0_1_1(self) -> None:
        from typer.testing import CliRunner

        from seekr.cli import app

        result = CliRunner().invoke(app, ["--version"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stdout.strip(), "seekr 0.1.1")

    def test_stale_files_removed(self) -> None:
        for name in ("uv.lock",):
            self.assertFalse((ROOT / name).exists(), name)
        self.assertTrue((ROOT / "CHANGELOG.md").is_file())
