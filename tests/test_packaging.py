"""Tests for package metadata, version alignment, and repository cleanup."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

import pytest

import seekr

ROOT = Path(__file__).resolve().parents[1]
_HAS_TYPER = importlib.util.find_spec("typer") is not None
_HAS_SETUPTOOLS = importlib.util.find_spec("setuptools") is not None


def _has_build_package() -> bool:
    """Return True when the PyPA ``build`` distribution is installed."""
    # Check distribution metadata rather than find_spec("build") because a
    # top-level ./build/ directory in the repo looks like a namespace package.
    try:
        importlib.metadata.version("build")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _build_wheel(dist_dir: Path, stage_root: Path) -> Path:
    """Build a wheel into ``dist_dir`` and return its path."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if _has_build_package():
        cmd = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
            str(ROOT),
        ]
        cwd = stage_root
    else:
        stage = stage_root / "stage"
        stage.mkdir(parents=True, exist_ok=True)
        for filename in ("pyproject.toml", "LICENSE", "README.md"):
            src_file = ROOT / filename
            if src_file.is_file():
                shutil.copy2(src_file, stage / filename)
        shutil.copytree(
            ROOT / "src",
            stage / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
        cmd = [
            sys.executable,
            "-c",
            "import sys, setuptools.build_meta as bm; print(bm.build_wheel(sys.argv[1]))",
            str(dist_dir),
        ]
        cwd = stage

    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)

    wheels = list(dist_dir.glob("seekr_search-*.whl"))
    if len(wheels) != 1:
        raise AssertionError(f"expected one seekr_search wheel, found {list(dist_dir.iterdir())}")
    return wheels[0]


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


@pytest.mark.slow
class WheelContentTests(unittest.TestCase):
    @unittest.skipUnless(_HAS_SETUPTOOLS, "setuptools is required to build a wheel offline")
    def test_wheel_contains_assets_and_demo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dist_dir = tmp / "dist"
            dist_dir.mkdir()
            wheel_path = _build_wheel(dist_dir, tmp)
            with zipfile.ZipFile(wheel_path) as zf:
                names = set(zf.namelist())

        for asset in ("app.js", "index.html", "styles.css"):
            self.assertIn(f"seekr/web_assets/{asset}", names)
        for meta in ("SOURCES.md", "queries.jsonl"):
            self.assertIn(f"seekr/demo_course/{meta}", names)
        for topic in ("machine-learning", "data-structures", "4CCS1LGC-logic"):
            self.assertTrue(
                any(n.startswith(f"seekr/demo_course/{topic}/") for n in names),
                topic,
            )
        self.assertTrue(
            any(n.startswith("seekr/demo_course/") and n.endswith(".pdf") for n in names)
        )

        web_root = ROOT / "src" / "seekr" / "web_assets"
        expected_web = {
            f"seekr/web_assets/{p.relative_to(web_root).as_posix()}"
            for p in web_root.rglob("*")
            if p.is_file()
        }
        demo_root = ROOT / "src" / "seekr" / "demo_course"
        expected_demo = {
            f"seekr/demo_course/{p.relative_to(demo_root).as_posix()}"
            for p in demo_root.rglob("*")
            if p.is_file()
        }
        self.assertTrue(expected_web)
        self.assertTrue(expected_demo)
        self.assertEqual(expected_web - names, set())
        self.assertEqual(expected_demo - names, set())


if __name__ == "__main__":
    unittest.main()
