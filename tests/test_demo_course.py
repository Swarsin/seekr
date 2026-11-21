"""Tests for the packaged demo course, evaluation queries, and wheel contents."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import seekr
from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.evaluation import load_queries
from seekr.extract import extract
from seekr.indexer import Indexer
from seekr.search import Searcher
from seekr.store import Store

REPO_ROOT = Path(__file__).resolve().parents[1]


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_HAS_FTS5 = _has_fts5()
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None
_HAS_PYMUPDF = importlib.util.find_spec("pymupdf") is not None
_HAS_SETUPTOOLS = importlib.util.find_spec("setuptools") is not None

_NEEDS_REAL_STORE = unittest.skipUnless(
    _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_FTS5,
    "numpy, sqlite-vec and a loadable FTS5 build are required",
)


class DemoCourseTests(unittest.TestCase):
    def test_layout(self) -> None:
        root = seekr.demo_course_path()
        self.assertIsInstance(root, Path)
        self.assertTrue(root.is_dir())
        expected_dirs = (
            "machine-learning/week1",
            "machine-learning/week2",
            "machine-learning/week3",
            "machine-learning/week4",
            "data-structures/week1",
            "data-structures/week2",
            "data-structures/week3",
            "4CCS1LGC-logic/week1",
            "4CCS1LGC-logic/week2",
        )
        for rel in expected_dirs:
            self.assertTrue((root / rel).is_dir(), rel)
        self.assertTrue((root / "SOURCES.md").is_file())
        self.assertTrue((root / "queries.jsonl").is_file())
        self.assertEqual(list(root.rglob("*.py")), [])

    def test_size_and_mix(self) -> None:
        files = [p for p in seekr.demo_course_path().rglob("*") if p.is_file()]
        course = [p for p in files if p.name not in {"SOURCES.md", "queries.jsonl"}]
        self.assertGreaterEqual(len(course), 15)
        self.assertLessEqual(len(course), 30)
        pdfs = [p for p in course if p.suffix == ".pdf"]
        self.assertGreaterEqual(len(pdfs), 2)
        self.assertLessEqual(len(pdfs), 3)
        self.assertLess(sum(p.stat().st_size for p in files), 1_000_000)

        for path in course:
            if path.suffix == ".md":
                words = len(path.read_text(encoding="utf-8").split())
                self.assertGreaterEqual(words, 150, path.name)
                self.assertLessEqual(words, 600, path.name)
            elif path.suffix == ".pdf" and _HAS_PYMUPDF:
                result = extract(path, "pdf")
                self.assertEqual(result.warnings, [], path.name)
                text = " ".join(unit.text for unit in result.units)
                words = len(text.split())
                self.assertGreaterEqual(words, 150, path.name)
                self.assertLessEqual(words, 600, path.name)

    def test_queries_reference_existing_files(self) -> None:
        root = seekr.demo_course_path()
        lines = (root / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(len(lines), 12)
        self.assertLessEqual(len(lines), 20)
        for line in lines:
            q = json.loads(line)
            self.assertTrue(q["relevant"])
            for rel in q["relevant"]:
                self.assertTrue((root / rel).is_file(), rel)

        queries = load_queries(root / "queries.jsonl")
        self.assertEqual(len(queries), len(lines))

    @_NEEDS_REAL_STORE
    def test_keyword_search_finds_expected_file_in_top3(self) -> None:
        root = seekr.demo_course_path().resolve()
        queries = load_queries(root / "queries.jsonl")
        embedder = FakeEmbedder(dim=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "index.db"
            cfg = Config(roots=(root,), db_path=db_path, embed_model="fake")
            store = Store(
                db_path,
                "fake",
                embedder.dim,
                vector_quant=cfg.vector_quant,
            )
            try:
                report = Indexer(store, embedder, cfg).sync([root])
                self.assertEqual(report.errors, [])
                searcher = Searcher(store, embedder)
                matched = 0
                for query in queries:
                    hits = searcher.search(
                        query.query, k=3, mode="keyword", per_file=1
                    )
                    top3 = {
                        Path(hit.path).resolve().relative_to(root).as_posix()
                        for hit in hits
                    }
                    if top3 & query.relevant:
                        matched += 1
            finally:
                store.close()

        self.assertGreaterEqual(matched / len(queries), 0.8)

    @unittest.skipUnless(_HAS_SETUPTOOLS, "setuptools is required to build a wheel")
    def test_built_wheel_contains_web_assets_and_demo_course(self) -> None:
        demo_root = seekr.demo_course_path()
        expected_demo = {
            f"seekr/demo_course/{p.relative_to(demo_root).as_posix()}"
            for p in demo_root.rglob("*")
            if p.is_file()
        }
        web_root = REPO_ROOT / "src" / "seekr" / "web_assets"
        expected_web = {
            f"seekr/web_assets/{p.relative_to(web_root).as_posix()}"
            for p in web_root.rglob("*")
            if p.is_file()
        }
        self.assertTrue(expected_web)
        self.assertTrue(expected_demo)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stage = tmp / "stage"
            stage.mkdir()
            for filename in ("pyproject.toml", "README.md", "LICENSE"):
                src_file = REPO_ROOT / filename
                if src_file.is_file():
                    shutil.copy2(src_file, stage / filename)
            shutil.copytree(
                REPO_ROOT / "src",
                stage / "src",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
            )
            dist_dir = tmp / "dist"
            dist_dir.mkdir()

            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "import setuptools.build_meta as bm; "
                        "print(bm.build_wheel(sys.argv[1]))"
                    ),
                    str(dist_dir),
                ],
                cwd=stage,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
            wheels = list(dist_dir.glob("*.whl"))
            self.assertEqual(len(wheels), 1, wheels)

            with zipfile.ZipFile(wheels[0]) as zf:
                names = set(zf.namelist())

        missing_web = sorted(expected_web - names)
        missing_demo = sorted(expected_demo - names)
        self.assertEqual(missing_web, [], f"missing web_assets in wheel: {missing_web}")
        self.assertEqual(
            missing_demo, [], f"missing demo_course files in wheel: {missing_demo}"
        )


if __name__ == "__main__":
    unittest.main()
