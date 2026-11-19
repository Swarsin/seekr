"""Regression tests for the separated reindex override guards.

``--force`` is only allowed to override per-file indexing errors, while the
empty-rebuild guard has its own ``--allow-empty`` opt-in.  These tests pin that
separation at both the lifecycle and CLI layers.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from seekr.config import Config
from seekr.indexer import SyncReport
from seekr.lifecycle import RebuildError, rebuild_index

_HAS_TYPER = importlib.util.find_spec("typer") is not None
ROOT = Path(__file__).resolve().parents[1]
CLI_INDEX_PATH = ROOT / "src" / "seekr" / "cli_index.py"


def _make_database(path: Path, documents: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                mtime_ns INTEGER NOT NULL DEFAULT 0,
                size INTEGER NOT NULL DEFAULT 0,
                sha256 TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT 'text'
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        for number in range(documents):
            conn.execute(
                "INSERT INTO documents(path) VALUES (?)",
                (f"/old/document-{number}.txt",),
            )
        conn.commit()
    finally:
        conn.close()


def _document_count(path: Path) -> int | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def _add_live_document(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("INSERT INTO documents(path) VALUES (?)", ("/daemon/new.txt",))
        conn.commit()
    finally:
        conn.close()


class _FakeEmbedder:
    model_name = "fake"
    dim = 8

    def embed_documents(self, texts):
        return [f"vec:{text}" for text in texts]

    def embed_query(self, text):
        return f"vec:{text}"


class _FakeStore:
    def __init__(self, path: Path, embed_model: str, dim: int, *, documents: int = 1) -> None:
        self.path = Path(path)
        self.embed_model = embed_model
        self.dim = dim
        self.documents = documents
        _make_database(self.path, documents)

    def stats(self) -> dict[str, int]:
        return {"documents": self.documents, "chunks": self.documents}

    def close(self) -> None:
        pass


class _FakeIndexer:
    def __init__(self, report: SyncReport) -> None:
        self.report = report

    def sync(self, roots: list[Path]) -> SyncReport:
        return self.report


class _LiveWritingIndexer(_FakeIndexer):
    """Simulate a daemon adding a document while the rebuild is building."""

    def __init__(self, report: SyncReport, live_path: Path) -> None:
        super().__init__(report)
        self.live_path = live_path

    def sync(self, roots: list[Path]) -> SyncReport:
        _add_live_document(self.live_path)
        return self.report


class RebuildGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.db = self.tmp / "index.db"
        self.cfg = Config(roots=(self.root,), db_path=self.db)
        self.embedder = _FakeEmbedder()

    def _rebuild(
        self,
        report: SyncReport,
        *,
        new_documents: int = 1,
        force: bool = False,
        allow_empty: bool = False,
        indexer_factory=None,
    ) -> SyncReport:
        store_factory = lambda path, model, dim: _FakeStore(
            path, model, dim, documents=new_documents
        )
        factory = indexer_factory or (lambda store, embedder, cfg: _FakeIndexer(report))
        return rebuild_index(
            self.cfg,
            embedder=self.embedder,
            store_factory=store_factory,
            indexer_factory=factory,
            force=force,
            allow_empty=allow_empty,
        )

    def test_force_does_not_override_the_empty_rebuild_guard(self) -> None:
        _make_database(self.db, documents=2)

        with self.assertRaises(RebuildError):
            self._rebuild(SyncReport(), new_documents=0, force=True)

        self.assertEqual(_document_count(self.db), 2)

    def test_allow_empty_overrides_the_empty_rebuild_guard(self) -> None:
        _make_database(self.db, documents=2)

        self._rebuild(SyncReport(), new_documents=0, allow_empty=True)

        self.assertEqual(_document_count(self.db), 0)

    def test_allow_empty_does_not_override_indexing_errors(self) -> None:
        _make_database(self.db, documents=2)
        report = SyncReport(added=1, errors=[("bad.txt", "boom")])

        with self.assertRaises(RebuildError):
            self._rebuild(report, allow_empty=True)

        self.assertEqual(_document_count(self.db), 2)

    def test_force_still_overrides_indexing_errors(self) -> None:
        _make_database(self.db, documents=2)
        report = SyncReport(added=1, errors=[("bad.txt", "boom")])

        self._rebuild(report, force=True)

        self.assertEqual(_document_count(self.db), 1)

    def test_allow_empty_overrides_the_post_upgrade_race_guard(self) -> None:
        _make_database(self.db, documents=0)

        report = self._rebuild(
            SyncReport(),
            new_documents=0,
            allow_empty=True,
            indexer_factory=lambda store, embedder, cfg: _LiveWritingIndexer(SyncReport(), self.db),
        )

        self.assertEqual(report.added, 0)
        self.assertEqual(_document_count(self.db), 0)

    def test_race_guard_still_refuses_an_empty_rebuild_by_default(self) -> None:
        _make_database(self.db, documents=0)

        with self.assertRaises(RebuildError):
            self._rebuild(
                SyncReport(),
                new_documents=0,
                indexer_factory=lambda store, embedder, cfg: _LiveWritingIndexer(
                    SyncReport(), self.db
                ),
            )

        self.assertEqual(_document_count(self.db), 1)


class CliReindexGuardSourceTests(unittest.TestCase):
    """Cheap contracts that also run when Typer is unavailable."""

    def test_cli_exposes_allow_empty_separately_from_force(self) -> None:
        source = CLI_INDEX_PATH.read_text(encoding="utf-8")

        self.assertIn('"--allow-empty"', source)
        self.assertIn("allow_empty", source)
        self.assertIn("force=force_flag", source)
        self.assertIn("allow_empty=allow_empty_flag", source)


@unittest.skipUnless(_HAS_TYPER, "typer is required for CLI tests")
class CliReindexGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        unused = Path(tempfile.gettempdir()) / "seekr-unused"
        self.cfg = Config(roots=(unused,), db_path=unused / "index.db")

    @staticmethod
    def _runner() -> Any:
        from typer.testing import CliRunner

        return CliRunner()

    def _app(self) -> Any:
        import typer

        from seekr.cli_index import register

        app = typer.Typer()
        register(app)
        return app

    def _invoke(self, args: list[str]) -> Any:
        with (
            mock.patch("seekr.cli_index._load_config", return_value=self.cfg),
            mock.patch("seekr.cli_index.rebuild_index", return_value=SyncReport()) as rebuild,
        ):
            result = self._runner().invoke(self._app(), args)
        return result, rebuild

    def test_default_reindex_passes_both_overrides_disabled(self) -> None:
        result, rebuild = self._invoke(["reindex"])

        self.assertEqual(result.exit_code, 0, result.output)
        rebuild.assert_called_once_with(self.cfg, force=False, allow_empty=False)

    def test_force_only_disables_the_error_guard(self) -> None:
        result, rebuild = self._invoke(["reindex", "--force"])

        self.assertEqual(result.exit_code, 0, result.output)
        rebuild.assert_called_once_with(self.cfg, force=True, allow_empty=False)

    def test_allow_empty_only_disables_the_empty_guard(self) -> None:
        result, rebuild = self._invoke(["reindex", "--allow-empty"])

        self.assertEqual(result.exit_code, 0, result.output)
        rebuild.assert_called_once_with(self.cfg, force=False, allow_empty=True)

    def test_help_documents_both_guards(self) -> None:
        result = self._runner().invoke(self._app(), ["reindex", "--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        output = " ".join(result.output.split())
        self.assertIn("--force", output)
        self.assertIn("--allow-empty", output)
        self.assertIn("indexing errors", output)
        self.assertIn("empty", output)


if __name__ == "__main__":
    unittest.main()
