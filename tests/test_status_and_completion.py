"""Tests for shell completions and polished seekr status output."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from seekr.cli import app
from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer
from seekr.store import SCHEMA_VERSION, Store

_HAS_TYPER = importlib.util.find_spec("typer") is not None
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_NEEDS_REAL_STORE = _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_TYPER and _has_fts5()


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class ShellCompletionTests(unittest.TestCase):
    def test_completion_generates_scripts_for_bash_zsh_and_fish(self) -> None:
        from typer.testing import CliRunner

        runner = CliRunner()
        for shell in ("bash", "zsh", "fish", " BASH "):
            with self.subTest(shell=shell):
                result = runner.invoke(app, ["completion", shell])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("_SEEKR_COMPLETE", result.stdout)
                self.assertGreater(len(result.stdout.strip()), 200)

    def test_completion_rejects_unsupported_shell_with_actionable_hint(self) -> None:
        from typer.testing import CliRunner

        result = CliRunner().invoke(app, ["completion", "powershell"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("unsupported shell", result.output)
        self.assertIn("bash", result.output)
        self.assertIn("zsh", result.output)
        self.assertIn("fish", result.output)


@unittest.skipUnless(_NEEDS_REAL_STORE, "numpy, sqlite-vec, typer and FTS5 are required")
class PolishedStatusTests(unittest.TestCase):
    def test_status_text_and_json_report_counts_kinds_metadata_and_daemon(self) -> None:
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "repo"
            root.mkdir()
            (root / "notes.md").write_text("# Doc\n\nhello world", encoding="utf-8")
            (root / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            db_path = base / "index.db"
            cfg = Config(roots=(root,), db_path=db_path, embed_model="fake", vector_quant="int8")

            store = Store(db_path, "fake", 8, vector_quant="int8")
            try:
                Indexer(store, FakeEmbedder(dim=8), cfg).sync([root])
                store.set_last_error_count(1)
            finally:
                store.close()

            sock = base / "seekr.sock"
            with (
                mock.patch("seekr.cli_index._load_config", return_value=cfg),
                mock.patch("seekr.cli_index._daemon_status", return_value=(True, sock)),
            ):
                runner = CliRunner()
                text_res = runner.invoke(app, ["status"])
                json_res = runner.invoke(app, ["status", "--json"])

            self.assertEqual(text_res.exit_code, 0, text_res.output)
            self.assertIn("documents: 2", text_res.output)
            self.assertIn("chunks: 2", text_res.output)
            self.assertIn("last error count: 1", text_res.output)
            self.assertIn("database:", text_res.output)
            self.assertIn(
                f"model: fake (dim=8, quant=int8, schema=v{SCHEMA_VERSION})",
                text_res.output,
            )
            self.assertIn("by kind: code=1, markdown=1", text_res.output)
            self.assertIn(f"daemon: running ({sock})", text_res.output)

            self.assertEqual(json_res.exit_code, 0, json_res.output)
            parsed = json.loads(json_res.stdout)
            self.assertTrue(parsed["exists"])
            self.assertEqual(parsed["documents"], 2)
            self.assertEqual(parsed["chunks"], 2)
            self.assertEqual(parsed["last_error_count"], 1)
            self.assertEqual(parsed["kinds"], {"code": 1, "markdown": 1})
            self.assertEqual(parsed["embed_model"], "fake")
            self.assertEqual(parsed["dim"], 8)
            self.assertEqual(parsed["vector_quant"], "int8")
            self.assertEqual(parsed["schema_version"], SCHEMA_VERSION)
            self.assertEqual(parsed["daemon"], {"running": True, "socket": str(sock)})
            self.assertGreater(parsed["size_bytes"], 0)

    def test_status_json_on_missing_database(self) -> None:
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=Path(tmp) / "missing.db", embed_model="fake")
            sock = Path(tmp) / "seekr.sock"
            with (
                mock.patch("seekr.cli_index._load_config", return_value=cfg),
                mock.patch("seekr.cli_index._daemon_status", return_value=(False, sock)),
            ):
                result = CliRunner().invoke(app, ["status", "--json"])

        self.assertEqual(result.exit_code, 0, result.output)
        parsed = json.loads(result.stdout)
        self.assertFalse(parsed["exists"])
        self.assertEqual(parsed["documents"], 0)
        self.assertEqual(parsed["chunks"], 0)
        self.assertEqual(parsed["last_error_count"], 0)
        self.assertFalse(parsed["daemon"]["running"])

    def test_status_reports_stopped_for_long_socket_path(self) -> None:
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runtime_dir = base / ("r" * 120)
            runtime_dir.mkdir()
            cfg = Config(db_path=base / "missing.db", embed_model="fake")
            env = {
                "XDG_RUNTIME_DIR": str(runtime_dir),
                "XDG_CONFIG_HOME": str(base / "config"),
                "XDG_DATA_HOME": str(base / "data"),
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch("seekr.cli_index._load_config", return_value=cfg),
            ):
                runner = CliRunner()
                result = runner.invoke(app, ["status"])
                json_result = runner.invoke(app, ["status", "--json"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("daemon: stopped", result.output)
        self.assertIn("socket path too long", result.output)
        self.assertIn(str(runtime_dir / "seekr.sock"), result.output)

        self.assertEqual(json_result.exit_code, 0, json_result.output)
        parsed = json.loads(json_result.stdout)
        self.assertFalse(parsed["daemon"]["running"])

    def test_status_existing_db_reports_stopped_for_long_socket_path(self) -> None:
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "repo"
            root.mkdir()
            (root / "notes.md").write_text("# Doc\n\nhello world", encoding="utf-8")
            db_path = base / "index.db"
            cfg = Config(roots=(root,), db_path=db_path, embed_model="fake")
            store = Store(db_path, "fake", 8)
            try:
                Indexer(store, FakeEmbedder(dim=8), cfg).sync([root])
            finally:
                store.close()

            runtime_dir = base / ("r" * 120)
            runtime_dir.mkdir()
            env = {
                "XDG_RUNTIME_DIR": str(runtime_dir),
                "XDG_CONFIG_HOME": str(base / "config"),
                "XDG_DATA_HOME": str(base / "data"),
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch("seekr.cli_index._load_config", return_value=cfg),
            ):
                result = CliRunner().invoke(app, ["status"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("daemon: stopped", result.output)
        self.assertIn("socket path too long", result.output)
        self.assertIn(str(runtime_dir / "seekr.sock"), result.output)

    def _create_course_index(self, base: Path) -> Config:
        root = base / "course"
        (root / "machine-learning" / "week1").mkdir(parents=True)
        (root / "machine-learning" / "week2").mkdir(parents=True)
        (root / "4CCS1ML" / "week-01").mkdir(parents=True)
        for i in range(1, 4):
            (root / "machine-learning" / "week1" / f"w1_{i}.md").write_text(
                f"ml week1 {i}", encoding="utf-8"
            )
        for i in range(1, 5):
            (root / "machine-learning" / "week2" / f"w2_{i}.md").write_text(
                f"ml week2 {i}", encoding="utf-8"
            )
        (root / "machine-learning" / "syllabus.md").write_text("syllabus", encoding="utf-8")
        (root / "4CCS1ML" / "week-01" / "notes.md").write_text("4ccs1ml", encoding="utf-8")
        (root / "general.md").write_text("general", encoding="utf-8")

        db_path = base / "index.db"
        cfg = Config(roots=(root,), db_path=db_path, embed_model="fake")
        store = Store(db_path, "fake", 8, vector_quant="int8")
        try:
            Indexer(store, FakeEmbedder(dim=8), cfg).sync([root])
        finally:
            store.close()
        return cfg

    def test_status_prints_course_table(self) -> None:
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = self._create_course_index(base)
            sock = base / "seekr.sock"
            with (
                mock.patch("seekr.cli_index._load_config", return_value=cfg),
                mock.patch("seekr.cli_index._daemon_status", return_value=(False, sock)),
            ):
                result = CliRunner().invoke(app, ["status"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("course:", result.output)
        self.assertIn("  4CCS1ML: week 1 (1)", result.output)
        self.assertIn("  machine learning: week 1 (3), week 2 (4), other (1)", result.output)
        self.assertIn("  other: other (1)", result.output)

    def test_status_json_has_course(self) -> None:
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = self._create_course_index(base)
            sock = base / "seekr.sock"
            with (
                mock.patch("seekr.cli_index._load_config", return_value=cfg),
                mock.patch("seekr.cli_index._daemon_status", return_value=(False, sock)),
            ):
                result = CliRunner().invoke(app, ["status", "--json"])

        self.assertEqual(result.exit_code, 0, result.output)
        parsed = json.loads(result.stdout)
        self.assertEqual(
            parsed["course"],
            [
                {"module": "4CCS1ML", "week": 1, "files": 1},
                {"module": "machine-learning", "week": 1, "files": 3},
                {"module": "machine-learning", "week": 2, "files": 4},
                {"module": "machine-learning", "week": None, "files": 1},
                {"module": None, "week": None, "files": 1},
            ],
        )
