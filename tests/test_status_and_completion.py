"""Tests for shell completions and polished seekr status output."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from seekr.cli import app
from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer
from seekr.store import Store

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
            self.assertIn("model: fake (dim=8, quant=int8, schema=v2)", text_res.output)
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
            self.assertEqual(parsed["schema_version"], 2)
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
