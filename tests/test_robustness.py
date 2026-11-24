"""Tests for structured --verbose logging and actionable user-facing errors."""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from seekr import daemon
from seekr.cli import app
from seekr.client import DaemonError
from seekr.config import Config, ConfigError, load_config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer
from seekr.logging_config import LOGGER_NAME, configure_logging, log_event
from seekr.opener import Argv
from seekr.search import Searcher
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


_NEEDS_REAL_STORE = _HAS_NUMPY and _HAS_SQLITE_VEC and _has_fts5()


class StructuredLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_logging(verbose=False)

    def test_default_logging_is_silent_and_verbose_formats_fields(self) -> None:
        logger = logging.getLogger("seekr.test")

        configure_logging(verbose=False)
        log_event(logger, logging.INFO, "should.not.emit", count=1)

        buf = io.StringIO()
        configure_logging(verbose=True, stream=buf)
        log_event(
            logger,
            logging.INFO,
            "indexer.sync.done",
            added=3,
            ok=True,
            empty=None,
            path=Path("/tmp/my notes/doc.md"),  # rendered with the host's separators
            note='a=b "q"',
        )
        output = buf.getvalue()

        self.assertIn("level=INFO", output)
        self.assertIn("logger=seekr.test", output)
        self.assertIn("event=indexer.sync.done", output)
        self.assertIn("added=3", output)
        self.assertIn("ok=true", output)
        self.assertIn("empty=null", output)
        self.assertIn(f"path={json.dumps(str(Path('/tmp/my notes/doc.md')))}", output)
        self.assertIn('note="a=b \\"q\\""', output)

    @unittest.skipUnless(_NEEDS_REAL_STORE, "numpy, sqlite-vec and FTS5 are required")
    def test_store_indexer_and_searcher_emit_structured_events_when_verbose(self) -> None:
        buf = io.StringIO()
        configure_logging(verbose=True, stream=buf)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "notes"
            root.mkdir()
            (root / "a.md").write_text("alpha beta gamma", encoding="utf-8")
            db_path = base / "index.db"
            cfg = Config(roots=(root,), db_path=db_path, embed_model="fake")
            embedder = FakeEmbedder(dim=8)
            store = Store(db_path, "fake", 8, vector_quant="int8")
            try:
                Indexer(store, embedder, cfg).sync([root])
                Searcher(store, embedder).search("alpha", mode="hybrid", k=5)
            finally:
                store.close()

        logs = buf.getvalue()
        self.assertIn("event=store.open", logs)
        self.assertIn("event=indexer.sync.start", logs)
        self.assertIn("event=indexer.sync.done", logs)
        self.assertIn("event=search.query", logs)
        self.assertEqual(logging.getLogger(LOGGER_NAME).level, logging.DEBUG)


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class SubprocessVerboseCliTests(unittest.TestCase):
    def _run_cli(self, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
        src = Path(__file__).resolve().parents[1] / "src"
        merged_env = dict(os.environ)
        merged_env.update(env)
        merged_env["PYTHONPATH"] = os.pathsep.join(
            [str(src), merged_env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        return subprocess.run(
            [sys.executable, "-m", "seekr.cli", *args],
            env=merged_env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_verbose_logs_to_stderr_without_polluting_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg_dir = base / "config" / "seekr"
            cfg_dir.mkdir(parents=True)
            (cfg_dir / "config.toml").write_text(
                f'db_path = "{(base / "index.db").as_posix()}"\n',
                encoding="utf-8",
            )
            env = {
                "XDG_CONFIG_HOME": str(base / "config"),
                "XDG_DATA_HOME": str(base / "data"),
                "XDG_RUNTIME_DIR": str(base / "runtime"),
                # Windows and macOS do not read XDG_CONFIG_HOME.
                "SEEKR_CONFIG": str(cfg_dir / "config.toml"),
            }

            quiet = self._run_cli(env, "search", "alpha", "--mode", "keyword")
            self.assertEqual(quiet.returncode, 0, quiet.stderr)
            self.assertEqual(quiet.stdout.strip(), "No results.")
            self.assertEqual(quiet.stderr.strip(), "")

            verbose = self._run_cli(env, "--verbose", "search", "alpha", "--mode", "keyword")
            self.assertEqual(verbose.returncode, 0, verbose.stderr)
            self.assertEqual(verbose.stdout.strip(), "No results.")
            self.assertIn("event=cli_search.local_open", verbose.stderr)
            self.assertIn("mode=keyword", verbose.stderr)


class ActionableErrorTests(unittest.TestCase):
    def test_config_errors_explain_cause_and_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_toml = Path(tmp) / "bad.toml"
            bad_toml.write_text("roots = [\n", encoding="utf-8")
            with self.assertRaises(ConfigError) as bad_exc:
                load_config(bad_toml)
            self.assertIn("invalid TOML", str(bad_exc.exception))
            self.assertIn("fix the syntax", str(bad_exc.exception))

            unknown_toml = Path(tmp) / "unknown.toml"
            unknown_toml.write_text('bogus_key = "x"\n', encoding="utf-8")
            with self.assertRaises(ConfigError) as unknown_exc:
                load_config(unknown_toml)
            self.assertIn("unknown config key", str(unknown_exc.exception))
            self.assertIn("valid keys are", str(unknown_exc.exception))

        with self.assertRaises(ConfigError) as quant_exc:
            Config(vector_quant="binary")
        self.assertIn("vector_quant must be one of", str(quant_exc.exception))
        self.assertIn("config.toml", str(quant_exc.exception))

    def test_daemon_bind_listener_refuses_non_socket_with_actionable_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            regular = Path(tmp) / "not-a-socket"
            regular.write_text("keep", encoding="utf-8")
            with self.assertRaises(daemon.DaemonError) as raised:
                daemon._bind_listener(regular)
            message = str(raised.exception)
            self.assertIn("non-socket", message)
            self.assertIn("--socket", message)

    @unittest.skipUnless(_HAS_TYPER, "typer is not installed")
    def test_cli_errors_are_actionable_without_tracebacks(self) -> None:
        from typer.testing import CliRunner

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(roots=(), db_path=Path(tmp) / "index.db")

            with mock.patch("seekr.cli_index._load_config", return_value=cfg):
                no_roots = runner.invoke(app, ["index"])
            self.assertEqual(no_roots.exit_code, 1)
            self.assertIn("no index roots", no_roots.output)
            self.assertIn("config.toml", no_roots.output)

            bad_mode = runner.invoke(app, ["search", "needle", "--mode", "bogus"])
            self.assertEqual(bad_mode.exit_code, 1)
            self.assertIn("expected 'keyword', 'vector' or 'hybrid'", bad_mode.output)

            with (
                mock.patch("seekr.cli_open.parse_pick_line", return_value=("/tmp/a.md", "L1")),
                mock.patch("seekr.cli_open.open_target", return_value=Argv(["definitely-missing-bin", "/tmp/a.md"])),
                mock.patch("seekr.opener.subprocess.run", side_effect=FileNotFoundError),
            ):
                missing_exe = runner.invoke(app, ["open", "/tmp/a.md:L1"])
            self.assertEqual(missing_exe.exit_code, 1)
            self.assertIn("executable not found", missing_exe.output)
            self.assertIn("$EDITOR", missing_exe.output)

            with mock.patch("seekr.cli_open.get_hits", side_effect=DaemonError("daemon timed out")):
                pick_err = runner.invoke(app, ["pick", "needle"])
            self.assertEqual(pick_err.exit_code, 1)
            self.assertIsNone(pick_err.exception if not isinstance(pick_err.exception, SystemExit) else None)
            self.assertIn("daemon timed out", pick_err.output)
