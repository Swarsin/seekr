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
from seekr.platform_info import Mount
from seekr.service_discovery import ServiceInfo
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
                mock.patch(
                    "seekr.service.find_running_service",
                    return_value=ServiceInfo(
                        pid=4242,
                        port=8123,
                        token="t",
                        version="9.9.9",
                        started_at="2026-01-01T00:00:00+00:00",
                    ),
                ),
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
                f"model: fake (dim=8, quant=int8, schema=v{SCHEMA_VERSION}, "
                f"backend={cfg.embed_backend})",
                text_res.output,
            )
            self.assertIn("by kind: code=1, markdown=1", text_res.output)
            self.assertIn(
                "service: running at 127.0.0.1:8123 (pid 4242, version 9.9.9)",
                text_res.output,
            )

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
            self.assertEqual(parsed["embed_backend"], cfg.embed_backend)
            self.assertEqual(parsed["daemon"], {"running": True, "socket": str(sock)})
            self.assertEqual(
                parsed["service"],
                {"running": True, "port": 8123, "pid": 4242, "version": "9.9.9"},
            )
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

    def test_status_shows_backend(self) -> None:
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as tmp:
            sock = Path(tmp) / "seekr.sock"
            for backend in ("onnx", "torch"):
                cfg = Config(
                    db_path=Path(tmp) / "missing.db", embed_model="fake", embed_backend=backend
                )
                with (
                    self.subTest(backend=backend),
                    mock.patch("seekr.cli_index._load_config", return_value=cfg),
                    mock.patch("seekr.cli_index._daemon_status", return_value=(False, sock)),
                ):
                    text = CliRunner().invoke(app, ["status"])
                    data = CliRunner().invoke(app, ["status", "--json"])
                    self.assertEqual(text.exit_code, 0, text.output)
                    self.assertIn(f"backend={backend}", text.output)
                    self.assertEqual(json.loads(data.stdout)["embed_backend"], backend)

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
        self.assertIn("service: not running", result.output)
        self.assertNotIn("daemon:", result.output)

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
        self.assertIn("service: not running", result.output)
        self.assertNotIn("daemon:", result.output)

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

    def test_status_prints_polled_roots_when_service_running(self) -> None:
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            fast_root = base / "repo"
            fast_root.mkdir()
            slow_root = Path("/mnt/c/Users/alice/notes")
            cfg = Config(
                roots=(fast_root, slow_root),
                db_path=base / "missing.db",
                embed_model="fake",
                poll_interval_seconds=25,
            )
            mounts = [
                Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
                Mount(source="C:\\", mount_point="/mnt/c", fstype="9p"),
            ]
            service = {"running": True, "port": 8000, "pid": 7, "version": "1"}
            with (
                mock.patch("seekr.cli_index._load_config", return_value=cfg),
                mock.patch(
                    "seekr.cli_index._daemon_status",
                    return_value=(False, base / "seekr.sock"),
                ),
                mock.patch("seekr.cli_index._service_status", return_value=service),
                mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
            ):
                running = CliRunner().invoke(app, ["status"])
            with (
                mock.patch("seekr.cli_index._load_config", return_value=cfg),
                mock.patch(
                    "seekr.cli_index._daemon_status",
                    return_value=(False, base / "seekr.sock"),
                ),
                mock.patch(
                    "seekr.cli_index._service_status",
                    return_value={"running": False},
                ),
                mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
            ):
                stopped = CliRunner().invoke(app, ["status"])

        self.assertEqual(running.exit_code, 0, running.output)
        self.assertIn("service: running at 127.0.0.1:8000", running.output)
        self.assertIn(f"watch: {slow_root} (polling every 25s)", running.output)
        self.assertNotIn(f"watch: {fast_root}", running.output)
        self.assertNotIn("watch:", stopped.output)

    def test_status_prints_polled_roots_when_daemon_running(self) -> None:
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            fast_root = base / "repo"
            fast_root.mkdir()
            (fast_root / "notes.md").write_text("# Doc\n\nhello", encoding="utf-8")
            slow_root = Path("/mnt/c/Users/alice/notes")
            db_path = base / "index.db"
            cfg_default = Config(
                roots=(fast_root, slow_root),
                db_path=db_path,
                embed_model="fake",
            )
            store = Store(db_path, "fake", 8)
            try:
                Indexer(store, FakeEmbedder(dim=8), cfg_default).sync([fast_root])
            finally:
                store.close()

            mounts = [
                Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
                Mount(source="C:\\", mount_point="/mnt/c", fstype="9p"),
            ]
            sock = base / "seekr.sock"
            with (
                mock.patch("seekr.cli_index._load_config", return_value=cfg_default),
                mock.patch("seekr.cli_index._daemon_status", return_value=(True, sock)),
                mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
            ):
                res_running = CliRunner().invoke(app, ["status"])

            self.assertEqual(res_running.exit_code, 0, res_running.output)
            self.assertIn(
                f"watch: {slow_root} (polling every 15s)",
                res_running.output,
            )
            self.assertNotIn(f"watch: {fast_root}", res_running.output)

            cfg_custom = Config(
                roots=(fast_root, slow_root),
                db_path=db_path,
                embed_model="fake",
                poll_interval_seconds=25,
            )
            with (
                mock.patch("seekr.cli_index._load_config", return_value=cfg_custom),
                mock.patch("seekr.cli_index._daemon_status", return_value=(True, sock)),
                mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
            ):
                res_custom = CliRunner().invoke(app, ["status"])

            self.assertEqual(res_custom.exit_code, 0, res_custom.output)
            self.assertIn(
                f"watch: {slow_root} (polling every 25s)",
                res_custom.output,
            )

            with (
                mock.patch("seekr.cli_index._load_config", return_value=cfg_default),
                mock.patch("seekr.cli_index._daemon_status", return_value=(False, sock)),
                mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
            ):
                res_stopped = CliRunner().invoke(app, ["status"])

            self.assertEqual(res_stopped.exit_code, 0, res_stopped.output)
            self.assertNotIn("polling every", res_stopped.output)

    def test_status_uses_live_daemon_polling_metadata(self) -> None:
        import threading
        import time

        from typer.testing import CliRunner

        from seekr import daemon

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"
            cfg = Config(roots=(), db_path=db_path, embed_model="fake")
            store = Store(db_path, "fake", 8)
            self.addCleanup(store.close)
            searcher = mock.Mock()
            searcher.store = store
            searcher.daemon_metadata = {
                "polled_roots": ["/mnt/c/live"],
                "poll_interval_seconds": 30,
            }
            sock = base / "seekr.sock"
            stop_event = threading.Event()
            thread = threading.Thread(
                target=daemon.serve,
                args=(searcher, sock, stop_event),
                daemon=True,
            )
            thread.start()
            try:
                for _ in range(100):
                    if sock.exists():
                        break
                    time.sleep(0.02)
                with (
                    mock.patch("seekr.cli_index._load_config", return_value=cfg),
                    mock.patch("seekr.cli_index._daemon_status", return_value=(True, sock)),
                ):
                    result = CliRunner().invoke(app, ["status"])
            finally:
                stop_event.set()
                thread.join(timeout=5.0)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("watch: /mnt/c/live (polling every 30s)", result.output)


@unittest.skipUnless(_HAS_TYPER, "typer is not installed")
class MissingDbWatchStatusTests(unittest.TestCase):
    def test_status_without_db_prints_polled_roots_when_daemon_running(self) -> None:
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            slow_root = Path("/mnt/c/notes")
            cfg = Config(
                roots=(slow_root,),
                db_path=base / "missing.db",
                embed_model="fake",
            )
            mounts = [Mount(source="C:\\", mount_point="/mnt/c", fstype="9p")]
            sock = base / "seekr.sock"
            with (
                mock.patch("seekr.cli_index._load_config", return_value=cfg),
                mock.patch("seekr.cli_index._daemon_status", return_value=(True, sock)),
                mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
            ):
                result = CliRunner().invoke(app, ["status"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("watch: /mnt/c/notes (polling every 15s)", result.output)
