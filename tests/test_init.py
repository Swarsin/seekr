"""Tests for ``seekr init``, setup helpers, and demo-course provisioning."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from seekr.config import Config, load_config
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


class SetupFlowTests(unittest.TestCase):
    def test_scan_roots_counts_kinds_and_modules(self) -> None:
        from seekr.setup_flow import ScanSummary, scan_roots

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "course"
            (root / "machine-learning" / "week1").mkdir(parents=True)
            (root / "machine-learning" / "week3").mkdir(parents=True)
            (root / "linear-algebra").mkdir(parents=True)
            (root / "node_modules").mkdir(parents=True)
            (root / ".git").mkdir(parents=True)

            (root / "README.md").write_text("root overview", encoding="utf-8")
            (root / "machine-learning" / "week1" / "intro.md").write_text(
                "intro notes", encoding="utf-8"
            )
            (root / "machine-learning" / "week1" / "lab.py").write_text(
                "def f():\n    return 1\n", encoding="utf-8"
            )
            (root / "machine-learning" / "week3" / "notes.txt").write_text(
                "week 3 text", encoding="utf-8"
            )
            (root / "linear-algebra" / "syllabus.md").write_text(
                "syllabus without week folder", encoding="utf-8"
            )
            (root / "machine-learning" / "week1" / "diagram.png").write_bytes(b"png")
            (root / "node_modules" / "dep.md").write_text("ignored", encoding="utf-8")
            (root / ".git" / "HEAD.txt").write_text("ignored", encoding="utf-8")

            outside = base / "outside"
            outside.mkdir()
            secret = outside / "secret.md"
            secret.write_text("outside secret", encoding="utf-8")
            os.symlink(secret, root / "outside_link.md")

            symlink_dir = base / "linked_dir"
            symlink_dir.mkdir()
            (symlink_dir / "hidden.md").write_text("linked dir", encoding="utf-8")
            os.symlink(symlink_dir, root / "linked_dir")

            cfg = Config(roots=(root,))
            summary = scan_roots([root], cfg)

            self.assertIsInstance(summary, ScanSummary)
            self.assertEqual(
                summary.files_by_kind,
                {"code": 1, "markdown": 3, "text": 1},
            )
            self.assertEqual(summary.total, 5)
            self.assertEqual(
                summary.modules,
                {
                    "linear-algebra": (),
                    "machine-learning": (1, 3),
                },
            )

            # Verify scan_roots total matches Indexer.sync on_file notifications.
            on_file_paths: list[Path] = []
            tmp_db = base / "scan_check.db"
            if _NEEDS_REAL_STORE:
                store = Store(tmp_db, cfg.embed_model, 8, vector_quant=cfg.vector_quant)
                try:
                    Indexer(store, FakeEmbedder(dim=8), cfg).sync(
                        [root], on_file=on_file_paths.append
                    )
                finally:
                    store.close()
                self.assertEqual(len(on_file_paths), summary.total)

    def test_example_commands(self) -> None:
        from seekr.setup_flow import ScanSummary, example_commands

        with_weeks = example_commands(
            ScanSummary(
                files_by_kind={"markdown": 3},
                modules={"machine-learning": (1, 2)},
                total=3,
            )
        )
        self.assertEqual(len(with_weeks), 3)
        self.assertTrue(any("seekr search" in cmd for cmd in with_weeks))
        self.assertTrue(any("--module machine-learning" in cmd and "--week 1" in cmd for cmd in with_weeks))
        self.assertIn("seekr status", with_weeks)

        no_weeks = example_commands(
            ScanSummary(
                files_by_kind={"markdown": 2},
                modules={"linear algebra": ()},
                total=2,
            )
        )
        self.assertEqual(len(no_weeks), 3)
        self.assertTrue(any("linear algebra" in cmd for cmd in no_weeks))
        self.assertIn("seekr status", no_weeks)

        empty_mods = example_commands(
            ScanSummary(files_by_kind={"pdf": 2}, modules={}, total=2)
        )
        self.assertEqual(len(empty_mods), 3)
        self.assertIn("seekr status", empty_mods)

    def test_render_config_roundtrips_tricky_paths(self) -> None:
        from seekr.setup_flow import render_config, write_config

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            tricky_roots = [
                base / 'quote"dir',
                base / "back\\slash",
                base / "unicodé" / "日本語",
            ]
            tricky_db = base / 'db"dir\\store' / "index.db"

            rendered = render_config(tricky_roots, db_path=tricky_db)
            parsed = tomllib.loads(rendered)
            self.assertEqual(parsed["roots"], [str(path) for path in tricky_roots])
            self.assertEqual(parsed["db_path"], str(tricky_db))

            cfg_path = base / "config.toml"
            write_config(cfg_path, rendered, force=False)
            loaded = load_config(cfg_path)
            self.assertEqual(loaded.roots, tuple(tricky_roots))
            self.assertEqual(loaded.db_path, tricky_db)

            with self.assertRaises(FileExistsError):
                write_config(cfg_path, rendered, force=False)

    def test_suggest_windows_documents(self) -> None:
        from seekr.setup_flow import suggest_windows_documents

        def unexpected_run(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("run must not be called outside WSL")

        with mock.patch("seekr.platform_info.is_wsl", return_value=False):
            self.assertIsNone(suggest_windows_documents(run=unexpected_run))

        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp) / "Users" / "Alice"
            docs_dir = profile_dir / "Documents"
            docs_dir.mkdir(parents=True)

            calls: list[tuple[list[str], dict[str, Any]]] = []

            def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
                calls.append((argv, kwargs))
                if argv[0] == "cmd.exe":
                    return subprocess.CompletedProcess(
                        args=argv,
                        returncode=0,
                        stdout="C:\\Users\\Alice\r\n",
                        stderr="",
                    )
                if argv[:2] == ["wslpath", "-u"]:
                    return subprocess.CompletedProcess(
                        args=argv,
                        returncode=0,
                        stdout=f"{profile_dir}\n",
                        stderr="",
                    )
                raise AssertionError(f"unexpected command: {argv}")

            with mock.patch("seekr.platform_info.is_wsl", return_value=True):
                found = suggest_windows_documents(run=fake_run)

            self.assertEqual(found, docs_dir)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0], ["cmd.exe", "/c", "echo", "%USERPROFILE%"])
            self.assertEqual(calls[0][1]["timeout"], 3)
            self.assertEqual(calls[1][0], ["wslpath", "-u", "C:\\Users\\Alice"])
            self.assertEqual(calls[1][1]["timeout"], 3)

            docs_dir.rmdir()
            with mock.patch("seekr.platform_info.is_wsl", return_value=True):
                self.assertIsNone(suggest_windows_documents(run=fake_run))

            def failing_run(*_args: Any, **_kwargs: Any) -> Any:
                raise subprocess.TimeoutExpired(cmd=["cmd.exe"], timeout=3)

            with mock.patch("seekr.platform_info.is_wsl", return_value=True):
                self.assertIsNone(suggest_windows_documents(run=failing_run))

            def error_exit(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(args=argv, returncode=1, stdout="", stderr="err")

            with mock.patch("seekr.platform_info.is_wsl", return_value=True):
                self.assertIsNone(suggest_windows_documents(run=error_exit))


@unittest.skipUnless(_NEEDS_REAL_STORE, "numpy, sqlite-vec, typer and FTS5 are required")
class InitCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.home = self.base / "home"
        self.xdg_config = self.base / "xdg_config"
        self.xdg_data = self.base / "xdg_data"
        self.home.mkdir()
        self.xdg_config.mkdir()
        self.xdg_data.mkdir()

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("SEEKR_CONFIG", None)
        env.pop("HF_HUB_OFFLINE", None)
        env["HOME"] = str(self.home)
        env["XDG_CONFIG_HOME"] = str(self.xdg_config)
        env["XDG_DATA_HOME"] = str(self.xdg_data)
        return env

    @staticmethod
    def _runner() -> Any:
        from typer.testing import CliRunner

        return CliRunner()

    def _make_notes(self, name: str = "notes") -> Path:
        notes = self.base / name
        (notes / "ml" / "week1").mkdir(parents=True)
        (notes / "ml" / "week2").mkdir(parents=True)
        (notes / "ml" / "week1" / "intro.md").write_text(
            "# Gradient Descent\n\nGradient descent minimizes loss iteratively.",
            encoding="utf-8",
        )
        (notes / "ml" / "week2" / "bias.md").write_text(
            "# Bias Variance\n\nBias variance tradeoff controls generalization.",
            encoding="utf-8",
        )
        return notes

    def test_init_writes_config_pulls_and_indexes(self) -> None:
        from seekr.cli import app

        notes = self._make_notes()
        pull_calls: list[tuple[list[str], dict[str, Any]]] = []
        offline_seen_during_pull: list[str | None] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            pull_calls.append((list(cmd), dict(kwargs)))
            offline_seen_during_pull.append(os.environ.get("HF_HUB_OFFLINE"))
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        with (
            mock.patch.dict(os.environ, self._env(), clear=True),
            mock.patch("seekr.cli_init.subprocess.run", side_effect=fake_run),
            mock.patch("seekr.cli_index._build_embedder", side_effect=lambda _cfg: FakeEmbedder(dim=8)),
        ):
            result = self._runner().invoke(app, ["init", str(notes), "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(pull_calls), 1)
        self.assertEqual(
            pull_calls[0][0],
            [sys.executable, "-m", "seekr", "models", "pull", "--grammars", "--yes"],
        )
        self.assertEqual(offline_seen_during_pull, [None])

        cfg_path = self.xdg_config / "seekr" / "config.toml"
        self.assertTrue(cfg_path.is_file())
        cfg = load_config(cfg_path)
        self.assertEqual(cfg.roots, (notes.resolve(),))

        db_path = self.xdg_data / "seekr" / "index.db"
        self.assertTrue(db_path.is_file())
        store = Store(db_path, cfg.embed_model, 8, vector_quant=cfg.vector_quant)
        try:
            self.assertEqual(store.stats()["documents"], 2)
        finally:
            store.close()

        self.assertIn("Found 2 files", result.output)
        self.assertIn("Modules:", result.output)
        self.assertIn("ml (weeks 1-2)", result.output)
        self.assertIn("Indexed", result.output)
        self.assertIn("seekr search", result.output)
        self.assertIn("seekr status", result.output)

    def test_init_refuses_to_overwrite(self) -> None:
        from seekr.cli import app

        notes = self._make_notes()
        cfg_path = self.xdg_config / "seekr" / "config.toml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        original = 'roots = ["/tmp/original"]\n'
        cfg_path.write_text(original, encoding="utf-8")

        with mock.patch.dict(os.environ, self._env(), clear=True):
            result = self._runner().invoke(
                app, ["init", str(notes), "--yes", "--no-download"]
            )

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("already exists", result.output)
        self.assertIn("--force", result.output)
        self.assertEqual(cfg_path.read_text(encoding="utf-8"), original)

    def test_init_force_keeps_other_keys(self) -> None:
        from seekr.cli import app

        new_notes = self._make_notes("new_notes")
        old_notes = self._make_notes("old_notes")
        cfg_path = self.xdg_config / "seekr" / "config.toml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            (
                "# Keep this comment\n"
                "poll_interval_seconds = 30\n"
                f"roots = [{json.dumps(str(old_notes))}]  # trailing comment\n"
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.dict(os.environ, self._env(), clear=True),
            mock.patch("seekr.cli_index._build_embedder", side_effect=lambda _cfg: FakeEmbedder(dim=8)),
        ):
            result = self._runner().invoke(
                app, ["init", str(new_notes), "--force", "--yes", "--no-download"]
            )

        self.assertEqual(result.exit_code, 0, result.output)
        text = cfg_path.read_text(encoding="utf-8")
        self.assertIn("# Keep this comment", text)
        self.assertIn("poll_interval_seconds = 30", text)
        self.assertIn("# trailing comment", text)

        loaded = load_config(cfg_path)
        self.assertEqual(loaded.roots, (new_notes.resolve(),))
        self.assertEqual(loaded.poll_interval_seconds, 30)

    def test_init_no_download_skips_pull(self) -> None:
        from seekr.cli import app

        notes = self._make_notes()
        with (
            mock.patch.dict(os.environ, self._env(), clear=True),
            mock.patch("seekr.cli_init.subprocess.run") as mock_run,
            mock.patch("seekr.cli_index._build_embedder", side_effect=lambda _cfg: FakeEmbedder(dim=8)),
        ):
            result = self._runner().invoke(
                app, ["init", str(notes), "--yes", "--no-download"]
            )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_run.assert_not_called()

        # Also verify non-zero pull exit aborts cleanly before indexing.
        notes2 = self._make_notes("notes2")
        env2 = self._env()
        cfg2 = self.xdg_config / "seekr" / "config.toml"
        cfg2.unlink()
        with (
            mock.patch.dict(os.environ, env2, clear=True),
            mock.patch(
                "seekr.cli_init.subprocess.run",
                return_value=subprocess.CompletedProcess(args=[], returncode=1),
            ),
            mock.patch("seekr.cli_index._build_embedder") as mock_embed,
        ):
            fail_res = self._runner().invoke(app, ["init", str(notes2), "--yes"])

        self.assertEqual(fail_res.exit_code, 1, fail_res.output)
        mock_embed.assert_not_called()

    def test_init_yes_without_paths_errors(self) -> None:
        from seekr.cli import app

        with mock.patch.dict(os.environ, self._env(), clear=True):
            result = self._runner().invoke(app, ["init", "--yes"])

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn(
            "pass the folder to index, e.g. seekr init ~/Documents/ai-course",
            result.output,
        )

        # Invalid path and interactive prompt paths.
        with mock.patch.dict(os.environ, self._env(), clear=True):
            missing = self._runner().invoke(
                app, ["init", str(self.base / "missing"), "--yes", "--no-download"]
            )
            self.assertEqual(missing.exit_code, 1, missing.output)

            notes = self._make_notes("prompted_notes")
            with (
                mock.patch("seekr.setup_flow.suggest_windows_documents", return_value=notes),
                mock.patch(
                    "seekr.cli_index._build_embedder",
                    side_effect=lambda _cfg: FakeEmbedder(dim=8),
                ),
            ):
                prompted = self._runner().invoke(
                    app, ["init", "--no-download"], input="\n"
                )
            self.assertEqual(prompted.exit_code, 0, prompted.output)
            self.assertIn("Folder with your notes", prompted.output)
            self.assertEqual(
                load_config(self.xdg_config / "seekr" / "config.toml").roots,
                (notes.resolve(),),
            )

    def test_init_demo_without_config_uses_demo_as_root(self) -> None:
        from seekr.cli import app

        demo_dest = self.xdg_data / "seekr" / "demo-course"
        demo_dest.mkdir(parents=True)
        stale_file = demo_dest / "stale.md"
        stale_file.write_text("stale content", encoding="utf-8")

        with (
            mock.patch.dict(os.environ, self._env(), clear=True),
            mock.patch("seekr.cli_index._build_embedder", side_effect=lambda _cfg: FakeEmbedder(dim=8)),
        ):
            result = self._runner().invoke(
                app, ["init", "--demo", "--yes", "--no-download"]
            )
            cfg_path = self.xdg_config / "seekr" / "config.toml"
            self.assertTrue(cfg_path.is_file())
            cfg = load_config(cfg_path)
            search_res = self._runner().invoke(
                app, ["search", "bias variance", "--mode", "keyword"]
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(search_res.exit_code, 0, search_res.output)
        self.assertIn("bias-variance-tradeoff.md", search_res.output)
        self.assertTrue(demo_dest.is_dir())
        self.assertFalse(stale_file.exists())
        self.assertTrue(
            (demo_dest / "machine-learning" / "week2" / "bias-variance-tradeoff.md").is_file()
        )
        self.assertEqual(cfg.roots, (demo_dest,))
        self.assertEqual(cfg.db_path, self.xdg_data / "seekr" / "index.db")
        self.assertTrue(cfg.db_path.is_file())

        store = Store(cfg.db_path, cfg.embed_model, 8, vector_quant=cfg.vector_quant)
        try:
            self.assertEqual(store.stats()["documents"], 25)
        finally:
            store.close()

        self.assertIn("Found 25 files", result.output)
        self.assertIn("seekr search", result.output)
        self.assertIn("seekr status", result.output)

    def test_init_demo_with_config_writes_demo_toml(self) -> None:
        from seekr.cli import app

        cfg_path = self.xdg_config / "seekr" / "config.toml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        existing_text = '# personal config\nroots = ["/existing/notes"]\n'
        cfg_path.write_text(existing_text, encoding="utf-8")

        with (
            mock.patch.dict(os.environ, self._env(), clear=True),
            mock.patch("seekr.cli_index._build_embedder", side_effect=lambda _cfg: FakeEmbedder(dim=8)),
        ):
            result = self._runner().invoke(
                app, ["init", "--demo", "--yes", "--no-download"]
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(cfg_path.read_text(encoding="utf-8"), existing_text)

        demo_toml = cfg_path.parent / "demo.toml"
        self.assertTrue(demo_toml.is_file())
        demo_dest = self.xdg_data / "seekr" / "demo-course"
        demo_db = self.xdg_data / "seekr" / "demo.db"

        with mock.patch.dict(
            os.environ, {**self._env(), "SEEKR_CONFIG": str(demo_toml)}, clear=True
        ):
            demo_cfg = load_config(demo_toml)
            search_res = self._runner().invoke(
                app, ["search", "bias variance", "--mode", "keyword"]
            )

        self.assertEqual(search_res.exit_code, 0, search_res.output)
        self.assertIn("bias-variance-tradeoff.md", search_res.output)
        self.assertEqual(demo_cfg.roots, (demo_dest,))
        self.assertEqual(demo_cfg.db_path, demo_db)
        self.assertTrue(demo_db.is_file())
        self.assertFalse((self.xdg_data / "seekr" / "index.db").exists())

        store = Store(demo_db, demo_cfg.embed_model, 8, vector_quant=demo_cfg.vector_quant)
        try:
            self.assertEqual(store.stats()["documents"], 25)
        finally:
            store.close()

        self.assertIn(f'SEEKR_CONFIG={demo_toml} seekr search "bias variance"', result.output)


if __name__ == "__main__":
    unittest.main()
