"""Concurrency tests for daemon, watcher, indexer and lifecycle locks."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sqlite3
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path

from _platform import requires_legacy_daemon, short_tempdir
from seekr import cli_daemon, client
from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer
from seekr.lifecycle import LifecycleError, rebuild_index
from seekr.store import Store

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None
_HAS_WATCHDOG = importlib.util.find_spec("watchdog") is not None


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_HAS_FTS5 = _has_fts5()
_NEEDS_STACK = _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_WATCHDOG and _HAS_FTS5


def _wait_for(predicate: Callable[[], bool], timeout: float = 8.0, interval: float = 0.03) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _ping_daemon(sock_path: Path) -> bool:
    if not sock_path.exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.connect(str(sock_path))
        sock.sendall(b'{"op":"ping"}\n')
        line = sock.makefile("rb").readline(4096)
    except OSError:
        return False
    finally:
        sock.close()
    if not line:
        return False
    try:
        return json.loads(line).get("ok") is True
    except json.JSONDecodeError:
        return False


def _search_paths(sock_path: Path, query: str, *, mode: str = "hybrid") -> list[str]:
    hits = client.try_daemon_search(
        {
            "op": "search",
            "query": query,
            "k": 10,
            "mode": mode,
            "per_file": 2,
            "rerank": False,
            "path_prefix": None,
        },
        sock_path=sock_path,
        timeout=2.0,
    )
    if hits is None:
        return []
    return [hit.path for hit in hits]


@requires_legacy_daemon
@unittest.skipUnless(
    _NEEDS_STACK,
    "numpy, sqlite-vec, watchdog and FTS5 are required",
)
class DaemonWatcherIndexerConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = short_tempdir("seekr-c-")
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "docs"
        self.root.mkdir()
        self.db_path = self.base / "index.db"
        self.sock_path = self.base / "s.sock"
        self.cfg = Config(
            roots=(self.root,),
            db_path=self.db_path,
            embed_model="fake",
            vector_quant="int8",
        )
        self.embedder = FakeEmbedder(dim=8)

    def _start_daemon(
        self,
        *,
        debounce_s: float = 0.05,
    ) -> tuple[threading.Event, threading.Thread, list[BaseException]]:
        stop_event = threading.Event()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                cli_daemon.run_daemon(
                    cfg=self.cfg,
                    sock_path=self.sock_path,
                    debounce_s=debounce_s,
                    stop_event=stop_event,
                    embedder=self.embedder,
                    handle_signals=False,
                )
            except BaseException as exc:  # noqa: BLE001 - captured for thread assertion
                errors.append(exc)

        thread = threading.Thread(target=run, name="test-daemon", daemon=True)
        thread.start()
        self.addCleanup(self._stop_daemon, stop_event, thread)
        self.assertTrue(_wait_for(lambda: _ping_daemon(self.sock_path), timeout=5.0))
        return stop_event, thread, errors

    def _stop_daemon(self, stop_event: threading.Event, thread: threading.Thread) -> None:
        stop_event.set()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive(), "daemon thread failed to stop within 5s")

    def test_daemon_watcher_and_concurrent_readers_no_deadlock_or_lost_updates(self) -> None:
        seed = self.root / "seed.md"
        seed.write_text("seedbaseline initial token", encoding="utf-8")
        edit_targets: list[Path] = []
        for idx in range(4):
            path = self.root / f"edit_{idx}.md"
            path.write_text(f"beforeedit_{idx} old content", encoding="utf-8")
            edit_targets.append(path)
        rename_sources: list[Path] = []
        for idx in range(2):
            path = self.root / f"rename_src_{idx}.md"
            path.write_text(f"renametoken_{idx} stays searchable", encoding="utf-8")
            rename_sources.append(path)
        delete_targets: list[Path] = []
        for idx in range(3):
            path = self.root / f"delete_{idx}.md"
            path.write_text(f"deletedtoken_{idx} removable", encoding="utf-8")
            delete_targets.append(path)

        stop_daemon, daemon_thread, daemon_errors = self._start_daemon(debounce_s=0.05)

        stop_readers = threading.Event()
        reader_errors: list[tuple[int, BaseException]] = []
        queries_done = [0, 0, 0, 0]

        def reader(worker_id: int) -> None:
            modes = ("keyword", "vector", "hybrid")
            tokens = ("seedbaseline", "beforeedit_0", "afteredit_0", "createdtoken_0", "renametoken_0")
            try:
                step = 0
                while not stop_readers.is_set():
                    mode = modes[step % len(modes)]
                    token = tokens[step % len(tokens)]
                    hits = client.try_daemon_search(
                        {
                            "op": "search",
                            "query": token,
                            "k": 5,
                            "mode": mode,
                            "per_file": 2,
                            "rerank": False,
                            "path_prefix": None,
                        },
                        sock_path=self.sock_path,
                        timeout=2.0,
                    )
                    if hits is None:
                        raise RuntimeError("daemon returned None while expected active")
                    queries_done[worker_id] += 1
                    step += 1
                    time.sleep(0.005)
            except BaseException as exc:  # noqa: BLE001 - captured for thread assertion
                reader_errors.append((worker_id, exc))

        readers = [
            threading.Thread(target=reader, args=(i,), name=f"test-reader-{i}", daemon=True)
            for i in range(4)
        ]
        for thread in readers:
            thread.start()

        try:
            expected_tokens: dict[str, str] = {"seedbaseline": str(seed.resolve())}

            # Allow inotify watch threads to process queued startup events cleanly.
            time.sleep(0.1)

            for idx in range(8):
                path = self.root / f"created_{idx}.md"
                path.write_text(f"createdtoken_{idx} concurrent document", encoding="utf-8")
                expected_tokens[f"createdtoken_{idx}"] = str(path.resolve())
                time.sleep(0.01)

            for idx, path in enumerate(edit_targets):
                path.write_text(f"afteredit_{idx} updated document", encoding="utf-8")
                expected_tokens[f"afteredit_{idx}"] = str(path.resolve())
                time.sleep(0.01)

            for idx, src in enumerate(rename_sources):
                dst = self.root / f"renamed_dst_{idx}.md"
                os.rename(src, dst)
                expected_tokens[f"renametoken_{idx}"] = str(dst.resolve())

            for path in delete_targets:
                path.unlink()

            def all_updates_visible() -> bool:
                for token, expected_path in expected_tokens.items():
                    if expected_path not in _search_paths(self.sock_path, token, mode="keyword"):
                        return False
                for idx in range(len(delete_targets)):
                    if _search_paths(self.sock_path, f"deletedtoken_{idx}", mode="keyword"):
                        return False
                for idx, src in enumerate(rename_sources):
                    if str(src.resolve()) in _search_paths(
                        self.sock_path, f"renametoken_{idx}", mode="keyword"
                    ):
                        return False
                return True

            self.assertTrue(_wait_for(all_updates_visible, timeout=8.0, interval=0.05))
        finally:
            stop_readers.set()
            for thread in readers:
                thread.join(timeout=5.0)
                self.assertFalse(thread.is_alive(), "reader thread hung")

        self.assertEqual(reader_errors, [])
        self.assertTrue(all(count > 0 for count in queries_done), f"queries_done={queries_done}")

        for idx in range(4):
            hits = _search_paths(self.sock_path, f"afteredit_{idx}", mode="hybrid")
            self.assertIn(str(edit_targets[idx].resolve()), hits)

        self._stop_daemon(stop_daemon, daemon_thread)
        self.assertEqual(daemon_errors, [])
        self.assertFalse(self.sock_path.exists())

    def test_concurrent_indexer_sync_and_daemon_queries(self) -> None:
        initial = self.root / "initial.md"
        initial.write_text("initialsync baseline alpha", encoding="utf-8")
        extra_root = self.base / "extra"
        extra_root.mkdir()

        stop_daemon, daemon_thread, daemon_errors = self._start_daemon(debounce_s=0.1)

        stop_readers = threading.Event()
        reader_errors: list[tuple[int, BaseException]] = []
        writer_errors: list[BaseException] = []

        def reader(worker_id: int) -> None:
            modes = ("keyword", "vector", "hybrid")
            try:
                step = 0
                while not stop_readers.is_set():
                    hits = client.try_daemon_search(
                        {
                            "op": "search",
                            "query": "initialsync",
                            "k": 5,
                            "mode": modes[step % len(modes)],
                            "per_file": 2,
                            "rerank": False,
                            "path_prefix": None,
                        },
                        sock_path=self.sock_path,
                        timeout=2.0,
                    )
                    if hits is None:
                        raise RuntimeError("daemon unavailable during concurrent sync")
                    step += 1
                    time.sleep(0.005)
            except BaseException as exc:  # noqa: BLE001 - captured for thread assertion
                reader_errors.append((worker_id, exc))

        def concurrent_writer() -> None:
            store2 = Store(
                self.db_path,
                "fake",
                8,
                vector_quant=self.cfg.vector_quant,
            )
            try:
                indexer2 = Indexer(store2, self.embedder, self.cfg)
                for batch in range(3):
                    doc = extra_root / f"batch_{batch}.md"
                    doc.write_text(f"externalbatch_{batch} unique token", encoding="utf-8")
                    indexer2.sync([extra_root])
                    doc.write_text(f"externalbatch_{batch}_v2 updated token", encoding="utf-8")
                    indexer2.sync([extra_root])
            except BaseException as exc:  # noqa: BLE001 - captured for thread assertion
                writer_errors.append(exc)
            finally:
                store2.close()

        readers = [
            threading.Thread(target=reader, args=(i,), name=f"sync-reader-{i}", daemon=True)
            for i in range(3)
        ]
        writer = threading.Thread(target=concurrent_writer, name="sync-writer", daemon=True)
        for thread in readers:
            thread.start()
        writer.start()
        writer.join(timeout=10.0)
        stop_readers.set()
        for thread in readers:
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())

        self.assertFalse(writer.is_alive())
        self.assertEqual(writer_errors, [])
        self.assertEqual(reader_errors, [])

        for batch in range(3):
            expected_path = str((extra_root / f"batch_{batch}.md").resolve())
            hits = _search_paths(self.sock_path, f"externalbatch_{batch}_v2", mode="hybrid")
            self.assertIn(expected_path, hits)

        self._stop_daemon(stop_daemon, daemon_thread)
        self.assertEqual(daemon_errors, [])

    def test_daemon_lifecycle_lock_blocks_rebuild_until_stopped(self) -> None:
        (self.root / "note.md").write_text("lockguard test document", encoding="utf-8")
        stop_daemon, daemon_thread, daemon_errors = self._start_daemon(debounce_s=0.05)

        with self.assertRaises(LifecycleError) as raised:
            rebuild_index(self.cfg, embedder=self.embedder)
        message = str(raised.exception).lower()
        self.assertIn("in use", message)
        self.assertIn("daemon", message)

        self._stop_daemon(stop_daemon, daemon_thread)
        self.assertEqual(daemon_errors, [])

        report = rebuild_index(self.cfg, embedder=self.embedder)
        self.assertEqual(report.added, 1)
