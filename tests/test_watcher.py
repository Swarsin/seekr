"""Tests for the file watcher.

The default suite exercises debounce, delete and move routing through a small
stdlib-only fake observer, in-memory store and embedder.  Real watchdog
integration tests are skipped when watchdog (or the optional SQLite vector
backend) is not installed.
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from seekr import pathnorm
from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer
from seekr.platform_info import Mount
from seekr.search import Searcher
from seekr.store import DocRow, StoredChunk
from seekr.watcher import Watcher, WatcherError

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_WATCHDOG = importlib.util.find_spec("watchdog") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_HAS_FTS5 = _has_fts5()


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


@dataclass
class _FakeEvent:
    event_type: str
    src_path: str
    dest_path: str = ""
    is_directory: bool = False


class _FakeObserver:
    def __init__(self) -> None:
        self.schedules: list[tuple[object, str, bool]] = []
        self.started = False
        self.stopped = False

    def schedule(self, handler: object, path: str, recursive: bool) -> None:
        self.schedules.append((handler, path, recursive))

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout: float | None = None) -> None:
        return None


class _RecordingIndexer:
    """Duck-typed Indexer that records every watcher operation."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.sync_roots: list[tuple[Path, ...]] = []
        self.store = SimpleNamespace(list_paths=lambda prefix=None: [])

    def index_path(self, path: Path) -> str:
        self.calls.append(("index", Path(path)))
        return "added"

    def remove_path(self, path: Path) -> None:
        self.calls.append(("remove", Path(path)))

    def move_path(self, src: Path, dst: Path) -> str:
        self.calls.append(("move", Path(src), Path(dst)))
        return "unchanged"

    def sync(self, roots: list[Path]) -> object:
        resolved = tuple(Path(root) for root in roots)
        self.sync_roots.append(resolved)
        self.calls.append(("sync", resolved))
        return object()


class _MemoryStore:
    """Minimal Store implementation for real Indexer watcher tests."""

    def __init__(self) -> None:
        self.documents: dict[str, DocRow] = {}
        self.vectors: dict[str, object] = {}
        self.chunks: dict[int, tuple[int, StoredChunk]] = {}
        self.meta: dict[str, str] = {}
        self._next_id = 1
        self._next_chunk_id = 1

    @contextmanager
    def transaction(self, *, write: bool = False):
        yield

    def get_document(self, path: str) -> DocRow | None:
        return self.documents.get(str(path))

    def upsert_document(
        self,
        path: str,
        mtime_ns: int,
        size: int,
        sha256: str,
        kind: str,
        *,
        module: str | None = None,
        week: int | None = None,
    ) -> int:
        key = str(path)
        existing = self.documents.get(key)
        doc_id = existing.id if existing is not None else self._next_id
        if existing is None:
            self._next_id += 1
        self.documents[key] = DocRow(
            id=doc_id,
            path=key,
            mtime_ns=int(mtime_ns),
            size=int(size),
            sha256=sha256,
            kind=kind,
            module=None if module is None else str(module),
            week=None if week is None else int(week),
        )
        return doc_id

    def set_document_labels(
        self, labels: Iterable[tuple[str, str | None, int | None]]
    ) -> int:
        updated = 0
        for path, module, week in labels:
            doc = self.documents.get(str(path))
            if doc is None:
                continue
            self.documents[str(path)] = replace(
                doc,
                module=None if module is None else str(module),
                week=None if week is None else int(week),
            )
            updated += 1
        return updated

    def find_vector_by_text_sha(self, sha: str) -> object | None:
        return self.vectors.get(sha)

    def replace_chunks(self, doc_id: int, chunks: list[StoredChunk]) -> None:
        for chunk_id in [
            cid for cid, (owner, _chunk) in self.chunks.items() if owner == doc_id
        ]:
            del self.chunks[chunk_id]
        for chunk in chunks:
            self.vectors[chunk.text_sha] = chunk.vector
            self.chunks[self._next_chunk_id] = (doc_id, chunk)
            self._next_chunk_id += 1

    def delete_document(self, path: str) -> None:
        self.documents.pop(str(path), None)

    def move_document(self, src: str, dst: str, *, mtime_ns: int, size: int) -> int:
        source = str(src)
        destination = str(dst)
        doc = self.documents.get(source)
        if doc is None:
            raise KeyError(source)
        if destination != source and destination in self.documents:
            raise ValueError(destination)
        moved = replace(doc, path=destination, mtime_ns=int(mtime_ns), size=int(size))
        self.documents.pop(source, None)
        self.documents[destination] = moved
        return moved.id

    def list_paths(self, prefix: str | None = None) -> list[str]:
        paths = sorted(self.documents)
        if prefix is None:
            return paths
        # Component-aware, with the host's separator and case rules, like Store.
        return [path for path in paths if pathnorm.has_prefix(path, str(prefix))]

    def set_last_error_count(self, n: int) -> None:
        return None

    def get_meta_value(self, key: str) -> str | None:
        return self.meta.get(str(key))

    def set_meta_value(self, key: str, value: str) -> None:
        self.meta[str(key)] = str(value)


class _CountingEmbedder:
    """Stdlib-only embedder; counts document embedding batches."""

    model_name = "counting"
    dim = 4

    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[tuple[float, ...]]:
        self.document_calls += 1
        return [
            tuple(float(len(token) % 5) for token in text.split()) or (1.0,)
            for text in texts
        ]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0,)


class _FakeObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "notes"
        self.root.mkdir()
        self.outside = self.tmp / "outside"
        self.outside.mkdir()
        self.indexer = _RecordingIndexer()
        self.observer = _FakeObserver()
        self.watcher = Watcher(
            self.indexer,
            [self.root],
            debounce_s=0.15,
            observer_factory=lambda: self.observer,
        )
        self.addCleanup(self.watcher.stop)

    def _dispatch(
        self,
        event_type: str,
        src: Path,
        dst: Path | None = None,
        *,
        is_directory: bool = False,
    ) -> None:
        assert self.watcher._handler is not None
        self.watcher._handler.dispatch(
            _FakeEvent(
                event_type,
                str(src),
                str(dst) if dst is not None else "",
                is_directory,
            )
        )

    def test_start_schedules_roots_and_starts_observer(self) -> None:
        self.watcher.start()

        self.assertTrue(self.observer.started)
        self.assertEqual(self.observer.schedules[0][1], str(self.root.resolve()))
        self.assertTrue(self.observer.schedules[0][2])

    def test_new_file_is_indexed_after_debounce(self) -> None:
        path = self.root / "new.md"
        path.write_text("new material", encoding="utf-8")
        self.watcher.start()

        self._dispatch("created", path)

        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        self.assertEqual(self.indexer.calls, [("index", path.resolve())])

    def test_rapid_writes_are_debounced(self) -> None:
        path = self.root / "busy.md"
        path.write_text("busy", encoding="utf-8")
        self.watcher.start()

        for _ in range(10):
            self._dispatch("modified", path)
            time.sleep(0.03)

        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        time.sleep(0.35)
        index_calls = [call for call in self.indexer.calls if call[0] == "index"]
        self.assertGreaterEqual(len(index_calls), 1)
        self.assertLess(len(index_calls), 10)

    def test_deleted_file_calls_remove_path(self) -> None:
        path = self.root / "gone.md"
        self.watcher.start()

        self._dispatch("deleted", path)

        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        self.assertEqual(self.indexer.calls, [("remove", path.resolve())])

    def test_rename_routes_through_move_path_not_delete_then_index(self) -> None:
        src = self.root / "old.md"
        dst = self.root / "new.md"
        self.watcher.start()

        self._dispatch("moved", src, dst)

        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        self.assertEqual(self.indexer.calls, [("move", src.resolve(), dst.resolve())])

    def test_moves_to_the_same_destination_keep_each_source_pending(self) -> None:
        first = self.root / "first.md"
        second = self.root / "second.md"
        dst = self.root / "shared.md"
        self.watcher.start()

        self._dispatch("moved", first, dst)
        self._dispatch("moved", second, dst)

        self.assertTrue(
            _wait_for(
                lambda: len(
                    [call for call in self.indexer.calls if call[0] == "move"]
                )
                == 2
            )
        )
        moves = [call for call in self.indexer.calls if call[0] == "move"]
        self.assertEqual(
            moves,
            [
                ("move", first.resolve(), dst.resolve()),
                ("move", second.resolve(), dst.resolve()),
            ],
        )

    def test_chained_moves_merge_into_one_rename(self) -> None:
        first = self.root / "a.md"
        second = self.root / "b.md"
        third = self.root / "c.md"
        self.watcher.start()

        self._dispatch("moved", first, second)
        self._dispatch("moved", second, third)

        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        time.sleep(0.3)
        moves = [call for call in self.indexer.calls if call[0] == "move"]
        self.assertEqual(
            moves,
            [("move", first.resolve(), third.resolve())],
        )

    def test_paths_outside_roots_are_ignored(self) -> None:
        outside_file = self.outside / "secret.md"
        outside_file.write_text("outside", encoding="utf-8")
        self.watcher.start()

        self._dispatch("created", outside_file)

        time.sleep(0.3)
        self.assertEqual(self.indexer.calls, [])

    def test_symlink_escape_is_ignored(self) -> None:
        outside_file = self.outside / "secret.md"
        outside_file.write_text("outside", encoding="utf-8")
        link = self.root / "escape.md"
        link.symlink_to(outside_file)
        self.watcher.start()

        self._dispatch("created", link)

        time.sleep(0.3)
        self.assertEqual(self.indexer.calls, [])

    def test_directory_created_inside_root_is_scanned(self) -> None:
        directory = self.root / "newdir"
        directory.mkdir()
        self.watcher.start()

        self._dispatch("created", directory, is_directory=True)

        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        self.assertEqual(self.indexer.calls, [("sync", (directory.resolve(),))])

    def test_directory_move_maps_children_through_move_path(self) -> None:
        src = self.root / "before"
        dst = self.root / "after"
        dst.mkdir()
        (dst / "note.md").write_text("note", encoding="utf-8")
        self.watcher.start()

        self._dispatch("moved", src, dst, is_directory=True)

        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        moves = [call for call in self.indexer.calls if call[0] == "move"]
        self.assertEqual(
            moves,
            [("move", (src / "note.md").resolve(), (dst / "note.md").resolve())],
        )

    def test_paused_start_buffers_events_until_resume(self) -> None:
        path = self.root / "buffered.md"
        path.write_text("buffered", encoding="utf-8")
        self.watcher.start(paused=True)

        self._dispatch("created", path)
        time.sleep(0.3)
        self.assertEqual(self.indexer.calls, [])

        self.watcher.resume()
        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        self.assertEqual(self.indexer.calls, [("index", path.resolve())])

    def test_pause_buffers_events_until_resume(self) -> None:
        self.watcher.start()
        self.watcher.pause()
        path = self.root / "paused.md"
        path.write_text("paused", encoding="utf-8")

        self._dispatch("created", path)
        time.sleep(0.3)
        self.assertEqual(self.indexer.calls, [])

        self.watcher.resume()
        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        self.assertEqual(self.indexer.calls, [("index", path.resolve())])

    def test_stop_is_idempotent_and_discards_pending_events(self) -> None:
        watcher = Watcher(
            self.indexer,
            [self.root],
            debounce_s=10.0,
            observer_factory=lambda: _FakeObserver(),
        )
        watcher.start()
        path = self.root / "pending.md"
        path.write_text("pending", encoding="utf-8")
        watcher._handler.dispatch(_FakeEvent("created", str(path)))

        watcher.stop()
        watcher.stop()

        time.sleep(0.05)
        self.assertEqual(self.indexer.calls, [])

    def test_start_twice_raises(self) -> None:
        self.watcher.start()

        with self.assertRaises(WatcherError):
            self.watcher.start()


class _RootChangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.first = self.tmp / "first"
        self.first.mkdir()
        self.second = self.tmp / "second"
        self.second.mkdir()
        self.indexer = _RecordingIndexer()
        self.observers: list[_FakeObserver] = []

    def _factory(self) -> _FakeObserver:
        observer = _FakeObserver()
        self.observers.append(observer)
        return observer

    def _watcher(self, **kwargs: object) -> Watcher:
        watcher = Watcher(
            self.indexer,
            [self.first],
            debounce_s=0.05,
            observer_factory=self._factory,
            mounts=(),
            **kwargs,
        )
        self.addCleanup(watcher.stop)
        return watcher

    def _dispatch(self, watcher: Watcher, event_type: str, src: Path) -> None:
        assert watcher._handler is not None
        watcher._handler.dispatch(_FakeEvent(event_type, str(src)))

    def test_run_exclusive_wraps_every_indexing_action(self) -> None:
        inside = threading.local()
        seen: list[bool] = []
        wrapped: list[int] = []

        def run_exclusive(action) -> None:
            inside.active = True
            try:
                action()
            finally:
                inside.active = False
            wrapped.append(1)

        original = self.indexer.index_path

        def index_path(path: Path) -> str:
            seen.append(getattr(inside, "active", False))
            return original(path)

        self.indexer.index_path = index_path  # type: ignore[method-assign]
        watcher = self._watcher(run_exclusive=run_exclusive)
        watcher.start()
        path = self.first / "a.md"
        path.write_text("alpha", encoding="utf-8")

        self._dispatch(watcher, "created", path)

        self.assertTrue(_wait_for(lambda: bool(wrapped)))
        self.assertEqual(seen, [True])

    def test_set_roots_reschedules_running_watcher(self) -> None:
        watcher = self._watcher()
        watcher.start()
        self.assertEqual(len(self.observers), 1)

        watcher.set_roots([self.second])

        self.assertTrue(self.observers[0].stopped)
        newest = self.observers[-1]
        self.assertTrue(newest.started)
        self.assertEqual(
            [entry[1] for entry in newest.schedules], [str(self.second.resolve())]
        )
        moved_in = self.second / "b.md"
        moved_in.write_text("beta", encoding="utf-8")
        self._dispatch(watcher, "created", self.first / "ignored.md")
        self._dispatch(watcher, "created", moved_in)
        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        time.sleep(0.1)
        self.assertEqual(self.indexer.calls, [("index", moved_in.resolve())])

    def test_set_roots_drops_pending_events_outside_new_roots(self) -> None:
        watcher = self._watcher()
        watcher.start(paused=True)
        self._dispatch(watcher, "created", self.first / "pending.md")

        watcher.set_roots([self.second])
        watcher.resume()

        time.sleep(0.2)
        self.assertEqual(self.indexer.calls, [])

    def test_set_roots_before_start_only_changes_roots(self) -> None:
        watcher = self._watcher()
        watcher.set_roots([self.second])
        self.assertEqual(self.observers, [])

        watcher.start()

        self.assertEqual(
            [entry[1] for entry in self.observers[0].schedules],
            [str(self.second.resolve())],
        )


class _RealIndexerRenameTests(unittest.TestCase):
    """Rename through the real Indexer with no optional model packages."""

    def test_live_rename_makes_zero_embedding_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "notes"
            root.mkdir()
            src = root / "old.md"
            src.write_text("zebra unique rename material", encoding="utf-8")
            store = _MemoryStore()
            embedder = _CountingEmbedder()
            cfg = Config(roots=(root,), db_path=base / "index.db")
            indexer = Indexer(store, embedder, cfg)
            self.assertEqual(indexer.index_path(src), "added")
            self.assertEqual(embedder.document_calls, 1)

            observer = _FakeObserver()
            watcher = Watcher(
                indexer,
                [root],
                debounce_s=0.1,
                observer_factory=lambda: observer,
            )
            watcher.start()
            self.addCleanup(watcher.stop)
            dst = root / "new.md"
            os.rename(src, dst)
            assert watcher._handler is not None
            watcher._handler.dispatch(_FakeEvent("moved", str(src), str(dst)))

            self.assertTrue(
                _wait_for(lambda: store.list_paths() == [str(dst.resolve())])
            )
            self.assertEqual(embedder.document_calls, 1)

    def test_chained_live_rename_merges_and_preserves_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "notes"
            root.mkdir()
            first = root / "a.md"
            second = root / "b.md"
            third = root / "c.md"
            first.write_text("zebra unique chained rename material", encoding="utf-8")
            store = _MemoryStore()
            embedder = _CountingEmbedder()
            cfg = Config(roots=(root,), db_path=base / "index.db")
            indexer = Indexer(store, embedder, cfg)
            self.assertEqual(indexer.index_path(first), "added")
            original = store.get_document(str(first.resolve()))
            assert original is not None
            vectors_before = dict(store.vectors)
            chunk_ids_before = set(store.chunks)
            self.assertEqual(embedder.document_calls, 1)

            os.rename(first, second)
            os.rename(second, third)
            observer = _FakeObserver()
            watcher = Watcher(
                indexer,
                [root],
                debounce_s=0.1,
                observer_factory=lambda: observer,
            )
            watcher.start()
            self.addCleanup(watcher.stop)
            assert watcher._handler is not None
            watcher._handler.dispatch(
                _FakeEvent("moved", str(first), str(second))
            )
            watcher._handler.dispatch(
                _FakeEvent("moved", str(second), str(third))
            )

            self.assertTrue(
                _wait_for(
                    lambda: str(third.resolve()) in store.list_paths()
                )
            )
            time.sleep(0.25)
            self.assertEqual(store.list_paths(), [str(third.resolve())])
            moved = store.get_document(str(third.resolve()))
            assert moved is not None
            self.assertEqual(moved.id, original.id)
            self.assertEqual(store.vectors, vectors_before)
            self.assertEqual(set(store.chunks), chunk_ids_before)
            self.assertEqual(embedder.document_calls, 1)


def _network_mounts(test: unittest.TestCase, root: Path, fstype: str) -> list[Mount]:
    """Return mounts that put *root* on a network filesystem of *fstype*.

    Windows has no mount table (seekr asks for the drive type there), so on
    Windows *root* is reported as a network drive instead.
    """
    if sys.platform != "win32":
        return [Mount(source="C:\\", mount_point=str(root), fstype=fstype)]
    slow = str(root.resolve())
    patcher = mock.patch(
        "seekr.platform_info._is_slow_windows_path",
        lambda path: pathnorm.has_prefix(str(path), slow),
    )
    patcher.start()
    test.addCleanup(patcher.stop)
    return []


@unittest.skipUnless(_HAS_WATCHDOG, "watchdog is not installed")
class _RealWatchdogIntegrationTests(unittest.TestCase):
    """Real recursive observer, real Indexer, stdlib store and embedder."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "notes"
        self.root.mkdir()
        self.store = _MemoryStore()
        self.embedder = _CountingEmbedder()
        self.cfg = Config(roots=(self.root,), db_path=self.base / "index.db")
        self.indexer = Indexer(self.store, self.embedder, self.cfg)
        self.watcher: Watcher | None = None

    def tearDown(self) -> None:
        if self.watcher is not None:
            self.watcher.stop()

    def test_new_file_and_delete_are_reflected_in_the_index(self) -> None:
        self.watcher = Watcher(self.indexer, [self.root], debounce_s=0.2)
        self.watcher.start()
        time.sleep(0.2)
        path = self.root / "live.md"
        path.write_text("live addition", encoding="utf-8")

        self.assertTrue(
            _wait_for(
                lambda: self.store.get_document(str(path.resolve())) is not None
            )
        )

        path.unlink()
        self.assertTrue(
            _wait_for(lambda: self.store.get_document(str(path.resolve())) is None)
        )

    def test_live_rename_uses_move_path_with_zero_further_embeddings(self) -> None:
        src = self.root / "rename-me.md"
        src.write_text("alpha beta gamma", encoding="utf-8")
        self.assertEqual(self.indexer.index_path(src), "added")
        calls_after_index = self.embedder.document_calls
        self.watcher = Watcher(self.indexer, [self.root], debounce_s=0.2)
        self.watcher.start()
        time.sleep(0.2)

        dst = self.root / "renamed.md"
        os.rename(src, dst)

        self.assertTrue(
            _wait_for(lambda: self.store.list_paths() == [str(dst.resolve())])
        )
        self.assertEqual(self.embedder.document_calls, calls_after_index)

    def test_polling_observer_detects_create_and_delete(self) -> None:
        mounts = _network_mounts(self, self.root, "9p")
        self.watcher = Watcher(
            self.indexer,
            [self.root],
            debounce_s=0.1,
            poll_interval=0.2,
            mounts=mounts,
        )
        self.assertEqual(self.watcher.polled_roots, (self.root.resolve(),))
        self.watcher.start()
        time.sleep(0.25)

        path = self.root / "polled.md"
        path.write_text("polled content", encoding="utf-8")
        self.assertTrue(
            _wait_for(
                lambda: self.store.get_document(str(path.resolve())) is not None
            )
        )

        path.unlink()
        self.assertTrue(
            _wait_for(lambda: self.store.get_document(str(path.resolve())) is None)
        )


class _PollingObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.fast_root = self.tmp / "fast"
        self.fast_root.mkdir()
        self.slow_root = self.tmp / "slow"
        self.slow_root.mkdir()
        self.indexer = _RecordingIndexer()

    def test_slow_roots_use_polling_observer(self) -> None:
        native_obs = _FakeObserver()
        poll_obs = _FakeObserver()
        poll_intervals: list[float] = []
        mounts = [
            Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
            *_network_mounts(self, self.slow_root, "9p"),
        ]

        def poll_factory(interval: float) -> _FakeObserver:
            poll_intervals.append(interval)
            return poll_obs

        watcher = Watcher(
            self.indexer,
            [self.fast_root, self.slow_root],
            debounce_s=0.05,
            observer_factory=lambda: native_obs,
            poll_interval=20,
            mounts=mounts,
            polling_observer_factory=poll_factory,
        )
        self.addCleanup(watcher.stop)

        self.assertEqual(watcher.polled_roots, (self.slow_root.resolve(),))
        watcher.start()

        self.assertEqual(poll_intervals, [20.0])
        self.assertTrue(native_obs.started)
        self.assertTrue(poll_obs.started)
        self.assertEqual(
            [item[1] for item in native_obs.schedules],
            [str(self.fast_root.resolve())],
        )
        self.assertEqual(
            [item[1] for item in poll_obs.schedules],
            [str(self.slow_root.resolve())],
        )

        slow_file = self.slow_root / "note.md"
        slow_file.write_text("polled note", encoding="utf-8")
        assert watcher._handler is not None
        watcher._handler.dispatch(_FakeEvent("created", str(slow_file)))
        self.assertTrue(_wait_for(lambda: bool(self.indexer.calls)))
        self.assertEqual(self.indexer.calls, [("index", slow_file.resolve())])

        watcher.stop()
        self.assertTrue(native_obs.stopped)
        self.assertTrue(poll_obs.stopped)

    def test_no_slow_roots_creates_no_polling_observer(self) -> None:
        native_obs = _FakeObserver()
        polling_calls: list[float] = []
        mounts = [Mount(source="/dev/sdc", mount_point="/", fstype="ext4")]

        def poll_factory(interval: float) -> _FakeObserver:
            polling_calls.append(interval)
            return _FakeObserver()

        watcher = Watcher(
            self.indexer,
            [self.fast_root],
            observer_factory=lambda: native_obs,
            mounts=mounts,
            polling_observer_factory=poll_factory,
        )
        self.addCleanup(watcher.stop)

        self.assertEqual(watcher.polled_roots, ())
        watcher.start()
        self.assertEqual(polling_calls, [])
        self.assertIsNone(watcher._polling_observer)
        self.assertTrue(native_obs.started)
        watcher.stop()
        self.assertTrue(native_obs.stopped)

    def test_only_slow_roots_skips_native_observer(self) -> None:
        native_calls = 0
        poll_obs = _FakeObserver()
        mounts = [
            Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
            *_network_mounts(self, self.slow_root, "drvfs"),
        ]

        def native_factory() -> _FakeObserver:
            nonlocal native_calls
            native_calls += 1
            return _FakeObserver()

        watcher = Watcher(
            self.indexer,
            [self.slow_root],
            observer_factory=native_factory,
            mounts=mounts,
            polling_observer_factory=lambda _interval: poll_obs,
        )
        self.addCleanup(watcher.stop)

        self.assertEqual(watcher.polled_roots, (self.slow_root.resolve(),))
        watcher.start()
        self.assertEqual(native_calls, 0)
        self.assertIsNone(watcher._observer)
        self.assertTrue(poll_obs.started)
        self.assertEqual(
            [item[1] for item in poll_obs.schedules],
            [str(self.slow_root.resolve())],
        )
        watcher.stop()
        watcher.stop()
        self.assertTrue(poll_obs.stopped)

    def test_invalid_poll_interval_raises(self) -> None:
        for bad in (0, -1, float("nan"), float("inf"), "15", True):
            with self.subTest(poll_interval=bad), self.assertRaises(WatcherError):
                Watcher(
                    self.indexer,
                    [self.fast_root],
                    poll_interval=bad,  # type: ignore[arg-type]
                    mounts=[],
                )

    def test_polling_observer_start_failure_stops_native_observer(self) -> None:
        native_obs = _FakeObserver()
        poll_obs = _FakeObserver()

        def fail_start() -> None:
            raise RuntimeError("poll boom")

        poll_obs.start = fail_start  # type: ignore[method-assign]
        mounts = [
            Mount(source="/dev/sdc", mount_point="/", fstype="ext4"),
            *_network_mounts(self, self.slow_root, "cifs"),
        ]

        watcher = Watcher(
            self.indexer,
            [self.fast_root, self.slow_root],
            observer_factory=lambda: native_obs,
            mounts=mounts,
            polling_observer_factory=lambda _interval: poll_obs,
        )

        with self.assertRaises(WatcherError):
            watcher.start()

        self.assertTrue(native_obs.stopped)
        self.assertIsNone(watcher._observer)
        self.assertIsNone(watcher._polling_observer)


@unittest.skipUnless(
    _HAS_WATCHDOG and _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_FTS5,
    "watchdog, numpy, sqlite-vec and FTS5 are required",
)
class _RealStoreWatchdogIntegrationTests(unittest.TestCase):
    """Actual watchdog observer, production Store and FakeEmbedder."""

    def setUp(self) -> None:
        from seekr.store import Store

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "notes"
        self.root.mkdir()
        self.db_path = self.base / "index.db"
        self.cfg = Config(roots=(self.root,), db_path=self.db_path, embed_model="fake")
        self.embedder = FakeEmbedder(dim=8)
        self.store = Store(self.db_path, "fake", 8)
        self.addCleanup(self.store.close)
        self.indexer = Indexer(self.store, self.embedder, self.cfg)
        self.searcher = Searcher(self.store, self.embedder)
        self.watcher: Watcher | None = None

    def tearDown(self) -> None:
        if self.watcher is not None:
            self.watcher.stop()

    def _search_paths(self, query: str) -> list[str]:
        return [hit.path for hit in self.searcher.search(query, mode="keyword", k=5)]

    def test_new_file_becomes_searchable_then_rename_costs_no_embeddings(self) -> None:
        self.watcher = Watcher(self.indexer, [self.root], debounce_s=0.2)
        self.watcher.start()
        time.sleep(0.2)
        path = self.root / "live.md"
        path.write_text("zebra unique searchable", encoding="utf-8")

        self.assertTrue(
            _wait_for(
                lambda: str(path.resolve()) in self._search_paths("zebra")
            )
        )
        calls_after_new_file = len(self.embedder.calls)
        dst = self.root / "renamed.md"
        os.rename(path, dst)

        self.assertTrue(
            _wait_for(
                lambda: str(dst.resolve()) in self._search_paths("zebra")
                and str(path.resolve()) not in self._search_paths("zebra")
            )
        )
        self.assertEqual(len(self.embedder.calls), calls_after_new_file)

        dst.unlink()
        self.assertTrue(
            _wait_for(
                lambda: str(dst.resolve()) not in self._search_paths("zebra")
            )
        )


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    os.name == "posix",
    "macOS and Linux path rules use / boundaries, which never apply to Windows paths",
)
class CanonicalContainmentTests(unittest.TestCase):
    NFC = "M\u00f3dulo"
    NFD = "Mo\u0301dulo"

    def _watcher(self, root: Path) -> Watcher:
        return Watcher(_RecordingIndexer(), [root], 0.0, mounts=())

    def test_darwin_event_paths_match_roots_across_case_and_unicode_form(self) -> None:
        with mock.patch("seekr.pathnorm._PLATFORM", "darwin"):
            watcher = self._watcher(Path("/vol") / self.NFD.upper())
            self.assertEqual(watcher._resolved_roots, (Path("/vol") / "M\u00d3DULO",))
            inside = watcher._inside_roots(f"/vol/{self.NFC.lower()}/a.md")
            self.assertEqual(inside, Path(f"/vol/{self.NFC.lower()}/a.md"))
            nfd_event = watcher._inside_roots(f"/vol/{self.NFD}/b.md")
            self.assertEqual(nfd_event, Path(f"/vol/{self.NFC}/b.md"))
            self.assertIsNone(watcher._inside_roots(f"/vol/{self.NFC}2/a.md"))

    def test_linux_containment_is_exact(self) -> None:
        with mock.patch("seekr.pathnorm._PLATFORM", "linux"):
            watcher = self._watcher(Path("/vol") / self.NFD)
            self.assertEqual(watcher._resolved_roots, (Path("/vol") / self.NFD,))
            self.assertIsNone(watcher._inside_roots(f"/vol/{self.NFC}/a.md"))
            self.assertIsNone(watcher._inside_roots(f"/vol/{self.NFD.lower()}/a.md"))
            self.assertEqual(
                watcher._inside_roots(f"/vol/{self.NFD}/a.md"), Path(f"/vol/{self.NFD}/a.md")
            )
            whole = Watcher(_RecordingIndexer(), [Path("/")], 0.0, mounts=())
            self.assertEqual(whole._inside_roots("/x/y.md"), Path("/x/y.md"))
