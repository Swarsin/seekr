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
import tempfile
import time
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer
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
        )
        return doc_id

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
        value = str(prefix).rstrip("/")
        return [
            path
            for path in paths
            if path == value or path.startswith(value + "/")
        ]

    def set_last_error_count(self, n: int) -> None:
        return None


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
