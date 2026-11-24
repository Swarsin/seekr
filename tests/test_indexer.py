"""Tests for incremental indexing, failure isolation and move handling.

The indexer is written against the ``Store`` protocol, so most tests use a
small deterministic in-memory store and embedder made only from the standard
library.  A separate ``RealStoreIndexerTests`` class exercises the production
SQLite store and is skipped when numpy/sqlite-vec are unavailable.
"""

from __future__ import annotations

import copy
import errno
import importlib.util
import os
import sqlite3
import tempfile
import unittest
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

from seekr import indexer as indexer_module
from seekr.config import Config
from seekr.course import CourseLabel
from seekr.extract import ExtractionError
from seekr.indexer import Indexer
from seekr.models import text_sha
from seekr.store import DocRow

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_HAS_FTS5 = _has_fts5()


@dataclass
class _FakeChunk:
    id: int
    doc_id: int
    text: str
    text_sha: str
    locator: str
    loc_start: float
    vector: object


class FakeStore:
    """In-memory Store implementation using only the standard library."""

    def __init__(self) -> None:
        self.documents: dict[str, DocRow] = {}
        self.chunks: dict[int, _FakeChunk] = {}
        self.meta: dict[str, str] = {}
        self._next_doc_id = 1
        self._next_chunk_id = 1
        self.move_calls: list[tuple[str, str, int, int]] = []
        self.delete_calls: list[str] = []
        self.last_error_count = 0
        self.fail_replace = False
        self.fail_upsert = False
        self._snapshots: list[tuple[object, ...]] = []

    # -- test helpers ------------------------------------------------------
    def _snapshot(self) -> tuple[object, ...]:
        return (
            copy.deepcopy(self.documents),
            copy.deepcopy(self.chunks),
            copy.deepcopy(self.meta),
            self._next_doc_id,
            self._next_chunk_id,
        )

    def _restore(self, snapshot: tuple[object, ...]) -> None:
        documents, chunks, meta, next_doc_id, next_chunk_id = snapshot
        self.documents = documents  # type: ignore[assignment]
        self.chunks = chunks  # type: ignore[assignment]
        self.meta = meta  # type: ignore[assignment]
        self._next_doc_id = next_doc_id  # type: ignore[assignment]
        self._next_chunk_id = next_chunk_id  # type: ignore[assignment]

    @contextmanager
    def transaction(self, *, write: bool = False):
        snapshot = self._snapshot() if write else None
        if write:
            self._snapshots.append(snapshot)  # type: ignore[arg-type]
        try:
            yield
        except BaseException:
            if write:
                self._snapshots.pop()
                self._restore(snapshot)  # type: ignore[arg-type]
            raise
        else:
            if write:
                self._snapshots.pop()

    # -- Store protocol ----------------------------------------------------
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
        if self.fail_upsert:
            raise RuntimeError("injected document write failure")
        existing = self.documents.get(path)
        if existing is None:
            doc_id = self._next_doc_id
            self._next_doc_id += 1
        else:
            doc_id = existing.id
        self.documents[path] = DocRow(
            id=doc_id,
            path=path,
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

    def get_document(self, path: str) -> DocRow | None:
        return self.documents.get(str(path))

    def replace_chunks(self, doc_id: int, chunks: list[object]) -> None:
        for chunk_id, chunk in list(self.chunks.items()):
            if chunk.doc_id == doc_id:
                del self.chunks[chunk_id]
        for item in chunks:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            self.chunks[chunk_id] = _FakeChunk(
                id=chunk_id,
                doc_id=doc_id,
                text=item.spec.text,
                text_sha=item.text_sha,
                locator=item.spec.locator,
                loc_start=float(item.spec.loc_start),
                vector=item.vector,
            )
        if self.fail_replace:
            raise RuntimeError("injected chunk write failure")

    def delete_document(self, path: str) -> None:
        self.delete_calls.append(str(path))
        doc = self.documents.pop(str(path), None)
        if doc is None:
            return
        for chunk_id, chunk in list(self.chunks.items()):
            if chunk.doc_id == doc.id:
                del self.chunks[chunk_id]

    def move_document(self, src: str, dst: str, *, mtime_ns: int, size: int) -> int:
        self.move_calls.append((str(src), str(dst), int(mtime_ns), int(size)))
        doc = self.documents.get(str(src))
        if doc is None:
            raise KeyError(f"document not found: {src}")
        if src != dst and str(dst) in self.documents:
            raise ValueError(f"destination is already indexed: {dst}")
        del self.documents[str(src)]
        self.documents[str(dst)] = DocRow(
            id=doc.id,
            path=str(dst),
            mtime_ns=int(mtime_ns),
            size=int(size),
            sha256=doc.sha256,
            kind=doc.kind,
            module=doc.module,
            week=doc.week,
        )
        return doc.id

    def list_paths(self, prefix: str | None = None) -> list[str]:
        paths = sorted(self.documents)
        if prefix is None:
            return paths
        base = str(prefix).rstrip("/")
        if not base:
            return paths
        return [path for path in paths if path == base or path.startswith(base + "/")]

    def find_vector_by_text_sha(self, sha: str) -> object | None:
        for chunk in self.chunks.values():
            if chunk.text_sha == sha:
                return chunk.vector
        return None

    def stats(self) -> dict[str, int]:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}

    def set_last_error_count(self, n: int) -> None:
        self.last_error_count = int(n)

    def get_last_error_count(self) -> int:
        return self.last_error_count

    def get_meta_value(self, key: str) -> str | None:
        return self.meta.get(str(key))

    def set_meta_value(self, key: str, value: str) -> None:
        self.meta[str(key)] = str(value)


class RecordingEmbedder:
    """Deterministic embedder that records every batch."""

    model_name = "fake"
    dim = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    def embed_documents(self, texts: list[str]) -> list[str]:
        batch = list(texts)
        self.calls.append(batch)
        if self.fail:
            raise RuntimeError("injected embedding failure")
        return [f"vec:{text_sha(text)}" for text in batch]

    def embed_query(self, text: str) -> str:
        return f"vec:{text_sha(text)}"


class ByteEmbedder(RecordingEmbedder):
    """An embedder whose vectors can be stored by the stdlib Store harness."""

    def embed_documents(self, texts: list[str]) -> list[bytes]:
        batch = list(texts)
        self.calls.append(batch)
        if self.fail:
            raise RuntimeError("injected embedding failure")
        return [text_sha(text).encode("ascii")[: self.dim] for text in batch]


@contextmanager
def _stdlib_vector_backend():
    """Let the production Store run on plain SQLite when sqlite-vec is absent."""

    def create_table(conn: sqlite3.Connection, dim: int) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks_vec ("
            "rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
        )

    def serialize(vector: object) -> bytes:
        return bytes(vector)  # type: ignore[arg-type]

    def deserialize(payload: bytes | memoryview) -> bytes:
        return bytes(payload)

    with (
        mock.patch("seekr.store._load_vector_extension", lambda conn: None),
        mock.patch("seekr.store._create_vector_table", create_table),
        mock.patch("seekr.store._serialize_vector", serialize),
        mock.patch("seekr.store._deserialize_vector", deserialize),
    ):
        yield


def _paragraph(prefix: str) -> str:
    return " ".join([prefix, *(f"word{number}" for number in range(9))])


def _short_paragraph(prefix: str) -> str:
    return f"{prefix} short unique words"


class _IndexerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.store = FakeStore()
        self.embedder = RecordingEmbedder()
        self.cfg = self._config()
        self.indexer = Indexer(self.store, self.embedder, self.cfg)

    def _config(self, **overrides: object) -> Config:
        values: dict[str, object] = {
            "roots": (self.root,),
            "db_path": self.tmp / "index.db",
            "chunk_target_words": 10,
            "chunk_max_words": 10,
            "chunk_overlap_words": 0,
        }
        values.update(overrides)
        return Config(**values)  # type: ignore[arg-type]

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def stored(self, path: Path) -> DocRow | None:
        return self.store.get_document(str(path.resolve()))


class SyncTests(_IndexerTestCase):
    def test_sync_adds_files(self) -> None:
        self.write("a.txt", "alpha beta gamma")
        self.write("b.txt", "delta epsilon")
        self.write("c.txt", "zeta eta theta")

        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.updated, report.unchanged), (3, 0, 0))
        self.assertEqual(report.errors, [])
        self.assertEqual(self.store.stats(), {"documents": 3, "chunks": 3})
        self.assertEqual(sum(len(call) for call in self.embedder.calls), 3)
        self.assertEqual(len(self.embedder.calls), 1)

    def test_sync_batch_deduplicates_identical_chunks_across_files(self) -> None:
        self.write("a.txt", "shared chunk text across files")
        self.write("b.txt", "shared chunk text across files")
        self.write("c.txt", "distinct chunk text")

        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.errors), (3, []))
        self.assertEqual(self.store.stats(), {"documents": 3, "chunks": 3})
        self.assertEqual(len(self.embedder.calls), 1)
        self.assertEqual(len(self.embedder.calls[0]), 2)

    def test_resync_unchanged_makes_no_embed_calls(self) -> None:
        self.write("a.txt", "alpha beta gamma")
        self.write("b.txt", "delta epsilon")
        self.write("c.txt", "zeta eta theta")
        self.indexer.sync([self.root])
        calls = len(self.embedder.calls)

        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.updated, report.unchanged), (0, 0, 3))
        self.assertEqual(len(self.embedder.calls), calls)

    def test_touch_only_is_unchanged_and_not_reembedded(self) -> None:
        path = self.write("a.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        calls = len(self.embedder.calls)
        new_mtime = path.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(new_mtime, new_mtime))

        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.updated, report.unchanged), (0, 0, 1))
        self.assertEqual(len(self.embedder.calls), calls)
        doc = self.stored(path)
        assert doc is not None
        self.assertEqual(doc.mtime_ns, new_mtime)

    def test_edit_one_paragraph_embeds_only_changed_chunks(self) -> None:
        first = _short_paragraph("alpha")
        second = _short_paragraph("bravo")
        third = _short_paragraph("charlie")
        path = self.write(
            "notes.md",
            "\n".join(
                [
                    f"# Section A\n{first}",
                    f"# Section B\n{second}",
                    f"# Section C\n{third}",
                ]
            ),
        )
        self.indexer.sync([self.root])
        self.assertEqual(len(self.embedder.calls), 1)
        self.assertEqual(len(self.embedder.calls[0]), 3)

        replacement = _short_paragraph("delta")
        path.write_text(
            "\n".join(
                [
                    f"# Section A\n{first}",
                    f"# Section B\n{replacement}",
                    f"# Section C\n{third}",
                ]
            ),
            encoding="utf-8",
        )
        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.updated, report.unchanged), (0, 1, 0))
        self.assertEqual(len(self.embedder.calls), 2)
        self.assertEqual(len(self.embedder.calls[1]), 1)
        self.assertIn("delta", self.embedder.calls[1][0])

    def test_rename_costs_zero_embeddings(self) -> None:
        first = _paragraph("alpha")
        second = _paragraph("bravo")
        path = self.write("old.txt", f"{first}\n{second}")
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        chunk_count = self.store.stats()["chunks"]

        renamed = self.root / "new.txt"
        path.rename(renamed)
        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.removed), (1, 1))
        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNone(self.stored(path))
        self.assertIsNotNone(self.stored(renamed))
        self.assertEqual(self.store.stats(), {"documents": 1, "chunks": chunk_count})

    def test_deleted_file_is_removed(self) -> None:
        path = self.write("a.txt", "alpha beta gamma")
        self.indexer.sync([self.root])

        path.unlink()
        report = self.indexer.sync([self.root])

        self.assertEqual((report.added, report.removed), (0, 1))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_ignore_globs_and_oversize_skipped(self) -> None:
        self.write("node_modules/dependency.md", "ignored dependency")
        self.write("large.txt", "x" * 100)
        cfg = self._config(max_file_bytes=32)
        indexer = Indexer(self.store, self.embedder, cfg)

        report = indexer.sync([self.root])

        self.assertEqual(report.skipped, 2)
        self.assertEqual(report.added, 0)
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})
        self.assertEqual(self.embedder.calls, [])

    def test_corrupt_file_recorded_and_sync_continues(self) -> None:
        self.write("bad.pdf", "%PDF-1.4 garbage")
        self.write("good.txt", "alpha beta gamma")
        real_extract = __import__("seekr.indexer", fromlist=["extract"]).extract

        def fake_extract(path: Path, kind: str):
            if path.suffix == ".pdf":
                raise ExtractionError("corrupt pdf")
            return real_extract(path, kind)

        with mock.patch("seekr.indexer.extract", side_effect=fake_extract):
            report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 1)
        self.assertEqual(len(report.errors), 1)
        self.assertIn("corrupt pdf", report.errors[0][1])
        self.assertEqual(self.store.stats(), {"documents": 1, "chunks": 1})

    def test_file_vanishes_between_scan_and_read_is_removed(self) -> None:
        path = self.write("vanish.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        path.write_text("changed content so extraction is attempted", encoding="utf-8")

        def vanishing_extract(source: Path, kind: str):
            source.unlink()
            raise FileNotFoundError(source)

        with mock.patch("seekr.indexer.extract", side_effect=vanishing_extract):
            report = self.indexer.sync([self.root])

        self.assertEqual(report.errors, [])
        self.assertEqual(report.removed, 1)
        self.assertIsNone(self.stored(path))

    def test_symlink_loop_does_not_hang_or_create_documents(self) -> None:
        self.write("real.txt", "alpha beta gamma")
        os.symlink(self.root, self.root / "self")
        os.symlink(self.root / "loop_b", self.root / "loop_a")
        os.symlink(self.root / "loop_a", self.root / "loop_b")

        report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 1)
        self.assertEqual(report.errors, [])
        self.assertEqual(self.store.stats(), {"documents": 1, "chunks": 1})

    def test_symlinked_file_outside_root_is_not_indexed(self) -> None:
        outside = self.tmp / "outside"
        outside.mkdir()
        secret = outside / "secret.md"
        secret.write_text("outside secret", encoding="utf-8")
        os.symlink(secret, self.root / "link.md")

        report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 0)
        self.assertEqual(report.skipped, 1)
        self.assertTrue(any("outside" in message for _, message in report.warnings))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_component_containment_prevents_pruning_sibling_root(self) -> None:
        sibling = self.tmp / "root2"
        sibling.mkdir()
        sibling_doc = str(sibling / "missing.md")
        self.store.upsert_document(sibling_doc, 1, 1, "a" * 64, "markdown")

        report = self.indexer.sync([self.root])

        self.assertEqual(report.removed, 0)
        self.assertIsNotNone(self.store.get_document(sibling_doc))

    def test_missing_root_is_not_pruned(self) -> None:
        doc_path = str(self.root / "gone.md")
        self.store.upsert_document(doc_path, 1, 1, "a" * 64, "markdown")
        self.root.rmdir()

        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.removed, 0)
        self.assertIsNotNone(self.store.get_document(doc_path))

    def test_walk_error_disables_pruning_for_that_root(self) -> None:
        doc_path = str(self.root / "never-seen.md")
        self.store.upsert_document(doc_path, 1, 1, "a" * 64, "markdown")

        def failing_walk(root, topdown=True, onerror=None, followlinks=False):
            if onerror is not None:
                onerror(OSError(errno.EACCES, "permission denied", str(root)))
            return iter(())

        with mock.patch("seekr.indexer.os.walk", side_effect=failing_walk):
            report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.removed, 0)
        self.assertIsNotNone(self.store.get_document(doc_path))

    def test_empty_roots_do_not_prune_anything(self) -> None:
        doc_path = str(self.root / "gone.md")
        self.store.upsert_document(doc_path, 1, 1, "a" * 64, "markdown")
        self.root.rmdir()

        report = self.indexer.sync([])

        self.assertEqual(report.removed, 0)
        self.assertEqual(report.errors, [])
        self.assertIsNotNone(self.store.get_document(doc_path))

    def test_sync_report_includes_extraction_warnings(self) -> None:
        (self.root / "latin.txt").write_bytes(b"caf\xe9 notes")

        report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 1)
        self.assertEqual(report.errors, [])
        self.assertTrue(
            any("latin-1" in message for _, message in report.warnings)
        )

    def test_get_report_returns_latest_report_and_error_count(self) -> None:
        self.assertIsNone(self.indexer.get_report())
        self.write("bad.txt", "")
        (self.root / "bad.txt").write_bytes(b"\x00\x01binary")

        report = self.indexer.sync([self.root])

        self.assertIs(self.indexer.get_report(), report)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(self.store.get_last_error_count(), 1)


class PreservationAndRetryTests(_IndexerTestCase):
    def test_extraction_failure_keeps_last_good_metadata_and_chunks(self) -> None:
        path = self.write("notes.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        old_doc = self.stored(path)
        assert old_doc is not None
        old_chunks = copy.deepcopy(self.store.chunks)

        path.write_bytes(b"\x00\x01binary")
        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        new_doc = self.stored(path)
        assert new_doc is not None
        self.assertEqual(new_doc.mtime_ns, old_doc.mtime_ns)
        self.assertEqual(new_doc.sha256, old_doc.sha256)
        self.assertEqual(self.store.chunks, old_chunks)

    def test_embedding_failure_keeps_last_good_metadata_and_chunks(self) -> None:
        path = self.write("notes.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        old_doc = self.stored(path)
        old_chunks = copy.deepcopy(self.store.chunks)
        path.write_text("delta epsilon zeta", encoding="utf-8")
        self.embedder.fail = True

        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        new_doc = self.stored(path)
        assert new_doc is not None and old_doc is not None
        self.assertEqual(new_doc.mtime_ns, old_doc.mtime_ns)
        self.assertEqual(new_doc.sha256, old_doc.sha256)
        self.assertEqual(self.store.chunks, old_chunks)

    def test_atomic_write_failure_rolls_back_metadata_and_chunks(self) -> None:
        path = self.write("notes.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        old_doc = self.stored(path)
        old_chunks = copy.deepcopy(self.store.chunks)
        path.write_text("delta epsilon zeta", encoding="utf-8")
        self.store.fail_replace = True

        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        new_doc = self.stored(path)
        assert new_doc is not None and old_doc is not None
        self.assertEqual(new_doc.mtime_ns, old_doc.mtime_ns)
        self.assertEqual(new_doc.sha256, old_doc.sha256)
        self.assertEqual(self.store.chunks, old_chunks)

    def test_failed_new_document_write_leaves_no_partial_document(self) -> None:
        self.write("notes.txt", "alpha beta gamma")
        self.store.fail_replace = True

        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_partial_batch_embedding_failure_isolates_failing_file(self) -> None:
        self.write("a_good.txt", "alpha beta gamma")
        self.write("b_poison.txt", "poison chunk text")
        self.write("c_good.txt", "delta epsilon zeta")
        original_embed = self.embedder.embed_documents

        def guarded_embed(texts: list[str]) -> list[str]:
            if any("poison" in text for text in texts):
                raise RuntimeError("poison chunk failed")
            return original_embed(texts)

        with mock.patch.object(self.embedder, "embed_documents", side_effect=guarded_embed):
            report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 2)
        self.assertEqual(len(report.errors), 1)
        self.assertIn("b_poison.txt", report.errors[0][0])
        self.assertIsNotNone(self.stored(self.root / "a_good.txt"))
        self.assertIsNone(self.stored(self.root / "b_poison.txt"))
        self.assertIsNotNone(self.stored(self.root / "c_good.txt"))

    def test_partial_batch_write_failure_isolates_failing_file(self) -> None:
        self.write("a_good.txt", "alpha beta gamma")
        bad_path = self.write("b_write_fail.txt", "delta epsilon zeta")
        self.write("c_good.txt", "eta theta iota")
        bad_key = str(bad_path.resolve())
        original_upsert = self.store.upsert_document

        def guarded_upsert(
            path: str, mtime_ns: int, size: int, sha256: str, kind: str
        ) -> int:
            if path == bad_key:
                raise RuntimeError("write failed for bad file")
            return original_upsert(path, mtime_ns, size, sha256, kind)

        with mock.patch.object(self.store, "upsert_document", side_effect=guarded_upsert):
            report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 2)
        self.assertEqual(len(report.errors), 1)
        self.assertIn("b_write_fail.txt", report.errors[0][0])
        self.assertIsNotNone(self.stored(self.root / "a_good.txt"))
        self.assertIsNone(self.stored(bad_path))
        self.assertIsNotNone(self.stored(self.root / "c_good.txt"))

    def test_transient_change_is_retried_then_indexed(self) -> None:
        self.write("churn.txt", "initial text")
        real_extract = __import__("seekr.indexer", fromlist=["extract"]).extract
        calls = 0

        def churning_extract(source: Path, kind: str):
            nonlocal calls
            calls += 1
            if calls == 1:
                source.write_text("settled text after one write", encoding="utf-8")
                stamp = 2_000_000_000
                os.utime(source, ns=(stamp, stamp))
            return real_extract(source, kind)

        with mock.patch("seekr.indexer.extract", side_effect=churning_extract):
            report = self.indexer.sync([self.root])

        self.assertEqual(calls, 2)
        self.assertEqual(report.added, 1)
        self.assertEqual(report.errors, [])
        self.assertEqual(len(self.embedder.calls), 1)

    def test_changing_file_is_retried_a_bounded_number_of_times(self) -> None:
        self.write("churn.txt", "initial text")
        real_extract = __import__("seekr.indexer", fromlist=["extract"]).extract
        calls = 0

        def churning_extract(source: Path, kind: str):
            nonlocal calls
            calls += 1
            source.write_text(f"attempt {calls} text {'x' * calls}", encoding="utf-8")
            stamp = 1_000_000_000 + calls * 1_000_000
            os.utime(source, ns=(stamp, stamp))
            return real_extract(source, kind)

        with mock.patch("seekr.indexer.extract", side_effect=churning_extract):
            report = self.indexer.sync([self.root])

        self.assertEqual(calls, 3)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(self.embedder.calls, [])
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})


class MovePathTests(_IndexerTestCase):
    def test_unchanged_rename_preserves_chunks_without_embedding(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        old_doc = self.stored(src)
        assert old_doc is not None
        old_chunk_ids = set(self.store.chunks)
        dst = self.root / "new.txt"
        src.rename(dst)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "unchanged")
        self.assertEqual(self.embedder.calls, calls)
        self.assertEqual(len(self.store.move_calls), 1)
        self.assertIsNone(self.stored(src))
        new_doc = self.stored(dst)
        assert new_doc is not None
        self.assertEqual(new_doc.id, old_doc.id)
        self.assertEqual(set(self.store.chunks), old_chunk_ids)

    def test_changed_content_reuses_source_until_destination_is_prepared(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = len(self.embedder.calls)
        dst = self.root / "new.txt"
        src.rename(dst)
        dst.write_text(_paragraph("bravo"), encoding="utf-8")

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "added")
        self.assertEqual(len(self.embedder.calls), calls + 1)
        self.assertIsNone(self.stored(src))
        self.assertIsNotNone(self.stored(dst))

    def test_destination_already_prepared_costs_zero_embeddings(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        dst = self.root / "new.txt"
        src.rename(dst)
        self.indexer.index_path(dst)
        calls = len(self.embedder.calls)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "removed")
        self.assertEqual(len(self.embedder.calls), calls)
        self.assertEqual(self.store.move_calls, [])
        self.assertIsNone(self.stored(src))
        self.assertIsNotNone(self.stored(dst))

    def test_failed_destination_preparation_keeps_source_document(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        dst = self.root / "new.txt"
        src.rename(dst)
        dst.write_text(_paragraph("bravo"), encoding="utf-8")

        def failing_extract(path: Path, kind: str):
            raise ExtractionError("cannot read destination")

        with mock.patch("seekr.indexer.extract", side_effect=failing_extract):
            result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "skipped")
        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNotNone(self.stored(src))
        self.assertIsNone(self.stored(dst))

    def test_missing_destination_removes_source_document(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        dst = self.root / "new.txt"
        src.unlink()

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "removed")
        self.assertIsNone(self.stored(src))
        self.assertIsNone(self.stored(dst))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_chained_move_with_vanished_destination_removes_source(self) -> None:
        src = self.write("a.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        second = self.root / "b.txt"
        third = self.root / "c.txt"
        src.rename(second)
        second.rename(third)

        result = self.indexer.move_path(src, second)

        self.assertEqual(result, "removed")
        self.assertIsNone(self.stored(src))
        self.assertIsNone(self.stored(second))
        self.assertIsNone(self.stored(third))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_changed_extension_reindexes_with_new_kind(self) -> None:
        src = self.write("old.txt", "alpha beta gamma")
        self.indexer.sync([self.root])
        dst = self.root / "new.md"
        src.rename(dst)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "added")
        self.assertEqual(self.store.move_calls, [])
        self.assertIsNone(self.stored(src))
        doc = self.stored(dst)
        assert doc is not None
        self.assertEqual(doc.kind, "markdown")

    def test_overwrite_with_stale_destination_row_is_reindexed(self) -> None:
        src = self.write("source.txt", "alpha alpha alpha")
        dst = self.write("dest.txt", "bravo bravo bravo")
        self.indexer.sync([self.root])
        src_doc = self.stored(src)
        dst_doc = self.stored(dst)
        assert src_doc is not None and dst_doc is not None
        old_mtime = dst_doc.mtime_ns
        old_size = dst_doc.size
        calls = len(self.embedder.calls)

        src.rename(dst)
        os.utime(dst, ns=(old_mtime, old_mtime))
        self.assertEqual(dst.stat().st_size, old_size)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "updated")
        self.assertEqual(len(self.embedder.calls), calls)
        new_doc = self.stored(dst)
        assert new_doc is not None
        self.assertEqual(new_doc.sha256, src_doc.sha256)
        self.assertIsNone(self.stored(src))
        texts = {
            chunk.text
            for chunk in self.store.chunks.values()
            if chunk.doc_id == new_doc.id
        }
        self.assertEqual(texts, {"alpha alpha alpha"})

    def test_move_to_ignored_destination_removes_stale_source(self) -> None:
        cfg = self._config(ignore_globs=("ignored.txt",))
        self.indexer = Indexer(self.store, self.embedder, cfg)
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        dst = self.root / "ignored.txt"
        src.rename(dst)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "removed")
        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNone(self.stored(src))
        self.assertIsNone(self.stored(dst))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_move_to_unsupported_destination_removes_stale_source(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        dst = self.root / "new.unknown"
        src.rename(dst)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "removed")
        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNone(self.stored(src))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_move_to_oversized_destination_removes_stale_source(self) -> None:
        src = self.write("old.txt", "alpha")
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        self.indexer = Indexer(
            self.store, self.embedder, self._config(max_file_bytes=16)
        )
        dst = self.root / "new.txt"
        src.rename(dst)
        dst.write_text("alpha plus far too much extra content", encoding="utf-8")

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "removed")
        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNone(self.stored(src))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0})

    def test_chained_moves_reconcile_each_source_in_order(self) -> None:
        first = self.write("a.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = list(self.embedder.calls)
        second = self.root / "b.txt"
        third = self.root / "c.txt"

        first.rename(second)
        self.assertEqual(self.indexer.move_path(first, second), "unchanged")
        second.rename(third)
        self.assertEqual(self.indexer.move_path(second, third), "unchanged")

        self.assertEqual(self.embedder.calls, calls)
        self.assertIsNone(self.stored(first))
        self.assertIsNone(self.stored(second))
        self.assertIsNotNone(self.stored(third))

    def test_code_language_change_is_not_a_plain_move(self) -> None:
        src = self.write("old.py", "def alpha():\n    return 1\n")
        self.indexer.sync([self.root])
        dst = self.root / "new.js"
        src.rename(dst)

        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "added")
        self.assertEqual(self.store.move_calls, [])
        self.assertIsNone(self.stored(src))
        doc = self.stored(dst)
        assert doc is not None
        self.assertEqual(doc.kind, "code")


class GrammarSummaryTests(_IndexerTestCase):
    def test_sync_reports_one_summary_for_many_files(self) -> None:
        for index in range(5):
            self.write(f"pkg/mod_{index}.py", f"def fn_{index}():\n    return {index}\n")
        self.write("web/app.js", "function hello() {\n  return 1;\n}\n")

        with mock.patch(
            "seekr.chunking._load_parser",
            return_value=(None, "missing grammar"),
        ):
            report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 6)
        grammar_warnings = [m for _, m in report.warnings if "tree-sitter grammars" in m]
        self.assertEqual(len(grammar_warnings), 1)
        self.assertEqual(report.missing_grammars, {"javascript": 1, "python": 5})
        self.assertEqual(list(report.missing_grammars), ["javascript", "python"])
        self.assertIn("6 code file(s)", grammar_warnings[0])
        expected_message = (
            "6 code file(s) used 60-line windows because tree-sitter grammars "
            "are not installed (javascript: 1, python: 5); install them with: "
            "seekr models pull --grammars"
        )
        self.assertEqual(
            report.warnings,
            [(str(self.root.resolve()), expected_message)],
        )

    def test_sync_reports_one_summary_for_30_python_files(self) -> None:
        paths = [
            self.write(f"tree/file_{index:02d}.py", f"def f_{index}():\n    return {index}\n")
            for index in range(30)
        ]

        with mock.patch(
            "seekr.chunking._load_parser",
            return_value=(
                None,
                "tree-sitter-language-pack 1.25.2 has no locally cached 'python' parser.",
            ),
        ):
            report = self.indexer.sync([self.root])

            self.assertEqual(report.added, 30)
            self.assertEqual(report.missing_grammars, {"python": 30})
            self.assertEqual(len(report.warnings), 1)
            self.assertIn("30 code file(s)", report.warnings[0][1])
            self.assertIn("(python: 30)", report.warnings[0][1])

            for index, path in enumerate(paths):
                path.write_text(
                    f"def f_{index}_v2():\n    return {index + 100}\n",
                    encoding="utf-8",
                )
            resync = self.indexer.sync([self.root])

        self.assertEqual(resync.updated, 30)
        self.assertEqual(resync.missing_grammars, {"python": 30})
        self.assertEqual(len(resync.warnings), 1)
        self.assertIn("30 code file(s)", resync.warnings[0][1])

    def test_other_chunking_warnings_stay_per_file(self) -> None:
        class FakeNode:
            def __init__(self, size: int) -> None:
                self.type = "module"
                self.start_byte = 0
                self.end_byte = size
                self.start_point = (0, 0)
                self.end_point = (0, 1)
                self.named_children: list[object] = []
                self.has_error = True

        class FakeTree:
            def __init__(self, size: int) -> None:
                self.root_node = FakeNode(size)

        class FakeParser:
            def parse(self, source: bytes) -> FakeTree:
                return FakeTree(len(source))

        p1 = self.write("broken_a.py", "def broken_a(:\n    pass\n")
        p2 = self.write("broken_b.py", "def broken_b(:\n    pass\n")

        with mock.patch(
            "seekr.chunking._load_parser",
            return_value=(FakeParser(), "controlled local parser"),
        ):
            report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 2)
        self.assertEqual(report.missing_grammars, {})
        self.assertEqual(len(report.warnings), 2)
        self.assertEqual(
            {path for path, _ in report.warnings},
            {str(p1.resolve()), str(p2.resolve())},
        )
        self.assertTrue(all("parse errors" in msg.lower() for _, msg in report.warnings))
        self.assertFalse(any("tree-sitter grammars" in msg for _, msg in report.warnings))

    def test_grammar_summary_message_formats_and_sorts_alphabetically(self) -> None:
        message = indexer_module.grammar_summary_message(
            {"rust": 2, "go": 4, "python": 10}
        )
        self.assertEqual(
            message,
            "16 code file(s) used 60-line windows because tree-sitter grammars are not "
            "installed (go: 4, python: 10, rust: 2); install them with: "
            "seekr models pull --grammars",
        )

    def test_single_file_root_records_one_grammar_summary(self) -> None:
        py_file = self.write("single.py", "def one():\n    return 1\n")
        with mock.patch(
            "seekr.chunking._load_parser",
            return_value=(None, "no cached 'python' parser"),
        ):
            report = self.indexer.sync([py_file])

        self.assertEqual(report.added, 1)
        self.assertEqual(report.missing_grammars, {"python": 1})
        self.assertEqual(len(report.warnings), 1)
        self.assertEqual(report.warnings[0][0], str(py_file.resolve()))
        self.assertIn("1 code file(s)", report.warnings[0][1])


@unittest.skipUnless(_HAS_FTS5, "FTS5 is required for the SQLite Store harness")
class StdlibStoreIndexerTests(unittest.TestCase):
    """Exercise the production Store SQLite code without numpy/sqlite-vec."""

    def setUp(self) -> None:
        from seekr.store import Store

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.cfg = Config(
            roots=(self.root,),
            db_path=self.tmp / "index.db",
            chunk_target_words=10,
            chunk_max_words=10,
            chunk_overlap_words=0,
        )
        self._backend = _stdlib_vector_backend()
        self._backend.__enter__()
        self.addCleanup(self._backend.__exit__, None, None, None)
        self.store = Store(self.cfg.db_path, self.cfg.embed_model, 8)
        self.addCleanup(self.store.close)
        self.embedder = ByteEmbedder()
        self.indexer = Indexer(self.store, self.embedder, self.cfg)

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_add_resync_rename_delete_and_failure_with_real_store(self) -> None:
        first = self.write("first.txt", _paragraph("alpha"))
        second = self.write("second.txt", _paragraph("bravo"))
        report = self.indexer.sync([self.root])
        self.assertEqual(report.added, 2)
        calls = len(self.embedder.calls)

        report = self.indexer.sync([self.root])
        self.assertEqual(report.unchanged, 2)
        self.assertEqual(len(self.embedder.calls), calls)

        renamed = self.root / "renamed.txt"
        first.rename(renamed)
        report = self.indexer.sync([self.root])
        self.assertEqual((report.added, report.removed), (1, 1))
        self.assertEqual(len(self.embedder.calls), calls)

        second.write_bytes(b"\x00\x01binary")
        report = self.indexer.sync([self.root])
        self.assertEqual(len(report.errors), 1)
        old = self.store.get_document(str(second.resolve()))
        assert old is not None
        self.assertEqual(len(self.store.keyword_search("bravo", 10)), 1)

        second.unlink()
        report = self.indexer.sync([self.root])
        self.assertEqual(report.removed, 1)
        self.assertIsNone(self.store.get_document(str(second.resolve())))

    def test_real_store_move_path_preserves_chunk_ids_without_embedding(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = len(self.embedder.calls)
        old_id = self.store.keyword_search('"alpha"', 10)[0][0]

        dst = self.root / "new.txt"
        src.rename(dst)
        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "unchanged")
        self.assertEqual(len(self.embedder.calls), calls)
        hits = self.store.keyword_search('"alpha"', 10)
        self.assertEqual(hits[0][0], old_id)
        chunk = self.store.fetch_chunks([old_id])[old_id]
        self.assertEqual(chunk.path, str(dst.resolve()))


@unittest.skipUnless(
    _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_FTS5,
    "the production sqlite-vec/numpy stack is required",
)
class RealStoreIndexerTests(unittest.TestCase):
    def setUp(self) -> None:
        from seekr.embed import FakeEmbedder
        from seekr.store import Store

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.cfg = Config(
            roots=(self.root,),
            db_path=self.tmp / "index.db",
            chunk_target_words=10,
            chunk_max_words=10,
            chunk_overlap_words=0,
        )
        self.store = Store(self.cfg.db_path, self.cfg.embed_model, 8)
        self.addCleanup(self.store.close)
        self.embedder = FakeEmbedder(dim=8)
        self.indexer = Indexer(self.store, self.embedder, self.cfg)

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_full_sync_add_resync_rename_and_delete(self) -> None:
        first = self.write("first.txt", _paragraph("alpha"))
        second = self.write("second.txt", _paragraph("bravo"))
        report = self.indexer.sync([self.root])
        self.assertEqual(report.added, 2)
        calls = len(self.embedder.calls)

        report = self.indexer.sync([self.root])
        self.assertEqual(report.unchanged, 2)
        self.assertEqual(len(self.embedder.calls), calls)

        renamed = self.root / "renamed.txt"
        first.rename(renamed)
        report = self.indexer.sync([self.root])
        self.assertEqual((report.added, report.removed), (1, 1))
        self.assertEqual(len(self.embedder.calls), calls)

        second.unlink()
        report = self.indexer.sync([self.root])
        self.assertEqual(report.removed, 1)
        self.assertEqual(self.store.stats(), {"documents": 1, "chunks": 1})

    def test_real_move_path_preserves_chunk_ids_and_vectors(self) -> None:
        src = self.write("old.txt", _paragraph("alpha"))
        self.indexer.sync([self.root])
        calls = len(self.embedder.calls)
        old_chunk_id = self.store.keyword_search('"alpha"', 10)[0][0]

        dst = self.root / "new.txt"
        src.rename(dst)
        result = self.indexer.move_path(src, dst)

        self.assertEqual(result, "unchanged")
        self.assertEqual(len(self.embedder.calls), calls)
        hits = self.store.keyword_search('"alpha"', 10)
        self.assertEqual(hits[0][0], old_chunk_id)
        self.assertEqual(self.store.fetch_chunks([old_chunk_id])[old_chunk_id].path, str(dst))

    def test_real_failure_preserves_last_good_document(self) -> None:
        path = self.write("notes.txt", "keepword alpha beta")
        self.indexer.sync([self.root])
        old_doc = self.store.get_document(str(path))
        assert old_doc is not None

        path.write_bytes(b"\x00\x01binary")
        report = self.indexer.sync([self.root])

        self.assertEqual(len(report.errors), 1)
        current = self.store.get_document(str(path))
        assert current is not None
        self.assertEqual(current.sha256, old_doc.sha256)
        self.assertEqual(len(self.store.keyword_search("keepword", 10)), 1)


class CourseLabelIndexTests(_IndexerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.course = self.tmp / "course"
        self.course.mkdir()

    def write_course(self, relative: str, text: str) -> Path:
        path = self.course / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def doc(self, relative: str) -> DocRow:
        row = self.stored(self.course / relative)
        assert row is not None
        return row

    def doc_label(self, relative: str) -> tuple[str | None, int | None]:
        row = self.doc(relative)
        return (row.module, row.week)

    def test_sync_labels_documents(self) -> None:
        a_md = self.write_course("ml/week2/a.md", "gradient descent notes")
        self.write_course("syllabus.md", "course overview")
        self.write_course("ml/notes.md", "general machine learning")
        self.write_course("Week 01/intro.md", "week one introduction")
        cfg = self._config(roots=(self.course,))
        indexer = Indexer(self.store, self.embedder, cfg)

        report = indexer.sync([self.course])

        self.assertEqual(report.added, 4)
        self.assertEqual(report.errors, [])
        self.assertEqual(indexer.label(a_md), CourseLabel("ml", 2))
        self.assertEqual(self.doc_label("ml/week2/a.md"), ("ml", 2))
        self.assertIsNone(self.doc("syllabus.md").module)
        self.assertIsNone(self.doc("syllabus.md").week)
        self.assertEqual(self.doc_label("ml/notes.md"), ("ml", None))
        self.assertEqual(self.doc_label("Week 01/intro.md"), ("course", 1))

    def test_changing_week_pattern_relabels_without_embedding(self) -> None:
        self.write_course("ml/week2/a.md", "week two notes")
        self.write_course("ml/lecture3/b.md", "lecture three notes")
        cfg = self._config(roots=(self.course,))
        indexer = Indexer(self.store, self.embedder, cfg)

        indexer.sync([self.course])
        calls = len(self.embedder.calls)
        self.assertEqual(self.doc_label("ml/week2/a.md"), ("ml", 2))
        self.assertEqual(self.doc_label("ml/lecture3/b.md"), ("ml", None))
        self.assertFalse(indexer.relabel_if_needed())

        cfg2 = replace(cfg, week_pattern=r"^lecture[\s_-]*(\d+)\b")
        indexer2 = Indexer(self.store, self.embedder, cfg2)
        report2 = indexer2.sync([self.course])

        self.assertEqual(len(self.embedder.calls), calls)
        self.assertEqual((report2.added, report2.updated, report2.unchanged), (0, 0, 2))
        self.assertEqual(self.doc_label("ml/week2/a.md"), ("ml", None))
        self.assertEqual(self.doc_label("ml/lecture3/b.md"), ("ml", 3))
        self.assertFalse(indexer2.relabel_if_needed())

    def test_course_structure_false_clears_labels(self) -> None:
        a_md = self.write_course("ml/week2/a.md", "ml notes")
        self.write_course("ds/week1/b.md", "ds notes")
        cfg = self._config(roots=(self.course,))
        Indexer(self.store, self.embedder, cfg).sync([self.course])
        calls = len(self.embedder.calls)
        self.assertEqual(self.doc_label("ml/week2/a.md"), ("ml", 2))

        cfg_off = replace(cfg, course_structure=False)
        indexer_off = Indexer(self.store, self.embedder, cfg_off)
        self.assertEqual(indexer_off.label(a_md), CourseLabel())

        report = indexer_off.sync([self.course])
        self.assertEqual(len(self.embedder.calls), calls)
        self.assertEqual(report.unchanged, 2)
        self.assertEqual(self.doc_label("ml/week2/a.md"), (None, None))
        self.assertEqual(self.doc_label("ds/week1/b.md"), (None, None))

        Indexer(self.store, self.embedder, cfg).sync([self.course])
        self.assertEqual(len(self.embedder.calls), calls)
        self.assertEqual(self.doc_label("ml/week2/a.md"), ("ml", 2))

    def test_move_recomputes_label_for_destination(self) -> None:
        src = self.write_course("ml/week1/a.md", _paragraph("alpha"))
        other_src = self.write_course("ml/week2/dup.md", _paragraph("bravo"))
        existing_dst = self.write_course("ds/week5/dup.md", _paragraph("bravo"))
        cfg = self._config(roots=(self.course,))
        indexer = Indexer(self.store, self.embedder, cfg)

        indexer.sync([self.course])
        calls = len(self.embedder.calls)
        self.assertEqual(self.doc_label("ml/week1/a.md"), ("ml", 1))

        dst = self.course / "ml/week3/a.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        result = indexer.move_path(src, dst)

        self.assertEqual(result, "unchanged")
        self.assertEqual(len(self.embedder.calls), calls)
        self.assertEqual(self.doc_label("ml/week3/a.md"), ("ml", 3))

        cross_dst = self.course / "ds/week4/a.md"
        cross_dst.parent.mkdir(parents=True, exist_ok=True)
        dst.rename(cross_dst)
        self.assertEqual(indexer.move_path(dst, cross_dst), "unchanged")
        self.assertEqual(len(self.embedder.calls), calls)
        self.assertEqual(self.doc_label("ds/week4/a.md"), ("ds", 4))

        # Moving over an already-indexed identical destination keeps destination label.
        other_src.rename(existing_dst)
        self.assertEqual(indexer.move_path(other_src, existing_dst), "removed")
        self.assertEqual(len(self.embedder.calls), calls)
        self.assertEqual(self.doc_label("ds/week5/dup.md"), ("ds", 5))

    def test_on_file_callback_called_per_processed_file(self) -> None:
        a_md = self.write_course("ml/week1/a.md", "alpha beta")
        b_txt = self.write_course("ml/week1/b.txt", "gamma delta")
        self.write_course("node_modules/ignored.md", "ignored text")
        image_png = self.write_course("ml/week1/diagram.png", "not supported")
        cfg = self._config(roots=(self.course,))
        indexer = Indexer(self.store, self.embedder, cfg)

        seen_first: list[Path] = []
        report1 = indexer.sync([self.course], on_file=seen_first.append)

        self.assertEqual((report1.added, report1.skipped), (2, 2))
        self.assertEqual(seen_first, sorted([a_md.resolve(), b_txt.resolve()]))

        a_md.write_text("alpha beta updated", encoding="utf-8")
        c_md = self.write_course("ds/week2/c.md", "epsilon zeta")
        bad_txt = self.course / "ds/week2/bad.txt"
        bad_txt.parent.mkdir(parents=True, exist_ok=True)
        bad_txt.write_bytes(b"\x00\x01binary")

        seen_second: list[Path] = []
        report2 = indexer.sync([self.course], on_file=seen_second.append)

        self.assertEqual((report2.added, report2.updated, report2.unchanged), (1, 1, 1))
        self.assertEqual(len(report2.errors), 1)
        expected_second = sorted(
            [a_md.resolve(), b_txt.resolve(), bad_txt.resolve(), c_md.resolve()]
        )
        self.assertEqual(seen_second, expected_second)

        seen_single: list[Path] = []
        indexer.sync([a_md], on_file=seen_single.append)
        self.assertEqual(seen_single, [a_md.resolve()])

        seen_unsupported: list[Path] = []
        indexer.sync([image_png], on_file=seen_unsupported.append)
        self.assertEqual(seen_unsupported, [])

    @unittest.skipUnless(_HAS_FTS5, "FTS5 is required for the SQLite Store harness")
    def test_relabel_persists_with_sqlite_store(self) -> None:
        from seekr.store import Store

        self.write_course("ml/week2/a.md", "gradient descent notes")
        self.write_course("ml/lecture4/b.md", "lecture four notes")
        cfg = Config(
            roots=(self.course,),
            db_path=self.tmp / "real.db",
            chunk_target_words=10,
            chunk_max_words=10,
            chunk_overlap_words=0,
        )
        with _stdlib_vector_backend():
            store = Store(cfg.db_path, cfg.embed_model, 8)
            self.addCleanup(store.close)
            embedder = ByteEmbedder()
            Indexer(store, embedder, cfg).sync([self.course])
            calls = len(embedder.calls)
            self.assertEqual(store.list_modules(), ["ml"])
            self.assertEqual(
                store.get_document(str((self.course / "ml/week2/a.md").resolve())).week,
                2,
            )

            cfg2 = replace(cfg, week_pattern=r"^lecture[\s_-]*(\d+)\b")
            Indexer(store, embedder, cfg2).sync([self.course])
            self.assertEqual(len(embedder.calls), calls)
            self.assertIsNone(
                store.get_document(str((self.course / "ml/week2/a.md").resolve())).week
            )
            self.assertEqual(
                store.get_document(str((self.course / "ml/lecture4/b.md").resolve())).week,
                4,
            )


class OfficeAndNotebookIndexTests(_IndexerTestCase):
    def test_legacy_doc_is_warned_and_skipped(self) -> None:
        self.write("notes.txt", "valid indexable text")
        legacy_doc = self.root / "old.doc"
        legacy_doc.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1legacy doc")
        legacy_ppt = self.root / "old.ppt"
        legacy_ppt.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1legacy ppt")

        report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 1)
        self.assertEqual(report.skipped, 2)
        self.assertEqual(report.errors, [])
        expected_msg = (
            "legacy Office format is not indexed; save it as .docx/.pptx to include it"
        )
        warnings_by_path = dict(report.warnings)
        self.assertEqual(warnings_by_path.get(str(legacy_doc.resolve())), expected_msg)
        self.assertEqual(warnings_by_path.get(str(legacy_ppt.resolve())), expected_msg)
        self.assertIsNone(self.stored(legacy_doc))
        self.assertIsNone(self.stored(legacy_ppt))

    def test_corrupt_office_and_bad_notebook_are_isolated_per_file(self) -> None:
        import json

        import docx
        import pptx

        good_doc = docx.Document()
        good_doc.add_paragraph("Valid word document content for indexing.")
        good_docx_path = self.root / "good.docx"
        good_doc.save(good_docx_path)

        good_prs = pptx.Presentation()
        slide = good_prs.slides.add_slide(good_prs.slide_layouts[0])
        slide.shapes.title.text = "Valid slide deck content."
        good_pptx_path = self.root / "good.pptx"
        good_prs.save(good_pptx_path)

        good_nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{"cell_type": "markdown", "source": "Valid notebook cell."}],
        }
        good_ipynb_path = self.root / "good.ipynb"
        good_ipynb_path.write_text(json.dumps(good_nb), encoding="utf-8")

        bad_docx = self.root / "corrupt.docx"
        bad_docx.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 128)
        bad_pptx = self.root / "truncated.pptx"
        bad_pptx.write_bytes(b"PK\x03\x04truncated")
        bad_ipynb = self.root / "broken.ipynb"
        bad_ipynb.write_text(
            json.dumps({"nbformat": 4, "cells": [{"cell_type": "code", "source": 123}]}),
            encoding="utf-8",
        )

        report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 3)
        self.assertEqual(report.skipped, 3)
        self.assertEqual(len(report.errors), 3)
        error_paths = {path for path, _ in report.errors}
        self.assertEqual(
            error_paths,
            {
                str(bad_docx.resolve()),
                str(bad_pptx.resolve()),
                str(bad_ipynb.resolve()),
            },
        )
        self.assertIsNotNone(self.stored(good_docx_path))
        self.assertIsNotNone(self.stored(good_pptx_path))
        self.assertIsNotNone(self.stored(good_ipynb_path))

    def test_docx_schema_invalid_xml_is_isolated_per_file(self) -> None:
        import io
        import zipfile

        import docx

        doc = docx.Document()
        doc.add_heading("Section", level=1)
        doc.add_paragraph("Valid body text.")
        buf = io.BytesIO()
        doc.save(buf)

        bad_docx = self.root / "missing_attr.docx"
        with (
            zipfile.ZipFile(io.BytesIO(buf.getvalue()), "r") as zin,
            zipfile.ZipFile(bad_docx, "w") as zout,
        ):
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = data.replace(b'w:val="Heading1"', b"")
                zout.writestr(item, data)

        good_txt = self.write("sibling.txt", "sibling valid content")

        report = self.indexer.sync([self.root])

        self.assertEqual(report.added, 1)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.errors[0][0], str(bad_docx.resolve()))
        self.assertIsNotNone(self.stored(good_txt))
        self.assertIsNone(self.stored(bad_docx))


if __name__ == "__main__":
    unittest.main()


class CanonicalPathTests(_IndexerTestCase):
    NFD = "Módulo"
    NFC = "Módulo"

    def test_linux_stored_paths_are_unchanged_resolve_output(self) -> None:
        nfd_file = self.write(f"{self.NFD}/a.txt", "alpha beta gamma")
        link = self.tmp / "link-root"
        link.symlink_to(self.root, target_is_directory=True)

        self.indexer.sync([link])
        self.indexer.index_path(nfd_file)

        # Byte-for-byte what Path.resolve() gives: no Unicode normalisation.
        self.assertEqual(sorted(self.store.documents), [str(nfd_file.resolve())])
        self.assertIn(self.NFD, next(iter(self.store.documents)))

    def test_darwin_resolves_and_walks_to_nfc_paths(self) -> None:
        # The NFC spelling opens the same file on APFS/HFS+; on this test host
        # only the recorded strings are checked.
        self.write(f"{self.NFD}/a.txt", "alpha beta gamma")
        expected = self.root.resolve() / self.NFC / "a.txt"
        with mock.patch.object(indexer_module, "_PLATFORM", "darwin"):
            resolved, issue = Indexer._resolve(self.root / self.NFD / "a.txt")
            files, ok = self.indexer._walk_root(self.root.resolve(), indexer_module.SyncReport())
        self.assertIsNone(issue)
        self.assertEqual(str(resolved), str(expected))
        self.assertTrue(ok)
        self.assertEqual([str(item) for item in files], [str(expected)])
