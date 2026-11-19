"""Offline regression tests for seekr's cached-model indexing/search pipeline.

Three goals:

1. Unit-test the standalone socket/DNS audit hook helper.  It blocks and
   records audited ``socket.connect`` / ``socket.create_connection`` /
   ``socket.getaddrinfo`` attempts to non-local targets while letting AF_UNIX
   paths and loopback traffic through.  Everything below is exercised via
   direct calls, including ``sys.audit`` dispatch of the audited events, so no
   real network packets are ever sent.  (Outbound connect attempts are blocked
   by the audit hook before CPython performs the syscall - verified against an
   unroutable TEST-NET-3 address.)
2. Keep the default suite offline and stdlib-only: index+search of a tiny
   generated markdown fixture run under the audit hook must record zero
   outbound attempts.  Extraction/chunking/indexing is exercised with the
   standard library alone via a stub store; the FakeEmbedder + real Store +
   Searcher-variant mirrors the existing real-Store convention and skips when
   numpy, the sqlite-vec extension or FTS5 are missing.  seekr modules are
   imported lazily inside tests so unrelated in-progress modules are never
   imported at collection time.
3. Opt-in end-to-end test with a real sentence-transformers model, skipped
   unless ``SEEKR_TEST_EMBED_MODEL_DIR`` names a pre-provisioned local model
   directory.  Once the environment variable is set, this is a regression
   suite: missing dependencies or a broken model fail the test instead of
   being converted into skips.

Run the scoped suite with::

    PYTHONPATH=src python3 -m unittest tests.test_cached_pipeline -v

Opt in to the real-model run with::

    SEEKR_TEST_EMBED_MODEL_DIR=/path/to/model \
        PYTHONPATH=src python3 -m unittest tests.test_cached_pipeline -v
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import shutil
import socket
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

try:  # pragma: no cover - pytest is an optional development dependency
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

try:  # package-style import (unittest from the repo root / pytest)
    from tests import _cached_pipeline_netaudit as netaudit
except ImportError:  # pragma: no cover - loose-file layout (e.g. pytest inserts tests/)
    import _cached_pipeline_netaudit as netaudit

OfflineNetworkAttempt = netaudit.OfflineNetworkAttempt


def _slow(test_class: type[unittest.TestCase]) -> type[unittest.TestCase]:
    """Mark a class ``slow`` for pytest without breaking plain unittest runs."""
    if pytest is None:  # pragma: no cover - depends on the test runner
        return test_class
    return pytest.mark.slow(test_class)


def _has_fts5() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
    except sqlite3.Error:  # pragma: no cover - defensive
        return False
    try:
        options = [row[0] for row in conn.execute("PRAGMA compile_options")]
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None
_HAS_FTS5 = _has_fts5()
_HAS_REAL_STORE = _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_FTS5

_MODEL_DIR_ENV = "SEEKR_TEST_EMBED_MODEL_DIR"
_MODEL_DIR = os.environ.get(_MODEL_DIR_ENV)
_OPTED_IN = bool(_MODEL_DIR and _MODEL_DIR.strip())

_TEST_NET = "203.0.113.9"  # TEST-NET-3; never routes. The hook blocks before
# the connect syscall, so no packet can leave regardless.

#: Prefer the pre-created, approved external fixture area over generic /tmp.
_TEMP_ROOT = "/tmp/opencode" if os.path.isdir("/tmp/opencode") else None

_MARKDOWN = """# Alpha Notes

Intro paragraph mentions the xylophone archive once.

## Timeline

The xylophone deadline gantt chart slipped by a week.

## Materials

Unrelated barrel, ladder and hinge words live here.
"""


def _make_fixture_dir(context: unittest.TestCase) -> tuple[Path, str]:
    """Create a tiny markdown fixture and return ``(root, resolved_file_path)``."""
    root = Path(tempfile.mkdtemp(prefix="seekr_cached_pipeline_", dir=_TEMP_ROOT))
    context.addCleanup(shutil.rmtree, root, ignore_errors=True)
    (root / "notes.md").write_text(_MARKDOWN, encoding="utf-8")
    return root, str((root / "notes.md").resolve())


class AuditHookTests(unittest.TestCase):
    """Direct-call unit tests for the offline network audit hook helper."""

    def test_is_local_target_classification(self) -> None:
        is_local = netaudit.is_local_target
        self.assertTrue(is_local("/tmp/opencode/some.sock"))  # AF_UNIX path
        self.assertTrue(is_local(b"@abstract"))
        self.assertTrue(is_local("localhost"))
        self.assertTrue(is_local(("127.0.0.1", 80)))
        self.assertTrue(is_local(("::1", 80)))
        self.assertFalse(is_local(("203.0.113.9", 80)))
        self.assertFalse(is_local(("10.1.2.3", 80)))
        self.assertFalse(is_local(("fe80::1", 80)))
        self.assertFalse(is_local(("example.invalid", 443)))
        self.assertFalse(is_local(None))

    def test_outbound_connect_blocked_and_recorded_no_packets(self) -> None:
        with netaudit.offline_network_audit() as record:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.addCleanup(probe.close)
            with self.assertRaises(OfflineNetworkAttempt):
                probe.connect((_TEST_NET, 80))
        self.assertEqual(record.blocked, [("socket.connect", (_TEST_NET, 80))])
        self.assertEqual(record.allowed, [])

    def test_create_connection_blocked_at_first_audited_stage(self) -> None:
        with netaudit.offline_network_audit() as record, self.assertRaises(
            OfflineNetworkAttempt
        ):
            socket.create_connection((_TEST_NET, 80))
        # ``socket.create_connection`` fires no dedicated audit event itself;
        # the block must land on the inner getaddrinfo before any DNS or
        # connect work happens.
        self.assertEqual(
            [event for event, _ in record.blocked], ["socket.getaddrinfo"]
        )
        self.assertEqual(record.allowed, [])

    def test_sys_audit_dispatch_of_audited_event_names(self) -> None:
        with netaudit.offline_network_audit() as record:
            with self.assertRaises(OfflineNetworkAttempt):
                sys.audit(
                    "socket.getaddrinfo",
                    "example.invalid",
                    443,
                    socket.AF_UNSPEC,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                )
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.addCleanup(probe.close)
            with self.assertRaises(OfflineNetworkAttempt):
                sys.audit("socket.connect", probe, ("10.1.2.3", 80))
            with self.assertRaises(OfflineNetworkAttempt):
                sys.audit(
                    "socket.create_connection", ("myhost.example.invalid", 5432), 5
                )
        self.assertEqual(
            [event for event, _ in record.blocked],
            ["socket.getaddrinfo", "socket.connect", "socket.create_connection"],
        )
        self.assertEqual(record.allowed, [])

    def test_af_unix_and_loopback_allowed_and_recorded(self) -> None:
        # Real AF_UNIX IPC under the hook - local only, no network packets.
        root = Path(tempfile.mkdtemp(prefix="seekr_audit_unix_", dir=_TEMP_ROOT))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        unix_path = str(root / "hook.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(unix_path)
        server.listen(1)
        self.addCleanup(server.close)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(client.close)

        with netaudit.offline_network_audit() as record:
            client.connect(unix_path)
            # Loopback/non-outbound dispatches must be allowed too.
            sys.audit(
                "socket.getaddrinfo",
                "localhost",
                0,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
            )
            sys.audit("socket.connect", client, ("127.0.0.1", 0))
            sys.audit("socket.create_connection", ("::1", 6379), 5, None)
        self.assertEqual(record.blocked, [])
        self.assertEqual(
            sorted(event for event, _ in record.allowed),
            [
                "socket.connect",  # the real AF_UNIX connect above
                "socket.connect",  # the 127.0.0.1 dispatch
                "socket.create_connection",
                "socket.getaddrinfo",
            ],
        )

    def test_hook_is_inert_when_no_scope_is_active(self) -> None:
        # Without an active scope the singleton hook must neither block nor
        # record, so it cannot disturb other tests sharing the process.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(probe.close)
        sys.audit("socket.connect", probe, (_TEST_NET, 80))
        sys.audit("socket.getaddrinfo", "example.invalid", 443, 0, 0, 0)

    def test_records_do_not_leak_between_scopes(self) -> None:
        with netaudit.offline_network_audit() as first:
            pass
        self.assertEqual(first.blocked, [])
        with netaudit.offline_network_audit() as second, self.assertRaises(
            OfflineNetworkAttempt
        ):
            sys.audit("socket.getaddrinfo", "example.invalid", 80, 0, 0, 0)
        self.assertEqual(len(second.blocked), 1)
        self.assertEqual(first.blocked, [])


class _PipelineStubDoc:
    """Minimal document row for :class:`_PipelineStubStore`."""

    def __init__(self, *, id: int, mtime_ns: int, size: int, sha256: str, kind: str):
        self.id = id
        self.mtime_ns = mtime_ns
        self.size = size
        self.sha256 = sha256
        self.kind = kind


class _PipelineStubChunk:
    """In-memory chunk row recorded by :class:`_PipelineStubStore`."""

    def __init__(self, doc_id: int, stored: object) -> None:
        self.doc_id = doc_id
        spec = stored.spec  # type: ignore[attr-defined]
        self.text = spec.text
        self.text_sha = stored.text_sha  # type: ignore[attr-defined]
        self.locator = spec.locator
        self.loc_start = float(spec.loc_start)
        self.vector = stored.vector  # type: ignore[attr-defined]


class _PipelineStubStore:
    """Stdlib-only stand-in capturing exactly what the Indexer persists."""

    def __init__(self) -> None:
        self.documents: dict[str, _PipelineStubDoc] = {}
        self.chunks: dict[int, _PipelineStubChunk] = {}
        self._next_doc_id = 1
        self._next_chunk_id = 1

    @contextlib.contextmanager
    def transaction(self, *, write: bool = False):
        yield

    def get_document(self, path: str) -> _PipelineStubDoc | None:
        return self.documents.get(str(path))

    def upsert_document(
        self, path: str, mtime_ns: int, size: int, sha256: str, kind: str
    ) -> int:
        doc = self.documents.get(str(path))
        if doc is None:
            doc_id = self._next_doc_id
            self._next_doc_id += 1
        else:
            doc_id = doc.id
        self.documents[str(path)] = _PipelineStubDoc(
            id=doc_id, mtime_ns=int(mtime_ns), size=int(size), sha256=sha256, kind=kind
        )
        return doc_id

    def replace_chunks(self, doc_id: int, chunks: list[object]) -> None:
        for chunk_id in [
            cid for cid, chunk in self.chunks.items() if chunk.doc_id == doc_id
        ]:
            del self.chunks[chunk_id]
        for stored in chunks:
            self.chunks[self._next_chunk_id] = _PipelineStubChunk(doc_id, stored)
            self._next_chunk_id += 1

    def find_vector_by_text_sha(self, sha: str) -> object | None:
        for chunk in self.chunks.values():
            if chunk.text_sha == sha:
                return chunk.vector
        return None

    def list_paths(self, prefix: str | None = None) -> list[str]:
        paths = sorted(self.documents)
        if prefix is None:
            return paths
        base = str(prefix).rstrip("/")
        if not base:
            return paths
        return [path for path in paths if path == base or path.startswith(base + "/")]

    def delete_document(self, path: str) -> None:
        self.documents.pop(str(path), None)

    def set_last_error_count(self, n: int) -> None:
        pass


class _HashVectorsEmbedder:
    """Deterministic pure-python embedder (opaque tuples; no numpy needed)."""

    model_name = "fake"

    @staticmethod
    def _vector(text: str) -> tuple[int, ...]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
        return (int.from_bytes(digest, "big") & 0xFF,)

    def embed_documents(self, texts: list[str]) -> list[tuple[int, ...]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> tuple[int, ...]:
        return self._vector(text)


class StdlibPipelineOfflineTests(unittest.TestCase):
    """Default suite, stdlib only: markdown indexing stays offline.

    Exercises the real extraction/chunking/indexer path (markdown -> ``§``
    heading locators) with a stub store and a pure-python embedder so the
    plain offline suite genuinely runs the pipeline without numpy/sqlite-vec.
    """

    def test_indexing_markdown_produces_section_locators_offline(self) -> None:
        from seekr.config import Config
        from seekr.indexer import Indexer

        root, resolved_path = _make_fixture_dir(self)
        store = _PipelineStubStore()
        with netaudit.offline_network_audit() as record:
            cfg = Config(roots=(root,), db_path=root / ".seekr" / "index.db")
            report = Indexer(store, _HashVectorsEmbedder(), cfg).sync([root])

        self.assertEqual(report.error_count, 0)
        self.assertEqual(report.warning_count, 0)
        self.assertEqual(report.added, 1)
        self.assertEqual(list(store.documents), [resolved_path])
        locators = [chunk.locator for chunk in store.chunks.values()]
        self.assertIn("\u00a7Alpha Notes > Timeline", locators)
        self.assertTrue(
            all(locator.startswith("\u00a7") for locator in locators),
            f"every markdown chunk must carry a § locator, got {locators}",
        )
        self.assertEqual(
            record.blocked + record.allowed,
            [],
            "stdlib indexing must make zero outbound socket/DNS attempts",
        )


@unittest.skipUnless(
    _HAS_REAL_STORE,
    "numpy, sqlite-vec and an FTS5-capable sqlite build are required; "
    "the default suite stays stdlib-only without them",
)
class FakeEmbedderPipelineOfflineTests(unittest.TestCase):
    """Default-suite regression: FakeEmbedder index+search stays offline."""

    def test_index_and_search_record_zero_outbound_attempts(self) -> None:
        from seekr.config import Config
        from seekr.embed import FakeEmbedder
        from seekr.indexer import Indexer
        from seekr.rerank import FakeReranker
        from seekr.search import Searcher
        from seekr.store import Store

        root, resolved_path = _make_fixture_dir(self)

        with netaudit.offline_network_audit() as record:
            embedder = FakeEmbedder(dim=64)
            store = Store(
                db_path=root / ".seekr" / "index.db",
                embed_model=embedder.model_name,
                dim=embedder.dim,
            )
            self.addCleanup(store.close)
            reranker = FakeReranker({"Timeline": 2.0})
            cfg = Config(roots=(root,), db_path=root / ".seekr" / "index.db")
            report = Indexer(store, embedder, cfg).sync([root])
            hits = Searcher(store, embedder, reranker).search(
                "xylophone deadline", k=5, mode="hybrid", rerank=True
            )

        self.assertEqual(report.error_count, 0)
        self.assertEqual(report.warning_count, 0)
        self.assertEqual(report.added, 1)
        self.assertTrue(embedder.calls, "indexing should have embedded documents")
        self.assertTrue(hits, "hybrid search should find the fixture")
        self.assertTrue(
            any(
                hit.path == resolved_path
                and hit.locator.startswith("\u00a7")
                and "Timeline" in hit.locator
                for hit in hits
            ),
            f"expected a hit at {resolved_path} with a §Timeline locator, got {hits}",
        )
        self.assertEqual(
            record.blocked + record.allowed,
            [],
            "the offline pipeline must make zero outbound socket/DNS attempts",
        )


@unittest.skipUnless(
    _OPTED_IN,
    f"set {_MODEL_DIR_ENV} to a pre-provisioned local model directory",
)
@_slow
class CachedRealModelOfflineTests(unittest.TestCase):
    """Opt-in: real cached-model indexing + hybrid search, fully offline.

    Once ``SEEKR_TEST_EMBED_MODEL_DIR`` is set this is a regression suite, not
    a skip target: missing/broken dependencies or a bad model FAIL the test.
    """

    def _require_dependencies(self) -> None:
        """Fail loudly on missing heavy deps instead of skipping."""
        missing = [
            name
            for name in ("numpy", "sqlite_vec", "sentence_transformers")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            self.fail(
                f"{_MODEL_DIR_ENV} is set (opted in) but required packages are "
                f"missing: {', '.join(missing)}. Install them or unset the "
                "environment variable; a missing dependency must not silently "
                "turn this regression test into a skip."
            )

    def test_cached_model_index_search_is_offline_with_locators(self) -> None:
        self._require_dependencies()

        model_dir = Path(str(_MODEL_DIR)).expanduser()
        if not model_dir.is_dir():
            self.fail(
                f"{_MODEL_DIR_ENV} must point to an existing model directory, "
                f"got {model_dir}"
            )

        # Lazy imports: heavy seekr modules load only inside the opted-in test.
        from seekr.config import Config
        from seekr.embed import SentenceTransformerEmbedder
        from seekr.indexer import Indexer
        from seekr.rerank import FakeReranker
        from seekr.search import Searcher
        from seekr.store import Store

        root, resolved_path = _make_fixture_dir(self)

        with netaudit.offline_network_audit() as record:
            # Constructing the embedder and reading ``dim`` trigger the first
            # model load; the reranker is built here too so initial lazy setup
            # is covered by the outbound-attempt audit.
            embedder = SentenceTransformerEmbedder(str(model_dir))
            store = Store(
                db_path=root / ".seekr" / "index.db",
                embed_model=embedder.model_name,
                dim=embedder.dim,
            )
            self.addCleanup(store.close)
            reranker = FakeReranker({"Timeline": 2.0})
            cfg = Config(roots=(root,), db_path=root / ".seekr" / "index.db")
            report = Indexer(store, embedder, cfg).sync([root])
            hits = Searcher(store, embedder, reranker).search(
                "xylophone deadline", k=5, mode="hybrid", rerank=True
            )

        self.assertEqual(report.error_count, 0)
        self.assertEqual(report.warning_count, 0)
        self.assertGreaterEqual(embedder.dim, 1)
        self.assertTrue(hits, "hybrid search should find the fixture")
        self.assertTrue(
            any(
                hit.path == resolved_path
                and hit.locator.startswith("\u00a7")
                and "Timeline" in hit.locator
                for hit in hits
            ),
            f"expected a hit at {resolved_path} with a §Timeline locator, got {hits}",
        )
        self.assertEqual(
            record.blocked + record.allowed,
            [],
            "cached-model indexing/search must make zero outbound attempts",
        )


if __name__ == "__main__":  # pragma: no cover - manual standalone run
    unittest.main()
