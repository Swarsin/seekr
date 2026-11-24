"""End-to-end acceptance tests for the local web interface backend.

Every test drives a real ``127.0.0.1`` loopback server (``create_server`` /
``run_web``).  Where the flow matters, the backend is a real
:class:`seekr.web_service.WebService` built against a temporary config file,
temporary database and generated temporary document roots: the embedding
backend is always :class:`seekr.embed.FakeEmbedder` with ``embed_model = "fake"``
in the config, so no real model cache, real config, real database, real
document root and no outbound network are ever touched.  Loopback HTTP is
permitted; outbound attempts are blocked by ``tests._cached_pipeline_netaudit``.

Tests assert the frozen contract in ``docs/web-api.md`` — reachable states,
error codes, counts, ordering and time behaviour — not implementation
internals.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import importlib.resources
import json
import os
import socket
import stat
import threading
import time
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _platform import LEGACY_DAEMON
from seekr.cli_search import get_hits
from seekr.config import load_config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer, SyncReport
from seekr.lifecycle import LifecycleLock
from seekr.locations import format_location
from seekr.models import ChunkSpec, text_sha
from seekr.store import Store, StoredChunk
from seekr.web_config import toml_string
from seekr.web_contract import LIMITS, HitDTO
from seekr.web_server import SeekrHTTPServer, create_server
from seekr.web_service import WebService

_POSIX_MODES = os.name == "posix"  # Windows has no POSIX permission bits

try:  # The offline audit hook lives next to the tests, not in a package.
    from tests import _cached_pipeline_netaudit as _netaudit
except ImportError:  # pragma: no cover - depends on the import mode
    import _cached_pipeline_netaudit as _netaudit


# --------------------------------------------------------------------------
# small 127.0.0.1-only HTTP client
# --------------------------------------------------------------------------


@dataclass
class Response:
    status: int
    headers: dict[str, list[str]]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.body)

    def header(self, name: str) -> list[str]:
        return self.headers.get(name.lower(), [])

    def one(self, name: str) -> str:
        values = self.header(name)
        assert len(values) == 1, (name, values)
        return values[0]

    def error_code(self) -> str:
        assert set(self.json()["error"]) <= {"code", "message", "fields"}
        return self.json()["error"]["code"]


def _request(
    server: Any,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 10.0,
) -> Response:
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.bound_port, timeout=timeout
    )
    try:
        connection.request(method, path, body=body, headers=dict(headers or {}))
        raw = connection.getresponse()
        data = raw.read()
    finally:
        connection.close()
    collected: dict[str, list[str]] = {}
    for name, value in raw.getheaders():
        collected.setdefault(name.lower(), []).append(value)
    return Response(raw.status, collected, data)


def _raw_exchange(
    server: Any,
    lines: list[str],
    *,
    body: bytes = b"",
    read_timeout: float = 3.0,
) -> Response:
    """Send one hand-built request over a fresh socket; parse the reply."""
    payload = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body
    with socket.create_connection(
        ("127.0.0.1", server.bound_port), timeout=read_timeout
    ) as conn:
        conn.settimeout(read_timeout)
        conn.sendall(payload)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = conn.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
    data = b"".join(chunks)
    head, separator, body_read = data.partition(b"\r\n\r\n")
    if not separator:  # pragma: no cover - defensive
        return Response(0, {}, data)
    head_lines = head.split(b"\r\n")
    status = int(head_lines[0].split()[1]) if len(head_lines[0].split()) >= 2 else 0
    headers: dict[str, list[str]] = {}
    for line in head_lines[1:]:
        name, present, value = line.partition(b":")
        if present:
            headers.setdefault(name.decode("latin-1").lower(), []).append(
                value.strip().decode("latin-1")
            )
    return Response(status, headers, body_read)


def _request_lines(server: Any, method: str, path: str, *extra: str) -> list[str]:
    return [f"{method} {path} HTTP/1.1", f"Host: 127.0.0.1:{server.bound_port}", *extra]


# --------------------------------------------------------------------------
# temp config / docs / service / server helpers
# --------------------------------------------------------------------------


def _write_config(
    tmp_path: Path,
    *,
    roots: list[Path] | None = None,
    db_path: Path | None = None,
    embed_model: str = "fake",
    raw: str = "",
    directory: str = "",
) -> Path:
    path = (
        tmp_path / (directory + "/config.toml") if directory else tmp_path / "config.toml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    db = db_path or (tmp_path / "index.db")
    lines = [
        f"embed_model = {json.dumps(embed_model)}\n",
        f"db_path = {json.dumps(str(db))}\n",
        'vector_quant = "int8"\n',
    ]
    if roots:
        lines.append(
            "roots = [" + ", ".join(json.dumps(str(root)) for root in roots) + "]\n"
        )
    path.write_text("".join(lines) + raw, encoding="utf-8")
    return path


def _root(tmp_path: Path, name: str = "documents") -> Path:
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


NOVICE_DOCS = {
    "notes.md": "The xylophone archive notes are stored in this file.\n",
    "guide.txt": "A barrel and a hinge walked into a xylophone shop.\n",
}
BATCH_WORD = "batchword"
PAUSED_INDEX_FILES = 65


def _write_docs(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _doc_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _doc_mtimes(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat().st_mtime_ns
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _make_service(
    tmp_path: Path,
    config_path: Path,
    *,
    embedder_factory: Any = None,
    **options: Any,
) -> WebService:
    factory = embedder_factory or (lambda cfg: FakeEmbedder(dim=64))
    options.setdefault("daemon_socket_path", tmp_path / "seekr.sock")
    options.setdefault("daemon_probe", lambda path: False)
    return WebService(config_path, embedder_factory=factory, **options)


@contextlib.contextmanager
def _running(service: WebService) -> Iterator[SeekrHTTPServer]:
    server = create_server(service, port=0)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.02},
        name="seekr-w4-test-serve",
        daemon=True,
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def _session_headers(server: Any) -> dict[str, str]:
    return {"Origin": server.url, "X-Seekr-Client": "1"}


def _bootstrap(server: Any) -> str:
    response = _request(server, "GET", "/api/session", headers=_session_headers(server))
    assert response.status == 200, response.text
    payload = response.json()
    assert set(payload) == {"token", "limits"}
    assert payload["limits"] == LIMITS
    assert response.one("content-type") == "application/json; charset=utf-8"
    assert response.one("cache-control") == "no-store"
    return payload["token"]


def _token(token: str) -> dict[str, str]:
    return {"X-Seekr-Token": token}


def _json_post(token: str, payload: dict[str, Any]) -> tuple[dict[str, str], bytes]:
    return (
        {"X-Seekr-Token": token, "Content-Type": "application/json"},
        json.dumps(payload).encode("utf-8"),
    )


def _empty_index(server: Any, token: str) -> Response:
    headers, body = _json_post(token, {})
    return _request(server, "POST", "/api/index", headers=headers, body=body)


def _search(server: Any, token: str, **fields: Any) -> Response:
    headers, body = _json_post(token, fields)
    return _request(server, "POST", "/api/search", headers=headers, body=body)


def _wait_job(
    server: Any,
    token: str,
    job_id: str,
    deadline: float = 30.0,
    *,
    poll_interval: float = 0.01,
) -> dict[str, Any]:
    """Poll the job endpoint over HTTP until a terminal state is reached."""
    limit = time.monotonic() + deadline
    seen: list[str] = []
    while time.monotonic() < limit:
        response = _request(
            server, "GET", f"/api/jobs/{job_id}", headers=_token(token)
        )
        assert response.status == 200, response.text
        body = response.json()
        assert set(body) == {
            "job_id",
            "operation",
            "state",
            "started_at",
            "finished_at",
            "elapsed_ms",
            "report",
            "error",
        }
        assert body["job_id"] == job_id
        assert body["operation"] in ("index", "model_check")
        assert body["state"] in ("running", "succeeded", "partial", "failed")
        assert body["state"] != "queued", body
        assert isinstance(body["elapsed_ms"], int) and body["elapsed_ms"] >= 0
        if body["state"] == "running":
            assert body["finished_at"] is None
            assert body["report"] is None  # no invented progress while running
        else:
            assert body["finished_at"]
            return body
        time.sleep(poll_interval)
    raise AssertionError(f"job {job_id} did not finish: {seen[-5:]}")


def _add_indexed_document(
    db_path: Path,
    *,
    model: str = "fake",
    path: str = "/docs/alpha.txt",
    text: str = "alpha xylophone notes",
) -> None:
    embedder = FakeEmbedder(dim=64)
    store = Store(db_path, model, 64, vector_quant="int8")
    try:
        doc_id = store.upsert_document(path, 1, len(text), text_sha(text), "text")
        store.replace_chunks(
            doc_id,
            [
                StoredChunk(
                    spec=ChunkSpec(text=text, locator="L1", loc_start=1.0),
                    text_sha=text_sha(text),
                    vector=embedder.embed_documents([text])[0],
                )
            ],
        )
    finally:
        store.close()


class _PausingRealIndexer(Indexer):
    """A real ``Indexer.sync`` paused after its first committed batch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.paused = threading.Event()
        self.release = threading.Event()
        self._paused_once = False

    def _flush_batch(self, batch: Any, report: SyncReport, **kwargs: Any) -> None:
        super()._flush_batch(batch, report, **kwargs)
        if not self._paused_once:
            self._paused_once = True
            self.paused.set()
            if not self.release.wait(15.0):
                raise RuntimeError("paused indexer was not released")


class _StaticIndexer:
    """A stand-in whose ``sync`` returns fixed reports, one per call."""

    def __init__(self, reports: list[SyncReport]) -> None:
        self.reports = list(reports)

    def sync(self, roots: Any) -> SyncReport:
        del roots
        if not self.reports:
            raise RuntimeError("no more static reports")
        return self.reports.pop(0)


class _GatedEmbedder:
    """An embedder whose construction blocks until released."""

    def __init__(self, entered: threading.Event, gate: threading.Event) -> None:
        self.entered = entered
        self.gate = gate
        self.embedder = FakeEmbedder(dim=64)

    def __call__(self, cfg: Any) -> FakeEmbedder:
        del cfg
        self.entered.set()
        if not self.gate.wait(15.0):
            raise RuntimeError("gated embedder was never released")
        return self.embedder


# --------------------------------------------------------------------------
# the full novice flow
# --------------------------------------------------------------------------


def test_novice_flow_over_http_with_real_service(tmp_path: Path) -> None:
    docs = _root(tmp_path)
    _write_docs(docs, NOVICE_DOCS)
    config_path = _write_config(tmp_path, db_path=tmp_path / "index.db")
    builds: list[str] = []

    def factory(cfg: Any):
        builds.append(cfg.embed_model)
        return FakeEmbedder(dim=64)

    with _netaudit.offline_network_audit() as audit, _running(
        _make_service(tmp_path, config_path, embedder_factory=factory)
    ) as server:
        assert server.bound_port > 0
        assert server.url == f"http://127.0.0.1:{server.bound_port}"

        # -- session bootstrap
        token = _bootstrap(server)

        # -- status before anything is configured or indexed
        response = _request(server, "GET", "/api/status", headers=_token(token))
        assert response.status == 200, response.text
        snapshot = response.json()
        assert set(snapshot) == {"index", "model", "daemon", "config", "activity"}
        assert snapshot["index"]["state"] == "missing"
        assert snapshot["index"]["documents"] == 0
        assert snapshot["index"]["chunks"] == 0
        assert snapshot["model"] == {
            "state": "unchecked",
            "model": "fake",
            "dim": None,
            "message": None,
        }
        assert snapshot["daemon"] == {
            "present": False,
            # Off Linux there is no legacy daemon, so no socket to report.
            "socket": str(tmp_path / "seekr.sock") if LEGACY_DAEMON else None,
        }
        assert snapshot["config"]["state"] == "ok"
        assert snapshot["activity"] == {"lane": "idle", "job_id": None}
        assert builds == [], "status must never load a model"

        # -- roots: absent key and the raw-bytes revision
        response = _request(server, "GET", "/api/roots", headers=_token(token))
        assert response.status == 200, response.text
        view = response.json()
        assert view["roots"] == []
        assert view["config_path"] == str(config_path)
        assert view["daemon_restart_required"] is False
        assert view["warning"] is None
        revision = view["revision"]
        assert revision == hashlib.sha256(config_path.read_bytes()).hexdigest()

        # -- add an existing generated folder; source files are untouched
        hashes_before = _doc_hashes(docs)
        mtimes_before = _doc_mtimes(docs)
        headers, raw = _json_post(token, {"path": str(docs), "revision": revision})
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 200, response.text
        view = response.json()
        assert len(view["roots"]) == 1
        assert Path(view["roots"][0]).resolve() == docs.resolve()
        assert view["config_path"] == str(config_path)
        assert view["daemon_restart_required"] is False
        assert view["warning"] is None
        new_revision = view["revision"]
        assert new_revision == hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert new_revision != revision

        # -- index job targets only the validated config roots
        response = _empty_index(server, token)
        assert response.status == 202, response.text
        job_id = response.json()["job_id"]
        finished = _wait_job(server, token, job_id)
        assert finished["operation"] == "index"
        assert finished["state"] == "succeeded"
        assert finished["error"] is None
        assert finished["report"] == {
            "added": 2,
            "updated": 0,
            "unchanged": 0,
            "removed": 0,
            "skipped": 0,
            "errors": [],
            "warnings": [],
            "error_total": 0,
            "warning_total": 0,
            "errors_truncated": False,
            "warnings_truncated": False,
        }
        assert _doc_hashes(docs) == hashes_before
        assert _doc_mtimes(docs) == mtimes_before

        # -- status after indexing
        response = _request(server, "GET", "/api/status", headers=_token(token))
        snapshot = response.json()
        assert snapshot["index"]["state"] == "ready"
        assert snapshot["index"]["documents"] == 2
        assert snapshot["index"]["chunks"] >= 2
        assert snapshot["index"]["size_bytes"] > 0
        assert snapshot["index"]["last_error_count"] == 0
        assert snapshot["index"]["model"] == "fake"
        assert snapshot["index"]["dim"] == 64
        assert snapshot["index"]["vector_quant"] == "int8"
        assert snapshot["model"]["state"] == "ready"
        assert snapshot["model"]["dim"] == 64
        assert snapshot["model"]["message"] is None
        assert snapshot["activity"] == {"lane": "idle", "job_id": None}

        # -- search: keyword, hybrid and vector
        modes = ("keyword", "hybrid", "vector")
        results: dict[str, list[dict[str, Any]]] = {}
        for mode in modes:
            response = _search(
                server, token, query="xylophone", mode=mode, k=5, per_file=2
            )
            assert response.status == 200, response.text
            payload = response.json()
            assert set(payload) == {"hits", "took_ms"}
            assert isinstance(payload["took_ms"], (int, float))
            assert payload["took_ms"] >= 0
            hits = payload["hits"]
            assert hits
            assert {hit["kind"] for hit in hits} == {"markdown", "text"}
            for hit in hits:
                assert set(hit) == {
                    "chunk_id",
                    "path",
                    "locator",
                    "loc_start",
                    "snippet",
                    "score",
                    "kind",
                    "module",
                    "week",
                    "highlights",
                    "location",
                    "posix_location",
                }
                assert Path(hit["path"]).resolve() in (
                    docs / "notes.md",
                    docs / "guide.txt",
                )
                # the locator may legitimately be empty for these passages;
                # keyword parity with `get_hits` above asserts its value
                assert hit["snippet"]
                assert "explain" not in hit
            results[mode] = hits

        # -- keyword parity with the shared local path, including ordering
        cfg = load_config(config_path)
        expected = get_hits(
            "xylophone",
            mode="keyword",
            k=5,
            per_file=2,
            cfg=cfg,
            embedder=FakeEmbedder(dim=64),
            use_daemon=False,
        )
        assert results["keyword"] == [
            HitDTO(
                chunk_id=hit.chunk_id,
                path=hit.path,
                locator=hit.locator,
                loc_start=hit.loc_start,
                snippet=hit.snippet,
                score=hit.score,
                kind=hit.kind,
                module=hit.module,
                week=hit.week,
                highlights=hit.highlights,
                location=format_location(hit.path, hit.locator, windows_path=None),
                posix_location=format_location(hit.path, hit.locator, windows_path=None),
            ).to_dict()
            for hit in expected
        ]
        assert [hit["path"] for hit in results["keyword"]] == [
            str(hit.path) for hit in expected
        ]

        # -- model check job through its job endpoint
        headers, raw = _json_post(token, {})
        response = _request(
            server, "POST", "/api/model/check", headers=headers, body=raw
        )
        assert response.status == 202, response.text
        check_id = response.json()["job_id"]
        finished = _wait_job(server, token, check_id)
        assert finished["operation"] == "model_check"
        assert finished["state"] == "succeeded"
        assert finished["report"] == {"model": "fake", "dim": 64}
        assert finished["error"] is None

        # -- the finished job is still served by the job endpoint
        response = _request(
            server, "GET", f"/api/jobs/{job_id}", headers=_token(token)
        )
        assert response.status == 200, response.text
        assert response.json()["state"] == "succeeded"

    # exactly two builds: the index job and the explicit model check share the
    # lane but not the cache; every search reuses the cached backend
    assert len(builds) == 2 and set(builds) == {"fake"}, builds
    assert audit.blocked == [], audit.blocked
    allowed = dict(audit.allowed)
    assert allowed, "loopback traffic should have been recorded as allowed"
    for event, target in audit.allowed:
        if event == "socket.getaddrinfo":
            assert target in ("127.0.0.1", "localhost"), target
        else:
            assert _netaudit.is_local_target(target), target


# --------------------------------------------------------------------------
# status states and search preflight
# --------------------------------------------------------------------------


def test_status_index_states_and_search_preflight_over_http(tmp_path: Path) -> None:
    builds: list[str] = []

    def explode(cfg: Any):
        builds.append(cfg.embed_model)
        raise AssertionError("status and search preflight must never load a model")

    # -- missing: nothing exists yet; every search mode returns empty hits
    missing_db = tmp_path / "missing" / "index.db"
    config_path = _write_config(tmp_path, db_path=missing_db)
    with _running(
        _make_service(tmp_path, config_path, embedder_factory=explode)
    ) as server:
        token = _bootstrap(server)
        response = _request(server, "GET", "/api/status", headers=_token(token))
        assert response.status == 200, response.text
        snapshot = response.json()
        assert snapshot["index"]["state"] == "missing"
        assert snapshot["index"]["documents"] == 0
        assert snapshot["index"]["chunks"] == 0
        assert snapshot["model"]["state"] == "unchecked"
        assert not missing_db.exists()
        for mode in ("keyword", "hybrid", "vector"):
            response = _search(server, token, query="xylophone", mode=mode)
            assert response.status == 200, response.text
            payload = response.json()
            assert payload["hits"] == []
            assert payload["took_ms"] >= 0
        assert builds == []

    # -- empty: the database exists but holds no documents
    empty_db = tmp_path / "empty" / "index.db"
    empty_db.parent.mkdir(parents=True, exist_ok=True)
    Store(empty_db, "fake", 64, vector_quant="int8").close()
    with _running(
        _make_service(
            tmp_path,
            _write_config(tmp_path, db_path=empty_db),
            embedder_factory=explode,
        )
    ) as server:
        token = _bootstrap(server)
        response = _request(server, "GET", "/api/status", headers=_token(token))
        snapshot = response.json()
        assert snapshot["index"]["state"] == "empty"
        assert snapshot["index"]["documents"] == 0
        assert snapshot["index"]["chunks"] == 0
        assert snapshot["index"]["schema_version"] is not None
        for mode in ("keyword", "hybrid", "vector"):
            response = _search(server, token, query="xylophone", mode=mode)
            assert response.status == 200, response.text
            assert response.json()["hits"] == []
        assert builds == []

    # -- ready: an indexed document is searchable without loading a model
    ready_db = tmp_path / "ready" / "index.db"
    _add_indexed_document(ready_db, path="/docs/alpha.txt")
    with _running(
        _make_service(
            tmp_path,
            _write_config(tmp_path, db_path=ready_db),
            embedder_factory=explode,
        )
    ) as server:
        token = _bootstrap(server)
        response = _request(server, "GET", "/api/status", headers=_token(token))
        snapshot = response.json()
        assert snapshot["index"]["state"] == "ready"
        assert snapshot["index"]["documents"] == 1
        assert snapshot["index"]["chunks"] == 1
        response = _search(server, token, query="xylophone", mode="keyword")
        assert response.status == 200, response.text
        assert [hit["path"] for hit in response.json()["hits"]] == ["/docs/alpha.txt"]
        assert builds == [], "keyword search must never load a model"

    # -- incompatible: a stored model mismatch is refused, never rebuilt
    mismatch_db = tmp_path / "mismatch" / "index.db"
    _add_indexed_document(
        mismatch_db, model="different/model", path="/docs/beta.txt"
    )
    with _running(
        _make_service(
            tmp_path,
            _write_config(
                tmp_path,
                db_path=mismatch_db,
                roots=[_root(tmp_path, "mismatch-root")],
            ),
            embedder_factory=explode,
        )
    ) as server:
        token = _bootstrap(server)
        response = _request(server, "GET", "/api/status", headers=_token(token))
        assert response.status == 200, response.text
        snapshot = response.json()
        assert snapshot["index"]["state"] == "incompatible"
        assert snapshot["index"]["message"]
        for mode in ("keyword", "hybrid", "vector"):
            response = _search(server, token, query="xylophone", mode=mode)
            assert response.status == 409, (mode, response.text)
            assert response.error_code() == "incompatible_index"
        response = _empty_index(server, token)
        assert response.status == 409
        assert response.error_code() == "incompatible_index"
        assert builds == []

    # -- error: a corrupt database stays a 200 status with index.state error,
    #    and search/index preflight maps it to 503 index_unavailable
    corrupt_db = tmp_path / "corrupt" / "index.db"
    corrupt_db.parent.mkdir(parents=True, exist_ok=True)
    corrupt_db.write_bytes(b"this is not a sqlite database")
    with _running(
        _make_service(
            tmp_path,
            _write_config(
                tmp_path,
                db_path=corrupt_db,
                roots=[_root(tmp_path, "corrupt-root")],
            ),
            embedder_factory=explode,
        )
    ) as server:
        token = _bootstrap(server)
        response = _request(server, "GET", "/api/status", headers=_token(token))
        assert response.status == 200, response.text
        snapshot = response.json()
        assert snapshot["index"]["state"] == "error"
        assert snapshot["index"]["message"]
        assert snapshot["index"]["documents"] is None
        for mode in ("keyword", "hybrid", "vector"):
            response = _search(server, token, query="xylophone", mode=mode)
            assert response.status == 503, (mode, response.text)
            assert response.error_code() == "index_unavailable"
        response = _empty_index(server, token)
        assert response.status == 503
        assert response.error_code() == "index_unavailable"


def test_status_config_model_and_daemon_states_over_http(tmp_path: Path) -> None:
    builds: list[str] = []

    def failing(cfg: Any):
        builds.append(cfg.embed_model)
        raise RuntimeError("model files are missing")

    # -- config ok / model unchecked / daemon absent, on a ready index so the
    #    later hybrid/vector preflight actually reaches the model check
    first_db = tmp_path / "first" / "index.db"
    _add_indexed_document(first_db, path="/docs/alpha.txt")
    config_path = _write_config(tmp_path, db_path=first_db)
    with _running(_make_service(tmp_path, config_path, embedder_factory=failing)) as server:
        token = _bootstrap(server)
        response = _request(server, "GET", "/api/status", headers=_token(token))
        snapshot = response.json()
        assert snapshot["config"]["state"] == "ok"
        assert snapshot["config"]["path"] == str(config_path)
        assert snapshot["config"]["revision"] == hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest()
        assert snapshot["config"]["roots"] == 0
        assert snapshot["config"]["message"] is None
        assert snapshot["daemon"]["present"] is False
        assert snapshot["model"]["state"] == "unchecked"

        # a failed model check is the only retry path away from unavailable
        headers, raw = _json_post(token, {})
        response = _request(
            server, "POST", "/api/model/check", headers=headers, body=raw
        )
        assert response.status == 202, response.text
        check_id = response.json()["job_id"]
        finished = _wait_job(server, token, check_id)
        assert finished["state"] == "failed"
        assert finished["report"] is None
        assert finished["error"] == "model files are missing"
        assert builds == ["fake"]

        response = _request(server, "GET", "/api/status", headers=_token(token))
        snapshot = response.json()
        assert snapshot["model"]["state"] == "unavailable"
        assert snapshot["model"]["model"] == "fake"
        assert snapshot["model"]["dim"] is None
        assert snapshot["model"]["message"] == "model files are missing"

        # hybrid/vector are refused without retrying; keyword never needs it
        for mode in ("hybrid", "vector"):
            response = _search(server, token, query="xylophone", mode=mode)
            assert response.status == 503, (mode, response.text)
            assert response.error_code() == "model_unavailable"
        assert builds == ["fake"], "model_unavailable must never retry"

    # -- keyword still works with an unavailable model on a ready index
    ready_db = tmp_path / "keyword" / "index.db"
    _add_indexed_document(ready_db, path="/docs/kw.txt")
    with _running(
        _make_service(
            tmp_path,
            _write_config(tmp_path, db_path=ready_db),
            embedder_factory=failing,
        )
    ) as server:
        token = _bootstrap(server)
        response = _search(server, token, query="xylophone", mode="keyword")
        assert response.status == 200, response.text
        assert [hit["path"] for hit in response.json()["hits"]] == ["/docs/kw.txt"]
        assert builds == ["fake"], builds

    if LEGACY_DAEMON:  # the daemon only exists on Linux
        # -- a live daemon socket is reported; absence is not
        config_path = _write_config(tmp_path, roots=[_root(tmp_path, "daemon-root")])
        with _running(
            _make_service(tmp_path, config_path, daemon_probe=lambda path: True)
        ) as server:
            token = _bootstrap(server)
            response = _request(server, "GET", "/api/status", headers=_token(token))
            snapshot = response.json()
            assert snapshot["daemon"]["present"] is True
            assert snapshot["daemon"]["socket"] == str(tmp_path / "seekr.sock")
            response = _empty_index(server, token)
            assert response.status == 409
            assert response.error_code() == "daemon_active"
            assert "seekr daemon" in response.json()["error"]["message"]
            revision = _request(
                server, "GET", "/api/roots", headers=_token(token)
            ).json()["revision"]
            extra = _root(tmp_path, "daemon-extra")
            headers, raw = _json_post(token, {"path": str(extra), "revision": revision})
            response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
            assert response.status == 200, response.text
            view = response.json()
            assert view["daemon_restart_required"] is False
            assert view["warning"] is None
            assert len(view["roots"]) == 2
            assert Path(view["roots"][0]).resolve() == (tmp_path / "daemon-root").resolve()
            assert Path(view["roots"][1]).resolve() == extra.resolve()


def test_status_skips_the_lifecycle_lock_and_keyword_maps_busy_over_http(
    tmp_path: Path,
) -> None:
    ready_db = tmp_path / "index.db"
    _add_indexed_document(ready_db, path="/docs/alpha.txt")
    config_path = _write_config(tmp_path, db_path=ready_db)

    lock = LifecycleLock(ready_db, shared=False, blocking=False)
    lock.acquire()
    try:
        with _running(_make_service(tmp_path, config_path)) as server:
            token = _bootstrap(server)
            # status reads outside the shared lifecycle lock
            started = time.monotonic()
            response = _request(server, "GET", "/api/status", headers=_token(token))
            elapsed = time.monotonic() - started
            assert response.status == 200, response.text
            assert elapsed < 2.0
            assert response.json()["index"]["state"] == "ready"

            # keyword waits within its bounded budget, then reports busy
            response = _search(server, token, query="xylophone", mode="keyword")
            assert response.status == 409, response.text
            assert response.error_code() == "busy"
    finally:
        lock.release()

    with _running(_make_service(tmp_path, config_path)) as server:
        token = _bootstrap(server)
        response = _search(server, token, query="xylophone", mode="keyword")
        assert response.status == 200, response.text
        assert response.json()["hits"]


def test_model_loading_lane_and_busy_second_engine_operation_over_http(
    tmp_path: Path,
) -> None:
    docs = _root(tmp_path)
    _write_docs(docs, NOVICE_DOCS)
    ready_db = tmp_path / "ready.db"
    _add_indexed_document(ready_db, path="/docs/alpha.txt")
    config_path = _write_config(tmp_path, roots=[docs], db_path=ready_db)
    entered = threading.Event()
    gate = threading.Event()
    gated = _GatedEmbedder(entered, gate)
    builds: list[FakeEmbedder] = []

    def factory(cfg: Any):
        builds.append(gated(cfg))
        return builds[-1]

    with _running(_make_service(tmp_path, config_path, embedder_factory=factory)) as server:
        token = _bootstrap(server)

        results: list[Response | None] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def hybrid() -> None:
            try:
                results.append(_search(server, token, query="xylophone", mode="hybrid"))
            except BaseException as exc:  # noqa: BLE001 - collected below
                with lock:
                    errors.append(exc)

        started_at = time.monotonic()
        client = threading.Thread(target=hybrid, daemon=False)
        client.start()
        assert entered.wait(10.0)

        # -- status reports loading + lane search without waiting behind it
        response = _request(server, "GET", "/api/status", headers=_token(token))
        elapsed = time.monotonic() - started_at
        assert response.status == 200, response.text
        assert elapsed < 2.0
        snapshot = response.json()
        assert snapshot["model"]["state"] == "loading"
        assert snapshot["model"]["message"] is None
        assert snapshot["index"]["state"] == "ready"  # a read-only inspection
        assert snapshot["activity"] == {"lane": "search", "job_id": None}

        # -- concurrent searches wait for the one model load, never 409
        waiting: list[threading.Thread] = []
        for mode in ("hybrid", "vector"):

            def concurrent(mode: str = mode) -> None:
                try:
                    results.append(_search(server, token, query="xylophone", mode=mode))
                except BaseException as exc:  # noqa: BLE001 - collected below
                    with lock:
                        errors.append(exc)

            waiting.append(threading.Thread(target=concurrent, daemon=False))
            waiting[-1].start()

        # -- other engine operations get an immediate 409 busy
        response = _empty_index(server, token)
        assert response.status == 409
        assert response.error_code() == "busy"
        headers, raw = _json_post(token, {})
        response = _request(
            server,
            "POST",
            "/api/model/check",
            headers=headers,
            body=raw,
        )
        assert response.status == 409
        assert response.error_code() == "busy"
        headers, raw = _json_post(
            token, {"path": str(tmp_path / "busy-extra"), "revision": "stale"}
        )
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 409
        assert response.error_code() == "busy"

        # -- keyword bypasses the lane while the first hybrid is loading
        response = _search(server, token, query="xylophone", mode="keyword")
        assert response.status == 200, response.text
        assert response.json()["hits"]

        gate.set()
        for thread in (client, *waiting):
            thread.join(timeout=20.0)
            assert not thread.is_alive()

        # -- the released searches returned hits and the model is ready + cached
        assert errors == []
        assert len(results) == 3
        assert all(item is not None and item.status == 200 for item in results)
        response = _request(server, "GET", "/api/status", headers=_token(token))
        snapshot = response.json()
        assert snapshot["model"]["state"] == "ready"
        assert snapshot["model"]["dim"] == 64
        assert snapshot["activity"] == {"lane": "idle", "job_id": None}

        # -- a second hybrid search works with the warm, cached backend
        response = _search(server, token, query="xylophone", mode="hybrid")
        assert response.status == 200, response.text

    assert builds and len(builds) == 1, builds


# --------------------------------------------------------------------------
# search error mapping
# --------------------------------------------------------------------------


def test_search_error_mapping_over_http(tmp_path: Path) -> None:
    builds: list[str] = []

    def explode(cfg: Any):
        builds.append(cfg.embed_model)
        raise AssertionError("empty and missing preflights must not load a model")

    # -- missing and empty short-circuit to 200 empty hits for every mode
    for name in ("missing", "empty"):
        db = tmp_path / name / "index.db"
        if name == "empty":
            db.parent.mkdir(parents=True, exist_ok=True)
            Store(db, "fake", 64, vector_quant="int8").close()
        with _running(
            _make_service(
                tmp_path, _write_config(tmp_path, db_path=db), embedder_factory=explode
            )
        ) as server:
            token = _bootstrap(server)
            for mode in ("keyword", "hybrid", "vector"):
                response = _search(server, token, query="xylophone", mode=mode)
                assert response.status == 200, (name, mode, response.text)
                assert response.json()["hits"] == []
        assert builds == []

    # -- incompatible_index for a mismatched model, for every mode
    mismatch_db = tmp_path / "mismatch" / "index.db"
    _add_indexed_document(mismatch_db, model="different/model", path="/docs/m.txt")
    with _running(
        _make_service(
            tmp_path,
            _write_config(
                tmp_path,
                db_path=mismatch_db,
                roots=[_root(tmp_path, "mismatch-root")],
            ),
            embedder_factory=explode,
        )
    ) as server:
        token = _bootstrap(server)
        for mode in ("keyword", "hybrid", "vector"):
            response = _search(server, token, query="xylophone", mode=mode)
            assert response.status == 409, (mode, response.text)
            assert response.error_code() == "incompatible_index"
        response = _empty_index(server, token)
        assert response.status == 409
        assert response.error_code() == "incompatible_index"
    assert builds == []

    # -- index_unavailable for a corrupt database, for every mode
    corrupt_db = tmp_path / "corrupt" / "index.db"
    corrupt_db.parent.mkdir(parents=True, exist_ok=True)
    corrupt_db.write_bytes(b"not a sqlite database")
    with _running(
        _make_service(
            tmp_path,
            _write_config(
                tmp_path,
                db_path=corrupt_db,
                roots=[_root(tmp_path, "corrupt-root")],
            ),
            embedder_factory=explode,
        )
    ) as server:
        token = _bootstrap(server)
        for mode in ("keyword", "hybrid", "vector"):
            response = _search(server, token, query="xylophone", mode=mode)
            assert response.status == 503, (mode, response.text)
            assert response.error_code() == "index_unavailable"
        response = _empty_index(server, token)
        assert response.status == 503
        assert response.error_code() == "index_unavailable"

    # -- model_unavailable for hybrid/vector while keyword still works
    ready_db = tmp_path / "model" / "index.db"
    _add_indexed_document(ready_db, path="/docs/m.txt")
    attempts: list[str] = []

    def failing(cfg: Any):
        attempts.append(cfg.embed_model)
        raise RuntimeError("model files are missing")

    with _running(
        _make_service(
            tmp_path, _write_config(tmp_path, db_path=ready_db), embedder_factory=failing
        )
    ) as server:
        token = _bootstrap(server)
        headers, raw = _json_post(token, {})
        response = _request(
            server, "POST", "/api/model/check", headers=headers, body=raw
        )
        assert response.status == 202, response.text
        _wait_job(server, token, response.json()["job_id"])
        for mode in ("hybrid", "vector"):
            response = _search(server, token, query="xylophone", mode=mode)
            assert response.status == 503, (mode, response.text)
            assert response.error_code() == "model_unavailable"
        response = _search(server, token, query="xylophone", mode="keyword")
        assert response.status == 200, response.text
        assert response.json()["hits"]
    assert attempts == ["fake"], "hybrid/vector with an unavailable model must not retry"

    if LEGACY_DAEMON:  # the daemon only exists on Linux
        # -- daemon_failure for a reachable broken daemon
        from seekr.client import DaemonError

        def broken_daemon(*args: Any, **kwargs: Any):
            del args, kwargs
            raise DaemonError("daemon protocol failed")

        config_path = _write_config(tmp_path, roots=[_root(tmp_path, "daemon-docs")])
        with _running(
            _make_service(
                tmp_path,
                config_path,
                daemon_probe=lambda path: True,
                daemon_search=broken_daemon,
            )
        ) as server:
            token = _bootstrap(server)
            for mode in ("keyword", "hybrid", "vector"):
                response = _search(server, token, query="xylophone", mode=mode)
                assert response.status == 503, (mode, response.text)
                assert response.error_code() == "daemon_failure"
            response = _search(server, token, query="xylophone")
            assert response.status == 503, response.text
            assert response.error_code() == "daemon_failure"

    # -- config_invalid for every mode, always naming the config path
    invalid = tmp_path / "invalid" / "config.toml"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text("embed_model = \"fake\"\ndb_path = \"x.db\"\nroots = [broken\n")
    with _running(_make_service(tmp_path, invalid, embedder_factory=explode)) as server:
        token = _bootstrap(server)
        response = _request(server, "GET", "/api/roots", headers=_token(token))
        assert response.status == 409
        assert response.error_code() == "config_invalid"
        assert str(invalid) in response.json()["error"]["message"]
        for mode in ("keyword", "hybrid", "vector"):
            response = _search(server, token, query="xylophone", mode=mode)
            assert response.status == 409, (mode, response.text)
            assert response.error_code() == "config_invalid"
        response = _empty_index(server, token)
        assert response.status == 409
        assert response.error_code() == "config_invalid"
        headers, raw = _json_post(token, {})
        response = _request(
            server, "POST", "/api/model/check", headers=headers, body=raw
        )
        assert response.status == 409
        assert response.error_code() == "config_invalid"

    # -- invalid_request is the body/field validation code
    with _running(_make_service(tmp_path, _write_config(tmp_path))) as server:
        token = _bootstrap(server)
        response = _search(server, token)
        assert response.status == 400
        assert response.error_code() == "invalid_request"


def test_invalid_search_filters_are_rejected_before_daemon_over_http(
    tmp_path: Path,
) -> None:
    daemon_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    probes: list[Path] = []

    def daemon_search(*args: Any, **kwargs: Any) -> list[object]:
        daemon_calls.append((args, kwargs))
        return []

    def daemon_probe(path: Path) -> bool:
        probes.append(path)
        return True

    with _running(
        _make_service(
            tmp_path,
            _write_config(tmp_path),
            daemon_probe=daemon_probe,
            daemon_search=daemon_search,
        )
    ) as server:
        token = _bootstrap(server)
        cases = (
            ("keyword", {"kind": "bogus"}, "kind"),
            ("hybrid", {"kind": "bogus"}, "kind"),
            ("keyword", {"since": "7x"}, "since"),
            ("hybrid", {"since": "7x"}, "since"),
            ("keyword", {"since": "0d"}, "since"),
            ("hybrid", {"since": "0d"}, "since"),
        )
        for mode, fields, field_name in cases:
            response = _search(
                server, token, query="xylophone", mode=mode, **fields
            )
            assert response.status == 400, (mode, fields, response.text)
            assert response.error_code() == "invalid_request"
            error = response.json()["error"]
            assert set(error) <= {"code", "message", "fields"}
            assert isinstance(error["fields"][field_name], str)
            assert error["fields"][field_name]

        # the filters are validated before a live daemon is even probed
        assert probes == []
        assert daemon_calls == []


# --------------------------------------------------------------------------
# engine busy and keyword during a real paused index
# --------------------------------------------------------------------------


def test_keyword_and_lane_conflicts_during_real_paused_index_over_http(
    tmp_path: Path,
) -> None:
    docs = _root(tmp_path)
    _write_docs(
        docs,
        {
            f"{BATCH_WORD}-{index:03d}.txt": f"{BATCH_WORD} item {index}\n"
            for index in range(PAUSED_INDEX_FILES)
        },
    )
    config_path = _write_config(tmp_path, roots=[docs])
    created: list[_PausingRealIndexer] = []

    def factory(store: Any, embedder: Any, cfg: Any) -> _PausingRealIndexer:
        del embedder
        created.append(_PausingRealIndexer(store, FakeEmbedder(dim=64), cfg))
        return created[-1]

    with _running(_make_service(tmp_path, config_path, indexer_factory=factory)) as server:
        token = _bootstrap(server)
        response = _empty_index(server, token)
        assert response.status == 202, response.text
        job_id = response.json()["job_id"]

        # -- the real Indexer.sync pauses deterministically after its first
        #    committed batch (64 files); the job thread constructs it
        #    asynchronously, so wait for both construction and the pause.
        construct_deadline = time.monotonic() + 10.0
        while not created and time.monotonic() < construct_deadline:
            time.sleep(0.01)
        assert created, "the real indexer was never constructed"
        assert created[0].paused.wait(10.0), "the indexer never paused"

        deadline = time.monotonic() + 10.0
        activity: dict[str, Any] = {}
        while time.monotonic() < deadline:
            response = _request(server, "GET", "/api/status", headers=_token(token))
            current = response.json()["activity"]
            if current["lane"] == "index":
                activity.update(current)
                break
            time.sleep(0.01)
        assert activity, "the paused index job never became visible in status"
        assert activity == {"lane": "index", "job_id": job_id}

        snapshot = _request(server, "GET", "/api/status", headers=_token(token)).json()
        assert snapshot["index"]["state"] == "ready"  # a committed batch is visible
        assert snapshot["index"]["documents"] == PAUSED_INDEX_FILES - 1

        # -- engine operations and root writes are immediately busy
        for path in ("/api/index", "/api/model/check"):
            headers, raw = _json_post(token, {})
            response = _request(server, "POST", path, headers=headers, body=raw)
            assert response.status == 409, (path, response.text)
            assert response.error_code() == "busy"
        headers, raw = _json_post(
            token, {"path": str(tmp_path / "busy-extra"), "revision": "stale"}
        )
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 409, response.text
        assert response.error_code() == "busy"

        # -- every search mode reads concurrently with the index write
        for mode in ("hybrid", "vector"):
            response = _search(server, token, query=BATCH_WORD, mode=mode)
            assert response.status == 200, (mode, response.text)
            assert response.json()["hits"]
        started = time.monotonic()
        response = _search(
            server, token, query=BATCH_WORD, mode="keyword", k=5
        )
        assert time.monotonic() - started < 2.0
        keyword_during_pause = [
            {key: hit[key] for key in hit} for hit in response.json()["hits"]
        ]

        # -- parity with the local keyword path over the same committed state
        cfg = load_config(config_path)
        expected = get_hits(
            BATCH_WORD,
            mode="keyword",
            k=5,
            per_file=2,
            cfg=cfg,
            embedder=FakeEmbedder(dim=64),
            use_daemon=False,
        )
        assert keyword_during_pause == [
            HitDTO(
                chunk_id=hit.chunk_id,
                path=hit.path,
                locator=hit.locator,
                loc_start=hit.loc_start,
                snippet=hit.snippet,
                score=hit.score,
                kind=hit.kind,
                module=hit.module,
                week=hit.week,
                highlights=hit.highlights,
                location=format_location(hit.path, hit.locator, windows_path=None),
                posix_location=format_location(hit.path, hit.locator, windows_path=None),
            ).to_dict()
            for hit in expected
        ]

        # -- a root write is also busy even with a stale revision
        headers, raw = _json_post(
            token, {"path": str(tmp_path / "extra"), "revision": "stale"}
        )
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 409
        assert response.error_code() == "busy"

        # -- release the writer; the job finishes with the real report
        created[0].release.set()
        _wait_job(server, token, job_id)
        finished = _request(
            server, "GET", f"/api/jobs/{job_id}", headers=_token(token)
        ).json()
        assert finished["state"] == "succeeded"
        assert finished["report"]["added"] == PAUSED_INDEX_FILES
        assert finished["report"]["unchanged"] == 0
        response = _request(server, "GET", "/api/status", headers=_token(token))
        snapshot = response.json()
        assert snapshot["index"]["state"] == "ready"
        assert snapshot["index"]["documents"] == PAUSED_INDEX_FILES
        assert snapshot["activity"] == {"lane": "idle", "job_id": None}


# --------------------------------------------------------------------------
# TOML/root flows over HTTP
# --------------------------------------------------------------------------


def test_toml_root_flows_over_http(tmp_path: Path) -> None:
    builds: list[str] = []

    def explode(cfg: Any):
        builds.append(cfg.embed_model)
        raise AssertionError("root flows must never load a model")

    # -- shape (a): an absent top-level roots key in a flat file appends at EOF
    text = "# keep the model comment and blank line\n\n"
    config_path = _write_config(
        tmp_path, raw=text, db_path=tmp_path / "a" / "index.db"
    )
    original_bytes = config_path.read_bytes()
    if _POSIX_MODES:
        assert config_path.stat().st_mode & 0o777 == 0o644  # existing modes are preserved
    docs = _root(tmp_path, "first-root")
    _write_docs(docs, NOVICE_DOCS)
    with _running(_make_service(tmp_path, config_path, embedder_factory=explode)) as server:
        token = _bootstrap(server)
        response = _request(server, "GET", "/api/roots", headers=_token(token))
        assert response.status == 200, response.text
        view = response.json()
        assert view["roots"] == []
        assert view["config_path"] == str(config_path)
        revision = view["revision"]
        assert revision == hashlib.sha256(original_bytes).hexdigest()
        headers, raw = _json_post(token, {"path": str(docs), "revision": revision})
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 200, response.text
        view = response.json()
        canonical = str(docs.resolve())

        # every byte outside the appended span is preserved, including the
        # key spelling, assignment whitespace, trailing comment and line ending
        expected = original_bytes + f"roots = [{toml_string(canonical)}]\n".encode()
        assert config_path.read_bytes() == expected

        # a tomllib + load_config round-trip equality
        assert view["roots"] == [canonical]
        assert view["revision"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert tomllib.loads(config_path.read_text())["roots"] == [canonical]
        assert load_config(config_path).roots == (Path(canonical),)

        # the refreshed roots snapshot shows the stored name, form and revision
        response = _request(server, "GET", "/api/roots", headers=_token(token))
        view = response.json()
        assert view["roots"] == [canonical]
        assert view["revision"] != revision
        assert builds == []

    # -- shape (b): a single-line array with a trailing comment and CRLF endings:
    #    only the array value span is replaced; nothing else changes at all.
    text = (
        'db_path = "b.db"\r\n'
        "# leading comment\r\n"
        'roots = ["/tmp/opencode/a-original"] # first root\r\n'
        "# trailing comment\r\n"
    )
    config_path = tmp_path / "crlf" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(text.encode())
    original_bytes = config_path.read_bytes()
    second = _root(tmp_path, "second-root")
    _write_docs(second, NOVICE_DOCS)
    with _running(_make_service(tmp_path, config_path, embedder_factory=explode)) as server:
        token = _bootstrap(server)
        revision = _request(server, "GET", "/api/roots", headers=_token(token)).json()[
            "revision"
        ]
        headers, raw = _json_post(token, {"path": str(second), "revision": revision})
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 200, response.text
        view = response.json()
        canonical = str(second.resolve())

        start = text.index('["/tmp/opencode/a-original"]')
        end = text.index("]", start) + 1
        expected = (
            text[:start] + f'["/tmp/opencode/a-original", {toml_string(canonical)}]' + text[end:]
        ).encode()
        assert config_path.read_bytes() == expected
        assert view["roots"] == ["/tmp/opencode/a-original", canonical]
        assert view["revision"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
        loaded = load_config(config_path)
        assert loaded.roots == (
            Path("/tmp/opencode/a-original"),
            second.resolve(),
        )

    # -- TOML-safe escaping of quotes, backslashes, Unicode and tabs
    weird = tmp_path / 'we"ird\\dir-café\tswirl'
    weird.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(tmp_path)
    with _running(_make_service(tmp_path, config_path)) as server:
        token = _bootstrap(server)
        revision = _request(server, "GET", "/api/roots", headers=_token(token)).json()[
            "revision"
        ]
        headers, raw = _json_post(token, {"path": str(weird), "revision": revision})
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 200, response.text
        view = response.json()
        canonical = view["roots"][0]
        assert canonical == str(weird.resolve())
        assert tomllib.loads(config_path.read_text())["roots"] == [canonical]
        assert load_config(config_path).roots == (weird.resolve(),)

    # -- duplicate detection, also through a symlink alias
    alias = tmp_path / "alias"
    alias.symlink_to(docs)
    config_path = _write_config(tmp_path, roots=[docs])
    with _running(_make_service(tmp_path, config_path)) as server:
        token = _bootstrap(server)
        revision = _request(server, "GET", "/api/roots", headers=_token(token)).json()[
            "revision"
        ]
        for submitted in (str(docs), str(alias)):
            headers, raw = _json_post(token, {"path": submitted, "revision": revision})
            response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
            assert response.status == 409, submitted
            assert response.error_code() == "duplicate_root"
            assert response.json()["error"]["fields"]["path"]
        assert builds == []

    # -- the 50-root limit is roots_limit and the config is untouched
    fifty = [tmp_path / f"root-{index}" for index in range(LIMITS["roots_max"])]
    already = ", ".join(json.dumps(str(path)) for path in fifty)
    config_path = _write_config(
        tmp_path, raw=f"roots = [{already}]\n", directory="limit"
    )
    original_bytes = config_path.read_bytes()
    extra = _root(tmp_path, "extra")
    with _running(_make_service(tmp_path, config_path, embedder_factory=explode)) as server:
        token = _bootstrap(server)
        revision = _request(server, "GET", "/api/roots", headers=_token(token)).json()[
            "revision"
        ]
        headers, raw = _json_post(token, {"path": str(extra), "revision": revision})
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 400
        assert response.error_code() == "roots_limit"
        assert config_path.read_bytes() == original_bytes
        assert builds == []

    # -- unsupported and ambiguous TOML shapes are refused for writes, naming
    #    the path; shapes that remain readable configs are still reported by GET
    cases = {
        "multiline": 'embed_model = "fake"\nroots = [\n"a",\n"b",\n]\n',
        "internal-comment": 'embed_model = "fake"\nroots = ["a", # hmm\n"b"]\n',
        "quoted": 'embed_model = "fake"\n"roots" = ["a"]\n',
        "dotted": 'embed_model = "fake"\na.roots = ["a"]\n',
        "table": 'embed_model = "fake"\n[roots]\nfiles = 1\n',
        "array-of-tables": 'embed_model = "fake"\n[[roots]]\n',
        "inside-table": 'embed_model = "fake"\n[tool]\nroots = ["a"]\n',
        "table-and-absent": 'embed_model = "fake"\n[tool]\n',
    }
    readable_shapes = {
        "multiline": ["a", "b"],
        "internal-comment": ["a", "b"],
        "quoted": ["a"],
    }
    for name, shape in cases.items():
        directory = "invalid-" + name
        config_path = tmp_path / directory / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(shape.encode())
        with _running(
            _make_service(tmp_path, config_path, embedder_factory=explode)
        ) as server:
            token = _bootstrap(server)
            response = _request(server, "GET", "/api/roots", headers=_token(token))
            if name in readable_shapes:
                assert response.status == 200, (name, response.text)
                view = response.json()
                assert view["roots"] == readable_shapes[name]
                assert view["config_path"] == str(config_path)
                assert view["revision"] == hashlib.sha256(shape.encode()).hexdigest()
            else:
                assert response.status == 409, (name, response.text)
                assert response.error_code() == "config_invalid"
                assert str(config_path) in response.json()["error"]["message"]
            revision = hashlib.sha256(shape.encode()).hexdigest()
            headers, raw = _json_post(
                token, {"path": str(extra), "revision": revision}
            )
            response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
            assert response.status == 409, (name, response.text)
            assert response.error_code() == "config_invalid"
            assert str(config_path) in response.json()["error"]["message"]
            assert config_path.read_text() == shape
        assert tomllib.loads(shape)  # each refusal is valid TOML, not a parse error
    assert builds == []

    # -- missing config creation: 0700 parent, 0600 file, roots only
    created = tmp_path / "created/deep/config.toml"
    parent = created.parent
    assert not parent.exists()
    fresh = _root(tmp_path, "fresh-root")
    _write_docs(fresh, NOVICE_DOCS)
    with _running(_make_service(tmp_path, created)) as server:
        token = _bootstrap(server)
        revision = _request(server, "GET", "/api/roots", headers=_token(token)).json()[
            "revision"
        ]
        headers, raw = _json_post(token, {"path": str(fresh), "revision": revision})
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 200, response.text
        view = response.json()
        assert view["config_path"] == str(created)
    assert created.is_file()
    if _POSIX_MODES:
        assert stat.S_IMODE(parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(created.stat().st_mode) == 0o600
    assert created.read_text() == f"roots = [{toml_string(str(fresh.resolve()))}]\n"
    assert tomllib.loads(created.read_text())["roots"] == [str(fresh.resolve())]
    assert load_config(created).roots == (fresh.resolve(),)

    # -- an existing non-0600 mode is preserved; a symlink target is refused
    with _running(_make_service(tmp_path, _write_config(tmp_path))) as server:
        token = _bootstrap(server)
        revision = _request(server, "GET", "/api/roots", headers=_token(token)).json()[
            "revision"
        ]
        headers, raw = _json_post(token, {"path": str(extra), "revision": revision})
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 200, response.text

        target = tmp_path / "target" / "config.toml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('embed_model = "fake"\n', encoding="utf-8")
        stat_before = stat.S_IMODE(target.stat().st_mode)
        linked = tmp_path / "linked.toml"
        linked.symlink_to(target)
        with _running(_make_service(tmp_path, linked)) as server:
            token = _bootstrap(server)
            revision = _request(
                server, "GET", "/api/roots", headers=_token(token)
            ).json()["revision"]
            headers, raw = _json_post(
                token, {"path": str(extra), "revision": revision}
            )
            response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
            assert response.status == 409
            assert response.error_code() == "config_invalid"
            assert target.read_text() == 'embed_model = "fake"\n'
            assert stat.S_IMODE(target.stat().st_mode) == stat_before

    # -- stale revisions and external edits never clobber newer file contents
    config_path = _write_config(tmp_path)
    with _running(_make_service(tmp_path, config_path)) as server:
        token = _bootstrap(server)
        revision = _request(server, "GET", "/api/roots", headers=_token(token)).json()[
            "revision"
        ]
        # an external edit that keeps the file valid adds a comment line
        with config_path.open("a", encoding="utf-8") as handle:
            handle.write("# an external comment\n")
        external = config_path.read_bytes()
        headers, raw = _json_post(token, {"path": str(extra), "revision": revision})
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 409
        assert response.error_code() == "stale_config"
        assert config_path.read_bytes() == external
        refreshed = _request(server, "GET", "/api/roots", headers=_token(token)).json()
        headers, raw = _json_post(
            token, {"path": str(extra), "revision": refreshed["revision"]}
        )
        response = _request(server, "POST", "/api/roots", headers=headers, body=raw)
        assert response.status == 200, response.text
        assert extra.resolve() in (
            Path(value) for value in _request(
                server, "GET", "/api/roots", headers=_token(token)
            ).json()["roots"]
        )

    assert builds == []


# --------------------------------------------------------------------------
# job lifecycle: states, reports, truncation, retention
# --------------------------------------------------------------------------


def test_job_lifecycle_states_reports_truncation_and_retention_over_http(
    tmp_path: Path,
) -> None:
    reports = [
        SyncReport(warnings=[("w.txt", "warning only")]),
        SyncReport(
            updated=1,
            errors=[("broken.txt", "cannot read")],
            warnings=[("w.txt", "warning too")],
        ),
        SyncReport(
            errors=[(f"p{index:03d}.txt", "boom") for index in range(101)],
            warnings=[(f"w{index:03d}.txt", "hm") for index in range(2)],
        ),
        SyncReport(),
    ]

    indexer = _StaticIndexer(reports)

    def factory(store: Any, embedder: Any, cfg: Any) -> _StaticIndexer:
        del store, embedder, cfg
        return indexer

    config_path = _write_config(tmp_path, roots=[_root(tmp_path, "job-docs")])
    with _running(_make_service(tmp_path, config_path, indexer_factory=factory)) as server:
        token = _bootstrap(server)
        job_ids: list[str] = []
        for expected in (
            "succeeded",
            "partial",
            "partial",
            "succeeded",
            "failed",
        ):
            response = _empty_index(server, token)
            assert response.status == 202, response.text
            job_id = response.json()["job_id"]
            finished = _wait_job(server, token, job_id)
            assert finished["state"] == expected
            assert finished["state"] != "queued"
            job_ids.append(job_id)

        # -- warnings alone succeed; partial iff errors
        response = _request(
            server, "GET", f"/api/jobs/{job_ids[0]}", headers=_token(token)
        )
        view = response.json()
        assert view["state"] == "succeeded"
        assert view["report"]["error_total"] == 0 and view["report"]["warning_total"] == 1
        assert view["report"]["errors"] == [] and view["report"]["warnings"] == [
            {"path": "w.txt", "message": "warning only"}
        ]
        assert (
            view["report"]["errors_truncated"] is False
            and view["report"]["warnings_truncated"] is False
        )
        assert view["error"] is None

        # -- partial iff errors; diagnostics keep the first hits
        response = _request(
            server, "GET", f"/api/jobs/{job_ids[1]}", headers=_token(token)
        )
        view = response.json()
        assert view["state"] == "partial"
        assert view["report"]["error_total"] == 1 and view["report"]["warning_total"] == 1
        assert view["report"]["errors"] == [
            {"path": "broken.txt", "message": "cannot read"}
        ]
        assert view["report"]["warnings"] == [
            {"path": "w.txt", "message": "warning too"}
        ]

        # -- diagnostics retain the first 100 errors/warnings with full totals
        response = _request(
            server, "GET", f"/api/jobs/{job_ids[2]}", headers=_token(token)
        )
        view = response.json()
        assert view["state"] == "partial"
        assert view["report"]["error_total"] == 101 and len(view["report"]["errors"]) == 100
        assert view["report"]["errors_truncated"] is True
        assert view["report"]["warning_total"] == 2 and not view["report"]["warnings_truncated"]
        assert len(view["report"]["warnings"]) == 2
        assert [entry["path"] for entry in view["report"]["errors"][:3]] == [
            "p000.txt",
            "p001.txt",
            "p002.txt",
        ]
        assert view["report"]["errors"][0] == {"path": "p000.txt", "message": "boom"}

        # -- a failed job has report: null and a safe error string
        response = _request(
            server, "GET", f"/api/jobs/{job_ids[4]}", headers=_token(token)
        )
        view = response.json()
        assert view["state"] == "failed"
        assert view["report"] is None and view["error"] == "no more static reports"

        # -- retention: at most 20 jobs, dropping the oldest finished first
        for _ in range(LIMITS["jobs_retained"] - len(job_ids) + 1):
            response = _empty_index(server, token)
            assert response.status == 202, response.text
            expired_id = response.json()["job_id"]
            _wait_job(server, token, expired_id)
            job_ids.append(expired_id)
            assert "queued" not in {state for state in (finished["state"],)}

        response = _request(
            server, "GET", f"/api/jobs/{job_ids[0]}", headers=_token(token)
        )
        assert response.status == 404
        assert response.error_code() == "job_not_found"

        # -- unknown and malformed job IDs are 404 job_not_found
        for path in ("/api/jobs/not-a-job", "/api/jobs/%2e%2e"):
            response = _request(server, "GET", path, headers=_token(token))
            assert response.status == 404, path
            assert response.error_code() == "job_not_found"

    assert reports, "every configured report was served"


# --------------------------------------------------------------------------
# offline packaging: declared web assets and importable resources
# --------------------------------------------------------------------------


def test_web_assets_are_declared_and_importable_offline() -> None:
    project_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    package_data = project["tool"]["setuptools"]["package-data"]["seekr"]
    assert "web_assets/*" in package_data

    assets = importlib.resources.files("seekr") / "web_assets"
    for name in ("index.html", "styles.css", "app.js"):
        resource = assets / name
        assert resource.is_file(), name
        assert resource.read_bytes(), name
