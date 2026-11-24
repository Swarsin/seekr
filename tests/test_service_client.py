"""Tests for the loopback service client and CLI/MCP routing through it.

The service runs in-process with :class:`FakeEmbedder` and an in-memory
observer, or a tiny stub HTTP server stands in for it.  Config, data and state
live in temporary directories; only loopback HTTP is used.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from seekr import __version__, cli_index, cli_search, paths, service_client
from seekr.cli import app
from seekr.cli_search import SearchError, get_hits, get_similar_hits
from seekr.config import load_config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer, SyncReport
from seekr.mcp_server import McpServer
from seekr.models import Hit, HitExplain
from seekr.service import SeekrService
from seekr.service_client import (
    ServiceClient,
    ServiceError,
    ServiceUnavailable,
    hit_from_wire,
)
from seekr.service_discovery import ServiceInfo, write_service_info
from seekr.store import Store
from seekr.web_contract import SearchRequest, SimilarRequest
from seekr.web_service import WebService


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


class FakeObserver:
    def schedule(self, handler: object, path: str, recursive: bool) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None


@pytest.fixture(autouse=True)
def no_leaked_threads():
    before = set(threading.enumerate())
    yield

    def leaked() -> list[threading.Thread]:
        return [
            thread
            for thread in threading.enumerate()
            if thread not in before
            and thread.is_alive()
            and thread.name != "seekr-embed-prewarm"
        ]

    assert _wait_for(lambda: not leaked(), timeout=2.0), leaked()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        target = tmp_path / name.lower()
        target.mkdir(mode=0o700)
        monkeypatch.setenv(name, str(target))
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("alpha notes about xylophones\n", encoding="utf-8")
    (root / "b.md").write_text("beta notes about xylophones and drums\n", encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(
        'embed_model = "fake"\n'
        f"db_path = {json.dumps(str(tmp_path / 'index.db'))}\n"
        'vector_quant = "int8"\n'
        f"roots = [{json.dumps(str(root))}]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SEEKR_CONFIG", str(config))
    monkeypatch.setattr(service_client, "_SKEW_LOGGED", set())
    return tmp_path


def _service(request: pytest.FixtureRequest) -> SeekrService:
    svc = SeekrService(
        embedder_factory=lambda cfg: FakeEmbedder(),
        observer_factory=FakeObserver,
        debounce_s=0.05,
        port=0,
    )
    request.addfinalizer(svc.stop)
    return svc


def _started_service(request: pytest.FixtureRequest) -> SeekrService:
    svc = _service(request)
    svc.start()
    assert _wait_for(lambda: int(svc.backend.status().index["documents"] or 0) == 2)
    assert _wait_for(lambda: svc.backend.status().activity["lane"] == "idle")
    return svc


def _index_locally(env: Path) -> None:
    cfg = load_config()
    embedder = FakeEmbedder()
    store = Store(cfg.db_path, cfg.embed_model, embedder.dim, vector_quant=cfg.vector_quant)
    try:
        Indexer(store, embedder, cfg).sync(list(cfg.roots))
    finally:
        store.close()


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_discovery(port: int, *, version: str = __version__, pid: int | None = None) -> None:
    write_service_info(
        ServiceInfo(
            pid=os.getpid() if pid is None else pid,
            port=port,
            token="test-token",
            version=version,
            started_at="2026-01-01T00:00:00Z",
        ),
        paths.service_file(),
    )


class _Spy:
    """Count calls to a ServiceClient method while delegating to it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        self.calls: list[dict[str, Any]] = []
        original = getattr(ServiceClient, name)
        spy = self

        def wrapper(client: ServiceClient, request: Any) -> Any:
            spy.calls.append(request)
            return original(client, request)

        monkeypatch.setattr(ServiceClient, name, wrapper)


# -- stub server --------------------------------------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    responses: ClassVar[list[tuple[int, object]]] = []
    seen: ClassVar[list[tuple[str, str, object, str | None]]] = []

    def log_message(self, format: str, *args: object) -> None:
        return None

    def _reply(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw) if raw else None
        type(self).seen.append(
            (self.command, self.path, body, self.headers.get("X-Seekr-Token"))
        )
        status, payload = type(self).responses.pop(0) if type(self).responses else (200, {})
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = _reply
    do_POST = _reply


@pytest.fixture
def stub(env: Path, request: pytest.FixtureRequest) -> type[_StubHandler]:
    handler = type("Handler", (_StubHandler,), {"responses": [], "seen": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()

    def close() -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    request.addfinalizer(close)
    _write_discovery(int(server.server_address[1]))
    return handler


def _error(code: str, message: str) -> dict[str, object]:
    return {"error": {"code": code, "message": message}}


# -- wire conversion ----------------------------------------------------------


def test_hit_round_trip_through_wire_dto() -> None:
    plain = Hit(
        chunk_id=7,
        path="/notes/a.md",
        locator="§Intro",
        loc_start=1.5,
        snippet="alpha notes",
        score=0.25,
        kind="markdown",
        module="CS101",
        week=3,
        highlights=((0, 5), (6, 11)),
    )
    explained = Hit(
        **{
            **plain.__dict__,
            "explain": HitExplain(
                keyword_rank=1, keyword_bm25=-2.5, vector_rank=2,
                vector_distance=0.4, rrf_score=0.03,
            ),
        }
    )
    for hit, explain in ((plain, False), (explained, True)):
        wire = json.loads(json.dumps(WebService._hit_dto(hit, explain=explain).to_dict()))
        assert hit_from_wire(wire) == hit


# -- discovery ----------------------------------------------------------------


def test_from_discovery_none_without_file(env: Path) -> None:
    assert ServiceClient.from_discovery() is None


def test_version_skew_ignored(env: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_discovery(_closed_port(), version="0.0.0-other")
    with caplog.at_level(logging.DEBUG, logger="seekr.service_client"):
        assert ServiceClient.from_discovery() is None
        assert ServiceClient.from_discovery() is None
    skew = [r for r in caplog.records if "version_skew" in r.getMessage()]
    assert len(skew) == 1


def test_unreachable_service_falls_back_quickly(env: Path) -> None:
    _index_locally(env)
    _write_discovery(_closed_port())
    assert ServiceClient.from_discovery() is not None

    started = time.monotonic()
    hits = get_hits("xylophones", mode="keyword")
    elapsed = time.monotonic() - started

    assert {Path(hit.path).name for hit in hits} == {"a.md", "b.md"}
    assert elapsed < 1.5


def test_unreachable_client_raises_service_unavailable(env: Path) -> None:
    _write_discovery(_closed_port())
    client = ServiceClient.from_discovery()
    assert client is not None
    with pytest.raises(ServiceUnavailable):
        client.status()


# -- search routing -----------------------------------------------------------


def test_search_uses_running_service(env: Path, request, monkeypatch) -> None:
    _started_service(request)
    spy = _Spy(monkeypatch, "search")

    hits = get_hits("xylophones", mode="keyword", k=5, kind="markdown", since="30d")
    explained = get_hits("xylophones", mode="hybrid", explain=True, path_prefix=str(env))

    assert all(isinstance(hit, Hit) for hit in hits + explained)
    assert {Path(hit.path).name for hit in hits} == {"a.md", "b.md"}
    assert explained and all(isinstance(hit.explain, HitExplain) for hit in explained)
    assert len(spy.calls) == 2
    assert spy.calls[0] == {
        "query": "xylophones", "k": 5, "mode": "keyword", "per_file": 2,
        "rerank": False, "explain": False, "kind": "markdown", "since": "30d",
    }
    assert spy.calls[1]["path"] == str(env)


def test_service_never_calls_itself(env: Path, request, monkeypatch) -> None:
    svc = _started_service(request)
    search_spy = _Spy(monkeypatch, "search")
    similar_spy = _Spy(monkeypatch, "similar")

    svc.backend.search(SearchRequest(query="xylophones", mode="hybrid"))
    svc.backend.search(SearchRequest(query="xylophones", mode="keyword"))
    svc.backend.similar(SimilarRequest(path=str(env / "docs" / "a.md")))
    assert search_spy.calls == []
    assert similar_spy.calls == []

    # One client call through HTTP is exactly one service request: the
    # service handles it locally instead of calling itself again.
    get_hits("xylophones", mode="hybrid")
    get_similar_hits(env / "docs" / "a.md")
    assert len(search_spy.calls) == 1
    assert len(similar_spy.calls) == 1


def test_similar_uses_service_with_absolute_path(env: Path, request, monkeypatch) -> None:
    _started_service(request)
    spy = _Spy(monkeypatch, "similar")
    monkeypatch.chdir(env / "docs")

    hits = get_similar_hits("a.md", k=3)

    assert [Path(hit.path).name for hit in hits] == ["b.md"]
    assert spy.calls == [{"path": str(env / "docs" / "a.md"), "k": 3, "per_file": 1}]


def test_k_above_wire_limit_skips_service(env: Path, request, monkeypatch) -> None:
    _started_service(request)
    spy = _Spy(monkeypatch, "search")

    hits = get_hits("xylophones", mode="keyword", k=51)

    assert {Path(hit.path).name for hit in hits} == {"a.md", "b.md"}
    assert spy.calls == []


def test_injected_config_skips_service(env: Path, request, monkeypatch) -> None:
    _started_service(request)
    spy = _Spy(monkeypatch, "search")

    get_hits("xylophones", mode="keyword", cfg=load_config())
    get_hits("xylophones", mode="keyword", use_service=False)

    assert spy.calls == []


def test_mcp_search_uses_service(env: Path, request, monkeypatch) -> None:
    _started_service(request)
    spy = _Spy(monkeypatch, "search")

    _payload, count = McpServer()._dispatch_search({"query": "xylophones", "mode": "keyword"})

    assert count == 2
    assert len(spy.calls) == 1


# -- error mapping ------------------------------------------------------------


def test_service_4xx_is_search_error(stub, monkeypatch) -> None:
    stub.responses.append((400, _error("invalid_request", "module filter is ambiguous")))
    monkeypatch.setattr(
        cli_search, "_build_searcher", lambda *a, **k: pytest.fail("fell back to local")
    )

    with pytest.raises(SearchError, match="module filter is ambiguous"):
        get_hits("xylophones", mode="keyword")
    method, path, body, token = stub.seen[0]
    assert (method, path, token) == ("POST", "/api/search", "test-token")
    assert body["query"] == "xylophones"


@pytest.mark.parametrize(
    ("status", "code", "message"),
    [
        (409, "busy", "a model download is running"),
        (500, "internal_error", "search failed unexpectedly"),
        (503, "index_unavailable", "cannot read the index: boom"),
    ],
)
def test_service_busy_and_5xx_surface(stub, monkeypatch, status, code, message) -> None:
    stub.responses.append((status, _error(code, message)))
    monkeypatch.setattr(
        cli_search, "_build_searcher", lambda *a, **k: pytest.fail("fell back to local")
    )
    with pytest.raises(SearchError, match=message):
        get_hits("xylophones")


def test_shutting_down_service_falls_back(stub, env: Path) -> None:
    _index_locally(env)
    stub.responses.append((409, _error("busy", "the web service is shutting down")))

    hits = get_hits("xylophones", mode="keyword")

    assert {Path(hit.path).name for hit in hits} == {"a.md", "b.md"}


def test_client_error_carries_status_and_code(stub) -> None:
    stub.responses.append((404, _error("job_not_found", "job was not found or has expired")))
    client = ServiceClient.from_discovery()
    assert client is not None
    with pytest.raises(ServiceError) as caught:
        client.job("abc")
    assert caught.value.status == 404
    assert caught.value.code == "job_not_found"
    assert str(caught.value) == "job was not found or has expired"
    assert not isinstance(caught.value, ServiceUnavailable)


def test_shutdown_and_index_calls(stub) -> None:
    stub.responses.extend(
        [(202, {"stopping": True}), (202, {"job_id": "j1"}), (202, {"job_id": "j2"})]
    )
    client = ServiceClient.from_discovery()
    assert client is not None
    client.shutdown()
    assert client.index_roots() == "j1"
    assert client.index_paths(["/abs/x"]) == "j2"
    assert [(m, p, b) for m, p, b, _ in stub.seen] == [
        ("POST", "/api/shutdown", {}),
        ("POST", "/api/index", {}),
        ("POST", "/api/index/paths", {"paths": ["/abs/x"]}),
    ]


# -- index delegation ---------------------------------------------------------


def test_report_dict_printer_matches_report_printer(monkeypatch) -> None:
    report = SyncReport(
        added=2, updated=1, unchanged=4, removed=1, skipped=3,
        errors=[("/a.pdf", "broken")], warnings=[("/b.docx", "legacy"), ("/c", "odd")],
    )
    lines: list[str] = []
    monkeypatch.setattr(
        cli_index.typer, "echo", lambda message="", **kwargs: lines.append(str(message))
    )
    cli_index._print_report(report)
    expected = list(lines)
    lines.clear()
    cli_index._print_report_dict(WebService._serialise_report(report))
    assert lines == expected


def test_report_dict_printer_notes_truncation(monkeypatch) -> None:
    report = SyncReport(errors=[(f"/f{i}", "bad") for i in range(105)])
    lines: list[str] = []
    monkeypatch.setattr(
        cli_index.typer, "echo", lambda message="", **kwargs: lines.append(str(message))
    )
    cli_index._print_report_dict(WebService._serialise_report(report))
    assert "errors: 105" in lines
    assert sum(line.startswith("error: /f") for line in lines) == 100
    assert lines[-1] == "error: ... 5 more not shown"


def test_index_delegates_to_service(env: Path, request, monkeypatch) -> None:
    svc = _started_service(request)
    monkeypatch.setattr(cli_index, "_JOB_POLL_SECONDS", 0.05)
    monkeypatch.setattr(
        cli_index, "_build_embedder", lambda cfg: pytest.fail("indexed locally")
    )
    (env / "docs" / "c.md").write_text("gamma xylophones\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["index"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[:7] == [
        "added: 1", "updated: 0", "unchanged: 2", "removed: 0", "skipped: 0",
        "errors: 0", "warnings: 0",
    ]

    (env / "docs" / "d.md").write_text("delta xylophones\n", encoding="utf-8")
    monkeypatch.chdir(env / "docs")
    result = runner.invoke(app, ["index", "d.md"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == "added: 1"
    assert int(svc.backend.status().index["documents"] or 0) == 4


def test_index_job_records_last_error_count(env: Path, request, monkeypatch) -> None:
    _started_service(request)
    monkeypatch.setattr(cli_index, "_JOB_POLL_SECONDS", 0.05)
    (env / "docs" / "broken.pdf").write_bytes(b"not a pdf at all")

    result = CliRunner().invoke(app, ["index"])

    assert result.exit_code == 0, result.output
    assert "errors: 1" in result.output.splitlines()
    status = CliRunner().invoke(app, ["status"])
    assert "last error count: 1" in status.output.splitlines()


def test_index_surfaces_service_errors(stub, monkeypatch) -> None:
    stub.responses.append((409, _error("daemon_active", "a seekr daemon is running")))
    monkeypatch.setattr(
        cli_index, "_build_embedder", lambda cfg: pytest.fail("indexed locally")
    )
    result = CliRunner().invoke(app, ["index"])
    assert result.exit_code == 1
    assert "a seekr daemon is running" in result.output


def test_index_reports_failed_job(stub, monkeypatch) -> None:
    monkeypatch.setattr(cli_index, "_JOB_POLL_SECONDS", 0.01)
    stub.responses.extend(
        [
            (202, {"job_id": "j1"}),
            (200, {"job_id": "j1", "state": "running", "report": None, "error": None}),
            (200, {"job_id": "j1", "state": "failed", "report": None, "error": "disk full"}),
        ]
    )
    result = CliRunner().invoke(app, ["index"])
    assert result.exit_code == 1
    assert "disk full" in result.output
    assert [p for _, p, _, _ in stub.seen] == ["/api/index", "/api/jobs/j1", "/api/jobs/j1"]


def test_index_falls_back_when_service_unreachable(env: Path, monkeypatch) -> None:
    _write_discovery(_closed_port())
    monkeypatch.setattr(cli_index, "_build_embedder", lambda cfg: FakeEmbedder())
    result = CliRunner().invoke(app, ["index"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == "added: 2"
