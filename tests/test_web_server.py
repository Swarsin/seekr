"""Loopback HTTP tests for the local Seekr web transport.

Everything here runs against a real ``127.0.0.1`` server and the synthetic
``FakeWebBackend``; no application service, Store, model, or external network
is used.
"""

from __future__ import annotations

import http.client
import importlib.resources
import inspect
import io
import json
import logging
import os
import select
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest

from seekr import web_server
from seekr.web_contract import (
    ASSET_MIME_TYPES,
    ERROR_STATUS,
    LIMITS,
    ApiError,
    HitDTO,
    SearchRequest,
    SearchResult,
)
from seekr.web_server import SECURITY_HEADERS, create_server, run_web
from web_fixtures import (
    COURSE_VIEW,
    JOB_FAILED,
    JOB_MODEL_SUCCEEDED,
    JOB_PARTIAL,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    PREVIEW_PNG_BYTES,
    PREVIEW_VIEW,
    ROOTS_VIEW,
    SEARCH_RESULT,
    STATUS_READY,
    FakeWebBackend,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


@dataclass
class HttpResponse:
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

    def one_header(self, name: str) -> str:
        values = self.header(name)
        assert len(values) == 1, (name, values)
        return values[0]


def _response_headers(response: http.client.HTTPResponse) -> dict[str, list[str]]:
    headers: dict[str, list[str]] = {}
    for name, value in response.getheaders():
        headers.setdefault(name.lower(), []).append(value)
    return headers


def request(
    server: Any,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 5.0,
) -> HttpResponse:
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        server.bound_port,
        timeout=timeout,
    )
    try:
        connection.request(method, path, body=body, headers=dict(headers or {}))
        raw = connection.getresponse()
        data = raw.read()
        return HttpResponse(raw.status, _response_headers(raw), data)
    finally:
        connection.close()


def session_headers(server: Any) -> dict[str, str]:
    return {"Origin": server.url, "X-Seekr-Client": "1"}


def fetch_token(server: Any) -> str:
    response = request(server, "GET", "/api/session", headers=session_headers(server))
    assert response.status == 200
    return response.json()["token"]


def json_headers(token: str, **extra: str) -> dict[str, str]:
    headers = {"X-Seekr-Token": token, "Content-Type": "application/json"}
    headers.update(extra)
    return headers


def raw_exchange(
    server: Any,
    lines: list[str],
    *,
    body: bytes = b"",
    read_timeout: float = 3.0,
) -> HttpResponse:
    payload = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body
    with socket.create_connection(
        ("127.0.0.1", server.bound_port),
        timeout=read_timeout,
    ) as connection:
        connection.settimeout(read_timeout)
        connection.sendall(payload)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = connection.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
    return _parse_raw_response(b"".join(chunks))


def _parse_raw_response(data: bytes) -> HttpResponse:
    head, separator, body = data.partition(b"\r\n\r\n")
    if not separator:
        return HttpResponse(0, {}, data)
    lines = head.split(b"\r\n")
    status = int(lines[0].split()[1]) if len(lines[0].split()) >= 2 else 0
    headers: dict[str, list[str]] = {}
    for line in lines[1:]:
        name, present, value = line.partition(b":")
        if present:
            headers.setdefault(name.decode("latin-1").lower(), []).append(
                value.strip().decode("latin-1")
            )
    return HttpResponse(status, headers, body)


def request_lines(
    server: Any,
    method: str,
    path: str,
    *extra: str,
    version: str = "HTTP/1.1",
    include_host: bool = True,
) -> list[str]:
    lines = [f"{method} {path} {version}"]
    if include_host:
        lines.append(f"Host: 127.0.0.1:{server.bound_port}")
    lines.extend(extra)
    return lines


def drip_until_closed(
    server: Any,
    prefix: bytes,
    *,
    interval: float = 0.04,
    max_seconds: float = 2.0,
) -> tuple[float | None, bytes]:
    """Drip bytes slowly and return (seconds-until-close, received-response)."""
    with socket.create_connection(
        ("127.0.0.1", server.bound_port),
        timeout=3.0,
    ) as connection:
        connection.sendall(prefix)
        started = time.monotonic()
        received = bytearray()
        while time.monotonic() - started < max_seconds:
            try:
                connection.sendall(b"x")
            except OSError:
                return time.monotonic() - started, bytes(received)
            time.sleep(interval)
            readable, _, _ = select.select([connection], [], [], 0)
            if not readable:
                continue
            chunk = connection.recv(65536)
            if not chunk:
                return time.monotonic() - started, bytes(received)
            received.extend(chunk)
        return None, bytes(received)


@contextmanager
def running_server(
    backend: FakeWebBackend | None = None,
    *,
    read_timeout: float = 1.0,
    request_deadline: float | None = None,
    join_timeout: float = 2.0,
) -> Iterator[tuple[web_server.SeekrHTTPServer, FakeWebBackend]]:
    fake = backend if backend is not None else FakeWebBackend()
    server = create_server(
        fake,
        port=0,
        read_timeout=read_timeout,
        request_deadline=request_deadline,
        join_timeout=join_timeout,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.02},
        name="seekr-test-serve",
        daemon=True,
    )
    thread.start()
    try:
        yield server, fake
    finally:
        if thread.is_alive():
            server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


@contextmanager
def capture_web_log() -> Iterator[io.StringIO]:
    """Capture ``seekr.web`` records without relying on root propagation.

    Other tests call ``configure_logging``, which sets the parent ``seekr``
    logger's ``propagate`` flag to False and can leave a global disable level
    behind.  A handler attached directly to ``seekr.web`` keeps this assertion
    meaningful regardless of that shared logging state; all temporary changes
    are restored before the context exits.
    """
    logger = logging.getLogger(web_server.ACCESS_LOG_NAME)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)

    previous_level = logger.level
    previous_disabled = logger.disabled
    previous_disable = logging.root.manager.disable

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.disabled = False
    logging.disable(logging.NOTSET)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
        logger.disabled = previous_disabled
        logging.disable(previous_disable)


def _api_error(code: str) -> ApiError:
    return ApiError(ERROR_STATUS[code], code, f"fixture {code}")


class ClosingBackend(FakeWebBackend):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.begin_calls = 0
        self.shutdown_calls = 0
        self.close_calls = 0
        self.order: list[str] = []

    def begin_shutdown(self) -> None:
        self.begin_calls += 1
        self.order.append("begin")

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.order.append("shutdown")

    def close(self) -> None:
        self.close_calls += 1
        self.order.append("close")


class BlockingShutdownBackend(ClosingBackend):
    """``shutdown`` blocks until released so ordering can be observed."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.shutdown_entered = threading.Event()
        self.release = threading.Event()

    def shutdown(self) -> None:
        super().shutdown()
        self.shutdown_entered.set()
        assert self.release.wait(timeout=10.0), "test backend shutdown was never released"
        self.order.append("shutdown_done")


class BlockingBackend(FakeWebBackend):
    """``status`` blocks until released, tracking active handler threads."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.release = threading.Event()
        self.entered = threading.Event()
        self.finished = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.daemon_flags: list[bool] = []

    def status(self) -> Any:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.daemon_flags.append(threading.current_thread().daemon)
        self.entered.set()
        try:
            assert self.release.wait(timeout=10.0), "test backend was never released"
        finally:
            with self.lock:
                self.active -= 1
            self.finished.set()
        return super().status()


class ExplodingBackend(FakeWebBackend):
    def status(self) -> Any:
        raise RuntimeError("SECRET_DOCUMENT_TEXT_MUST_NOT_LEAK")


# -- session, Host, Origin, token, and method policy ----------------------


def test_session_returns_same_token_and_exact_limits() -> None:
    with running_server() as (server, backend):
        first = request(server, "GET", "/api/session", headers=session_headers(server))
        assert first.status == 200
        payload = first.json()
        assert set(payload) == {"token", "limits"}
        assert payload["limits"] == LIMITS
        assert payload["token"] == server.token

        second = request(server, "GET", "/api/session", headers=session_headers(server))
        assert second.json()["token"] == server.token
        assert backend.calls == []

        assert first.one_header("content-type") == "application/json; charset=utf-8"
        assert first.one_header("cache-control") == "no-store"
        assert first.one_header("connection") == "close"
        for name, value in SECURITY_HEADERS:
            assert first.header(name) == [value]
        assert not any(name.startswith("access-control-") for name in first.headers)


def test_session_requires_browser_origin_proof() -> None:
    with running_server() as (server, _backend):
        bare = request(
            server,
            "GET",
            "/api/session",
            headers={"X-Seekr-Client": "1"},
        )
        assert bare.status == 403
        assert bare.json()["error"]["code"] == "forbidden"
        assert "browser-origin headers are required" in bare.text

        wrong_origin = request(
            server,
            "GET",
            "/api/session",
            headers={"Origin": "http://example.test", "X-Seekr-Client": "1"},
        )
        assert wrong_origin.status == 403

        for value in ("cross-site", "same-site"):
            response = request(
                server,
                "GET",
                "/api/session",
                headers={"Sec-Fetch-Site": value, "X-Seekr-Client": "1"},
            )
            assert response.status == 403

        none_site = request(
            server,
            "GET",
            "/api/session",
            headers={"Sec-Fetch-Site": "none", "X-Seekr-Client": "1"},
        )
        assert none_site.status == 200


def test_session_requires_exactly_one_client_header() -> None:
    with running_server() as (server, _backend):
        missing = request(server, "GET", "/api/session", headers={"Origin": server.url})
        assert missing.status == 400
        assert missing.json()["error"]["code"] == "invalid_request"

        wrong = request(
            server,
            "GET",
            "/api/session",
            headers={"Origin": server.url, "X-Seekr-Client": "true"},
        )
        assert wrong.status == 400

        duplicate = raw_exchange(
            server,
            request_lines(
                server,
                "GET",
                "/api/session",
                f"Origin: {server.url}",
                "X-Seekr-Client: 1",
                "X-Seekr-Client: 1",
            ),
        )
        assert duplicate.status == 400
        assert duplicate.json()["error"]["code"] == "invalid_request"


def test_bad_host_rejected_everywhere() -> None:
    with running_server() as (server, _backend):
        valid_headers = session_headers(server)
        for host in (
            f"localhost:{server.bound_port}",
            f"127.0.0.1:{server.bound_port + 1}",
            "127.0.0.1",
            f"127.0.0.1:{server.bound_port} ",
        ):
            response = request(
                server,
                "GET",
                "/api/session",
                headers={**valid_headers, "Host": host},
            )
            assert response.status == 403
            assert response.json()["error"]["code"] == "forbidden"

        missing = raw_exchange(
            server,
            request_lines(
                server,
                "GET",
                "/api/session",
                f"Origin: {server.url}",
                "X-Seekr-Client: 1",
                include_host=False,
            ),
        )
        assert missing.status == 403

        duplicate = raw_exchange(
            server,
            request_lines(
                server,
                "GET",
                "/api/session",
                f"Host: 127.0.0.1:{server.bound_port}",
                f"Origin: {server.url}",
                "X-Seekr-Client: 1",
            ),
        )
        assert duplicate.status == 403

        options = request(
            server,
            "OPTIONS",
            "/api/status",
            headers={"Host": f"localhost:{server.bound_port}"},
        )
        assert options.status == 403


def test_origin_and_fetch_site_rules_for_token_routes() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        wrong_origin = request(
            server,
            "GET",
            "/api/status",
            headers={"X-Seekr-Token": token, "Origin": "http://example.test"},
        )
        assert wrong_origin.status == 403

        for value in ("cross-site", "same-site"):
            response = request(
                server,
                "GET",
                "/api/status",
                headers={"X-Seekr-Token": token, "Sec-Fetch-Site": value},
            )
            assert response.status == 403

        duplicate_origin = raw_exchange(
            server,
            request_lines(
                server,
                "GET",
                "/api/status",
                f"X-Seekr-Token: {token}",
                f"Origin: {server.url}",
                f"Origin: {server.url}",
            ),
        )
        assert duplicate_origin.status == 403

        duplicate_site = raw_exchange(
            server,
            request_lines(
                server,
                "GET",
                "/api/status",
                f"X-Seekr-Token: {token}",
                "Sec-Fetch-Site: none",
                "Sec-Fetch-Site: same-origin",
            ),
        )
        assert duplicate_site.status == 403

        curl_like = request(
            server,
            "GET",
            "/api/status",
            headers={"X-Seekr-Token": token},
        )
        assert curl_like.status == 200


def test_token_rules() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)

        missing = request(server, "GET", "/api/status")
        assert missing.status == 403
        assert missing.json()["error"]["code"] == "forbidden"

        wrong = request(
            server,
            "GET",
            "/api/status",
            headers={"X-Seekr-Token": token + "x"},
        )
        assert wrong.status == 403

        non_ascii = request(
            server,
            "GET",
            "/api/status",
            headers={"X-Seekr-Token": "t\u00f6ken"},
        )
        assert non_ascii.status == 403

        duplicate = raw_exchange(
            server,
            request_lines(
                server,
                "GET",
                "/api/status",
                f"X-Seekr-Token: {token}",
                f"X-Seekr-Token: {token}",
            ),
        )
        assert duplicate.status == 403

        valid = request(
            server,
            "GET",
            "/api/status",
            headers={"X-Seekr-Token": token},
        )
        assert valid.status == 200


def test_token_compare_uses_compare_digest() -> None:
    source = inspect.getsource(web_server)
    assert "hmac.compare_digest" in source
    assert "self.server.token" in source


def test_options_is_405_without_cors() -> None:
    with running_server() as (server, _backend):
        response = request(server, "OPTIONS", "/api/status")
        assert response.status == 405
        assert response.json()["error"]["code"] == "method_not_allowed"
        assert not any(name.startswith("access-control-") for name in response.headers)

        unknown = request(server, "OPTIONS", "/not-a-route")
        assert unknown.status == 405

        cross_site = request(
            server,
            "OPTIONS",
            "/api/status",
            headers={"Origin": "http://example.test", "Sec-Fetch-Site": "cross-site"},
        )
        assert cross_site.status == 405
        assert not any(
            name.startswith("access-control-") for name in cross_site.headers
        )


def test_malformed_request_line_gets_json_error() -> None:
    with running_server() as (server, _backend):
        response = raw_exchange(server, ["NOT A REQUEST"])
        assert response.status == 400
        assert response.json()["error"]["code"] == "invalid_request"
        for name, value in SECURITY_HEADERS:
            assert response.header(name) == [value]
        assert response.one_header("cache-control") == "no-store"


def test_unknown_routes_and_wrong_methods() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        for path in (
            "/api/unknown",
            "/api/jobs/a/b",
            "/web_assets/app.js",
            "/../pyproject.toml",
            "/styles.css.bak",
        ):
            response = request(server, "GET", path)
            assert response.status == 404, path
            assert response.json()["error"]["code"] == "not_found"

        double_slash = raw_exchange(
            server,
            request_lines(
                server,
                "GET",
                "//index.html",
                f"Origin: {server.url}",
            ),
        )
        assert double_slash.status == 404
        assert double_slash.json()["error"]["code"] == "not_found"

        wrong_methods = (
            ("GET", "/api/search", None),
            ("POST", "/api/status", b"{}"),
            ("POST", "/api/session", b"{}"),
            ("DELETE", "/api/status", None),
            ("POST", "/styles.css", b"{}"),
        )
        for method, path, body in wrong_methods:
            headers = json_headers(token) if body is not None else {"X-Seekr-Token": token}
            response = request(server, method, path, headers=headers, body=body)
            assert response.status == 405, (method, path)
            assert response.json()["error"]["code"] == "method_not_allowed"


# -- endpoint success shapes ----------------------------------------------


def test_status_and_roots_get() -> None:
    with running_server() as (server, backend):
        token = fetch_token(server)

        status = request(
            server,
            "GET",
            "/api/status",
            headers={"X-Seekr-Token": token},
        )
        assert status.status == 200
        assert status.json() == STATUS_READY.to_dict()
        assert backend.calls == [("status", {})]

        roots = request(
            server,
            "GET",
            "/api/roots",
            headers={"X-Seekr-Token": token},
        )
        assert roots.status == 200
        assert roots.json() == ROOTS_VIEW.to_dict()
        assert set(roots.json()) == {
            "roots",
            "config_path",
            "revision",
            "daemon_restart_required",
            "warning",
        }


def test_add_root_maps_request_and_response() -> None:
    with running_server() as (server, backend):
        token = fetch_token(server)
        response = request(
            server,
            "POST",
            "/api/roots",
            headers=json_headers(token),
            body=json.dumps(
                {"path": "/tmp/opencode/new-root", "revision": "fixture-revision"}
            ).encode(),
        )
        assert response.status == 200
        assert response.json() == ROOTS_VIEW.to_dict()
        assert backend.calls[-1] == (
            "add_root",
            {"path": "/tmp/opencode/new-root", "revision": "fixture-revision"},
        )


def test_search_request_mapping_and_defaults() -> None:
    with running_server() as (server, backend):
        token = fetch_token(server)
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=json.dumps({"query": "xylophone"}).encode(),
        )
        assert response.status == 200
        assert response.json() == SEARCH_RESULT.to_dict()
        assert "explain" not in response.text
        assert backend.calls[-1][0] == "search"
        sent = backend.calls[-1][1]["request"]
        assert sent == SearchRequest(query="xylophone")


def test_search_full_filters_and_empty_hits() -> None:
    backend = FakeWebBackend(search_result=SearchResult(hits=(), took_ms=0.25))
    with running_server(backend) as (server, backend):
        token = fetch_token(server)
        payload = {
            "query": "needle",
            "mode": "keyword",
            "k": 3,
            "per_file": 1,
            "path": "/tmp/opencode/fixture",
            "kind": "markdown",
            "since": "7d",
        }
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=json.dumps(payload).encode(),
        )
        assert response.status == 200
        assert response.json() == {"hits": [], "took_ms": 0.25}
        assert backend.calls[-1][1]["request"] == SearchRequest(
            query="needle",
            mode="keyword",
            k=3,
            per_file=1,
            path="/tmp/opencode/fixture",
            kind="markdown",
            since="7d",
        )


def test_model_check_and_index_start_return_202() -> None:
    with running_server() as (server, backend):
        token = fetch_token(server)

        model = request(
            server,
            "POST",
            "/api/model/check",
            headers=json_headers(token),
            body=b"{}",
        )
        assert model.status == 202
        assert model.json() == {"job_id": backend.start_model_check_job_id}

        index = request(
            server,
            "POST",
            "/api/index",
            headers=json_headers(token),
            body=b"{}",
        )
        assert index.status == 202
        assert index.json() == {"job_id": backend.start_index_job_id}


def test_job_endpoint_and_not_found() -> None:
    jobs = {
        job.job_id: job
        for job in (
            JOB_RUNNING,
            JOB_SUCCEEDED,
            JOB_PARTIAL,
            JOB_FAILED,
            JOB_MODEL_SUCCEEDED,
        )
    }
    with running_server(FakeWebBackend(jobs=jobs)) as (server, _backend):
        token = fetch_token(server)
        for job in jobs.values():
            response = request(
                server,
                "GET",
                f"/api/jobs/{job.job_id}",
                headers={"X-Seekr-Token": token},
            )
            assert response.status == 200
            assert response.json() == job.to_dict()
            assert response.json()["state"] != "queued"

        missing = request(
            server,
            "GET",
            "/api/jobs/does-not-exist",
            headers={"X-Seekr-Token": token},
        )
        assert missing.status == 404
        assert missing.json()["error"]["code"] == "job_not_found"

        for path in ("/api/jobs/%2e%2e", "/api/jobs/"):
            malformed = request(
                server,
                "GET",
                path,
                headers={"X-Seekr-Token": token},
            )
            assert malformed.status == 404, path
            assert malformed.json()["error"]["code"] == "job_not_found"


# -- body policy ----------------------------------------------------------


def test_post_content_type_rules() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        body = b'{"query": "x"}'

        missing = request(
            server,
            "POST",
            "/api/search",
            headers={"X-Seekr-Token": token},
            body=body,
        )
        assert missing.status == 415
        assert missing.json()["error"]["code"] == "unsupported_media_type"

        for content_type in (
            "text/plain",
            "application/json; charset=latin-1",
            "application/json; profile=fixture",
            "application/json; charset=utf-8; extra=1",
        ):
            response = request(
                server,
                "POST",
                "/api/search",
                headers={"X-Seekr-Token": token, "Content-Type": content_type},
                body=body,
            )
            assert response.status == 415, content_type

        accepted = request(
            server,
            "POST",
            "/api/search",
            headers={
                "X-Seekr-Token": token,
                "Content-Type": "Application/JSON; charset=UTF-8",
            },
            body=body,
        )
        assert accepted.status == 200


def test_post_content_length_rules() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)

        missing = raw_exchange(
            server,
            request_lines(
                server,
                "POST",
                "/api/search",
                f"X-Seekr-Token: {token}",
                "Content-Type: application/json",
            ),
        )
        assert missing.status == 400
        assert missing.json()["error"]["code"] == "invalid_request"

        for value in ("abc", "-1", "1.0", "+1", ""):
            response = raw_exchange(
                server,
                request_lines(
                    server,
                    "POST",
                    "/api/search",
                    f"X-Seekr-Token: {token}",
                    "Content-Type: application/json",
                    f"Content-Length: {value}",
                ),
            )
            assert response.status == 400, value

        duplicate = raw_exchange(
            server,
            request_lines(
                server,
                "POST",
                "/api/search",
                f"X-Seekr-Token: {token}",
                "Content-Type: application/json",
                "Content-Length: 0",
                "Content-Length: 0",
            ),
        )
        assert duplicate.status == 400

        oversized = raw_exchange(
            server,
            request_lines(
                server,
                "POST",
                "/api/search",
                f"X-Seekr-Token: {token}",
                "Content-Type: application/json",
                f"Content-Length: {LIMITS['request_body_bytes'] + 1}",
            ),
        )
        assert oversized.status == 413
        assert oversized.json()["error"]["code"] == "payload_too_large"


def test_exact_body_limit_is_not_413() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        prefix = '{"query": "", "padding": "'
        suffix = '"}'
        padding = "x" * (
            LIMITS["request_body_bytes"] - len(prefix) - len(suffix)
        )
        body = (prefix + padding + suffix).encode()
        assert len(body) == LIMITS["request_body_bytes"]
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=body,
        )
        assert response.status == 400
        assert response.json()["error"]["code"] == "invalid_request"


def test_incomplete_body_hits_read_timeout() -> None:
    with running_server(read_timeout=0.2) as (server, _backend):
        token = fetch_token(server)
        started = time.monotonic()
        response = raw_exchange(
            server,
            request_lines(
                server,
                "POST",
                "/api/search",
                f"X-Seekr-Token: {token}",
                "Content-Type: application/json",
                "Content-Length: 10",
            ),
            body=b"{",
            read_timeout=3.0,
        )
        elapsed = time.monotonic() - started
        assert response.status == 400
        assert response.json()["error"]["code"] == "invalid_request"
        assert elapsed < 2.0


def test_slow_drip_headers_hit_total_deadline() -> None:
    with running_server(read_timeout=0.12, request_deadline=0.45) as (server, _backend):
        token = fetch_token(server)
        prefix = (
            "GET /api/status HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.bound_port}\r\n"
            f"X-Seekr-Token: {token}\r\n"
            "X-Drip: "
        ).encode("latin-1")
        closed_at, _received = drip_until_closed(server, prefix, interval=0.04)
        assert closed_at is not None
        assert closed_at < 2.0


def test_slow_drip_body_hits_total_deadline() -> None:
    with running_server(read_timeout=0.12, request_deadline=0.45) as (server, _backend):
        token = fetch_token(server)
        prefix = (
            "POST /api/search HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.bound_port}\r\n"
            f"X-Seekr-Token: {token}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 500\r\n"
            "\r\n"
        ).encode("latin-1")
        closed_at, received = drip_until_closed(server, prefix, interval=0.04)
        assert closed_at is not None
        assert closed_at < 2.0
        # A timeout while reading the body is reported as a safe JSON error
        # when the response can still be written.
        if received:
            assert b"invalid_request" in received or b"400" in received


def test_oversized_body_returns_json_413() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        body = b'{"query":"' + (b"x" * (200 * 1024)) + b'"}'
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=body,
            timeout=10.0,
        )
        assert response.status == 413
        assert response.json()["error"]["code"] == "payload_too_large"


def test_early_403_with_body_returns_json() -> None:
    with running_server() as (server, _backend):
        body = b'{"query":"' + (b"x" * (100 * 1024)) + b'"}'
        response = request(
            server,
            "POST",
            "/api/search",
            headers={
                "X-Seekr-Token": "wrong-token",
                "Content-Type": "application/json",
            },
            body=body,
            timeout=10.0,
        )
        assert response.status == 403
        assert response.json()["error"]["code"] == "forbidden"


def test_early_405_with_body_returns_json() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        body = b'{"query":"' + (b"x" * (100 * 1024)) + b'"}'
        response = request(
            server,
            "POST",
            "/api/status",
            headers=json_headers(token),
            body=body,
            timeout=10.0,
        )
        assert response.status == 405
        assert response.json()["error"]["code"] == "method_not_allowed"


def test_chunked_and_get_body_rules() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)

        chunked = raw_exchange(
            server,
            request_lines(
                server,
                "POST",
                "/api/search",
                f"X-Seekr-Token: {token}",
                "Content-Type: application/json",
                "Transfer-Encoding: chunked",
            ),
            body=b"2\r\n{}\r\n0\r\n\r\n",
        )
        assert chunked.status == 415
        assert chunked.json()["error"]["code"] == "unsupported_media_type"

        get_chunked = raw_exchange(
            server,
            request_lines(
                server,
                "GET",
                "/api/status",
                f"X-Seekr-Token: {token}",
                "Transfer-Encoding: chunked",
            ),
        )
        assert get_chunked.status == 400
        assert get_chunked.json()["error"]["code"] == "invalid_request"

        get_body = raw_exchange(
            server,
            request_lines(
                server,
                "GET",
                "/api/status",
                f"X-Seekr-Token: {token}",
                "Content-Length: 3",
            ),
            body=b"abc",
        )
        assert get_body.status == 400
        assert get_body.json()["error"]["code"] == "invalid_request"

        zero_body = request(
            server,
            "GET",
            "/api/status",
            headers={"X-Seekr-Token": token, "Content-Length": "0"},
        )
        assert zero_body.status == 200


@pytest.mark.parametrize(
    ("body", "expected_code"),
    [
        (b"{", "invalid_request"),
        (b"[]", "invalid_request"),
        (b"null", "invalid_request"),
        (b'{"query": "x", "query": "y"}', "invalid_request"),
        (b'{"query": "x", "unknown": 1}', "invalid_request"),
        (b'{"query": "x", "k": "5"}', "invalid_request"),
        (b'{"query": "x", "k": true}', "invalid_request"),
        (b'{"query": "x", "k": 0}', "invalid_request"),
        (b'{"query": "x", "k": 51}', "invalid_request"),
        (b'{"query": "x", "per_file": 11}', "invalid_request"),
        (b'{"query": "x", "mode": "bogus"}', "invalid_request"),
        (b'{"query": null}', "invalid_request"),
        (b"{}", "invalid_request"),
        (b'{"query": "x", "path": 5}', "invalid_request"),
        (b'{"query": "x", "since": 5}', "invalid_request"),
        (b'{"query": "x", "k": NaN}', "invalid_request"),
    ],
)
def test_search_json_shape_rules(body: bytes, expected_code: str) -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=body,
        )
        assert response.status == 400, body
        assert response.json()["error"]["code"] == expected_code


def test_client_supplied_field_names_are_not_echoed() -> None:
    marker = "<img src=x onerror=alert(1)>"
    with running_server() as (server, _backend):
        token = fetch_token(server)
        for path, payload in (
            ("/api/search", {"query": "x", marker: 1}),
            ("/api/model/check", {marker: 1}),
            ("/api/index", {marker: 1}),
        ):
            response = request(
                server,
                "POST",
                path,
                headers=json_headers(token),
                body=json.dumps(payload).encode(),
            )
            assert response.status == 400, path
            assert response.json()["error"]["code"] == "invalid_request"
            assert marker not in response.text


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "\ud800"},
        {"query": "x", "path": "\ud800"},
        {"query": "x", "kind": "\ud800"},
        {"query": "x", "since": "\ud800"},
        {"query": "x", "\ud800": 1},
    ],
)
def test_lone_surrogates_rejected(payload: dict[str, Any]) -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=json.dumps(payload).encode("ascii"),
        )
        assert response.status == 400
        assert response.json()["error"]["code"] == "invalid_request"


def test_valid_surrogate_pair_is_accepted() -> None:
    with running_server() as (server, backend):
        token = fetch_token(server)
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=b'{"query":"\\ud83d\\ude00"}',
        )
        assert response.status == 200
        assert backend.calls[-1][1]["request"].query == "\U0001f600"


def test_backend_response_with_surrogate_is_serialized_safely() -> None:
    hit = HitDTO(
        chunk_id=1,
        path="\ud800",
        locator="line 1",
        loc_start=1.0,
        snippet="fixture",
        score=0.5,
        kind="text",
    )
    backend = FakeWebBackend(
        search_result=SearchResult(hits=(hit,), took_ms=1.0),
    )
    with running_server(backend) as (server, _backend):
        token = fetch_token(server)
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=json.dumps({"query": "x"}).encode(),
        )
        assert response.status == 200
        assert response.json()["hits"][0]["path"] == "\ud800"
        assert b"\\ud800" in response.body


def test_query_length_limit() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=json.dumps({"query": "q" * (LIMITS["query_chars"] + 1)}).encode(),
        )
        assert response.status == 400
        assert response.json()["error"]["code"] == "invalid_request"


def test_roots_and_empty_job_bodies_reject_unknown_or_wrong_types() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        cases = (
            ("/api/roots", {"path": "/tmp/opencode/root"}),
            ("/api/roots", {"path": 5, "revision": "r"}),
            ("/api/roots", {"path": "/tmp/opencode/root", "revision": None}),
            ("/api/roots", {"path": "/tmp/opencode/root", "revision": "r", "extra": 1}),
            ("/api/model/check", {"extra": 1}),
            ("/api/index", {"extra": 1}),
        )
        for path, payload in cases:
            response = request(
                server,
                "POST",
                path,
                headers=json_headers(token),
                body=json.dumps(payload).encode(),
            )
            assert response.status == 400, (path, payload)
            assert response.json()["error"]["code"] == "invalid_request"


# -- error mapping and leakage -------------------------------------------


_ERROR_CASES = [
    ("status", "GET", "/api/status", None, "internal_error"),
    ("roots", "GET", "/api/roots", None, "config_invalid"),
    (
        "add_root",
        "POST",
        "/api/roots",
        {"path": "/tmp/opencode/root", "revision": "r"},
        "invalid_root",
    ),
    (
        "add_root",
        "POST",
        "/api/roots",
        {"path": "/tmp/opencode/root", "revision": "r"},
        "roots_limit",
    ),
    ("add_root", "POST", "/api/roots", {"path": "/p", "revision": "r"}, "busy"),
    (
        "add_root",
        "POST",
        "/api/roots",
        {"path": "/p", "revision": "r"},
        "stale_config",
    ),
    (
        "add_root",
        "POST",
        "/api/roots",
        {"path": "/p", "revision": "r"},
        "duplicate_root",
    ),
    (
        "add_root",
        "POST",
        "/api/roots",
        {"path": "/p", "revision": "r"},
        "config_invalid",
    ),
    (
        "add_root",
        "POST",
        "/api/roots",
        {"path": "/p", "revision": "r"},
        "config_write_failed",
    ),
    ("search", "POST", "/api/search", {"query": "x"}, "config_invalid"),
    ("search", "POST", "/api/search", {"query": "x"}, "busy"),
    ("search", "POST", "/api/search", {"query": "x"}, "incompatible_index"),
    ("search", "POST", "/api/search", {"query": "x"}, "index_unavailable"),
    ("search", "POST", "/api/search", {"query": "x"}, "model_unavailable"),
    ("search", "POST", "/api/search", {"query": "x"}, "daemon_failure"),
    ("search", "POST", "/api/search", {"query": "x"}, "internal_error"),
    ("start_model_check", "POST", "/api/model/check", {}, "busy"),
    ("start_model_check", "POST", "/api/model/check", {}, "config_invalid"),
    ("start_index", "POST", "/api/index", {}, "no_roots"),
    ("start_index", "POST", "/api/index", {}, "daemon_active"),
    ("start_index", "POST", "/api/index", {}, "incompatible_index"),
    ("start_index", "POST", "/api/index", {}, "config_invalid"),
    ("start_index", "POST", "/api/index", {}, "index_unavailable"),
    ("start_index", "POST", "/api/index", {}, "model_unavailable"),
    ("start_index", "POST", "/api/index", {}, "busy"),
    ("start_download", "POST", "/api/setup/download", {}, "busy"),
    ("start_download", "POST", "/api/setup/download", {"rerank": True}, "config_invalid"),
    ("course", "GET", "/api/course", None, "incompatible_index"),
    ("course", "GET", "/api/course", None, "index_unavailable"),
    ("course", "GET", "/api/course", None, "config_invalid"),
    ("course", "GET", "/api/course", None, "busy"),
    ("preview", "GET", "/api/preview?chunk=1", None, "not_found"),
    ("preview", "GET", "/api/preview?chunk=1", None, "incompatible_index"),
    ("preview", "GET", "/api/preview?chunk=1", None, "index_unavailable"),
    ("preview_page_png", "GET", "/api/preview/page?chunk=1", None, "not_found"),
    ("preview_page_png", "GET", "/api/preview/page?chunk=1", None, "incompatible_index"),
    ("open_chunk", "POST", "/api/open", {"chunk": 1}, "not_found"),
    ("open_chunk", "POST", "/api/open", {"chunk": 1}, "internal_error"),
    ("remove_root", "POST", "/api/roots/remove", {"path": "/p", "revision": "r"}, "invalid_root"),
    ("remove_root", "POST", "/api/roots/remove", {"path": "/p", "revision": "r"}, "stale_config"),
    ("remove_root", "POST", "/api/roots/remove", {"path": "/p", "revision": "r"}, "daemon_active"),
    ("remove_root", "POST", "/api/roots/remove", {"path": "/p", "revision": "r"}, "busy"),
    ("remove_root", "POST", "/api/roots/remove", {"path": "/p", "revision": "r"}, "config_invalid"),
    ("remove_root", "POST", "/api/roots/remove", {"path": "/p", "revision": "r"}, "config_write_failed"),
    ("job", "GET", "/api/jobs/fixture-job", None, "job_not_found"),
]


@pytest.mark.parametrize(
    ("method", "http_method", "path", "body", "code"),
    _ERROR_CASES,
    ids=[f"{case[4]}-{case[0]}" for case in _ERROR_CASES],
)
def test_api_error_envelope_matrix(
    method: str,
    http_method: str,
    path: str,
    body: dict[str, Any] | None,
    code: str,
) -> None:
    backend = FakeWebBackend(errors={method: _api_error(code)})
    with running_server(backend) as (server, _backend):
        token = fetch_token(server)
        headers = {"X-Seekr-Token": token}
        raw_body = None
        if body is not None:
            headers = json_headers(token)
            raw_body = json.dumps(body).encode()
        response = request(
            server,
            http_method,
            path,
            headers=headers,
            body=raw_body,
        )
        assert response.status == ERROR_STATUS[code]
        payload = response.json()
        assert payload["error"]["code"] == code
        assert payload["error"]["message"] == f"fixture {code}"


def test_500_does_not_leak_backend_text() -> None:
    with (
        capture_web_log() as log_stream,
        running_server(ExplodingBackend()) as (server, _backend),
    ):
        token = fetch_token(server)
        response = request(
            server,
            "GET",
            "/api/status",
            headers={"X-Seekr-Token": token},
        )
    assert response.status == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "SECRET_DOCUMENT_TEXT_MUST_NOT_LEAK" not in log_stream.getvalue()


def test_security_headers_and_no_store_on_error() -> None:
    expected_csp = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'"
    )
    with running_server() as (server, _backend):
        response = request(server, "GET", "/api/status")
        assert response.status == 403
        for name, value in SECURITY_HEADERS:
            assert response.header(name) == [value], name
        assert response.one_header("content-security-policy") == expected_csp
        assert response.one_header("x-frame-options") == "DENY"
        assert response.one_header("x-content-type-options") == "nosniff"
        assert response.one_header("referrer-policy") == "no-referrer"
        assert response.one_header("cache-control") == "no-store"
        assert response.one_header("content-type") == "application/json; charset=utf-8"
        assert response.one_header("connection") == "close"


# -- packaged assets ------------------------------------------------------


def _fake_asset(filename: str) -> bytes | None:
    return {
        "index.html": b"<html>fixture</html>",
        "styles.css": b"body { color: #111; }",
        "app.js": b"function fixture() {}",
    }.get(filename)


def test_asset_allowlist_and_mime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_server, "_read_asset", _fake_asset)
    with running_server() as (server, _backend):
        expected = {
            "/": ("index.html", ASSET_MIME_TYPES["/"]),
            "/index.html": ("index.html", ASSET_MIME_TYPES["/index.html"]),
            "/styles.css": ("styles.css", ASSET_MIME_TYPES["/styles.css"]),
            "/app.js": ("app.js", ASSET_MIME_TYPES["/app.js"]),
        }
        for route, (filename, mime) in expected.items():
            response = request(server, "GET", route, headers={"Origin": server.url})
            assert response.status == 200, route
            assert response.one_header("content-type") == mime
            assert response.body == _fake_asset(filename)
            assert response.one_header("cache-control") == "no-store"
            for name, value in SECURITY_HEADERS:
                assert response.header(name) == [value]


def test_missing_allowlisted_asset_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_server, "_read_asset", lambda _filename: None)
    with running_server() as (server, _backend):
        for route in ("/", "/index.html", "/styles.css", "/app.js"):
            response = request(server, "GET", route, headers={"Origin": server.url})
            assert response.status == 404, route
            assert response.json()["error"]["code"] == "not_found"


def test_actual_assets_tolerate_absence() -> None:
    asset_root = importlib.resources.files("seekr") / "web_assets"
    routes = {
        "/": "index.html",
        "/index.html": "index.html",
        "/styles.css": "styles.css",
        "/app.js": "app.js",
    }
    with running_server() as (server, _backend):
        for route, filename in routes.items():
            entry = asset_root / filename
            try:
                expected = entry.read_bytes() if entry.is_file() else None
            except OSError:
                expected = None
            response = request(server, "GET", route, headers={"Origin": server.url})
            if expected is None:
                assert response.status == 404, route
                assert response.json()["error"]["code"] == "not_found"
            else:
                assert response.status == 200, route
                assert response.body == expected
                assert response.one_header("content-type") == ASSET_MIME_TYPES[route]


def test_assets_require_origin_and_reject_wrong_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_server, "_read_asset", _fake_asset)
    with running_server() as (server, _backend):
        bare = request(server, "GET", "/styles.css")
        assert bare.status == 403
        assert "browser-origin headers are required" in bare.text

        wrong = request(
            server,
            "GET",
            "/styles.css",
            headers={"Origin": "http://example.test"},
        )
        assert wrong.status == 403

        cross_site = request(
            server,
            "GET",
            "/styles.css",
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert cross_site.status == 403

        post = request(
            server,
            "POST",
            "/styles.css",
            headers={"Origin": server.url},
            body=b"{}",
        )
        assert post.status == 405


def test_no_arbitrary_files_or_directory_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_server, "_read_asset", _fake_asset)
    with running_server() as (server, _backend):
        token = fetch_token(server)
        for path in (
            "/pyproject.toml",
            "/src/seekr/cli.py",
            "/web_assets/index.html",
            "/%2e%2e/pyproject.toml",
            "/styles.css/../../pyproject.toml",
            "/web_assets/",
        ):
            response = request(server, "GET", path, headers={"Origin": server.url})
            assert response.status == 404, path
            assert response.json()["error"]["code"] == "not_found"

        job_path = request(
            server,
            "GET",
            "/api/jobs/..",
            headers={"X-Seekr-Token": token},
        )
        assert job_path.status == 404
        assert job_path.json()["error"]["code"] == "job_not_found"


# -- keep-alive, concurrency, and shutdown -------------------------------


def test_connection_close_and_no_keep_alive() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        raw = raw_exchange(
            server,
            request_lines(
                server,
                "GET",
                "/api/status",
                f"X-Seekr-Token: {token}",
                "Connection: keep-alive",
            ),
        )
        assert raw.status == 200
        assert raw.one_header("connection") == "close"

        again = request(
            server,
            "GET",
            "/api/status",
            headers={"X-Seekr-Token": token},
        )
        assert again.status == 200


def test_ninth_sequential_request_succeeds() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)
        for _ in range(9):
            response = request(
                server,
                "GET",
                "/api/status",
                headers={"X-Seekr-Token": token},
            )
            assert response.status == 200


def test_idle_connections_do_not_starve_handler_slots() -> None:
    with running_server(read_timeout=0.25) as (server, _backend):
        token = fetch_token(server)
        idle: list[socket.socket] = []
        try:
            for _ in range(LIMITS["handler_threads"]):
                idle.append(
                    socket.create_connection(
                        ("127.0.0.1", server.bound_port),
                        timeout=2.0,
                    )
                )
            time.sleep(0.15)
            started = time.monotonic()
            response = request(
                server,
                "GET",
                "/api/status",
                headers={"X-Seekr-Token": token},
                timeout=5.0,
            )
            elapsed = time.monotonic() - started
            assert response.status == 200
            assert elapsed < 4.0
            assert server.handler_thread_count <= LIMITS["handler_threads"]
        finally:
            for connection in idle:
                connection.close()


def test_handler_thread_cap() -> None:
    backend = BlockingBackend()
    with running_server(backend) as (server, _backend):
        token = fetch_token(server)
        results: list[HttpResponse] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                response = request(
                    server,
                    "GET",
                    "/api/status",
                    headers={"X-Seekr-Token": token},
                    timeout=10.0,
                )
                with lock:
                    results.append(response)
            except BaseException as exc:  # noqa: BLE001 - test collects failures
                with lock:
                    errors.append(exc)

        workers = [threading.Thread(target=worker, daemon=True) for _ in range(10)]
        try:
            for thread in workers:
                thread.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with backend.lock:
                    if backend.max_active >= LIMITS["handler_threads"]:
                        break
                time.sleep(0.01)
            with backend.lock:
                assert backend.max_active == LIMITS["handler_threads"]
        finally:
            backend.release.set()
            for thread in workers:
                thread.join(timeout=10.0)

        assert errors == []
        assert len(results) == 10
        assert all(response.status == 200 for response in results)
        with backend.lock:
            assert backend.max_active <= LIMITS["handler_threads"]
            assert backend.daemon_flags
            assert all(flag is False for flag in backend.daemon_flags)


def test_bounded_shutdown_does_not_cancel_backend_work() -> None:
    backend = BlockingBackend()
    server = create_server(backend, port=0, read_timeout=1.0, join_timeout=0.2)
    serving = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.02},
        name="seekr-test-serve",
        daemon=True,
    )
    serving.start()
    token_response = request(server, "GET", "/api/session", headers=session_headers(server))
    token = token_response.json()["token"]

    client_errors: list[BaseException] = []

    def client() -> None:
        try:
            request(
                server,
                "GET",
                "/api/status",
                headers={"X-Seekr-Token": token},
                timeout=5.0,
            )
        except BaseException as exc:  # noqa: BLE001 - shutdown closes the socket
            client_errors.append(exc)

    client_thread = threading.Thread(target=client, daemon=True)
    try:
        client_thread.start()
        assert backend.entered.wait(5.0)
        server.shutdown()
        started = time.monotonic()
        server.server_close()
        elapsed = time.monotonic() - started
        assert elapsed < 2.0
        assert server.handler_thread_count >= 1
        assert not backend.finished.is_set()
    finally:
        backend.release.set()
        client_thread.join(timeout=5.0)
        serving.join(timeout=5.0)

    deadline = time.monotonic() + 5.0
    while server.handler_thread_count and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.handler_thread_count == 0
    server.server_close()


# -- run_web lifecycle ---------------------------------------------------


def test_run_web_reports_url_and_closes_backend() -> None:
    backend = ClosingBackend()
    stop = threading.Event()
    ready = threading.Event()
    state: dict[str, Any] = {}

    def on_listen(url: str) -> None:
        state["url"] = url
        ready.set()

    def target() -> None:
        state["code"] = run_web(
            backend,
            port=0,
            stop_event=stop,
            install_signal_handlers=False,
            on_listen=on_listen,
            read_timeout=1.0,
            join_timeout=2.0,
        )

    thread = threading.Thread(target=target, name="seekr-test-run-web", daemon=True)
    thread.start()
    try:
        assert ready.wait(5.0)
        url = state["url"]
        parsed = urlsplit(url)
        assert parsed.scheme == "http"
        assert parsed.hostname == "127.0.0.1"
        assert parsed.port and parsed.port > 0
        assert "token" not in url

        address = SimpleNamespace(bound_port=parsed.port, url=url)
        token_response = request(
            address,
            "GET",
            "/api/session",
            headers={"Origin": url, "X-Seekr-Client": "1"},
        )
        assert token_response.status == 200
        token = token_response.json()["token"]
        status = request(
            address,
            "GET",
            "/api/status",
            headers={"X-Seekr-Token": token},
        )
        assert status.status == 200
    finally:
        stop.set()
        thread.join(timeout=10.0)

    assert state["code"] == 0
    assert backend.begin_calls == 1
    assert backend.shutdown_calls == 1
    assert backend.close_calls == 1
    assert backend.order == ["begin", "shutdown", "close"]


def test_run_web_browser_failure_does_not_stop_server() -> None:
    backend = ClosingBackend()
    stop = threading.Event()
    ready = threading.Event()
    browser_failed = threading.Event()
    messages: list[str] = []
    state: dict[str, Any] = {}

    def on_listen(url: str) -> None:
        state["url"] = url
        ready.set()

    def on_browser_error(message: str) -> None:
        messages.append(message)
        browser_failed.set()

    def opener(_url: str) -> bool:
        raise RuntimeError("fixture headless environment")

    def target() -> None:
        state["code"] = run_web(
            backend,
            port=0,
            stop_event=stop,
            open_browser=True,
            browser_opener=opener,
            install_signal_handlers=False,
            on_listen=on_listen,
            on_browser_error=on_browser_error,
            read_timeout=1.0,
            join_timeout=2.0,
        )

    thread = threading.Thread(target=target, name="seekr-test-browser", daemon=True)
    thread.start()
    try:
        assert ready.wait(5.0)
        assert browser_failed.wait(5.0)
        assert "could not open a browser" in messages[0]
        address = SimpleNamespace(
            bound_port=urlsplit(state["url"]).port, url=state["url"]
        )
        response = request(
            address,
            "GET",
            "/api/session",
            headers={"Origin": state["url"], "X-Seekr-Client": "1"},
        )
        assert response.status == 200
    finally:
        stop.set()
        thread.join(timeout=10.0)

    assert state["code"] == 0
    assert backend.begin_calls == 1


def test_browser_opener_returning_false_is_reported() -> None:
    messages: list[str] = []
    web_server._try_open_browser(
        "http://127.0.0.1:1",
        lambda _url: False,
        messages.append,
    )
    assert messages
    assert "no browser was available" in messages[0]


def test_run_web_listen_callback_failure_closes_backend() -> None:
    backend = ClosingBackend()

    def on_listen(_url: str) -> None:
        raise RuntimeError("fixture listen failure")

    with pytest.raises(RuntimeError, match="fixture listen failure"):
        run_web(
            backend,
            port=0,
            install_signal_handlers=False,
            on_listen=on_listen,
        )
    assert backend.begin_calls == 1
    assert backend.shutdown_calls == 1
    assert backend.close_calls == 1
    assert backend.order == ["begin", "shutdown", "close"]


def test_begin_shutdown_precedes_blocking_backend_join() -> None:
    backend = BlockingShutdownBackend()
    stop = threading.Event()
    ready = threading.Event()
    state: dict[str, Any] = {}

    def on_listen(_url: str) -> None:
        ready.set()

    def target() -> None:
        state["code"] = run_web(
            backend,
            port=0,
            stop_event=stop,
            install_signal_handlers=False,
            on_listen=on_listen,
            join_timeout=1.0,
        )

    thread = threading.Thread(target=target, name="seekr-test-shutdown-order", daemon=True)
    thread.start()
    try:
        assert ready.wait(5.0)
        stop.set()
        assert backend.shutdown_entered.wait(5.0)
        assert backend.begin_calls == 1
        assert backend.order[:2] == ["begin", "shutdown"]
    finally:
        backend.release.set()
        thread.join(timeout=10.0)

    assert state["code"] == 0
    assert backend.order == ["begin", "shutdown", "shutdown_done", "close"]


def test_browser_launch_does_not_block_serving_or_shutdown() -> None:
    backend = ClosingBackend()
    stop = threading.Event()
    ready = threading.Event()
    opened = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    errors: list[str] = []
    state: dict[str, Any] = {}

    def on_listen(url: str) -> None:
        state["url"] = url
        ready.set()

    def opener(_url: str) -> bool:
        opened.set()
        assert release.wait(timeout=10.0), "browser opener was never released"
        finished.set()
        return True

    def target() -> None:
        state["code"] = run_web(
            backend,
            port=0,
            stop_event=stop,
            open_browser=True,
            browser_opener=opener,
            install_signal_handlers=False,
            on_listen=on_listen,
            on_browser_error=errors.append,
            join_timeout=1.0,
        )

    thread = threading.Thread(target=target, name="seekr-test-browser-delay", daemon=True)
    thread.start()
    try:
        assert ready.wait(5.0)
        assert opened.wait(5.0)
        address = SimpleNamespace(
            bound_port=urlsplit(state["url"]).port,
            url=state["url"],
        )
        response = request(
            address,
            "GET",
            "/api/session",
            headers={"Origin": state["url"], "X-Seekr-Client": "1"},
        )
        assert response.status == 200
        assert not finished.is_set()

        stop.set()
        thread.join(timeout=10.0)
        assert not thread.is_alive()
        assert state["code"] == 0
        assert not finished.is_set()
        assert errors == []
        assert backend.begin_calls == 1
        assert backend.shutdown_calls == 1
        assert backend.close_calls == 1
    finally:
        release.set()
        assert finished.wait(5.0)
        thread.join(timeout=5.0)


def test_run_web_port_in_use_closes_backend() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        backend = ClosingBackend()
        with pytest.raises(OSError):
            run_web(backend, port=port, install_signal_handlers=False)
        assert backend.shutdown_calls == 1
        assert backend.close_calls == 1
    finally:
        listener.close()


def test_signal_handlers_set_stop_and_second_warns() -> None:
    stop = threading.Event()
    notices: list[str] = []
    previous = web_server._install_signal_handlers(stop, notices.append)
    try:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert stop.is_set()
        assert len(notices) == 1
        assert "stopping" in notices[0]

        handler(signal.SIGINT, None)
        assert len(notices) == 2
        assert "still stopping" in notices[1]
    finally:
        for signum, original in previous.items():
            signal.signal(signum, original)


def test_signal_handlers_stay_installed_through_backend_shutdown() -> None:
    script = textwrap.dedent(
        """
        import os
        import signal
        import threading

        from seekr import web_server


        class Backend:
            def __init__(self):
                self.order = []
                self.begin_calls = 0
                self.close_calls = 0
                self.shutdown_entered = threading.Event()
                self.release = threading.Event()

            def begin_shutdown(self):
                self.begin_calls += 1
                self.order.append("begin")

            def shutdown(self):
                self.order.append("shutdown")
                self.shutdown_entered.set()
                if not self.release.wait(10):
                    raise RuntimeError("backend shutdown was not released")
                self.order.append("shutdown_done")

            def close(self):
                self.close_calls += 1
                self.order.append("close")


        backend = Backend()
        stop = threading.Event()
        ready = threading.Event()
        second_warning = threading.Event()
        notices = []
        state = {}


        def notice(message):
            notices.append(message)
            if len(notices) >= 2:
                second_warning.set()


        def on_listen(_url):
            ready.set()


        def coordinator():
            try:
                if not ready.wait(5):
                    return
                os.kill(os.getpid(), signal.SIGINT)
                if not backend.shutdown_entered.wait(5):
                    return
                state["during_default"] = (
                    signal.getsignal(signal.SIGINT) is signal.default_int_handler
                )
                os.kill(os.getpid(), signal.SIGINT)
                second_warning.wait(5)
            finally:
                backend.release.set()
                stop.set()


        threading.Thread(target=coordinator, daemon=True).start()
        code = web_server.run_web(
            backend,
            port=0,
            stop_event=stop,
            on_listen=on_listen,
            on_notice=notice,
            install_signal_handlers=True,
            join_timeout=1.0,
        )
        restored = signal.getsignal(signal.SIGINT) is signal.default_int_handler
        print(
            "RESULT",
            ",".join(backend.order),
            backend.begin_calls,
            backend.close_calls,
            len(notices),
            state.get("during_default"),
            restored,
            code,
            flush=True,
        )
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("RESULT ")
    )
    fields = line.split()
    assert fields[1] == "begin,shutdown,shutdown_done,close"
    assert fields[2] == "1"  # begin_shutdown called exactly once, before shutdown
    assert fields[3] == "1"  # close called once
    assert fields[4] == "2"  # first notice and second-signal warning
    assert fields[5] == "False"  # handler was still installed during backend join
    assert fields[6] == "True"  # original handler restored only after shutdown
    assert fields[7] == "0"


def test_access_log_has_no_token_or_body() -> None:
    marker = "UNIQUE_DOCUMENT_MARKER_MUST_NOT_BE_LOGGED"
    with running_server() as (server, _backend):
        token = fetch_token(server)
        with capture_web_log() as log_stream:
            response = request(
                server,
                "POST",
                "/api/search",
                headers=json_headers(token),
                body=json.dumps({"query": marker}).encode(),
            )
        assert response.status == 200
    log_text = log_stream.getvalue()
    assert log_text
    assert token not in log_text
    assert marker not in log_text


def test_search_request_accepts_module_week() -> None:
    backend = FakeWebBackend(search_result=SearchResult(hits=(), took_ms=0.5))
    with running_server(backend) as (server, backend):
        token = fetch_token(server)
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=json.dumps(
                {
                    "query": "trees",
                    "module": "machine-learning",
                    "week": "2-3",
                }
            ).encode(),
        )
        assert response.status == 200
        sent = backend.calls[-1][1]["request"]
        assert sent == SearchRequest(
            query="trees",
            module="machine-learning",
            week="2-3",
        )
        assert sent.to_query_kwargs() == {
            "query": "trees",
            "mode": "hybrid",
            "k": 10,
            "per_file": 2,
            "path_prefix": None,
            "kind": None,
            "since": None,
            "module": "machine-learning",
            "week": "2-3",
        }

        for bad_payload in (
            {"query": "trees", "module": 5},
            {"query": "trees", "week": 5},
            {"query": "trees", "module": "\ud800"},
            {"query": "trees", "week": "\ud800"},
        ):
            bad_resp = request(
                server,
                "POST",
                "/api/search",
                headers=json_headers(token),
                body=json.dumps(bad_payload).encode("ascii"),
            )
            assert bad_resp.status == 400
            assert bad_resp.json()["error"]["code"] == "invalid_request"


def test_hit_dto_has_labels() -> None:
    hit = HitDTO(
        chunk_id=101,
        path="/course/machine-learning/week4/trees.md",
        locator="§Trees",
        loc_start=12.0,
        snippet="decision trees",
        score=0.91,
        kind="markdown",
        module="machine-learning",
        week=4,
    )
    assert hit.to_dict() == {
        "chunk_id": 101,
        "path": "/course/machine-learning/week4/trees.md",
        "locator": "§Trees",
        "loc_start": 12.0,
        "snippet": "decision trees",
        "score": 0.91,
        "kind": "markdown",
        "module": "machine-learning",
        "week": 4,
        "highlights": [],
        "location": "",
        "posix_location": "",
    }

    backend = FakeWebBackend(search_result=SearchResult(hits=(hit,), took_ms=1.0))
    with running_server(backend) as (server, _backend):
        token = fetch_token(server)
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=json.dumps({"query": "trees"}).encode(),
        )
        assert response.status == 200
        assert response.json()["hits"] == [hit.to_dict()]


def test_index_html_kind_filter_includes_office_and_notebook_options() -> None:
    assets = importlib.resources.files("seekr") / "web_assets"
    html_text = (assets / "index.html").read_text(encoding="utf-8")
    js_text = (assets / "app.js").read_text(encoding="utf-8")
    for kind in ("pdf", "markdown", "text", "code", "transcript", "word", "slides", "notebook"):
        assert f'<option value="{kind}">' in html_text
        assert f"{kind}:" in js_text


def test_search_with_spread_returns_groups() -> None:
    spread_groups = (
        {"module": "machine-learning", "week": 2, "first": True, "chunk_ids": [2]},
        {"module": "machine-learning", "week": 4, "first": False, "chunk_ids": [1]},
    )
    with_spread = SearchResult(hits=(), took_ms=1.2, spread=spread_groups)
    assert with_spread.to_dict() == {
        "hits": [],
        "took_ms": 1.2,
        "spread": [
            {"module": "machine-learning", "week": 2, "first": True, "chunk_ids": [2]},
            {"module": "machine-learning", "week": 4, "first": False, "chunk_ids": [1]},
        ],
    }

    without_spread = SearchResult(hits=(), took_ms=1.2)
    assert "spread" not in without_spread.to_dict()

    with running_server(FakeWebBackend(search_result=without_spread)) as (server, no_spread_backend):
        token = fetch_token(server)
        no_spread_resp = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=json.dumps({"query": "trees", "spread": False}).encode(),
        )
        assert no_spread_resp.status == 200
        assert "spread" not in no_spread_resp.json()
        assert no_spread_backend.calls[-1][1]["request"] == SearchRequest(
            query="trees", spread=False
        )

    backend = FakeWebBackend(search_result=with_spread)
    with running_server(backend) as (server, backend):
        token = fetch_token(server)
        response = request(
            server,
            "POST",
            "/api/search",
            headers=json_headers(token),
            body=json.dumps({"query": "trees", "spread": True}).encode(),
        )
        assert response.status == 200
        assert response.json() == with_spread.to_dict()
        sent = backend.calls[-1][1]["request"]
        assert sent == SearchRequest(query="trees", spread=True)

        for bad_spread in ("true", 1, None, []):
            bad_resp = request(
                server,
                "POST",
                "/api/search",
                headers=json_headers(token),
                body=json.dumps({"query": "trees", "spread": bad_spread}).encode(),
            )
            assert bad_resp.status == 400
            assert bad_resp.json()["error"]["code"] == "invalid_request"


def test_course_preview_open_roots_remove_and_setup_download_routes() -> None:
    with running_server() as (server, backend):
        token = fetch_token(server)

        course = request(server, "GET", "/api/course", headers={"X-Seekr-Token": token})
        assert course.status == 200
        assert course.json() == COURSE_VIEW.to_dict()

        preview = request(
            server, "GET", "/api/preview?chunk=12", headers={"X-Seekr-Token": token}
        )
        assert preview.status == 200
        assert preview.json() == PREVIEW_VIEW.to_dict()
        assert backend.calls[-1] == ("preview", {"chunk_id": 12})

        page = request(
            server,
            "GET",
            "/api/preview/page?chunk=12",
            headers={"X-Seekr-Token": token},
        )
        assert page.status == 200
        assert page.body == PREVIEW_PNG_BYTES
        assert page.one_header("content-type") == "image/png"
        assert page.one_header("cache-control") == "no-store"
        assert page.one_header("x-content-type-options") == "nosniff"
        assert not any(name.startswith("access-control-") for name in page.headers)

        opened = request(
            server,
            "POST",
            "/api/open",
            headers=json_headers(token),
            body=json.dumps({"chunk": 12}).encode(),
        )
        assert opened.status == 200
        assert opened.json() == {"opened": True}
        assert backend.calls[-1] == ("open_chunk", {"chunk_id": 12})

        removed = request(
            server,
            "POST",
            "/api/roots/remove",
            headers=json_headers(token),
            body=json.dumps(
                {"path": "/tmp/opencode/docs", "revision": "rev-1"}
            ).encode(),
        )
        assert removed.status == 200
        assert removed.json() == ROOTS_VIEW.to_dict()
        assert backend.calls[-1] == (
            "remove_root",
            {"path": "/tmp/opencode/docs", "revision": "rev-1"},
        )

        dl_default = request(
            server,
            "POST",
            "/api/setup/download",
            headers=json_headers(token),
            body=b"{}",
        )
        assert dl_default.status == 202
        assert dl_default.json() == {"job_id": "job-download-fixture"}
        assert backend.calls[-1] == ("start_download", {"rerank": False})

        dl_rerank = request(
            server,
            "POST",
            "/api/setup/download",
            headers=json_headers(token),
            body=json.dumps({"rerank": True}).encode(),
        )
        assert dl_rerank.status == 202
        assert backend.calls[-1] == ("start_download", {"rerank": True})


def test_new_routes_validation_wrong_methods_and_error_envelopes() -> None:
    with running_server() as (server, _backend):
        token = fetch_token(server)

        bad_posts = [
            ("/api/open", {}),
            ("/api/open", {"chunk": 0}),
            ("/api/open", {"chunk": -5}),
            ("/api/open", {"chunk": True}),
            ("/api/open", {"chunk": "1"}),
            ("/api/open", {"chunk": None}),
            ("/api/open", {"chunk": 1, "extra": 1}),
            ("/api/roots/remove", {"path": "/tmp/a"}),
            ("/api/roots/remove", {"path": 5, "revision": "r"}),
            ("/api/roots/remove", {"path": "/tmp/a", "revision": None}),
            ("/api/roots/remove", {"path": "/tmp/a", "revision": "r", "extra": 1}),
            ("/api/setup/download", {"rerank": 1}),
            ("/api/setup/download", {"rerank": "true"}),
            ("/api/setup/download", {"rerank": None}),
            ("/api/setup/download", {"extra": True}),
        ]
        for path, payload in bad_posts:
            resp = request(
                server,
                "POST",
                path,
                headers=json_headers(token),
                body=json.dumps(payload).encode(),
            )
            assert resp.status == 400, (path, payload)
            assert resp.json()["error"]["code"] == "invalid_request"

        wrong_methods = [
            ("POST", "/api/course", b"{}"),
            ("POST", "/api/preview?chunk=1", b"{}"),
            ("POST", "/api/preview/page?chunk=1", b"{}"),
            ("GET", "/api/open", None),
            ("GET", "/api/roots/remove", None),
            ("GET", "/api/setup/download", None),
            ("OPTIONS", "/api/course", None),
            ("OPTIONS", "/api/preview?chunk=1", None),
            ("OPTIONS", "/api/preview/page?chunk=1", None),
            ("OPTIONS", "/api/open", None),
            ("OPTIONS", "/api/roots/remove", None),
            ("OPTIONS", "/api/setup/download", None),
        ]
        for method, path, body in wrong_methods:
            headers = json_headers(token) if body is not None else {"X-Seekr-Token": token}
            resp = request(server, method, path, headers=headers, body=body)
            assert resp.status == 405, (method, path)
            assert resp.json()["error"]["code"] == "method_not_allowed"
            assert not any(name.startswith("access-control-") for name in resp.headers)


def test_docs_web_api_documents_new_routes() -> None:
    doc = (ROOT / "docs" / "web-api.md").read_text(encoding="utf-8")
    for needle in (
        "/api/course",
        "/api/preview",
        "/api/preview/page",
        "/api/open",
        "/api/roots/remove",
        "/api/setup/download",
        "image/png",
        "2000",
        "download",
    ):
        assert needle in doc, needle
