"""End-to-end security tests for the local web interface transport.

Each exercise runs a real ``127.0.0.1`` loopback server — ``create_server`` or
``run_web`` — against either the synthetic ``FakeWebBackend`` or the real
:class:`seekr.web_service.WebService` with a temporary config, database and
roots and a :class:`seekr.embed.FakeEmbedder` (``embed_model = "fake"``).  No
real config, database, document root, model cache, or outbound network is ever
touched; loopback traffic is permitted.

Tests assert the frozen browser-origin, token, request-policy, error-leakage
and header rules in ``docs/web-api.md`` — reachable outcomes only, never
implementation internals.
"""

from __future__ import annotations

import http.client
import importlib.resources
import inspect
import json
import select
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from seekr import web_server
from seekr.indexer import Indexer, SyncReport
from seekr.web_contract import (
    ASSET_ALLOWLIST,
    ASSET_MIME_TYPES,
    LIMITS,
    HitDTO,
    SearchResult,
)
from seekr.web_server import SECURITY_HEADERS, create_server
from web_fixtures import FakeWebBackend

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
        error = self.json()["error"]
        assert set(error) <= {"code", "message", "fields"}
        return error["code"]

    def no_secrets(self, *secrets: str) -> None:
        text = self.text.lower()
        for marker in secrets:
            assert marker.lower() not in text, marker


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


def _drip_until_closed(
    server: Any,
    prefix: bytes,
    *,
    interval: float = 0.04,
    max_seconds: float = 2.0,
) -> tuple[float | None, bytes]:
    """Drip bytes slowly and return (seconds-until-close, received-response)."""
    with socket.create_connection(
        ("127.0.0.1", server.bound_port), timeout=3.0
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
            try:
                chunk = connection.recv(65536)
            except OSError:
                return time.monotonic() - started, bytes(received)
            if not chunk:
                return time.monotonic() - started, bytes(received)
            received.extend(chunk)
        return None, bytes(received)


# --------------------------------------------------------------------------
# temp service / server helpers
# --------------------------------------------------------------------------


def _write_config(
    tmp_path: Path,
    *,
    roots: list[Path] | None = None,
    db_path: Path | None = None,
    raw: str = "",
) -> Path:
    db = db_path or (tmp_path / "index.db")
    lines = [
        'embed_model = "fake"\n',
        f"db_path = {json.dumps(str(db))}\n",
        'vector_quant = "int8"\n',
    ]
    if roots:
        lines.append(
            "roots = [" + ", ".join(json.dumps(str(root)) for root in roots) + "]\n"
        )
    path = tmp_path / "config.toml"
    path.write_text("".join(lines) + raw, encoding="utf-8")
    return path


def _root(tmp_path: Path, name: str = "documents") -> Path:
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "note.txt").write_text("alpha xylophone notes\n", encoding="utf-8")
    return path


@contextmanager
def _running(
    backend: Any,
    **server_options: Any,
) -> Iterator[web_server.SeekrHTTPServer]:
    server = create_server(backend, port=0, **server_options)
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


def _bootstrap(server: Any) -> str:
    response = _request(
        server,
        "GET",
        "/api/session",
        headers={"Origin": server.url, "X-Seekr-Client": "1"},
    )
    assert response.status == 200, response.text
    payload = response.json()
    assert set(payload) == {"token", "limits"}
    assert payload["limits"] == LIMITS
    return payload["token"]


def _session_headers(server: Any) -> dict[str, str]:
    return {"Origin": server.url, "X-Seekr-Client": "1"}


def _token(token: str) -> dict[str, str]:
    return {"X-Seekr-Token": token}


def _json_post(token: str, payload: dict[str, Any]) -> tuple[dict[str, str], bytes]:
    return (
        {"X-Seekr-Token": token, "Content-Type": "application/json"},
        json.dumps(payload).encode("utf-8"),
    )


def _search(server: Any, token: str, **fields: Any) -> Response:
    headers, body = _json_post(token, fields)
    return _request(server, "POST", "/api/search", headers=headers, body=body)


def _empty_index(server: Any, token: str) -> Response:
    headers, body = _json_post(token, {})
    return _request(server, "POST", "/api/index", headers=headers, body=body)


def _index_body(token: str) -> tuple[dict[str, str], bytes]:
    return _json_post(token, {})


def _model_check(server: Any, token: str) -> Response:
    headers, body = _json_post(token, {})
    return _request(server, "POST", "/api/model/check", headers=headers, body=body)


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


class CountingBackend(FakeWebBackend):
    """The backend plus a per-method call counter used for preflight proofs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[str] = []

    def _record(self, method: str, **arguments: object) -> None:
        del arguments
        self.calls.append(method)


class ExplodingBackend(FakeWebBackend):
    """Every backend call raises a secret the responses must never leak."""

    SECRET = "SECRET_FIXTURE_TEXT_MUST_NOT_LEAK"

    def status(self) -> Any:
        raise RuntimeError(f"{self.SECRET} boom")

    def roots(self) -> Any:
        raise PermissionError(f"{self.SECRET} denied")

    def add_root(self, path: str, revision: str) -> Any:
        raise ValueError(f"{self.SECRET} {path} {revision}")

    def search(self, request: Any) -> Any:
        raise OSError(f"{self.SECRET} {request.query}")


class SurrogateBackend(FakeWebBackend):
    """Returns a hit with a lone surrogate to prove safe JSON serialization."""

    def search(self, request: Any) -> Any:
        del request
        return SearchResult(
            hits=(
                HitDTO(
                    chunk_id=9,
                    path="/docs/surrogate.txt",
                    locator="line 1",
                    loc_start=1.0,
                    snippet="surrogate \ud800 text",
                    score=0.5,
                    kind="text",
                ),
            ),
            took_ms=0.25,
        )


# --------------------------------------------------------------------------
# Host / Origin / fetch-metadata rules (FakeWebBackend, real loopback HTTP)
# --------------------------------------------------------------------------


def test_host_must_be_the_exact_loopback_address_and_actual_port() -> None:
    with _running(FakeWebBackend()) as server:
        assert server.bound_port > 0  # an ephemeral port is resolved before requests
        token = _bootstrap(server)
        valid = {"X-Seekr-Token": token}

        for host in (
            f"localhost:{server.bound_port}",
            f"127.0.0.1:{server.bound_port + 1}",
            "127.0.0.1",
            f"127.0.0.1:{server.bound_port} ",
            f"[::1]:{server.bound_port}",
        ):
            response = _request(
                server, "GET", "/api/status", headers={**valid, "Host": host}
            )
            assert response.status == 403, host
            assert response.error_code() == "forbidden"
            # the untrusted Host value is never echoed back
            assert host.strip() not in response.text

        missing = _raw_exchange(
            server,
            ["GET /api/status HTTP/1.1", f"X-Seekr-Token: {token}"],
        )
        assert missing.status == 403  # a request without any Host header

        duplicate = _raw_exchange(
            server,
            _request_lines(
                server,
                "GET",
                "/api/status",
                f"Host: 127.0.0.1:{server.bound_port}",
                f"Host: 127.0.0.1:{server.bound_port}",
                f"X-Seekr-Token: {token}",
            ),
        )
        assert duplicate.status == 403

        # a forwarded-host alias must not substitute for the exact Host
        forwarded = _raw_exchange(
            server,
            _request_lines(
                server,
                "GET",
                "/api/status",
                f"X-Forwarded-Host: 127.0.0.1:{server.bound_port}",
                f"X-Seekr-Token: {token}",
            ),
        )
        assert forwarded.status == 200  # an extra unvalidated header is ignored

        accepted = _request(server, "GET", "/api/status", headers=valid)
        assert accepted.status == 200


def test_origin_and_sec_fetch_site_rules() -> None:
    with _running(FakeWebBackend()) as server:
        token = _bootstrap(server)
        valid = {"X-Seekr-Token": token}

        for origin in (
            f"http://localhost:{server.bound_port}",
            f"https://127.0.0.1:{server.bound_port}",
            f"http://127.0.0.1:{server.bound_port + 1}",
            "http://example.test",
        ):
            response = _request(
                server, "GET", "/api/status", headers={**valid, "Origin": origin}
            )
            assert response.status == 403, origin
            assert response.error_code() == "forbidden"
            assert origin not in response.text

        for site in ("cross-site", "same-site"):
            response = _request(
                server, "GET", "/api/status", headers={**valid, "Sec-Fetch-Site": site}
            )
            assert response.status == 403, site
            assert response.error_code() == "forbidden"

        for site in ("same-origin", "none"):
            response = _request(
                server, "GET", "/api/status", headers={**valid, "Sec-Fetch-Site": site}
            )
            assert response.status == 200, site

        # a matching Origin together with cross-site fetch metadata is rejected
        response = _request(
            server,
            "GET",
            "/api/status",
            headers={
                **valid,
                "Origin": server.url,
                "Sec-Fetch-Site": "cross-site",
            },
        )
        assert response.status == 403


def test_headerless_session_and_asset_requests_get_a_browser_origin_hint() -> None:
    with _running(FakeWebBackend()) as server:
        for path in ("/api/session", "/", "/index.html", "/styles.css", "/app.js"):
            headerless = _request(server, "GET", path)
            assert headerless.status == 403, path
            assert headerless.error_code() == "forbidden"
            assert "browser-origin headers are required" in headerless.text

        # token-protected routes allow headerless requests with the token only
        token = _bootstrap(server)
        bare = _request(server, "GET", "/api/status", headers=_token(token))
        assert bare.status == 200

        sec_fetch_none = _request(
            server,
            "GET",
            "/api/status",
            headers={**_token(token), "Sec-Fetch-Site": "none"},
        )
        assert sec_fetch_none.status == 200

        origin_matched = _request(
            server, "GET", "/api/status", headers={**_token(token), "Origin": server.url}
        )
        assert origin_matched.status == 200

        asset = _request(
            server,
            "GET",
            "/",
            headers={"Origin": server.url, "Sec-Fetch-Site": "same-origin"},
        )
        assert asset.status == 200

        # duplicate browser-origin headers are rejected
        duplicate = _raw_exchange(
            server,
            _request_lines(
                server,
                "GET",
                "/",
                f"Origin: {server.url}",
                f"Origin: {server.url}",
            ),
        )
        assert duplicate.status == 403


# --------------------------------------------------------------------------
# session bootstrap and the ephemeral token
# --------------------------------------------------------------------------


def test_session_bootstrap_requires_exactly_one_client_header() -> None:
    with _running(FakeWebBackend()) as server:
        for headers in (
            {"Origin": server.url},
            {"Origin": server.url, "X-Seekr-Client": "2"},
            {"Origin": server.url, "X-Seekr-Client": "true"},
        ):
            response = _request(server, "GET", "/api/session", headers=headers)
            assert response.status == 400, headers
            assert response.error_code() == "invalid_request"

        duplicate = _raw_exchange(
            server,
            _request_lines(
                server,
                "GET",
                "/api/session",
                f"Origin: {server.url}",
                "X-Seekr-Client: 1",
                "X-Seekr-Client: 1",
            ),
        )
        assert duplicate.status == 400

        with_client = _request(
            server,
            "GET",
            "/api/session",
            headers={"Origin": server.url, "X-Seekr-Client": "1"},
        )
        assert with_client.status == 200


def test_token_rules_and_ephemeral_memory_only_token() -> None:
    tokens: list[str] = []
    with _running(FakeWebBackend()) as server:
        token = _bootstrap(server)
        tokens.append(token)

        for headers in ({}, {"X-Seekr-Token": ""}, {"X-Seekr-Token": token + "x"}):
            response = _request(server, "GET", "/api/status", headers=headers)
            assert response.status == 403, headers
            assert response.error_code() == "forbidden"

        # a similar token of different length is no shortcut; the comparison is
        # hmac.compare_digest (source-asserted in a dedicated test)
        mismatched = token[: len(token) // 2]
        response = _request(
            server, "GET", "/api/status", headers={"X-Seekr-Token": mismatched}
        )
        assert response.status == 403

        duplicate = _raw_exchange(
            server,
            _request_lines(
                server,
                "GET",
                "/api/status",
                f"X-Seekr-Token: {token}",
                f"X-Seekr-Token: {token}",
            ),
        )
        assert duplicate.status == 403

        # the token is one process-lifetime value, never rotated per request
        for _ in range(3):
            assert _bootstrap(server) == token

    # restarting the server invalidates the token
    with _running(FakeWebBackend()) as server:
        old_token = _bootstrap(server)
    with _running(FakeWebBackend()) as server:
        response = _request(server, "GET", "/api/status", headers=_token(old_token))
        assert response.status == 403
        assert response.error_code() == "forbidden"

        new_token = _bootstrap(server)
        assert new_token != old_token
        tokens.append(new_token)

    assert len(set(tokens)) == 2 and tokens[0] != tokens[1]
    assert token not in old_token


def test_token_comparison_uses_hmac_compare_digest() -> None:
    source = inspect.getsource(web_server)
    assert "hmac.compare_digest" in source
    assert "server.token.encode" in source


def test_token_stays_out_of_urls_and_html() -> None:
    leak_checks: list[bool] = []
    with _running(FakeWebBackend()) as server:
        token = _bootstrap(server)

        # a token as a query string must not open a shortcut around the header
        response = _request(
            server, "GET", f"/api/status?X-Seekr-Token={token}", headers=_token(token)
        )
        assert response.status == 200

        for path in ("/", "/index.html", "/styles.css", "/app.js"):
            asset = _request(server, "GET", path, headers={"Origin": server.url})
            assert asset.status == 200, path
            assert token.encode() not in asset.body  # never in a served asset
            leak_checks.append(token.encode() not in asset.body)

        response = _request(server, "GET", "/api/status", headers=_token(token))
        assert response.status == 200
        leak_checks.append(token not in response.text)

    assert leak_checks and all(leak_checks)


# --------------------------------------------------------------------------
# no CORS, OPTIONS, form posts, methods and routes
# --------------------------------------------------------------------------


def test_no_cors_headers_are_ever_emitted() -> None:
    with _running(FakeWebBackend()) as server:
        token = _bootstrap(server)
        responses = [
            _request(server, "GET", "/api/session", headers=_session_headers(server)),
            _request(server, "GET", "/api/status", headers=_token(token)),
            _request(server, "GET", "/", headers={"Origin": server.url}),
            _request(server, "OPTIONS", "/api/status"),
            _request(server, "GET", "/api/nope", headers=_token(token)),
            _request(server, "GET", "/api/status", headers={"X-Seekr-Token": "wrong"}),
        ]
        for response in responses:
            assert not any(
                name.startswith("access-control-") for name in response.headers
            ), response.headers

        options = responses[3]
        assert options.status == 405
        assert options.error_code() == "method_not_allowed"


def test_options_is_405_after_host_validation_and_preflight_fails() -> None:
    with _running(FakeWebBackend()) as server:
        preflight = _request(
            server,
            "OPTIONS",
            "/api/status",
            headers={
                "Host": f"localhost:{server.bound_port}",
                "Origin": "http://example.test",
                "Sec-Fetch-Site": "cross-site",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status == 403  # Host is validated before OPTIONS routing
        assert preflight.error_code() == "forbidden"
        assert not any(
            name.startswith("access-control-") for name in preflight.headers
        )

        valid_host = _request(server, "OPTIONS", "/api/status")
        assert valid_host.status == 405  # 405 before token or fetch-metadata checks
        assert valid_host.error_code() == "method_not_allowed"

        same_host_preflight = _request(
            server,
            "OPTIONS",
            "/api/status",
            headers={
                "Origin": server.url,
                "Sec-Fetch-Site": "same-origin",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert same_host_preflight.status == 405  # preflights are never answered
        assert not any(
            name.startswith("access-control-") for name in same_host_preflight.headers
        )


def test_simple_form_posts_are_rejected_as_unsupported_media() -> None:
    with _running(FakeWebBackend()) as server:
        token = _bootstrap(server)
        form = "query=xylophone&mode=keyword"
        for content_type in (
            "application/x-www-form-urlencoded",
            "text/plain",
            "multipart/form-data",
            "application/json; charset=latin-1",
            "APPLICATION/JSON; PROFILE=x",
            "application/json; charset=utf-8; boundary=x",
        ):
            headers = {**_token(token), "Content-Type": content_type}
            response = _request(
                server, "POST", "/api/search", headers=headers, body=form.encode()
            )
            assert response.status == 415, content_type
            assert response.error_code() == "unsupported_media_type"

        accepted = _request(
            server,
            "POST",
            "/api/search",
            headers={
                **_token(token),
                "Content-Type": "Application/JSON; charset=UTF-8",
            },
            body=b'{"query": "xylophone"}',
        )
        assert accepted.status == 200  # case-insensitive type, utf-8 charset only


def test_simple_form_posts_never_reach_the_backend() -> None:
    backend = CountingBackend()
    with _running(backend) as server:
        token = _bootstrap(server)
        headers = {
            **_token(token),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = _request(
            server,
            "POST",
            "/api/search",
            headers=headers,
            body=b"query=xylophone&mode=keyword",
        )
        assert response.status == 415
        assert response.error_code() == "unsupported_media_type"
        assert backend.calls == []  # rejected before any backend call


# --------------------------------------------------------------------------
# request limits: sizes, media types, framing, JSON shapes
# --------------------------------------------------------------------------


def test_oversized_bodies_are_413_before_any_backend_call() -> None:
    backend = CountingBackend()
    with _running(backend) as server:
        token = _bootstrap(server)
        limit = LIMITS["request_body_bytes"]

        oversized = bytes(limit + 1)
        response = _raw_exchange(
            server,
            _request_lines(
                server,
                "POST",
                "/api/search",
                f"X-Seekr-Token: {token}",
                "Content-Type: application/json",
                f"Content-Length: {limit + 1}",
            ),
            body=oversized,
        )
        assert response.status == 413
        assert response.error_code() == "payload_too_large"
        assert backend.calls == []  # rejected before any backend call

        limit_exact = (
            b'{"query": "", "path": "'
            + b"x" * (limit - len(b'{"query": "", "path": ""}'))
            + b'"}'
        )
        assert len(limit_exact) == limit
        response = _request(
            server,
            "POST",
            "/api/search",
            headers={**_token(token), "Content-Type": "application/json"},
            body=limit_exact,
        )
        assert response.status == 200  # exactly the limit is accepted
        assert set(response.json()) == {"hits", "took_ms"}
        assert backend.calls == ["search"]  # and never before the body is accepted


def test_content_length_ambiguity_and_framing_rules() -> None:
    with _running(FakeWebBackend()) as server:
        token = _bootstrap(server)

        for length in ("abc", "-1", "1.5", "+2", "0x10"):
            response = _raw_exchange(
                server,
                _request_lines(
                    server,
                    "POST",
                    "/api/search",
                    f"X-Seekr-Token: {token}",
                    "Content-Type: application/json",
                    f"Content-Length: {length}",
                ),
            )
            assert response.status == 400, length
            assert response.error_code() == "invalid_request"

        duplicate = _raw_exchange(
            server,
            _request_lines(
                server,
                "POST",
                "/api/search",
                f"X-Seekr-Token: {token}",
                "Content-Type: application/json",
                "Content-Length: 2",
                "Content-Length: 2",
            ),
        )
        assert duplicate.status == 400

        missing = _raw_exchange(
            server,
            _request_lines(
                server,
                "POST",
                "/api/search",
                f"X-Seekr-Token: {token}",
                "Content-Type: application/json",
            ),
        )
        assert missing.status == 400

        chunked = _raw_exchange(
            server,
            _request_lines(
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
        assert chunked.error_code() == "unsupported_media_type"

        get_chunked = _raw_exchange(
            server,
            _request_lines(
                server,
                "GET",
                "/api/status",
                f"X-Seekr-Token: {token}",
                "Transfer-Encoding: chunked",
            ),
        )
        assert get_chunked.status == 400
        assert get_chunked.error_code() == "invalid_request"

        # GET requests take no body beyond Content-Length: 0
        response = _request(
            server, "GET", "/api/status", headers=_token(token), body=b"{}"
        )
        assert response.status == 400

        zero = _request(
            server,
            "GET",
            "/api/status",
            headers={**_token(token), "Content-Length": "0"},
        )
        assert zero.status == 200


def test_slow_drip_headers_hit_total_request_deadline() -> None:
    with _running(
        FakeWebBackend(), read_timeout=0.2, request_deadline=0.5
    ) as server:
        token = _bootstrap(server)
        prefix = (
            "GET /api/status HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.bound_port}\r\n"
            f"X-Seekr-Token: {token}\r\n"
            "X-Drip: "
        ).encode("latin-1")
        closed_at, received = _drip_until_closed(server, prefix, interval=0.04)
        assert closed_at is not None
        assert closed_at < 2.0
        assert b"200 OK" not in received


@pytest.mark.parametrize(
    ("body", "description"),
    [
        (b"{", "truncated"),
        (b"[]", "array"),
        (b"null", "null"),
        (b'"a string"', "string"),
        (b'{"query": "x", "query": "y"}', "duplicate-key"),
        (b'{"query": "x", "unknown": 1}', "unknown-field"),
        (b'{"query": "x", "k": "5"}', "string-k"),
        (b'{"query": "x", "k": true}', "boolean-k"),
        (b'{"query": "x", "k": false}', "boolean-false-k"),
        (b'{"query": "x", "k": 0}', "low-k"),
        (b'{"query": "x", "k": 51}', "high-k"),
        (b'{"query": "x", "per_file": 11}', "high-per-file"),
        (b'{"query": "x", "mode": "bogus"}', "bad-mode"),
        (b'{"query": null}', "null-query"),
        (b"{}", "missing-query"),
        (b'{"query": "x", "path": 5}', "numeric-path"),
        (b'{"query": "x", "since": 5}', "numeric-since"),
        (b'{"query": "x", "k": NaN}', "nan"),
        (b'{"query": "\x00"}', "nul"),
    ],
    ids=lambda description: description,
)
def test_search_json_shapes_are_invalid_requests(body: bytes, description: str) -> None:
    with _running(FakeWebBackend()) as server:
        token = _bootstrap(server)
        headers = {**_token(token), "Content-Type": "application/json"}
        response = _request(
            server, "POST", "/api/search", headers=headers, body=body
        )
        assert response.status == 400, (description, body)
        assert response.error_code() == "invalid_request"
        # none of the malformed shapes echoes the request body back
        if len(body) > 8:
            assert body not in response.body


def test_lone_json_surrogates_are_invalid_requests_not_500() -> None:
    with _running(FakeWebBackend()) as server:
        token = _bootstrap(server)
        headers = {**_token(token), "Content-Type": "application/json"}
        for body in (
            b'{"query": "\\ud800"}',
            b'{"query": "x", "path": "\\udc00"}',
            b'{"query": "x", "kind": "\\ud800"}',
            b'{"query": "x", "since": "\\ud800"}',
        ):
            response = _request(
                server, "POST", "/api/search", headers=headers, body=body
            )
            assert response.status == 400, body
            assert response.error_code() == "invalid_request"
            assert b"internal_error" not in response.body

        # a complete valid UTF-16 surrogate pair is accepted and forwarded
        accepted = _request(
            server,
            "POST",
            "/api/search",
            headers=headers,
            body=b'{"query": "\\ud83d\\ude00"}',
        )
        assert accepted.status == 200
        assert set(accepted.json()) == {"hits", "took_ms"}


def test_backend_surrogates_are_json_escaped_not_500() -> None:
    with _running(SurrogateBackend()) as server:
        token = _bootstrap(server)
        response = _request(
            server,
            "POST",
            "/api/search",
            headers={**_token(token), "Content-Type": "application/json"},
            body=b'{"query": "x"}',
        )
        assert response.status == 200
        payload = response.json()
        assert payload["hits"][0]["snippet"] == "surrogate \ud800 text"
        assert b"\\ud800" in response.body
        assert b"internal_error" not in response.body


def test_empty_and_index_bodies_and_wrong_methods() -> None:
    with _running(FakeWebBackend()) as server:
        token = _bootstrap(server)
        for path in ("/api/model/check", "/api/index"):
            response = _request(
                server,
                "POST",
                path,
                headers={**_token(token), "Content-Type": "application/json"},
                body=b'{"extra": 1}',
            )
            assert response.status == 400, path
            assert response.error_code() == "invalid_request"

            response = _request(
                server,
                "POST",
                path,
                headers={**_token(token), "Content-Type": "application/json"},
                body=b"null",
            )
            assert response.status == 400, path
            assert response.error_code() == "invalid_request"

        wrong_methods = (
            ("GET", "/api/search"),
            ("POST", "/api/status"),
            ("POST", "/api/session"),
            ("DELETE", "/api/status"),
            ("PUT", "/api/index"),
            ("PATCH", "/api/roots"),
            ("POST", "/"),
            ("POST", "/styles.css"),
        )
        for method, path in wrong_methods:
            headers = (
                {**_token(token), "Content-Type": "application/json"}
                if method == "POST"
                else _token(token)
            )
            response = _request(server, method, path, headers=headers, body=b"{}")
            assert response.status == 405, (method, path)
            assert response.error_code() == "method_not_allowed"

        for path in (
            "/api/unknown",
            "/api",
            "/api/",
            "/api/session/extra",
            "/api/status/extra",
            "/api/jobs/a/b",
            "/web_assets/app.js",
            "/index.html.bak",
            "/app.js.map",
            "/nope",
        ):
            response = _request(server, "GET", path, headers={"Origin": server.url})
            assert response.status == 404, path
            assert response.error_code() == "not_found"

        # a malformed request line is an invalid_request, not an HTML page
        response = _raw_exchange(server, ["NOT A REQUEST"])
        assert response.status == 400
        assert response.error_code() == "invalid_request"
        assert b"<html" not in response.body


# --------------------------------------------------------------------------
# asset allowlist: traversal, encoded paths, no listing, packaged resources
# --------------------------------------------------------------------------


def test_asset_traversal_and_encoded_paths_are_not_served() -> None:
    with _running(FakeWebBackend()) as server:
        for path in (
            "/web_assets/index.html",
            "/web_assets/",
            "/pyproject.toml",
            "/src/seekr/cli.py",
            "/../pyproject.toml",
            "/../../seekr/config.py",
            "/%2e%2e/pyproject.toml",
            "/%73tyles.css",
            "/styles.css%00.png",
            "/app.js%2f..%2f..%2fpyproject.toml",
            "/./index.html",
            "/index%2ehtml",
            "/index.html/",
            "/INDEX.HTML",
            "/styles.css/",
        ):
            response = _request(server, "GET", path, headers={"Origin": server.url})
            assert response.status == 404, path
            assert response.error_code() == "not_found"
            assert b"def " not in response.body  # no source, no directory listing
            assert b"Directory listing" not in response.body

        # the frozen allowlist is unchanged and everything real is still served
        assert ASSET_ALLOWLIST == ("/", "/index.html", "/styles.css", "/app.js")
        for route, mime in ASSET_MIME_TYPES.items():
            response = _request(server, "GET", route, headers={"Origin": server.url})
            assert response.status == 200, route
            assert response.one("content-type") == mime


def test_assets_are_served_from_package_resources_with_exact_mime_types() -> None:
    with _running(FakeWebBackend()) as server:
        for route, expected in ASSET_MIME_TYPES.items():
            name = "index.html" if route == "/" else route.lstrip("/")
            resource = importlib.resources.files("seekr") / "web_assets" / name
            body = resource.read_bytes()
            response = _request(server, "GET", route, headers={"Origin": server.url})
            assert response.status == 200, route
            assert response.body == body  # packaged resources, byte for byte
            assert response.one("content-type") == expected
            assert response.one("cache-control") == "no-store"
            for name_, value in SECURITY_HEADERS:
                assert response.header(name_) == [value]
            assert not any(
                header.startswith("access-control-") for header in response.headers
            )

        # a missing packaged asset is 404 not_found, not a fallback file
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(web_server, "_read_asset", lambda filename: None)
            for route in ASSET_MIME_TYPES:
                response = _request(
                    server, "GET", route, headers={"Origin": server.url}
                )
                assert response.status == 404, route
                assert response.error_code() == "not_found"


def test_no_document_serving_route_or_directory_listing(tmp_path: Path) -> None:
    document = tmp_path / "document.txt"
    document.write_text("SECRET_DOCUMENT_TEXT_MUST_NOT_BE_SERVED\n", encoding="utf-8")
    with _running(FakeWebBackend()) as server:
        for path in (
            "/document.txt",
            f"/{document.name}",
            f"/documents/{document.name}",
            f"/docs/{document.name}",
            f"/files/{document.name}",
            f"/download?path={document}",
            f"/open?path={document}",
        ):
            response = _request(server, "GET", path, headers={"Origin": server.url})
            assert response.status == 404, path
            assert b"SECRET_DOCUMENT" not in response.body


# --------------------------------------------------------------------------
# frozen security headers and caching
# --------------------------------------------------------------------------


def test_frozen_security_headers_on_every_response_including_errors(
    tmp_path: Path,
) -> None:
    blocked: list[str] = []

    class LeakyService:  # the failing error here is a secret the response must
        SECRET = "SECRET_FIXTURE_DOCUMENT_TEXT"  # never include

        def status(self) -> Any:
            raise RuntimeError(f"leaking {self.SECRET}")

    with _running(FakeWebBackend()) as server:
        token = _bootstrap(server)
        responses = [
            _request(server, "GET", "/api/session", headers=_session_headers(server)),
            _request(server, "GET", "/api/status", headers=_token(token)),
            _request(server, "GET", "/", headers={"Origin": server.url}),
            _request(server, "OPTIONS", "/api/status"),
            _request(server, "GET", "/not-a-route", headers={"Origin": server.url}),
            _request(server, "GET", "/api/nope", headers=_token(token)),
        ]
        with _running(LeakyBackend()) as server:
            token = _bootstrap(server)
            responses = responses + [
                # an internal backend failure leaking only a safe message
                _request(server, "GET", "/api/status", headers=_token(token)),
                # a token-protected route with a wrong token
                _request(
                    server, "GET", "/api/nope", headers={"X-Seekr-Token": "wrong"}
                ),
            ]
        del blocked
        csp = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; form-action 'self'; "
            "frame-ancestors 'none'"
        )
        assert SECURITY_HEADERS == (
            ("Content-Security-Policy", csp),
            ("X-Frame-Options", "DENY"),
            ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"),
        )
        for response in responses:
            for name_, value in SECURITY_HEADERS:
                assert response.header(name_) == [value], name_
            assert response.one("cache-control") == "no-store"
            assert not any(
                name.startswith("access-control-") for name in response.headers
            )
            response.no_secrets(
                "SECRET_FIXTURE", "traceback", "Traceback", LeakyService.SECRET
            )
            # the exception text must not leak document text either
            assert b"SECRET" not in response.body


class LeakyBackend(FakeWebBackend):
    """Thé failing backend whose exception text is a documented-leak marker."""

    SECRET_TEXT = "SECRET_FIXTURE_DOCUMENT_TEXT"

    def status(self) -> Any:
        raise RuntimeError(f"{self.SECRET_TEXT} boom")
