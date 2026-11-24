"""Hardened, loopback-only HTTP transport for the local Seekr web interface.

The module owns HTTP parsing, browser-origin protections, the session token,
the fixed asset allowlist, and server lifecycle.  All domain work is delegated
to a constructor-injected :class:`seekr.web_contract.WebBackend`; handlers never
open a Store, load a model, or read configuration directly.

This module intentionally uses only the Python standard library and
``seekr.web_contract`` so HTTP tests can run without the application service.
"""

from __future__ import annotations

import hmac
import importlib.resources
import json
import logging
import os
import re
import secrets
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .web_contract import (
    ASSET_ALLOWLIST,
    ASSET_MIME_TYPES,
    LIMITS,
    ApiError,
    SearchRequest,
    SimilarRequest,
    WebBackend,
)

__all__ = [
    "ACCESS_LOG_NAME",
    "SECURITY_HEADERS",
    "SeekrHTTPServer",
    "WebRequestHandler",
    "create_server",
    "run_web",
]

ACCESS_LOG_NAME = "seekr.web"
_logger = logging.getLogger(ACCESS_LOG_NAME)

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)

#: Frozen response header baseline.  These are emitted exactly once per response.
SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("Content-Security-Policy", CSP),
    ("X-Frame-Options", "DENY"),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
)

_JSON_CONTENT_TYPE = "application/json; charset=utf-8"
_PNG_CONTENT_TYPE = "image/png"
_GET_METHOD = "GET"
_POST_METHOD = "POST"
_ORIGIN_FETCH_SITE_VALUES = frozenset({"same-origin", "none"})
_SEARCH_MODES = frozenset({"keyword", "vector", "hybrid"})
_CONTENT_LENGTH_RE = re.compile(r"[0-9]+")
_JOB_ID_RE = re.compile(r"[A-Za-z0-9_.~-]{1,200}")
_PREVIEW_QUERY_RE = re.compile(r"chunk=([1-9][0-9]{0,18})")
_PREVIEW_ROUTES = frozenset({"preview", "preview_page"})

#: Bounded post-response drain so an early error does not reset the client
#: before it can read the JSON envelope.  The one-second grace is shorter than
#: the request deadline and keeps error handling from becoming a slow-read DoS.
_MAX_DRAIN_BYTES = 1024 * 1024
_MAX_DRAIN_SECONDS = 1.0

_JSON_SURROGATE_MIN = 0xD800
_JSON_SURROGATE_MAX = 0xDFFF

_ASSET_FILENAMES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
}

#: Exact request-target -> (packaged filename, frozen MIME type).
_ASSET_ROUTES: dict[str, tuple[str, str]] = {
    route: (_ASSET_FILENAMES[route], ASSET_MIME_TYPES[route])
    for route in ASSET_ALLOWLIST
}

#: Exact request-target -> (route name, allowed methods).
_API_ROUTES: dict[str, tuple[str, frozenset[str]]] = {
    "/api/session": ("session", frozenset({_GET_METHOD})),
    "/api/status": ("status", frozenset({_GET_METHOD})),
    "/api/roots": ("roots", frozenset({_GET_METHOD, _POST_METHOD})),
    "/api/roots/remove": ("roots_remove", frozenset({_POST_METHOD})),
    "/api/course": ("course", frozenset({_GET_METHOD})),
    "/api/preview": ("preview", frozenset({_GET_METHOD})),
    "/api/preview/page": ("preview_page", frozenset({_GET_METHOD})),
    "/api/open": ("open", frozenset({_POST_METHOD})),
    "/api/search": ("search", frozenset({_POST_METHOD})),
    "/api/similar": ("similar", frozenset({_POST_METHOD})),
    "/api/shutdown": ("shutdown", frozenset({_POST_METHOD})),
    "/api/index/paths": ("index_paths", frozenset({_POST_METHOD})),
    "/api/model/check": ("model_check", frozenset({_POST_METHOD})),
    "/api/index": ("index", frozenset({_POST_METHOD})),
    "/api/reindex": ("reindex", frozenset({_POST_METHOD})),
    "/api/setup/download": ("setup_download", frozenset({_POST_METHOD})),
}


def _read_asset(filename: str) -> bytes | None:
    """Read one file from the packaged ``web_assets`` directory.

    The caller resolves only fixed allowlisted names, so this never exposes an
    arbitrary path.  A missing directory or file is reported as ``None``.
    """
    try:
        asset_root = importlib.resources.files("seekr") / "web_assets"
        asset = asset_root / filename
        if not asset.is_file():
            return None
        return asset.read_bytes()
    except (OSError, ModuleNotFoundError, ValueError):
        return None


def _forbidden(message: str) -> ApiError:
    return ApiError(403, "forbidden", message)


def _hint_forbidden() -> ApiError:
    return _forbidden(
        "browser-origin headers are required for this local request; "
        "send a matching Origin or Sec-Fetch-Site header"
    )


def _invalid(message: str, field: str | None = None) -> ApiError:
    fields = {field: message} if field else None
    return ApiError(400, "invalid_request", message, fields)


def _not_found() -> ApiError:
    return ApiError(404, "not_found", "unknown route or resource")


def _method_not_allowed() -> ApiError:
    return ApiError(405, "method_not_allowed", "method not allowed for this route")


def _parse_content_length(value: str) -> int:
    if not _CONTENT_LENGTH_RE.fullmatch(value):
        raise _invalid("Content-Length must be an unambiguous integer")
    try:
        return int(value, 10)
    except ValueError:  # pragma: no cover - regex plus CPython digit limit
        raise _invalid("Content-Length must be an unambiguous integer") from None


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> Any:
    raise ValueError("non-standard JSON constant")


def _has_surrogate(text: str) -> bool:
    return any(
        _JSON_SURROGATE_MIN <= ord(character) <= _JSON_SURROGATE_MAX
        for character in text
    )


def _reject_surrogates(value: Any) -> None:
    """Reject lone UTF-16 surrogates anywhere in a decoded JSON value.

    Python's JSON decoder combines valid surrogate pairs but leaves lone
    surrogates in strings (for example ``"\\ud800"``).  Those cannot be encoded
    as UTF-8 for SQLite, paths, or a UTF-8 JSON response.  The traversal is
    iterative so a deeply nested body cannot turn this check into a 500.
    """
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            if _has_surrogate(current):
                raise _invalid("request body contains an unpaired Unicode surrogate")
        elif isinstance(current, dict):
            for key, child in current.items():
                stack.append(key)
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)


def _decode_json_object(body: bytes) -> dict[str, Any]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise _invalid("request body must be valid UTF-8 JSON") from None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError):
        raise _invalid("request body must be valid JSON") from None
    if not isinstance(value, dict):
        raise _invalid("request body must be a JSON object")
    _reject_surrogates(value)
    return value


def _reject_unknown_fields(body: dict[str, Any], allowed: set[str]) -> None:
    if set(body) - allowed:
        # Do not echo an arbitrary client-provided field name back to the page.
        raise _invalid("request body contains an unknown field")


def _require_string(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if type(value) is not str:
        raise _invalid(f"{key} must be a string", field=key)
    return value


def _optional_string(body: dict[str, Any], key: str) -> str | None:
    if key not in body or body[key] is None:
        return None
    value = body[key]
    if type(value) is not str:
        raise _invalid(f"{key} must be a string or null", field=key)
    return value


def _optional_int(body: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    if key not in body:
        return default
    value = body[key]
    # ``bool`` is a subclass of ``int`` in Python; JSON booleans are not integers.
    if type(value) is not int:
        raise _invalid(f"{key} must be an integer", field=key)
    if not low <= value <= high:
        raise _invalid(f"{key} must be between {low} and {high}", field=key)
    return value


def _optional_bool(body: dict[str, Any], key: str, default: bool = False) -> bool:
    if key not in body:
        return default
    value = body[key]
    if type(value) is not bool:
        raise _invalid(f"{key} must be a boolean", field=key)
    return value


def _validate_search(body: dict[str, Any]) -> SearchRequest:
    _reject_unknown_fields(
        body,
        {
            "query",
            "mode",
            "k",
            "per_file",
            "path",
            "kind",
            "since",
            "module",
            "week",
            "spread",
            "rerank",
            "explain",
        },
    )
    query = _require_string(body, "query")
    if len(query) > LIMITS["query_chars"]:
        raise _invalid(
            f"query must be at most {LIMITS['query_chars']} characters",
            field="query",
        )
    mode = body.get("mode", "hybrid")
    if type(mode) is not str or mode not in _SEARCH_MODES:
        raise _invalid("mode must be one of keyword, vector, or hybrid", field="mode")
    k = _optional_int(body, "k", 10, LIMITS["k_min"], LIMITS["k_max"])
    per_file = _optional_int(
        body,
        "per_file",
        2,
        LIMITS["per_file_min"],
        LIMITS["per_file_max"],
    )
    path = _optional_string(body, "path")
    kind = _optional_string(body, "kind")
    since = _optional_string(body, "since")
    module = _optional_string(body, "module")
    week = _optional_string(body, "week")
    spread = _optional_bool(body, "spread", default=False)
    rerank = _optional_bool(body, "rerank", default=False)
    explain = _optional_bool(body, "explain", default=False)
    if rerank and mode == "keyword":
        raise _invalid("rerank is not available for keyword mode", field="rerank")
    return SearchRequest(
        query=query,
        mode=mode,  # type: ignore[arg-type]
        k=k,
        per_file=per_file,
        path=path,
        kind=kind,
        since=since,
        module=module,
        week=week,
        spread=spread,
        rerank=rerank,
        explain=explain,
    )


def _validate_similar(body: dict[str, Any]) -> SimilarRequest:
    _reject_unknown_fields(
        body,
        {"path", "k", "per_file", "path_prefix", "kind", "since", "module", "week"},
    )
    path = _require_string(body, "path")
    if not os.path.isabs(path):
        raise _invalid("path must be an absolute path", field="path")
    k = _optional_int(body, "k", 10, LIMITS["k_min"], LIMITS["k_max"])
    per_file = _optional_int(
        body, "per_file", 1, LIMITS["per_file_min"], LIMITS["per_file_max"]
    )
    return SimilarRequest(
        path=path,
        k=k,
        per_file=per_file,
        path_prefix=_optional_string(body, "path_prefix"),
        kind=_optional_string(body, "kind"),
        since=_optional_string(body, "since"),
        module=_optional_string(body, "module"),
        week=_optional_string(body, "week"),
    )


def _validate_index_paths(body: dict[str, Any]) -> list[str]:
    _reject_unknown_fields(body, {"paths"})
    value = body.get("paths")
    if type(value) is not list or not value:
        raise _invalid("paths must be a non-empty array of strings", field="paths")
    if len(value) > LIMITS["roots_max"]:
        raise _invalid(
            f"paths must contain at most {LIMITS['roots_max']} entries", field="paths"
        )
    paths: list[str] = []
    for item in value:
        if type(item) is not str or not item:
            raise _invalid("paths must be a non-empty array of strings", field="paths")
        if len(item.encode("utf-8")) > LIMITS["root_path_bytes"]:
            raise _invalid(
                f"each path must be at most {LIMITS['root_path_bytes']} bytes",
                field="paths",
            )
        if not os.path.isabs(item):
            raise _invalid("each path must be an absolute path", field="paths")
        paths.append(item)
    return paths


def _validate_roots(body: dict[str, Any]) -> tuple[str, str]:
    _reject_unknown_fields(body, {"path", "revision"})
    return _require_string(body, "path"), _require_string(body, "revision")


def _validate_open(body: dict[str, Any]) -> int:
    _reject_unknown_fields(body, {"chunk"})
    value = body.get("chunk")
    if type(value) is not int or value <= 0:
        raise _invalid("chunk must be a positive integer", field="chunk")
    return value


def _validate_download(body: dict[str, Any]) -> bool:
    _reject_unknown_fields(body, {"rerank"})
    return _optional_bool(body, "rerank", default=False)


def _validate_reindex(body: dict[str, Any]) -> tuple[bool, bool]:
    _reject_unknown_fields(body, {"force", "allow_empty"})
    return (
        _optional_bool(body, "force", default=False),
        _optional_bool(body, "allow_empty", default=False),
    )


def _validate_empty_body(body: dict[str, Any]) -> None:
    if body:
        raise _invalid("request body must be an empty object")


def _json_content_type(value: str) -> bool:
    parts = [part.strip() for part in value.split(";")]
    if not parts or parts[0].lower() != "application/json":
        return False
    parameters = parts[1:]
    if not parameters:
        return True
    if len(parameters) != 1:
        return False
    key, separator, raw_value = parameters[0].partition("=")
    if not separator or key.strip().lower() != "charset":
        return False
    charset = raw_value.strip().strip('"').strip("'").strip().lower()
    return charset == "utf-8"


class _DeadlineReader:
    """Request reader enforcing both per-read and total-read deadlines.

    ``BaseHTTPRequestHandler`` normally reads through a buffered file object;
    its socket timeout applies to each individual ``recv`` and a slow-drip
    client can therefore keep a handler slot forever.  This reader buffers the
    socket directly and re-arms ``settimeout`` to the smaller of the per-read
    timeout and the remaining total request deadline before every ``recv``.
    """

    def __init__(self, connection: socket.socket, deadline: float, read_timeout: float) -> None:
        self._connection = connection
        self._deadline = deadline
        self._read_timeout = max(0.001, read_timeout)
        self._buffer = bytearray()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _remaining(self) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("total request read deadline exceeded")
        return remaining

    def _recv(self) -> bytes:
        remaining = self._remaining()
        self._connection.settimeout(min(remaining, self._read_timeout))
        return self._connection.recv(65536)

    def readline(self, limit: int = -1) -> bytes:
        if limit == 0:
            return b""
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                end = newline + 1
                if limit >= 0:
                    end = min(end, limit)
                line = bytes(self._buffer[:end])
                del self._buffer[:end]
                return line
            if limit >= 0 and len(self._buffer) >= limit:
                line = bytes(self._buffer[:limit])
                del self._buffer[:limit]
                return line
            chunk = self._recv()
            if not chunk:
                line = bytes(self._buffer)
                self._buffer.clear()
                self._closed = True
                return line
            self._buffer.extend(chunk)

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size > 0:
            while len(self._buffer) < size:
                chunk = self._recv()
                if not chunk:
                    break
                self._buffer.extend(chunk)
            data = bytes(self._buffer[:size])
            del self._buffer[:size]
            return data

        chunks = [bytes(self._buffer)]
        self._buffer.clear()
        while True:
            chunk = self._recv()
            if not chunk:
                self._closed = True
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def read_with_deadline(self, size: int, deadline: float) -> bytes:
        """Read at most ``size`` bytes under a temporary tighter deadline."""
        previous = self._deadline
        self._deadline = min(previous, deadline)
        try:
            return self.read(size)
        finally:
            self._deadline = previous

    def close(self) -> None:
        self._closed = True
        self._buffer.clear()


class SeekrHTTPServer(ThreadingHTTPServer):
    """Loopback ``ThreadingHTTPServer`` with a bounded handler pool.

    At most ``LIMITS['handler_threads']`` request handlers exist at once.  A
    semaphore is acquired in the serving thread *before* a handler thread is
    created, so excess connections wait in the bounded accept backlog rather
    than creating unbounded threads.  Keep-alive is disabled by the HTTP/1.0
    handler and ``Connection: close`` responses, so idle clients cannot hold
    slots indefinitely; incomplete requests are closed by the read timeout.
    """

    daemon_threads = False
    block_on_close = False
    request_queue_size = 16
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        *,
        backend: WebBackend,
        token: str,
        read_timeout: float | None = None,
        request_deadline: float | None = None,
        join_timeout: float = 5.0,
    ) -> None:
        self.backend = backend
        self.token = token
        self.read_timeout = (
            float(LIMITS["read_timeout_s"]) if read_timeout is None else float(read_timeout)
        )
        self.request_deadline = (
            float(LIMITS["read_timeout_s"])
            if request_deadline is None
            else float(request_deadline)
        )
        self.join_timeout = join_timeout
        self._handler_slots = threading.BoundedSemaphore(LIMITS["handler_threads"])
        self._handler_threads: list[threading.Thread] = []
        self._active_sockets: set[socket.socket] = set()
        self._state_lock = threading.Lock()
        self._closing = threading.Event()
        super().__init__(server_address, RequestHandlerClass)

    @property
    def bound_port(self) -> int:
        return int(self.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.bound_port}"

    @property
    def handler_thread_count(self) -> int:
        with self._state_lock:
            return len(self._handler_threads)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        """Acquire a handler slot before creating a thread for ``request``."""
        acquired = False
        while not self._closing.is_set():
            if self._handler_slots.acquire(timeout=0.05):
                acquired = True
                break
        if not acquired:
            self.close_request(request)
            return

        thread = threading.Thread(
            target=self._run_handler,
            args=(request, client_address),
            name="seekr-web-handler",
            daemon=False,
        )
        with self._state_lock:
            if self._closing.is_set():
                start = False
            else:
                self._handler_threads.append(thread)
                self._active_sockets.add(request)
                start = True
        if not start:
            self._handler_slots.release()
            self.close_request(request)
            return
        try:
            thread.start()
        except BaseException:
            with self._state_lock:
                self._active_sockets.discard(request)
                try:
                    self._handler_threads.remove(thread)
                except ValueError:  # pragma: no cover - defensive
                    pass
            self._handler_slots.release()
            self.close_request(request)
            raise

    def _run_handler(self, request: socket.socket, client_address: Any) -> None:
        current = threading.current_thread()
        try:
            self.process_request_thread(request, client_address)
        finally:
            with self._state_lock:
                self._active_sockets.discard(request)
                try:
                    self._handler_threads.remove(current)
                except ValueError:  # pragma: no cover - defensive
                    pass
            self._handler_slots.release()

    def handle_error(self, _request: Any, _client_address: Any) -> None:
        """Log a fixed line rather than an exception message or traceback."""
        _logger.error("event=web_handler_failed")

    def shutdown(self) -> None:
        """Stop accepting; no new handler is created after this call."""
        self._closing.set()
        super().shutdown()

    def server_close(self) -> None:
        """Close the listener and active sockets, then join handlers with a bound."""
        self._closing.set()
        with self._state_lock:
            sockets = list(self._active_sockets)
            threads = list(self._handler_threads)

        try:
            super().server_close()
        except OSError:  # pragma: no cover - listener already closed
            pass

        for active in sockets:
            try:
                active.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                active.close()
            except OSError:
                pass

        deadline = time.monotonic() + max(0.0, self.join_timeout)
        current = threading.current_thread()
        for thread in threads:
            if thread is current or thread.ident is None:
                # A thread appended just before a concurrent close may not have
                # started yet; joining an unstarted thread raises.
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)


class WebRequestHandler(BaseHTTPRequestHandler):
    """One-shot HTTP handler implementing the frozen request policy."""

    protocol_version = "HTTP/1.0"
    default_request_version = "HTTP/1.0"
    server_version = "seekr"
    sys_version = ""

    def __getattr__(self, name: str) -> Any:
        # BaseHTTPRequestHandler looks up ``do_<METHOD>``.  Returning a generic
        # dispatcher for every method makes unsupported methods reach our frozen
        # 405/404 policy instead of the stdlib's HTML 501 response.
        if name.startswith("do_"):
            return self._handle_request
        raise AttributeError(name)

    def setup(self) -> None:
        super().setup()
        # Drop the stdlib buffered reader (it has not read anything yet) and
        # use a reader that enforces a total deadline across header/body reads.
        old_reader = self.rfile
        try:
            old_reader.close()
        except OSError:  # pragma: no cover - defensive close
            pass
        self._request_deadline = time.monotonic() + self.server.request_deadline
        self._body_consumed = False
        self.rfile = _DeadlineReader(
            self.connection,
            self._request_deadline,
            self.server.read_timeout,
        )

    def log_message(self, format: str, *args: object) -> None:
        # Never emit request lines, query strings, headers, bodies, or document
        # text through the stdlib's stderr logging.  Structured access logging
        # below records only a route label and status.
        return None

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        """Emit our JSON error envelope for parser-level failures."""
        if code in (HTTPStatus.METHOD_NOT_ALLOWED, HTTPStatus.NOT_IMPLEMENTED):
            status, error_code, text = 405, "method_not_allowed", "method not allowed"
        elif code == 413:
            status, error_code, text = 413, "payload_too_large", "request body is too large"
        elif code == 415:
            status, error_code, text = 415, "unsupported_media_type", "unsupported media type"
        else:
            status, error_code, text = 400, "invalid_request", "request is malformed"
        try:
            self._send_json(status, ApiError(status, error_code, text).to_envelope())
        except OSError:  # pragma: no cover - peer disappeared
            self.close_connection = True

    # -- response helpers -------------------------------------------------

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in SECURITY_HEADERS:
                self.send_header(name, value)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)
                self.wfile.flush()
        except OSError:
            self.close_connection = True

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        # ``ensure_ascii`` keeps response serialization safe even if backend
        # data contains an unpaired surrogate: JSON escapes stay ASCII, so the
        # UTF-8 encode cannot fail and the field remains round-trippable.
        body = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self._send_bytes(status, _JSON_CONTENT_TYPE, body)

    def _send_api_error(self, error: ApiError) -> None:
        self._send_json(error.status, error.to_envelope())

    # -- request policy ---------------------------------------------------

    def _request_target(self) -> tuple[str, bool, str] | None:
        # Use the raw request target so stdlib ``//`` normalization cannot turn
        # a non-canonical target into an exact allowlisted route.
        parts = self.requestline.split()
        target = parts[1] if len(parts) >= 2 else self.path
        if not target.startswith("/") or target.startswith("//") or "#" in target:
            return None
        try:
            urlsplit(target)
        except ValueError:
            return None
        raw_path, separator, raw_query = target.partition("?")
        return raw_path, bool(separator), raw_query

    def _single_header(self, name: str) -> str | None:
        values = self.headers.get_all(name) if self.headers is not None else None
        if not values:
            return None
        if len(values) != 1:
            return None
        return values[0]

    def _header_count(self, name: str) -> int:
        values = self.headers.get_all(name) if self.headers is not None else None
        return len(values or ())

    def _validate_host(self) -> None:
        values = self.headers.get_all("Host") if self.headers is not None else None
        expected = f"127.0.0.1:{self.server.bound_port}"
        if not values or len(values) != 1 or values[0] != expected:
            raise _forbidden("invalid Host header; expected the exact loopback address")

    def _check_fetch_metadata(self, *, require_proof: bool) -> bool:
        if self._header_count("Origin") > 1 or self._header_count("Sec-Fetch-Site") > 1:
            raise _forbidden("duplicate browser-origin headers are not allowed")

        origin = self._single_header("Origin")
        fetch_site = self._single_header("Sec-Fetch-Site")
        expected_origin = f"http://127.0.0.1:{self.server.bound_port}"

        if origin is not None and origin != expected_origin:
            raise _forbidden("request Origin does not match this local server")
        if fetch_site is not None and fetch_site not in _ORIGIN_FETCH_SITE_VALUES:
            raise _forbidden("cross-site and same-site requests are not allowed")

        has_proof = origin is not None or fetch_site is not None
        if require_proof and not has_proof:
            raise _hint_forbidden()
        return has_proof

    def _check_session_client_header(self) -> None:
        count = self._header_count("X-Seekr-Client")
        if count != 1:
            raise _invalid("exactly one X-Seekr-Client: 1 header is required")
        if self._single_header("X-Seekr-Client") != "1":
            raise _invalid("X-Seekr-Client must be 1")

    def _check_token(self) -> None:
        count = self._header_count("X-Seekr-Token")
        if count != 1:
            raise _forbidden("a single valid X-Seekr-Token header is required")
        provided = self._single_header("X-Seekr-Token")
        try:
            provided_bytes = provided.encode("latin-1") if provided is not None else None
            token_bytes = self.server.token.encode("latin-1")
        except UnicodeEncodeError:
            raise _forbidden("a single valid X-Seekr-Token header is required") from None
        if provided_bytes is None or not hmac.compare_digest(
            provided_bytes,
            token_bytes,
        ):
            raise _forbidden("a single valid X-Seekr-Token header is required")

    def _check_get_body(self) -> None:
        if self._header_count("Transfer-Encoding"):
            raise _invalid("GET requests must not use Transfer-Encoding")
        count = self._header_count("Content-Length")
        if count > 1:
            raise _invalid("duplicate Content-Length headers are not allowed")
        if count == 1:
            value = self._single_header("Content-Length")
            assert value is not None
            length = _parse_content_length(value)
            if length != 0:
                raise _invalid("GET requests must not include a request body")

    def _read_json_body(self) -> dict[str, Any]:
        if self._header_count("Transfer-Encoding"):
            raise ApiError(
                415,
                "unsupported_media_type",
                "Transfer-Encoding and chunked bodies are not supported",
            )

        content_type_count = self._header_count("Content-Type")
        if content_type_count > 1:
            raise _invalid("duplicate Content-Type headers are not allowed")
        content_type = self._single_header("Content-Type")
        if content_type is None or not _json_content_type(content_type):
            raise ApiError(
                415,
                "unsupported_media_type",
                "Content-Type must be application/json with an optional charset=utf-8",
            )

        length_count = self._header_count("Content-Length")
        if length_count != 1:
            raise _invalid("exactly one Content-Length header is required")
        length_value = self._single_header("Content-Length")
        assert length_value is not None
        length = _parse_content_length(length_value)
        if length > LIMITS["request_body_bytes"]:
            raise ApiError(413, "payload_too_large", "request body exceeds 64 KiB")

        try:
            body = self.rfile.read(length)
        except (TimeoutError, OSError):
            self._body_consumed = True
            raise _invalid("request body could not be read") from None
        self._body_consumed = True
        if len(body) != length:
            raise _invalid("request body is incomplete")
        return _decode_json_object(body)

    def _drain_unread_body(self) -> None:
        """Discard a bounded, declared body after an early error response.

        The JSON envelope is flushed before this runs, so the client can read
        it even when its write has not completed.  Reading some or all of the
        declared body avoids a close with unread data producing TCP RST and
        truncating the response.  The byte and time bounds keep this from
        becoming an unbounded slow-read resource.  Bodies larger than
        ``_MAX_DRAIN_BYTES`` (or chunked bodies) may still reset after the
        bounded drain; that trade-off is deliberate.
        """
        if getattr(self, "_body_consumed", True):
            return
        if self._header_count("Transfer-Encoding"):
            # Chunk framing would need a parser; closing is safer than reading
            # an attacker-controlled unbounded stream.
            return
        if self._header_count("Content-Length") != 1:
            return
        value = self._single_header("Content-Length")
        if value is None:
            return
        try:
            length = _parse_content_length(value)
        except ApiError:
            return
        if length <= 0:
            return
        amount = min(length, _MAX_DRAIN_BYTES)
        deadline = min(
            self._request_deadline,
            time.monotonic() + _MAX_DRAIN_SECONDS,
        )
        try:
            reader = self.rfile
            if isinstance(reader, _DeadlineReader):
                remaining = amount
                while remaining > 0:
                    chunk = reader.read_with_deadline(
                        min(remaining, 65536),
                        deadline,
                    )
                    if not chunk:
                        break
                    remaining -= len(chunk)
        except (TimeoutError, OSError):
            pass
        finally:
            self._body_consumed = True

    # -- dispatch ---------------------------------------------------------

    def _handle_request(self) -> None:
        route = "unknown"
        status = 500
        started = time.monotonic()
        try:
            route, status = self._dispatch()
        except ApiError as error:
            status = error.status
            self._send_api_error(error)
            self._drain_unread_body()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            return
        except Exception:  # noqa: BLE001 - last-resort safe 500 envelope
            status = 500
            _logger.error(
                "event=web_request_failed route=%s status=500",
                route,
            )
            try:
                self._send_api_error(
                    ApiError(500, "internal_error", "internal server error")
                )
            except OSError:  # pragma: no cover - peer disappeared
                self.close_connection = True
            self._drain_unread_body()
        finally:
            if _logger.isEnabledFor(logging.DEBUG):
                duration_ms = int((time.monotonic() - started) * 1000)
                _logger.debug(
                    "event=web_request route=%s status=%s duration_ms=%d",
                    route,
                    status,
                    duration_ms,
                )

    def _resolve_route(
        self,
    ) -> tuple[str, frozenset[str], str | None, bool, str] | None:
        parsed = self._request_target()
        if parsed is None:
            return None
        path, has_query, raw_query = parsed
        if path in _ASSET_ROUTES:
            filename, mime = _ASSET_ROUTES[path]
            return ("asset", frozenset({_GET_METHOD}), f"{filename}\x00{mime}", has_query, raw_query)
        if path in _API_ROUTES:
            name, methods = _API_ROUTES[path]
            return (name, methods, None, has_query, raw_query)
        if path == "/api/jobs/" or path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/") :]
            if "/" in job_id:
                return None
            return ("job", frozenset({_GET_METHOD}), job_id, has_query, raw_query)
        return None

    def _dispatch(self) -> tuple[str, int]:
        route = self._resolve_route()
        self._validate_host()

        # OPTIONS is rejected after Host validation but before routing, token,
        # or fetch-metadata checks.  This deliberately makes preflight fail
        # without any CORS response.
        if self.command == "OPTIONS":
            raise _method_not_allowed()

        if route is None:
            raise _not_found()

        name, allowed_methods, parameter, has_query, raw_query = route
        route_label = _route_label(name)
        if self.command not in allowed_methods:
            raise _method_not_allowed()

        if name == "asset":
            self._check_fetch_metadata(require_proof=True)
        elif name == "session":
            self._check_fetch_metadata(require_proof=True)
            self._check_session_client_header()
        else:
            self._check_fetch_metadata(require_proof=False)
            self._check_token()

        preview_chunk_id: int | None = None
        if name in _PREVIEW_ROUTES:
            if not has_query:
                raise _invalid("chunk query parameter is required", field="chunk")
            match = _PREVIEW_QUERY_RE.fullmatch(raw_query)
            if match is None:
                raise _invalid(
                    "chunk query parameter must be a positive integer",
                    field="chunk",
                )
            preview_chunk_id = int(match.group(1), 10)
        elif has_query:
            raise _invalid("query strings are not supported for this route")

        if self.command == _GET_METHOD:
            self._check_get_body()
        else:
            body = self._read_json_body()

        if name == "asset":
            assert parameter is not None
            filename, mime = parameter.split("\x00", 1)
            data = _read_asset(filename)
            if data is None:
                raise _not_found()
            self._send_bytes(200, mime, data)
            return route_label, 200

        backend = self.server.backend
        if name == "session":
            self._send_json(200, {"token": self.server.token, "limits": dict(LIMITS)})
            return route_label, 200
        if name == "status":
            self._send_json(200, backend.status().to_dict())
            return route_label, 200
        if name == "roots":
            if self.command == _GET_METHOD:
                self._send_json(200, backend.roots().to_dict())
                return route_label, 200
            assert body is not None
            path, revision = _validate_roots(body)
            self._send_json(200, backend.add_root(path, revision).to_dict())
            return route_label, 200
        if name == "roots_remove":
            assert body is not None
            path, revision = _validate_roots(body)
            self._send_json(200, backend.remove_root(path, revision).to_dict())
            return route_label, 200
        if name == "course":
            self._send_json(200, backend.course().to_dict())
            return route_label, 200
        if name == "preview":
            assert preview_chunk_id is not None
            self._send_json(200, backend.preview(preview_chunk_id).to_dict())
            return route_label, 200
        if name == "preview_page":
            assert preview_chunk_id is not None
            png_bytes = backend.preview_page_png(preview_chunk_id)
            self._send_bytes(200, _PNG_CONTENT_TYPE, png_bytes)
            return route_label, 200
        if name == "open":
            assert body is not None
            chunk_id = _validate_open(body)
            backend.open_chunk(chunk_id)
            self._send_json(200, {"opened": True})
            return route_label, 200
        if name == "search":
            assert body is not None
            request = _validate_search(body)
            self._send_json(200, backend.search(request).to_dict())
            return route_label, 200
        if name == "similar":
            assert body is not None
            similar_request = _validate_similar(body)
            self._send_json(200, backend.similar(similar_request).to_dict())
            return route_label, 200
        if name == "shutdown":
            assert body is not None
            _validate_empty_body(body)
            backend.request_shutdown()
            self._send_json(202, {"stopping": True})
            return route_label, 202
        if name == "index_paths":
            assert body is not None
            job_id = backend.start_index_paths(_validate_index_paths(body))
            self._send_json(202, {"job_id": job_id})
            return route_label, 202
        if name == "model_check":
            assert body is not None
            _validate_empty_body(body)
            job_id = backend.start_model_check()
            self._send_json(202, {"job_id": job_id})
            return route_label, 202
        if name == "index":
            assert body is not None
            _validate_empty_body(body)
            job_id = backend.start_index()
            self._send_json(202, {"job_id": job_id})
            return route_label, 202
        if name == "reindex":
            assert body is not None
            force, allow_empty = _validate_reindex(body)
            job_id = backend.start_reindex(force=force, allow_empty=allow_empty)
            self._send_json(202, {"job_id": job_id})
            return route_label, 202
        if name == "setup_download":
            assert body is not None
            rerank = _validate_download(body)
            job_id = backend.start_download(rerank=rerank)
            self._send_json(202, {"job_id": job_id})
            return route_label, 202
        if name == "job":
            assert parameter is not None
            if not _JOB_ID_RE.fullmatch(parameter):
                raise ApiError(404, "job_not_found", "job was not found or has expired")
            self._send_json(200, backend.job(parameter).to_dict())
            return route_label, 200
        raise _not_found()  # pragma: no cover - every route is dispatched above


def _route_label(name: str) -> str:
    return {
        "asset": "asset",
        "session": "/api/session",
        "status": "/api/status",
        "roots": "/api/roots",
        "roots_remove": "/api/roots/remove",
        "course": "/api/course",
        "preview": "/api/preview",
        "preview_page": "/api/preview/page",
        "open": "/api/open",
        "search": "/api/search",
        "similar": "/api/similar",
        "shutdown": "/api/shutdown",
        "index_paths": "/api/index/paths",
        "model_check": "/api/model/check",
        "index": "/api/index",
        "reindex": "/api/reindex",
        "setup_download": "/api/setup/download",
        "job": "/api/jobs/{job_id}",
    }.get(name, "unknown")


def create_server(
    backend: WebBackend,
    *,
    port: int = 8000,
    token: str | None = None,
    read_timeout: float | None = None,
    request_deadline: float | None = None,
    join_timeout: float = 5.0,
) -> SeekrHTTPServer:
    """Bind a loopback server and return it with the actual port recorded.

    ``port=0`` asks the OS for an ephemeral port; ``server.bound_port`` and
    ``server.url`` always contain the actual port used for Host/Origin checks.
    ``read_timeout`` bounds one socket read; ``request_deadline`` bounds the
    total time spent reading a request line, headers, and body.  The caller owns
    the returned server and must run and close it.
    """
    server = SeekrHTTPServer(
        ("127.0.0.1", port),
        WebRequestHandler,
        backend=backend,
        token=token or secrets.token_urlsafe(32),
        read_timeout=(
            float(LIMITS["read_timeout_s"]) if read_timeout is None else float(read_timeout)
        ),
        request_deadline=request_deadline,
        join_timeout=join_timeout,
    )
    return server


def _begin_backend_shutdown(backend: WebBackend) -> None:
    """Ask the backend to refuse new work before server/handler shutdown."""
    method = getattr(backend, "begin_shutdown", None)
    if not callable(method):
        return
    try:
        method()
    except Exception:  # noqa: BLE001 - admission close must not mask the exit path
        _logger.warning("event=web_backend_begin_shutdown_failed")


def _shutdown_backend(backend: WebBackend) -> None:
    """Call the backend's lifecycle hooks, tolerating partial implementations."""
    called: set[int] = set()
    for name in ("shutdown", "close"):
        method = getattr(backend, name, None)
        if not callable(method):
            continue
        target = id(getattr(method, "__func__", method))
        if target in called:
            continue
        called.add(target)
        try:
            method()
        except Exception:  # noqa: BLE001 - shutdown must not mask the exit path
            _logger.warning("event=web_backend_shutdown_failed hook=%s", name)


def _report_browser_error(
    message: str,
    on_error: Callable[[str], None] | None,
) -> None:
    try:
        if on_error is not None:
            on_error(message)
        else:
            print(f"seekr web: {message}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a reporting callback must not kill the server
        _logger.warning("event=web_browser_error_callback_failed")


def _try_open_browser(
    url: str,
    opener: Callable[[str], Any] | None,
    on_error: Callable[[str], None] | None,
) -> None:
    """Open ``url`` without ever failing the server."""
    try:
        if opener is None:
            import webbrowser

            result = webbrowser.open(url)
        else:
            result = opener(url)
    except Exception as exc:  # noqa: BLE001 - browser launch must never kill the server
        _report_browser_error(
            f"could not open a browser ({exc}); open {url} manually",
            on_error,
        )
        return
    if result is False:
        _report_browser_error(
            f"no browser was available; open {url} manually",
            on_error,
        )


def _install_signal_handlers(
    stop_event: threading.Event,
    notice: Callable[[str], None],
) -> dict[int, Any]:
    """Install first/second SIGINT/SIGTERM handlers; return originals."""
    previous: dict[int, Any] = {}
    count = 0

    def handler(_signum: int, _frame: Any) -> None:
        nonlocal count
        count += 1
        if count == 1:
            stop_event.set()
            notice(
                "Seekr web is stopping; waiting for any active operation to finish "
                "(a large index scan may take a while)..."
            )
        else:
            notice(
                "Seekr web is still stopping and is waiting for the active operation; "
                "a second interrupt does not cancel it."
            )

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        except (OSError, ValueError):
            previous.pop(signum, None)
    return previous


def run_web(
    backend: WebBackend,
    *,
    port: int = 8000,
    token: str | None = None,
    stop_event: threading.Event | None = None,
    open_browser: bool = False,
    install_signal_handlers: bool = True,
    on_listen: Callable[[str], None] | None = None,
    on_browser_error: Callable[[str], None] | None = None,
    on_notice: Callable[[str], None] | None = None,
    browser_opener: Callable[[str], Any] | None = None,
    read_timeout: float | None = None,
    request_deadline: float | None = None,
    join_timeout: float = 5.0,
) -> int:
    """Run the web server until ``stop_event`` (or SIGINT/SIGTERM) fires.

    The actual bound URL is reported through ``on_listen``.  Browser launching
    runs on a detached daemon thread so a slow launcher cannot delay serving or
    stopping.  The backend is asked to close admission before the listener and
    handler pool stop, and SIGINT/SIGTERM warning handlers remain installed
    until the backend shutdown join has completed.
    """
    try:
        server = create_server(
            backend,
            port=port,
            token=token,
            read_timeout=read_timeout,
            request_deadline=request_deadline,
            join_timeout=join_timeout,
        )
    except BaseException:
        _shutdown_backend(backend)
        raise

    stopped = stop_event if stop_event is not None else threading.Event()
    done = threading.Event()
    admission_lock = threading.Lock()
    admission_closed = False

    def notice(message: str) -> None:
        if on_notice is not None:
            on_notice(message)
        else:
            print(message, file=sys.stderr, flush=True)

    def close_admission() -> None:
        nonlocal admission_closed
        with admission_lock:
            if admission_closed:
                return
            admission_closed = True
        _begin_backend_shutdown(backend)

    def watcher() -> None:
        while not stopped.wait(0.05):
            if done.is_set():
                return
        if not done.is_set():
            # Refuse new jobs/searches before the listener and already-admitted
            # handlers are allowed to wind down.
            close_admission()
            try:
                server.shutdown()
            except Exception:  # noqa: BLE001 - pragma: no cover - defensive shutdown
                _logger.warning("event=web_shutdown_failed")

    previous_signals: dict[int, Any] = {}
    watcher_thread = threading.Thread(
        target=watcher,
        name="seekr-web-shutdown",
        daemon=True,
    )
    watcher_started = False
    try:
        if install_signal_handlers:
            previous_signals = _install_signal_handlers(stopped, notice)
        if on_listen is not None:
            on_listen(server.url)
        else:
            print(server.url, flush=True)
        watcher_thread.start()
        watcher_started = True
        if open_browser and not stopped.is_set():
            # A desktop browser launcher can wait for a process to exit; keep
            # it off the serving/shutdown path.  The thread is daemonized so a
            # stuck launcher cannot prevent process exit.
            browser_thread = threading.Thread(
                target=_try_open_browser,
                args=(server.url, browser_opener, on_browser_error),
                name="seekr-web-browser",
                daemon=True,
            )
            try:
                browser_thread.start()
            except Exception as exc:  # noqa: BLE001 - launch failure must not stop serving
                _report_browser_error(
                    f"could not start a browser launcher ({exc})",
                    on_browser_error,
                )
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            stopped.set()
    finally:
        done.set()
        stopped.set()
        close_admission()
        if watcher_started:
            watcher_thread.join(timeout=max(1.0, join_timeout))
        if watcher_started and watcher_thread.is_alive():  # pragma: no cover - defensive fallback
            try:
                server.shutdown()
            except Exception:  # noqa: BLE001 - best-effort defensive fallback
                _logger.warning("event=web_shutdown_fallback_failed")
            watcher_thread.join(timeout=1.0)
        # Keep the first/second-signal warning handlers installed through the
        # backend join: a repeated Ctrl-C during an index/DB write must only
        # warn, not interrupt or terminate the write.  Restore only after the
        # backend shutdown and final handler join have completed.
        try:
            server.server_close()
        finally:
            try:
                _shutdown_backend(backend)
            finally:
                try:
                    # A handler may have been waiting on a backend operation
                    # that completed during backend shutdown; join any such
                    # stragglers within the bound.
                    server.server_close()
                finally:
                    for signum, original in previous_signals.items():
                        try:
                            signal.signal(signum, original)
                        except (OSError, ValueError):  # pragma: no cover - non-main thread
                            pass
    return 0
