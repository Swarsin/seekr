"""Client for the local seekr service.

CLI commands and the MCP server use :class:`ServiceClient` to send searches and
index jobs to a running ``seekr service`` and fall back to working locally when
none answers.  Importing this module is cheap: it uses only the standard
library, the discovery file reader and the import-safe web contract.

Before its first real call a client checks, with a 1-second authenticated
``GET /api/status``, that a seekr service (not some other program that reused
a stale port) answers.  A service that cannot be reached or verified, or that
has begun shutting down, raises :class:`ServiceUnavailable`, which callers
treat as "no service".  An error response in the seekr error envelope raises
:class:`ServiceError` carrying the server's code and message, and callers
surface it rather than silently searching locally.
"""

from __future__ import annotations

import http.client
import json
import logging
from pathlib import Path
from typing import Any

from . import __version__, paths
from .models import Hit, HitExplain
from .service_discovery import ServiceInfo, read_service_info

__all__ = [
    "ServiceClient",
    "ServiceConnectError",
    "ServiceError",
    "ServiceUnavailable",
    "hit_from_wire",
    "service_answers",
    "service_request",
]

_LOGGER = logging.getLogger(__name__)

CONNECT_TIMEOUT = 1.0
STATUS_TIMEOUT = 1.0
REQUEST_TIMEOUT = 30.0

_STATUS_SECTIONS = ("index", "model", "daemon", "config", "activity")
_EXPLAIN_FIELDS = (
    "keyword_rank",
    "keyword_bm25",
    "vector_rank",
    "vector_distance",
    "rrf_score",
    "rerank_rank",
    "rerank_score",
)
# Versions whose skew has already been logged in this process.
_SKEW_LOGGED: set[str] = set()


class ServiceError(RuntimeError):
    """The service answered with an error, or could not be used."""

    def __init__(
        self, message: str, *, status: int | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class ServiceUnavailable(ServiceError):
    """No usable service answered; the caller should work locally."""


class ServiceConnectError(OSError):
    """The TCP connection failed, so the request was never sent."""


def service_request(
    port: int,
    method: str,
    path: str,
    *,
    token: str,
    body: object = None,
    timeout: float = 2.0,
    connect_timeout: float | None = None,
) -> tuple[int, Any]:
    """Make one authenticated loopback API call; return (status, JSON body).

    ``connect_timeout`` (default ``timeout``) bounds the TCP connect; ``timeout``
    bounds each later socket operation.  A failed connect raises
    :class:`ServiceConnectError`; later socket and protocol failures propagate
    as :class:`OSError` or :class:`http.client.HTTPException`.
    """
    data = None if body is None else json.dumps(body).encode("utf-8")
    conn = http.client.HTTPConnection(
        "127.0.0.1",
        int(port),
        timeout=timeout if connect_timeout is None else connect_timeout,
    )
    try:
        try:
            conn.connect()
        except OSError as exc:
            raise ServiceConnectError(str(exc) or exc.__class__.__name__) from exc
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
        headers = {"X-Seekr-Token": token}
        if data is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        status, raw = response.status, response.read()
    finally:
        conn.close()
    try:
        payload = json.loads(raw) if raw else None
    except ValueError:
        payload = None
    return status, payload


def service_answers(info: ServiceInfo, *, timeout: float = STATUS_TIMEOUT) -> bool:
    """Return whether ``info`` names a live seekr service that accepts its token.

    The token-authenticated ``GET /api/status`` must return 200 with the
    status object's sections; a service that is shutting down refuses it.
    """
    try:
        status, payload = service_request(
            info.port, "GET", "/api/status", token=info.token, timeout=timeout
        )
    except (OSError, ValueError, http.client.HTTPException):
        return False
    return (
        status == 200
        and isinstance(payload, dict)
        and all(isinstance(payload.get(name), dict) for name in _STATUS_SECTIONS)
    )


def _seekr_error(payload: object) -> tuple[str, str] | None:
    """Return (code, message) from a seekr error envelope, or ``None``."""
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None
    code, message = error.get("code"), error.get("message")
    if not isinstance(code, str) or not isinstance(message, str):
        return None
    return code, message


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[call-overload]


def hit_from_wire(data: dict[str, Any]) -> Hit:
    """Rebuild a :class:`~seekr.models.Hit` from a wire ``HitDTO`` dict."""
    raw_explain = data.get("explain")
    explain: HitExplain | None = None
    if isinstance(raw_explain, dict):
        explain = HitExplain(
            **{
                name: (
                    _optional_int(raw_explain.get(name))
                    if name.endswith("_rank")
                    else _optional_float(raw_explain.get(name))
                )
                for name in _EXPLAIN_FIELDS
            }
        )
    module = data.get("module")
    week = data.get("week")
    return Hit(
        chunk_id=int(data["chunk_id"]),
        path=str(data["path"]),
        locator=str(data["locator"]),
        loc_start=float(data["loc_start"]),
        snippet=str(data["snippet"]),
        score=float(data["score"]),
        kind=str(data["kind"]),
        explain=explain,
        module=None if module is None else str(module),
        week=None if week is None else int(week),
        highlights=tuple((int(a), int(b)) for a, b in data.get("highlights") or ()),
    )


class ServiceClient:
    """Authenticated calls to one running service."""

    def __init__(self, info: ServiceInfo) -> None:
        self.info = info
        self._verified = False

    @classmethod
    def from_discovery(cls, path: Path | None = None) -> ServiceClient | None:
        """Return a client for the discovered service, or ``None``.

        ``None`` means no live discovery file, or a service of another seekr
        version (logged once per version at debug level).  Reachability is
        checked by the first call; a dead or foreign listener shows up as
        :class:`ServiceUnavailable`.
        """
        info = read_service_info(path or paths.service_file())
        if info is None:
            return None
        if info.version != __version__:
            if info.version not in _SKEW_LOGGED:
                _SKEW_LOGGED.add(info.version)
                _LOGGER.debug(
                    "event=service_client.version_skew service=%s client=%s",
                    info.version,
                    __version__,
                )
            return None
        return cls(info)

    def _verify(self) -> None:
        """Probe ``/api/status`` once per client before slower calls."""
        if self._verified:
            return
        if not service_answers(self.info):
            raise ServiceUnavailable(
                f"no seekr service answers on port {self.info.port}"
            )
        self._verified = True

    def _call(
        self,
        method: str,
        path: str,
        body: object = None,
        *,
        timeout: float,
        retry_safe: bool = True,
        verify: bool = True,
    ) -> Any:
        """Make one verified call and return its JSON body.

        ``retry_safe`` says whether a failure after the request was sent may
        be treated as "no service".  It is false for calls that start work
        (an index job), where the service may have acted on the request.
        """
        if verify:
            self._verify()
        try:
            status, payload = service_request(
                self.info.port,
                method,
                path,
                token=self.info.token,
                body=body,
                timeout=timeout,
                connect_timeout=CONNECT_TIMEOUT,
            )
        except ServiceConnectError as exc:
            raise ServiceUnavailable(f"seekr service is not reachable: {exc}") from exc
        except (OSError, http.client.HTTPException) as exc:
            if retry_safe:
                raise ServiceUnavailable(
                    f"seekr service did not answer: {exc}"
                ) from exc
            raise ServiceError(
                f"the seekr service did not confirm {method} {path} ({exc}); "
                "it may still be working on it, see `seekr service status`"
            ) from exc
        if 200 <= status < 300:
            return payload
        error = _seekr_error(payload)
        if error is None:
            # Not a seekr answer: another program owns the port now.
            raise ServiceUnavailable(
                f"port {self.info.port} answered HTTP {status} without a seekr error",
                status=status,
            )
        code, message = error
        if code == "busy" and not service_answers(self.info):
            # The status probe is refused once shutdown has begun.
            raise ServiceUnavailable(message, status=status, code=code)
        raise ServiceError(message, status=status, code=code)

    def _hits(self, path: str, request: dict[str, Any]) -> list[Hit]:
        payload = self._call("POST", path, request, timeout=REQUEST_TIMEOUT)
        raw_hits = payload.get("hits") if isinstance(payload, dict) else None
        if not isinstance(raw_hits, list):
            raise ServiceError("seekr service returned a malformed search response")
        try:
            return [hit_from_wire(item) for item in raw_hits]
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceError(
                f"seekr service returned a malformed hit: {exc}"
            ) from exc

    def search(self, request: dict[str, Any]) -> list[Hit]:
        """``POST /api/search`` with a wire request body."""
        return self._hits("/api/search", request)

    def similar(self, request: dict[str, Any]) -> list[Hit]:
        """``POST /api/similar``; ``request["path"]`` must be absolute."""
        return self._hits("/api/similar", request)

    def status(self) -> dict[str, Any]:
        payload = self._call("GET", "/api/status", timeout=STATUS_TIMEOUT, verify=False)
        if not isinstance(payload, dict):
            raise ServiceError("seekr service returned a malformed status response")
        return payload

    def _job_id(self, payload: object) -> str:
        job_id = payload.get("job_id") if isinstance(payload, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise ServiceError("seekr service returned no job id")
        return job_id

    def index_roots(self) -> str:
        """Start indexing the configured roots; return the job id."""
        return self._job_id(
            self._call("POST", "/api/index", {}, timeout=REQUEST_TIMEOUT, retry_safe=False)
        )

    def index_paths(self, paths: list[str]) -> str:
        """Start indexing absolute ``paths``; return the job id."""
        return self._job_id(
            self._call(
                "POST",
                "/api/index/paths",
                {"paths": paths},
                timeout=REQUEST_TIMEOUT,
                retry_safe=False,
            )
        )

    def job(self, job_id: str) -> dict[str, Any]:
        payload = self._call("GET", f"/api/jobs/{job_id}", timeout=REQUEST_TIMEOUT)
        if not isinstance(payload, dict):
            raise ServiceError("seekr service returned a malformed job response")
        return payload

    def shutdown(self) -> None:
        self._call("POST", "/api/shutdown", {}, timeout=STATUS_TIMEOUT)
