"""Boundary shared by the local web interface's backend, HTTP layer, and tests.

This module is import-safe: it uses only the standard library and must never
import the HTTP server, a Store, an embedder, numpy, torch, or sqlite-vec.
``docs/web-api.md`` is the authoritative contract; this module is its
executable shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

__all__ = [
    "ASSET_ALLOWLIST",
    "ASSET_MIME_TYPES",
    "ERROR_STATUS",
    "INDEX_STATES",
    "JOB_OPERATIONS",
    "JOB_STATES",
    "LANES",
    "LIMITS",
    "MODEL_STATES",
    "ApiError",
    "CourseModuleDTO",
    "CourseView",
    "HitDTO",
    "JobView",
    "PreviewView",
    "RootsView",
    "SearchMode",
    "SearchRequest",
    "SearchResult",
    "SimilarRequest",
    "StatusView",
    "WebBackend",
]

#: Frozen request/retention limits.  ``GET /api/session`` returns exactly these keys.
LIMITS: dict[str, int] = {
    "request_body_bytes": 65536,
    "query_chars": 2000,
    "k_min": 1,
    "k_max": 50,
    "per_file_min": 1,
    "per_file_max": 10,
    "roots_max": 50,
    "root_path_bytes": 4096,
    "jobs_retained": 20,
    "errors_retained": 100,
    "warnings_retained": 100,
    "handler_threads": 8,
    "read_timeout_s": 10,
}

#: The frozen code -> HTTP status matrix.  No other API error codes exist.
ERROR_STATUS: dict[str, int] = {
    "invalid_request": 400,
    "invalid_root": 400,
    "roots_limit": 400,
    "no_roots": 400,
    "forbidden": 403,
    "not_found": 404,
    "job_not_found": 404,
    "method_not_allowed": 405,
    "busy": 409,
    "stale_config": 409,
    "daemon_active": 409,
    "incompatible_index": 409,
    "duplicate_root": 409,
    "config_invalid": 409,
    "payload_too_large": 413,
    "unsupported_media_type": 415,
    "internal_error": 500,
    "config_write_failed": 500,
    "index_unavailable": 503,
    "model_unavailable": 503,
    "daemon_failure": 503,
}

INDEX_STATES: tuple[str, ...] = ("missing", "empty", "ready", "incompatible", "error")
MODEL_STATES: tuple[str, ...] = ("unchecked", "loading", "ready", "unavailable")
JOB_OPERATIONS: tuple[str, ...] = ("model_check", "index", "download")
JOB_STATES: tuple[str, ...] = ("running", "succeeded", "partial", "failed")
LANES: tuple[str, ...] = ("idle", "search", "model_check", "index", "download")
ASSET_ALLOWLIST: tuple[str, ...] = ("/", "/index.html", "/styles.css", "/app.js")

#: Exact MIME types for the packaged asset allowlist.
ASSET_MIME_TYPES: dict[str, str] = {
    "/": "text/html; charset=utf-8",
    "/index.html": "text/html; charset=utf-8",
    "/styles.css": "text/css; charset=utf-8",
    "/app.js": "text/javascript; charset=utf-8",
}

SearchMode = Literal["keyword", "vector", "hybrid"]


class ApiError(Exception):
    """A safe API error with a frozen code/status pair.

    ``to_envelope()`` produces the sole error JSON shape; it omits ``fields``
    when there are no safe per-field details.
    """

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        fields: dict[str, str] | None = None,
    ) -> None:
        expected = ERROR_STATUS.get(code)
        if expected is None:
            raise ValueError(f"unknown web API error code: {code!r}")
        if status != expected:
            raise ValueError(
                f"web API error {code!r} must use status {expected}, got {status}"
            )
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.fields = fields

    def to_envelope(self) -> dict[str, object]:
        """Return ``{"error": {...}}`` without empty ``fields``."""
        error: dict[str, object] = {"code": self.code, "message": self.message}
        if self.fields:
            error["fields"] = dict(self.fields)
        return {"error": error}


@dataclass(frozen=True)
class HitDTO:
    """One search hit; ``explain`` is present on the wire only when requested."""

    chunk_id: int
    path: str
    locator: str
    loc_start: float
    snippet: str
    score: float
    kind: str
    module: str | None = None
    week: int | None = None
    highlights: tuple[tuple[int, int], ...] = ()
    location: str = ""
    posix_location: str = ""
    explain: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "locator": self.locator,
            "loc_start": self.loc_start,
            "snippet": self.snippet,
            "score": self.score,
            "kind": self.kind,
            "module": self.module,
            "week": self.week,
            "highlights": [[int(a), int(b)] for a, b in self.highlights],
            "location": self.location,
            "posix_location": self.posix_location,
        }
        if self.explain is not None:
            payload["explain"] = dict(self.explain)
        return payload


@dataclass(frozen=True)
class SearchRequest:
    """A validated ``POST /api/search`` request body."""

    query: str
    mode: SearchMode = "hybrid"
    k: int = 10
    per_file: int = 2
    path: str | None = None
    kind: str | None = None
    since: str | None = None
    module: str | None = None
    week: str | None = None
    spread: bool = False
    rerank: bool = False
    explain: bool = False

    def to_query_kwargs(self) -> dict[str, object]:
        """Map the wire field ``path`` to ``get_hits``'s ``path_prefix``.

        The caller chooses ``use_daemon`` and the read-only keyword path, and
        passes ``rerank`` and ``explain`` explicitly from the request fields;
        this helper never adds them.
        """
        return {
            "query": self.query,
            "mode": self.mode,
            "k": self.k,
            "per_file": self.per_file,
            "path_prefix": self.path,
            "kind": self.kind,
            "since": self.since,
            "module": self.module,
            "week": self.week,
        }


@dataclass(frozen=True)
class SimilarRequest:
    """A validated ``POST /api/similar`` request body."""

    path: str
    k: int = 10
    per_file: int = 1
    path_prefix: str | None = None
    kind: str | None = None
    since: str | None = None
    module: str | None = None
    week: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """A completed local or daemon-backed search."""

    hits: tuple[HitDTO, ...]
    took_ms: float
    spread: tuple[dict[str, object], ...] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "hits": [hit.to_dict() for hit in self.hits],
            "took_ms": self.took_ms,
        }
        if self.spread is not None:
            payload["spread"] = [dict(item) for item in self.spread]
        return payload


@dataclass(frozen=True)
class RootsView:
    """The config roots snapshot used by both GET and POST ``/api/roots``."""

    roots: tuple[str, ...]
    config_path: str
    revision: str
    daemon_restart_required: bool = False  # deprecated: always false
    warning: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "roots": list(self.roots),
            "config_path": self.config_path,
            "revision": self.revision,
            "daemon_restart_required": self.daemon_restart_required,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class JobView:
    """A running or finished background job; ``queued`` is not a valid state."""

    job_id: str
    operation: str
    state: str
    started_at: str
    finished_at: str | None
    elapsed_ms: int
    report: dict[str, object] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.operation not in JOB_OPERATIONS:
            raise ValueError(f"unknown job operation: {self.operation!r}")
        if self.state not in JOB_STATES:
            raise ValueError(f"unknown job state: {self.state!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "operation": self.operation,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": self.elapsed_ms,
            "report": self.report,
            "error": self.error,
        }


@dataclass(frozen=True)
class CourseModuleDTO:
    """One module summary entry returned by ``GET /api/course``."""

    name: str
    display: str
    weeks: tuple[int, ...] = ()
    files: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "display": self.display,
            "weeks": [int(week) for week in self.weeks],
            "files": int(self.files),
        }


@dataclass(frozen=True)
class CourseView:
    """Course module and week overview returned by ``GET /api/course``."""

    modules: tuple[CourseModuleDTO, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "modules": [module.to_dict() for module in self.modules],
        }


@dataclass(frozen=True)
class PreviewView:
    """Full chunk text and metadata returned by ``GET /api/preview``."""

    chunk: int
    kind: str
    path: str
    locator: str
    location: str
    text: str
    page_image: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk": int(self.chunk),
            "kind": self.kind,
            "path": self.path,
            "locator": self.locator,
            "location": self.location,
            "text": self.text,
            "page_image": bool(self.page_image),
        }


@dataclass(frozen=True)
class StatusView:
    """A model-free, read-only status snapshot."""

    index: dict[str, object]
    model: dict[str, object]
    daemon: dict[str, object]
    config: dict[str, object]
    activity: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": dict(self.index),
            "model": dict(self.model),
            "daemon": dict(self.daemon),
            "config": dict(self.config),
            "activity": dict(self.activity),
        }


class WebBackend(Protocol):
    """Constructor-injected backend consumed by the HTTP layer and browser tests.

    Implementations own all Store/model/config work.  No method blocks on HTTP
    parsing and no method takes an extra undocumented argument.
    """

    def status(self) -> StatusView:
        """Return a read-only status snapshot; never load a model."""
        ...

    def roots(self) -> RootsView:
        """Return stored roots in config order and the raw-bytes revision."""
        ...

    def add_root(self, path: str, revision: str) -> RootsView:
        """Validate and append one root using the stale-revision check."""
        ...

    def remove_root(self, path: str, revision: str) -> RootsView:
        """Validate and remove one root using the stale-revision check."""
        ...

    def course(self) -> CourseView:
        """Return course module/week/file summary from the index."""
        ...

    def preview(self, chunk_id: int) -> PreviewView:
        """Return full chunk text and metadata for ``chunk_id``."""
        ...

    def preview_page_png(self, chunk_id: int) -> bytes:
        """Render the PDF page for ``chunk_id`` as PNG bytes."""
        ...

    def open_chunk(self, chunk_id: int) -> None:
        """Open the source file for ``chunk_id`` at its locator."""
        ...

    def search(self, request: SearchRequest) -> SearchResult:
        """Run a validated search with the frozen daemon/read-only/lane policy."""
        ...

    def similar(self, request: SimilarRequest) -> SearchResult:
        """Return indexed hits similar to ``request.path`` under the lane policy."""
        ...

    def request_shutdown(self) -> None:
        """Ask the hosting process to stop; return without waiting for it."""
        ...

    def start_index_paths(self, paths: list[str]) -> str:
        """Start an async index job for specific files/directories; return its job ID."""
        ...

    def start_model_check(self) -> str:
        """Start the explicit model-check job and return its opaque job ID."""
        ...

    def start_index(self) -> str:
        """Start the async index job and return its opaque job ID."""
        ...

    def start_download(self, rerank: bool = False) -> str:
        """Start the explicit model/grammar download job and return its opaque job ID."""
        ...

    def job(self, job_id: str) -> JobView:
        """Return one retained job or raise ``ApiError(404, "job_not_found")``."""
        ...
