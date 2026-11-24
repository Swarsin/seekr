"""Shared, entirely synthetic fixtures for the local web interface tests.

Only :mod:`seekr.web_contract` is imported.  There is no real config, database,
document, model, socket, thread, or network activity here; every path and text
is obvious test data under the pre-created ``/tmp/opencode`` fixture area.
"""

from __future__ import annotations

from seekr.web_contract import (
    ApiError,
    CourseModuleDTO,
    CourseView,
    HitDTO,
    JobView,
    PreviewView,
    RootsView,
    SearchRequest,
    SearchResult,
    SimilarRequest,
    StatusView,
)

__all__ = [
    "COURSE_VIEW",
    "ERROR_BUSY",
    "ERROR_DAEMON_ACTIVE",
    "ERROR_FORBIDDEN",
    "ERROR_INCOMPATIBLE_INDEX",
    "ERROR_INDEX_UNAVAILABLE",
    "ERROR_INVALID_REQUEST",
    "ERROR_JOB_NOT_FOUND",
    "ERROR_MODEL_UNAVAILABLE",
    "FIXTURE_CONFIG_PATH",
    "FIXTURE_REVISION",
    "FIXTURE_ROOT",
    "FIXTURE_SOCKET_PATH",
    "JOB_DOWNLOAD_SUCCEEDED",
    "JOB_FAILED",
    "JOB_MODEL_SUCCEEDED",
    "JOB_PARTIAL",
    "JOB_RUNNING",
    "JOB_SUCCEEDED",
    "PREVIEW_PNG_BYTES",
    "PREVIEW_VIEW",
    "ROOTS_VIEW",
    "SEARCH_HITS",
    "SEARCH_RESULT",
    "STATUS_EMPTY",
    "STATUS_ERROR",
    "STATUS_READY",
    "FakeWebBackend",
]

FIXTURE_ROOT = "/tmp/opencode/seekr-web-fixtures"
FIXTURE_CONFIG_PATH = f"{FIXTURE_ROOT}/config.toml"
FIXTURE_SOCKET_PATH = f"{FIXTURE_ROOT}/seekr.sock"
FIXTURE_REVISION = "0123456789abcdef" * 4

STATUS_READY = StatusView(
    index={
        "state": "ready",
        "documents": 3,
        "chunks": 12,
        "size_bytes": 49152,
        "last_error_count": 0,
        "model": "fixture-embedder",
        "dim": 8,
        "vector_quant": "int8",
        "schema_version": 2,
        "message": None,
    },
    model={"state": "ready", "model": "fixture-embedder", "dim": 8, "message": None},
    daemon={"present": False, "socket": FIXTURE_SOCKET_PATH},
    config={
        "state": "ok",
        "path": FIXTURE_CONFIG_PATH,
        "revision": FIXTURE_REVISION,
        "roots": 2,
        "message": None,
    },
    activity={"lane": "idle", "job_id": None},
)

STATUS_EMPTY = StatusView(
    index={
        "state": "empty",
        "documents": 0,
        "chunks": 0,
        "size_bytes": 0,
        "last_error_count": 0,
        "model": None,
        "dim": None,
        "vector_quant": None,
        "schema_version": None,
        "message": None,
    },
    model={"state": "unchecked", "model": "fixture-embedder", "dim": None, "message": None},
    daemon={"present": False, "socket": FIXTURE_SOCKET_PATH},
    config={
        "state": "ok",
        "path": FIXTURE_CONFIG_PATH,
        "revision": FIXTURE_REVISION,
        "roots": 0,
        "message": None,
    },
    activity={"lane": "idle", "job_id": None},
)

STATUS_ERROR = StatusView(
    index={
        "state": "error",
        "documents": None,
        "chunks": None,
        "size_bytes": None,
        "last_error_count": None,
        "model": None,
        "dim": None,
        "vector_quant": None,
        "schema_version": None,
        "message": "fixture database is unreadable",
    },
    model={"state": "unchecked", "model": "fixture-embedder", "dim": None, "message": None},
    daemon={"present": False, "socket": FIXTURE_SOCKET_PATH},
    config={
        "state": "ok",
        "path": FIXTURE_CONFIG_PATH,
        "revision": FIXTURE_REVISION,
        "roots": 1,
        "message": None,
    },
    activity={"lane": "idle", "job_id": None},
)

ROOTS_VIEW = RootsView(
    roots=(
        f"{FIXTURE_ROOT}/docs",
        f"{FIXTURE_ROOT}/notes",
    ),
    config_path=FIXTURE_CONFIG_PATH,
    revision=FIXTURE_REVISION,
)

COURSE_VIEW = CourseView(
    modules=(
        CourseModuleDTO(
            name="machine-learning",
            display="machine learning",
            weeks=(1, 2),
            files=3,
        ),
    )
)

PREVIEW_VIEW = PreviewView(
    chunk=101,
    kind="markdown",
    path=f"{FIXTURE_ROOT}/notes.md",
    locator="line 12",
    location=f"{FIXTURE_ROOT}/notes.md (line 12)",
    text="Fixture xylophone notes from the synthetic test corpus.",
    page_image=False,
)

PREVIEW_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)

SEARCH_HITS = (
    HitDTO(
        chunk_id=101,
        path=f"{FIXTURE_ROOT}/notes.md",
        locator="line 12",
        loc_start=12.0,
        snippet="Fixture xylophone notes from the synthetic test corpus.",
        score=0.91,
        kind="markdown",
    ),
    HitDTO(
        chunk_id=102,
        path=f"{FIXTURE_ROOT}/guide.txt",
        locator="line 4",
        loc_start=4.0,
        snippet="A second generated passage about the fixture xylophone.",
        score=0.72,
        kind="text",
    ),
    HitDTO(
        chunk_id=103,
        path=f"{FIXTURE_ROOT}/sample.py",
        locator="line 27",
        loc_start=27.0,
        snippet="def fixture_xylophone() -> str: return 'synthetic'",
        score=0.55,
        kind="code",
    ),
)

SEARCH_RESULT = SearchResult(hits=SEARCH_HITS, took_ms=12.5)

_FINISHED_INDEX_REPORT = {
    "added": 2,
    "updated": 0,
    "unchanged": 1,
    "removed": 0,
    "skipped": 0,
    "errors": [],
    "warnings": [
        {
            "path": f"{FIXTURE_ROOT}/notes.md",
            "message": "fixture warning: skipped an empty fixture section",
        }
    ],
    "error_total": 0,
    "warning_total": 1,
    "errors_truncated": False,
    "warnings_truncated": False,
}

JOB_RUNNING = JobView(
    job_id="job-index-running",
    operation="index",
    state="running",
    started_at="2026-11-20T12:00:00Z",
    finished_at=None,
    elapsed_ms=1500,
    report=None,
    error=None,
)

JOB_SUCCEEDED = JobView(
    job_id="job-index-succeeded",
    operation="index",
    state="succeeded",
    started_at="2026-11-20T12:00:00Z",
    finished_at="2026-11-20T12:00:03Z",
    elapsed_ms=3000,
    report=_FINISHED_INDEX_REPORT,
    error=None,
)

JOB_PARTIAL = JobView(
    job_id="job-index-partial",
    operation="index",
    state="partial",
    started_at="2026-11-20T12:00:00Z",
    finished_at="2026-11-20T12:00:04Z",
    elapsed_ms=4000,
    report={
        "added": 1,
        "updated": 0,
        "unchanged": 0,
        "removed": 0,
        "skipped": 1,
        "errors": [
            {
                "path": f"{FIXTURE_ROOT}/broken.md",
                "message": "fixture error: generated file could not be read",
            }
        ],
        "warnings": [],
        "error_total": 1,
        "warning_total": 0,
        "errors_truncated": False,
        "warnings_truncated": False,
    },
    error=None,
)

JOB_FAILED = JobView(
    job_id="job-index-failed",
    operation="index",
    state="failed",
    started_at="2026-11-20T12:00:00Z",
    finished_at="2026-11-20T12:00:01Z",
    elapsed_ms=1000,
    report=None,
    error="fixture error: index job failed before producing a report",
)

JOB_MODEL_SUCCEEDED = JobView(
    job_id="job-model-succeeded",
    operation="model_check",
    state="succeeded",
    started_at="2026-11-20T12:00:00Z",
    finished_at="2026-11-20T12:00:02Z",
    elapsed_ms=2000,
    report={"model": "fixture-embedder", "dim": 8},
    error=None,
)

JOB_DOWNLOAD_SUCCEEDED = JobView(
    job_id="job-download-succeeded",
    operation="download",
    state="succeeded",
    started_at="2026-11-20T12:00:00Z",
    finished_at="2026-11-20T12:00:05Z",
    elapsed_ms=5000,
    report={"rerank": False, "returncode": 0, "lines": ["Done."]},
    error=None,
)

ERROR_INVALID_REQUEST = ApiError(400, "invalid_request", "invalid request")
ERROR_FORBIDDEN = ApiError(
    403,
    "forbidden",
    "browser-origin headers are required for this local request",
)
ERROR_JOB_NOT_FOUND = ApiError(404, "job_not_found", "job was not found or has expired")
ERROR_BUSY = ApiError(409, "busy", "another engine operation is already running")
ERROR_DAEMON_ACTIVE = ApiError(
    409,
    "daemon_active",
    "a seekr daemon is running; stop `seekr daemon` before a web index scan",
)
ERROR_INCOMPATIBLE_INDEX = ApiError(
    409,
    "incompatible_index",
    "the configured model or index schema is incompatible",
)
ERROR_INDEX_UNAVAILABLE = ApiError(
    503,
    "index_unavailable",
    "the index database is unreadable; run `seekr doctor`",
)
ERROR_MODEL_UNAVAILABLE = ApiError(
    503,
    "model_unavailable",
    "the embedding model is unavailable; run an explicit model check",
)


class FakeWebBackend:
    """A deterministic in-memory implementation of :class:`WebBackend`.

    ``errors`` maps a backend method name (``"status"``, ``"roots"``,
    ``"add_root"``, ``"remove_root"``, ``"course"``, ``"preview"``,
    ``"preview_page_png"``, ``"open_chunk"``, ``"search"``,
    ``"similar"``, ``"request_shutdown"``, ``"start_index_paths"``,
    ``"start_model_check"``, ``"start_index"``, ``"start_reindex"``,
    ``"start_download"``,
    or ``"job"``) to an :class:`ApiError` that method raises instead of
    returning.  Every call is appended to ``calls`` as
    ``(method_name, keyword_arguments)``.

    Starting a check, index, or download job automatically records a running
    ``JobView`` when the configured job ID is not already present in ``jobs``.
    """

    def __init__(
        self,
        *,
        status: StatusView | None = None,
        roots: RootsView | None = None,
        search_result: SearchResult | None = None,
        add_root_result: RootsView | None = None,
        remove_root_result: RootsView | None = None,
        course_view: CourseView | None = None,
        preview_view: PreviewView | None = None,
        preview_png: bytes = PREVIEW_PNG_BYTES,
        jobs: dict[str, JobView] | None = None,
        start_model_check_job_id: str = "job-model-check-fixture",
        start_index_job_id: str = "job-index-fixture",
        start_download_job_id: str = "job-download-fixture",
        similar_result: SearchResult | None = None,
        start_index_paths_job_id: str = "job-index-paths-fixture",
        start_reindex_job_id: str = "job-reindex-fixture",
        errors: dict[str, ApiError] | None = None,
    ) -> None:
        self.status_value = STATUS_READY if status is None else status
        self.roots_value = ROOTS_VIEW if roots is None else roots
        self.search_result_value = SEARCH_RESULT if search_result is None else search_result
        self.add_root_result_value = (
            ROOTS_VIEW if add_root_result is None else add_root_result
        )
        self.remove_root_result_value = (
            ROOTS_VIEW if remove_root_result is None else remove_root_result
        )
        self.course_view_value = COURSE_VIEW if course_view is None else course_view
        self.preview_view_value = PREVIEW_VIEW if preview_view is None else preview_view
        self.preview_png_value = preview_png
        self.jobs: dict[str, JobView] = dict(jobs or {})
        self.start_model_check_job_id = start_model_check_job_id
        self.start_index_job_id = start_index_job_id
        self.start_download_job_id = start_download_job_id
        self.similar_result_value = SEARCH_RESULT if similar_result is None else similar_result
        self.start_index_paths_job_id = start_index_paths_job_id
        self.start_reindex_job_id = start_reindex_job_id
        self.errors = dict(errors or {})
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _record(self, method: str, **arguments: object) -> None:
        self.calls.append((method, arguments))

    def _raise_configured(self, method: str) -> None:
        error = self.errors.get(method)
        if error is not None:
            raise error

    def _ensure_running_job(self, job_id: str, operation: str) -> None:
        if job_id in self.jobs:
            return
        self.jobs[job_id] = JobView(
            job_id=job_id,
            operation=operation,
            state="running",
            started_at="2026-11-20T12:00:00Z",
            finished_at=None,
            elapsed_ms=0,
            report=None,
            error=None,
        )

    def status(self) -> StatusView:
        self._record("status")
        self._raise_configured("status")
        return self.status_value

    def roots(self) -> RootsView:
        self._record("roots")
        self._raise_configured("roots")
        return self.roots_value

    def add_root(self, path: str, revision: str) -> RootsView:
        self._record("add_root", path=path, revision=revision)
        self._raise_configured("add_root")
        return self.add_root_result_value

    def remove_root(self, path: str, revision: str) -> RootsView:
        self._record("remove_root", path=path, revision=revision)
        self._raise_configured("remove_root")
        return self.remove_root_result_value

    def course(self) -> CourseView:
        self._record("course")
        self._raise_configured("course")
        return self.course_view_value

    def preview(self, chunk_id: int) -> PreviewView:
        self._record("preview", chunk_id=chunk_id)
        self._raise_configured("preview")
        return self.preview_view_value

    def preview_page_png(self, chunk_id: int) -> bytes:
        self._record("preview_page_png", chunk_id=chunk_id)
        self._raise_configured("preview_page_png")
        return self.preview_png_value

    def open_chunk(self, chunk_id: int) -> None:
        self._record("open_chunk", chunk_id=chunk_id)
        self._raise_configured("open_chunk")

    def search(self, request: SearchRequest) -> SearchResult:
        self._record("search", request=request)
        self._raise_configured("search")
        return self.search_result_value

    def similar(self, request: SimilarRequest) -> SearchResult:
        self._record("similar", request=request)
        self._raise_configured("similar")
        return self.similar_result_value

    def request_shutdown(self) -> None:
        self._record("request_shutdown")
        self._raise_configured("request_shutdown")

    def start_index_paths(self, paths: list[str]) -> str:
        self._record("start_index_paths", paths=list(paths))
        self._raise_configured("start_index_paths")
        self._ensure_running_job(self.start_index_paths_job_id, "index")
        return self.start_index_paths_job_id

    def start_model_check(self) -> str:
        self._record("start_model_check")
        self._raise_configured("start_model_check")
        self._ensure_running_job(self.start_model_check_job_id, "model_check")
        return self.start_model_check_job_id

    def start_index(self) -> str:
        self._record("start_index")
        self._raise_configured("start_index")
        self._ensure_running_job(self.start_index_job_id, "index")
        return self.start_index_job_id

    def start_reindex(self, force: bool = False, allow_empty: bool = False) -> str:
        self._record("start_reindex", force=force, allow_empty=allow_empty)
        self._raise_configured("start_reindex")
        self._ensure_running_job(self.start_reindex_job_id, "index")
        return self.start_reindex_job_id

    def start_download(self, rerank: bool = False) -> str:
        self._record("start_download", rerank=rerank)
        self._raise_configured("start_download")
        self._ensure_running_job(self.start_download_job_id, "download")
        return self.start_download_job_id

    def job(self, job_id: str) -> JobView:
        self._record("job", job_id=job_id)
        self._raise_configured("job")
        job = self.jobs.get(job_id)
        if job is None:
            raise ERROR_JOB_NOT_FOUND
        return job
