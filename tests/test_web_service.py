"""Offline end-to-end tests for the web application service.

Every test uses a temporary config, database and roots.  The embedding backend
is always :class:`seekr.embed.FakeEmbedder`; no real model or network is used.
"""

from __future__ import annotations

import contextlib
import json
import socket
import sqlite3
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from _platform import LEGACY_DAEMON, legacy_daemon_only
from seekr.cli_search import get_hits
from seekr.config import Config, load_config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer, SyncReport
from seekr.lifecycle import LifecycleLock
from seekr.locations import format_location
from seekr.models import ChunkSpec, Hit, HitExplain, text_sha
from seekr.platform_info import Mount
from seekr.store import SCHEMA_VERSION, Store, StoredChunk
from seekr.web_contract import (
    ApiError,
    CourseModuleDTO,
    CourseView,
    HitDTO,
    PreviewView,
    SearchRequest,
    SimilarRequest,
)
from seekr.web_service import WebService, _socket_is_live


def _toml(value: str) -> str:
    return json.dumps(value)


def _write_config(
    tmp_path: Path,
    *,
    roots: list[Path] | None = None,
    db_path: Path | None = None,
    embed_model: str = "fake",
    suffix: str = "",
) -> Path:
    lines = [
        f"embed_model = {_toml(embed_model)}",
        f"db_path = {_toml(str(db_path or (tmp_path / 'index.db')))}",
        'vector_quant = "int8"',
    ]
    if roots is not None:
        lines.append("roots = [" + ", ".join(_toml(str(root)) for root in roots) + "]")
    path = tmp_path / "config.toml"
    path.write_text("\n".join(lines) + "\n" + suffix, encoding="utf-8")
    return path


def _root(tmp_path: Path, name: str = "docs") -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def _write_docs(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


@contextlib.contextmanager
def _service(
    config_path: Path,
    tmp_path: Path,
    *,
    embedder_factory=None,
    daemon_probe=None,
    **kwargs,
):
    factory = embedder_factory or (lambda cfg: FakeEmbedder(dim=64))
    options = dict(kwargs)
    options.setdefault("daemon_socket_path", tmp_path / "seekr.sock")
    options.setdefault("daemon_probe", daemon_probe or (lambda path: False))
    service = WebService(config_path, embedder_factory=factory, **options)
    try:
        yield service
    finally:
        service.shutdown()


def _wait_job(service: WebService, job_id: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = service.job(job_id)
        if view.state != "running":
            return view
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {service.job(job_id)}")


def _flatten(calls: list[list[str]]) -> list[str]:
    return [text for batch in calls for text in batch]


def _add_indexed_document(
    db_path: Path,
    *,
    model: str = "fake",
    text: str = "alpha xylophone notes",
    path: str = "/notes/a.txt",
    vector_quant: str = "int8",
) -> None:
    embedder = FakeEmbedder(dim=64)
    store = Store(db_path, model, 64, vector_quant=vector_quant)
    try:
        doc_id = store.upsert_document(path, 1, len(text), "a" * 64, "text")
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


def _error_code(exc: ApiError) -> str:
    return exc.code


def _drop_index_table(db_path: Path, table: str) -> None:
    connection = sqlite3.connect(db_path)
    try:
        if table == "chunks_fts":
            for trigger in ("chunks_fts_ai", "chunks_fts_ad", "chunks_fts_au"):
                connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        if table == "chunks_vec":
            import sqlite_vec

            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
        connection.execute(f"DROP TABLE {table}")
        connection.commit()
    finally:
        connection.close()


def _drop_primary_tables(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE chunks")
        connection.execute("DROP TABLE documents")
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "unknown-kind"),
        ("kind", 123),
        ("since", "7x"),
        ("since", "0d"),
        ("since", 123),
    ],
)
def test_invalid_search_filters_are_400_before_daemon(
    tmp_path: Path, mode: str, field: str, value: object
) -> None:
    config_path = _write_config(tmp_path)
    daemon_calls: list[object] = []
    probes: list[Path] = []

    def daemon(*args, **kwargs):
        daemon_calls.append((args, kwargs))
        raise AssertionError("a daemon must not be called for invalid filters")

    def probe(path: Path) -> bool:
        probes.append(path)
        return True

    with (
        _service(
            config_path,
            tmp_path,
            daemon_probe=probe,
            daemon_search=daemon,
        ) as service,
        pytest.raises(ApiError) as raised,
    ):
        service.search(SearchRequest("alpha", mode=mode, **{field: value}))

    exc = raised.value
    assert exc.status == 400
    assert exc.code == "invalid_request"
    assert set(exc.fields or {}) == {field}
    assert probes == []
    assert daemon_calls == []


@pytest.mark.parametrize("error_type", [OverflowError, OSError])
def test_out_of_range_since_overflow_is_400_before_daemon(
    tmp_path: Path, error_type: type[BaseException]
) -> None:
    config_path = _write_config(tmp_path)
    daemon_calls: list[object] = []

    def daemon(*args, **kwargs):
        daemon_calls.append((args, kwargs))
        raise AssertionError("a daemon must not be called for invalid filters")

    def overflow(value: str | None) -> int | None:
        raise error_type("timestamp out of range")

    with (
        mock.patch("seekr.filters.parse_since", side_effect=overflow),
        _service(
            config_path,
            tmp_path,
            daemon_probe=lambda path: True,
            daemon_search=daemon,
        ) as service,
        pytest.raises(ApiError) as raised,
    ):
        service.search(SearchRequest("alpha", mode="hybrid", since="9999-12-31"))

    exc = raised.value
    assert exc.status == 400
    assert exc.code == "invalid_request"
    assert set(exc.fields or {}) == {"since"}
    assert daemon_calls == []


@pytest.mark.parametrize("mode", ["keyword", "hybrid"])
def test_real_out_of_sqlite_range_since_is_400_on_populated_index(
    tmp_path: Path, mode: str
) -> None:
    db_path = tmp_path / "index.db"
    _add_indexed_document(db_path)
    config_path = _write_config(tmp_path, db_path=db_path)

    with (
        _service(config_path, tmp_path) as service,
        pytest.raises(ApiError) as raised,
    ):
        service.search(SearchRequest("alpha", mode=mode, since="9999-12-31"))

    exc = raised.value
    assert exc.status == 400
    assert exc.code == "invalid_request"
    assert set(exc.fields or {}) == {"since"}


def test_shutdown_between_lane_start_and_job_publish_starts_no_thread(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    service = WebService(
        config_path,
        embedder_factory=lambda cfg: FakeEmbedder(dim=64),
        daemon_socket_path=tmp_path / "seekr.sock",
        daemon_probe=lambda path: False,
    )
    entered = threading.Event()
    release = threading.Event()
    captured: dict[str, str] = {}
    errors: list[BaseException] = []
    original_new_job = service._new_job

    def gated_new_job(operation: str):
        record = original_new_job(operation)
        captured["job_id"] = record.job_id
        entered.set()
        if not release.wait(5.0):
            raise RuntimeError("publish gate was not released")
        return record

    def start() -> None:
        try:
            service.start_model_check()
        except ApiError as exc:
            errors.append(exc)

    gate = mock.patch.object(service, "_new_job", side_effect=gated_new_job)
    starter = threading.Thread(target=start)
    try:
        with gate:
            starter.start()
            assert entered.wait(5.0)
            # Shutdown sees no published job thread and completes; the start
            # request must observe _shutting_down before it can start one.
            service.shutdown()
        release.set()
        starter.join(5.0)
    finally:
        release.set()
        service.shutdown()

    assert not starter.is_alive()
    assert len(errors) == 1
    exc = errors[0]
    assert isinstance(exc, ApiError)
    assert exc.status == 409
    assert exc.code == "busy"
    assert service.job(captured["job_id"]).state == "failed"
    assert not any(
        thread.name.startswith("seekr-web-") for thread in threading.enumerate()
    )


def test_terminal_model_check_releases_lane_before_job_is_visible(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)

    with _service(config_path, tmp_path) as service:
        first = _wait_job(service, service.start_model_check())
        assert first.state == "succeeded"
        assert service.status().activity == {"lane": "idle", "job_id": None}

        second = _wait_job(service, service.start_model_check())
        assert second.state == "succeeded"


def test_terminal_index_job_releases_lane_before_job_is_visible(
    tmp_path: Path,
) -> None:
    docs = _root(tmp_path)
    _write_docs(docs, {"a.txt": "alpha xylophone notes\n"})
    config_path = _write_config(tmp_path, roots=[docs])

    with _service(config_path, tmp_path) as service:
        first = _wait_job(service, service.start_index())
        assert first.state == "succeeded"
        assert service.status().activity == {"lane": "idle", "job_id": None}

        second = _wait_job(service, service.start_model_check())
        assert second.state == "succeeded"


def test_fake_model_index_search_add_root_and_unchanged_resync_without_reembedding(
    tmp_path: Path,
) -> None:
    docs = _root(tmp_path)
    _write_docs(
        docs,
        {
            "a.md": "# Alpha\n\nThe alpha xylophone notes live here.\n",
            "b.txt": "Beta xylophone guide for the synthetic corpus.\n",
        },
    )
    config_path = _write_config(tmp_path, roots=[docs])
    embedder = FakeEmbedder(dim=64)
    built: list[str] = []

    def factory(cfg: Config):
        built.append(cfg.embed_model)
        return embedder

    with _service(config_path, tmp_path, embedder_factory=factory) as service:
        first_job = service.start_index()
        first = _wait_job(service, first_job)
        assert first.state == "succeeded"
        assert first.report is not None
        assert first.report["added"] == 2
        assert first.report["unchanged"] == 0
        first_embedded_count = len(_flatten(embedder.calls))
        assert first_embedded_count >= 2
        assert built == ["fake"]

        status = service.status()
        assert status.index["state"] == "ready"
        assert status.index["documents"] == 2
        assert status.model == {
            "state": "ready",
            "model": "fake",
            "dim": 64,
            "message": None,
        }

        keyword = service.search(SearchRequest("xylophone", mode="keyword"))
        assert {hit.path for hit in keyword.hits} == {
            str(docs / "a.md"),
            str(docs / "b.txt"),
        }
        assert len(_flatten(embedder.calls)) == first_embedded_count

        hybrid = service.search(SearchRequest("xylophone", mode="hybrid"))
        assert len(hybrid.hits) == 2
        assert built == ["fake"]

        roots = service.roots()
        second_root = tmp_path / "more"
        second_root.mkdir()
        _write_docs(second_root, {"c.txt": "Gamma xylophone addendum.\n"})
        added = service.add_root(str(second_root), roots.revision)
        assert added.roots[-1] == str(second_root)
        assert added.daemon_restart_required is False
        assert added.warning is None

        second_job = service.start_index()
        second = _wait_job(service, second_job)
        assert second.state == "succeeded"
        assert second.report is not None
        assert second.report["added"] == 1
        assert second.report["unchanged"] == 2
        assert second.report["removed"] == 0
        new_texts = _flatten(embedder.calls)[first_embedded_count:]
        assert new_texts
        assert all("gamma" in text.lower() for text in new_texts)
        assert built == ["fake"]

        status = service.status()
        assert status.index["state"] == "ready"
        assert status.index["documents"] == 3
        assert status.index["chunks"] == 3


def test_missing_and_empty_index_short_circuit_without_model(tmp_path: Path) -> None:
    missing_config = _write_config(tmp_path, db_path=tmp_path / "missing.db")
    built: list[str] = []

    def explode(cfg: Config):
        built.append(cfg.embed_model)
        raise AssertionError("empty search must not build a model")

    with _service(missing_config, tmp_path, embedder_factory=explode) as service:
        result = service.search(SearchRequest("anything", mode="hybrid"))
        assert result.hits == ()
        assert isinstance(result.took_ms, float)
        assert built == []
        assert not (tmp_path / "missing.db").exists()

    empty_db = tmp_path / "empty.db"
    Store(empty_db, "fake", 64, vector_quant="int8").close()
    empty_config = _write_config(tmp_path, db_path=empty_db)
    with _service(empty_config, tmp_path, embedder_factory=explode) as service:
        for mode in ("keyword", "vector", "hybrid"):
            result = service.search(SearchRequest("anything", mode=mode))
            assert result.hits == ()
        assert built == []


def test_keyword_path_is_read_only_and_matches_local_get_hits(tmp_path: Path) -> None:
    docs = _root(tmp_path)
    _write_docs(docs, {"a.txt": "alpha xylophone notes\n"})
    db_path = tmp_path / "index.db"
    _add_indexed_document(db_path, path=str(docs / "a.txt"))
    config_path = _write_config(tmp_path, roots=[docs], db_path=db_path)
    cfg = load_config(config_path)

    expected_hits = get_hits(
        "xylophone",
        mode="keyword",
        k=10,
        per_file=2,
        cfg=cfg,
        embedder=FakeEmbedder(dim=64),
        use_daemon=False,
    )
    expected = [
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
        for hit in expected_hits
    ]

    recorded: list[dict[str, object]] = []

    def store_factory(*args, **kwargs):
        recorded.append(dict(kwargs))
        return Store(*args, **kwargs)

    def explode(cfg: Config):
        raise AssertionError("keyword search must not build a model")

    with _service(
        config_path,
        tmp_path,
        embedder_factory=explode,
        store_factory=store_factory,
    ) as service:
        with mock.patch(
            "seekr.cli_search._build_searcher",
            side_effect=AssertionError("keyword must not use _build_searcher"),
        ) as build_searcher:
            result = service.search(SearchRequest("xylophone", mode="keyword"))

        assert build_searcher.call_count == 0
        assert [hit.to_dict() for hit in result.hits] == expected
        assert recorded == [
            {"vector_quant": "int8", "read_only": True},
        ]


def test_status_missing_empty_ready_incompatible_and_error(tmp_path: Path) -> None:
    built: list[str] = []

    def explode(cfg: Config):
        built.append(cfg.embed_model)
        raise AssertionError("status must never load a model")

    missing_db = tmp_path / "missing.db"
    config_path = _write_config(tmp_path, db_path=missing_db)
    with _service(config_path, tmp_path, embedder_factory=explode) as service:
        missing = service.status()
        assert missing.index["state"] == "missing"
        assert missing.index["documents"] == 0
        assert missing.index["chunks"] == 0
        assert missing.model["state"] == "unchecked"
        assert missing.config["state"] == "ok"
        assert missing.daemon["present"] is False
        assert missing.activity == {"lane": "idle", "job_id": None}
        assert not missing_db.exists()
        assert built == []

    empty_db = tmp_path / "empty.db"
    Store(empty_db, "fake", 64, vector_quant="int8").close()
    with _service(
        _write_config(tmp_path, db_path=empty_db), tmp_path, embedder_factory=explode
    ) as service:
        empty = service.status()
        assert empty.index["state"] == "empty"
        assert empty.index["documents"] == 0
        assert empty.index["chunks"] == 0
        assert empty.index["schema_version"] == SCHEMA_VERSION

    ready_db = tmp_path / "ready.db"
    _add_indexed_document(ready_db)
    with _service(
        _write_config(tmp_path, db_path=ready_db), tmp_path, embedder_factory=explode
    ) as service:
        ready = service.status()
        assert ready.index["state"] == "ready"
        assert ready.index["documents"] == 1
        assert ready.index["chunks"] == 1
        assert ready.index["model"] == "fake"
        assert ready.index["dim"] == 64
        assert ready.index["vector_quant"] == "int8"
        assert ready.index["schema_version"] == SCHEMA_VERSION
        assert built == []

    mismatch_db = tmp_path / "mismatch.db"
    _add_indexed_document(mismatch_db, model="different/model")
    with _service(
        _write_config(tmp_path, db_path=mismatch_db), tmp_path, embedder_factory=explode
    ) as service:
        mismatch = service.status()
        assert mismatch.index["state"] == "incompatible"
        assert mismatch.index["message"]

    corrupt_db = tmp_path / "corrupt.db"
    corrupt_db.write_bytes(b"this is not a sqlite database")
    with _service(
        _write_config(tmp_path, db_path=corrupt_db), tmp_path, embedder_factory=explode
    ) as service:
        corrupt = service.status()
        assert corrupt.index["state"] == "error"
        assert corrupt.index["message"]
        assert built == []


def test_status_invalid_config_is_reported_without_raising(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("roots = [broken\n", encoding="utf-8")

    with _service(config_path, tmp_path) as service:
        status = service.status()

    assert status.config["state"] == "invalid"
    assert status.config["revision"] is not None
    assert status.config["roots"] is None
    assert status.config["message"]
    assert status.index["state"] == "error"
    assert status.model["state"] == "unchecked"


def test_status_does_not_take_the_lifecycle_lock(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    _add_indexed_document(db_path)
    config_path = _write_config(tmp_path, db_path=db_path)

    lock = LifecycleLock(db_path, shared=False, blocking=False)
    lock.acquire()
    try:
        with _service(config_path, tmp_path) as service:
            started = time.monotonic()
            status = service.status()
            elapsed = time.monotonic() - started
    finally:
        lock.release()

    assert status.index["state"] == "ready"
    assert elapsed < 1.0


def test_model_check_success_explicit_retry_and_failure_without_auto_retry(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    built: list[str] = []

    def good(cfg: Config):
        built.append(cfg.embed_model)
        return FakeEmbedder(dim=64)

    with _service(config_path, tmp_path, embedder_factory=good) as service:
        job = service.start_model_check()
        view = _wait_job(service, job)
        assert view.state == "succeeded"
        assert view.report == {"model": "fake", "dim": 64}
        assert service.status().model == {
            "state": "ready",
            "model": "fake",
            "dim": 64,
            "message": None,
        }
        assert built == ["fake"]

        again = _wait_job(service, service.start_model_check())
        assert again.state == "succeeded"
        assert built == ["fake", "fake"]

    ready_db = tmp_path / "ready.db"
    _add_indexed_document(ready_db)
    failing_config = _write_config(tmp_path, db_path=ready_db)
    attempts: list[str] = []

    def failing(cfg: Config):
        attempts.append(cfg.embed_model)
        raise RuntimeError("model files are missing")

    with _service(failing_config, tmp_path, embedder_factory=failing) as service:
        failed = _wait_job(service, service.start_model_check())
        assert failed.state == "failed"
        assert "model files are missing" in (failed.error or "")
        assert service.status().model["state"] == "unavailable"
        assert attempts == ["fake"]

        with pytest.raises(ApiError) as raised:
            service.search(SearchRequest("alpha", mode="hybrid"))
        assert _error_code(raised.value) == "model_unavailable"
        assert attempts == ["fake"]

        keyword = service.search(SearchRequest("alpha", mode="keyword"))
        assert keyword.hits


class _StaticIndexer:
    def __init__(self, report: SyncReport) -> None:
        self.report = report

    def sync(self, roots):
        del roots
        return self.report


def test_partial_if_and_only_if_errors_and_warnings_only_succeeds(tmp_path: Path) -> None:
    root = _root(tmp_path)
    config_path = _write_config(tmp_path, roots=[root])
    reports = [
        SyncReport(
            added=1,
            warnings=[("w.txt", "warning only")],
        ),
        SyncReport(
            added=1,
            skipped=1,
            errors=[("broken.txt", "cannot read")],
            warnings=[("w.txt", "warning too")],
        ),
        SyncReport(errors=[(f"p{i}.txt", "boom") for i in range(101)]),
    ]

    def factory(store, embedder, cfg):
        del store, embedder, cfg
        return _StaticIndexer(reports.pop(0))

    with _service(config_path, tmp_path, indexer_factory=factory) as service:
        warnings_only = _wait_job(service, service.start_index())
        assert warnings_only.state == "succeeded"
        assert warnings_only.report is not None
        assert warnings_only.report["warning_total"] == 1
        assert warnings_only.report["error_total"] == 0
        assert warnings_only.report["warnings"] == [
            {"path": "w.txt", "message": "warning only"}
        ]

        partial = _wait_job(service, service.start_index())
        assert partial.state == "partial"
        assert partial.report is not None
        assert partial.report["error_total"] == 1
        assert partial.report["warning_total"] == 1
        assert partial.report["errors"] == [
            {"path": "broken.txt", "message": "cannot read"}
        ]

        truncated = _wait_job(service, service.start_index())
        assert truncated.state == "partial"
        assert truncated.report is not None
        assert truncated.report["error_total"] == 101
        assert truncated.report["errors_truncated"] is True
        assert len(truncated.report["errors"]) == 100


class _PausingIndexer:
    def __init__(self, report: SyncReport | None = None) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.report = report or SyncReport(unchanged=1)

    def sync(self, roots):
        del roots
        self.started.set()
        if not self.release.wait(10.0):
            raise RuntimeError("pausing indexer was not released")
        return self.report


def test_engine_lane_busy_keyword_bypass_and_status_real_first_index(
    tmp_path: Path,
) -> None:
    docs = _root(tmp_path)
    _write_docs(docs, {"a.txt": "alpha xylophone notes\n"})
    config_path = _write_config(tmp_path, roots=[docs])
    embedder = FakeEmbedder(dim=64)
    paused = _PausingIndexer()
    calls = 0

    def factory(store, embedder_arg, cfg):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Indexer(store, embedder_arg, cfg)
        return paused

    with _service(
        config_path,
        tmp_path,
        embedder_factory=lambda cfg: embedder,
        indexer_factory=factory,
    ) as service:
        first = _wait_job(service, service.start_index())
        assert first.state == "succeeded"
        assert service.status().index["state"] == "ready"

        job_id = service.start_index()
        assert paused.started.wait(5.0)
        assert service.status().activity == {"lane": "index", "job_id": job_id}

        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert _error_code(raised.value) == "busy"
        with pytest.raises(ApiError) as raised:
            service.start_model_check()
        assert _error_code(raised.value) == "busy"
        # Searches read concurrently with the index write.
        hybrid = service.search(SearchRequest("xylophone", mode="hybrid"))
        assert hybrid.hits

        keyword = service.search(SearchRequest("xylophone", mode="keyword"))
        assert keyword.hits
        assert service.status().index["state"] == "ready"

        with pytest.raises(ApiError) as raised:
            service.add_root(str(_root(tmp_path, "extra")), service.roots().revision)
        assert _error_code(raised.value) == "busy"

        paused.release.set()
        finished = _wait_job(service, job_id)
        assert finished.state == "succeeded"
        assert service.status().activity == {"lane": "idle", "job_id": None}


class _PausingRealIndexer(Indexer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.paused = threading.Event()
        self.release = threading.Event()
        self._paused_once = False

    def _flush_batch(self, batch, report, *, docs_cache=None):
        super()._flush_batch(batch, report, docs_cache=docs_cache)
        if not self._paused_once:
            self._paused_once = True
            self.paused.set()
            if not self.release.wait(10.0):
                raise RuntimeError("real indexer pause was not released")


def test_keyword_during_real_indexer_pause_between_committed_batches(
    tmp_path: Path,
) -> None:
    docs = _root(tmp_path)
    _write_docs(docs, {f"f{i:03d}.txt": f"batchword item {i}\n" for i in range(65)})
    config_path = _write_config(tmp_path, roots=[docs])
    embedder = FakeEmbedder(dim=64)
    indexer: _PausingRealIndexer | None = None

    def factory(store, embedder_arg, cfg):
        nonlocal indexer
        indexer = _PausingRealIndexer(store, embedder_arg, cfg)
        return indexer

    with _service(
        config_path,
        tmp_path,
        embedder_factory=lambda cfg: embedder,
        indexer_factory=factory,
    ) as service:
        job_id = service.start_index()
        deadline = time.monotonic() + 10.0
        while indexer is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert indexer is not None
        assert indexer.paused.wait(10.0)
        embedded_before = len(_flatten(embedder.calls))
        assert embedded_before == 64

        started = time.monotonic()
        result = service.search(SearchRequest("batchword", mode="keyword", k=5))
        elapsed = time.monotonic() - started

        assert elapsed < 2.0
        assert len(result.hits) == 5
        assert service.status().index["state"] == "ready"
        assert len(_flatten(embedder.calls)) == embedded_before

        indexer.release.set()
        finished = _wait_job(service, job_id)
        assert finished.state == "succeeded"
        assert finished.report is not None
        assert finished.report["added"] == 65


def test_keyword_lifecycle_lock_busy_maps_to_409(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    _add_indexed_document(db_path)
    config_path = _write_config(tmp_path, db_path=db_path)

    lock = LifecycleLock(db_path, shared=False, blocking=False)
    lock.acquire()
    try:
        with _service(config_path, tmp_path, lock_wait_seconds=0.05) as service:
            with pytest.raises(ApiError) as raised:
                service.search(SearchRequest("alpha", mode="keyword"))
            assert _error_code(raised.value) == "busy"
    finally:
        lock.release()


@legacy_daemon_only
def test_daemon_first_bypasses_lane_and_local_fallback_uses_use_daemon_false(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "index.db"
    _add_indexed_document(db_path, path="/notes/a.txt", text="alpha xylophone notes")
    config_path = _write_config(tmp_path, db_path=db_path)
    daemon_hit = Hit(
        chunk_id=999,
        path="/daemon/hit.txt",
        locator="L1",
        loc_start=1.0,
        snippet="daemon result",
        score=1.0,
        kind="text",
    )
    local_calls: list[dict[str, object]] = []
    embedder = FakeEmbedder(dim=64)

    def local_search(query, **kwargs):
        local_calls.append(kwargs)
        return get_hits(query, **kwargs)

    with _service(
        config_path,
        tmp_path,
        embedder_factory=lambda cfg: embedder,
        daemon_probe=lambda path: True,
        daemon_search=lambda *args, **kwargs: [daemon_hit],
        local_search=local_search,
    ) as service:
        result = service.search(SearchRequest("alpha", mode="hybrid"))
        assert [hit.chunk_id for hit in result.hits] == [999]
        assert service.status().activity == {"lane": "idle", "job_id": None}
        assert local_calls == []

    with _service(
        config_path,
        tmp_path,
        embedder_factory=lambda cfg: embedder,
        daemon_probe=lambda path: True,
        daemon_search=lambda *args, **kwargs: [],
        local_search=local_search,
    ) as service:
        assert service.search(SearchRequest("alpha", mode="hybrid")).hits == ()
        assert local_calls == []

    with _service(
        config_path,
        tmp_path,
        embedder_factory=lambda cfg: embedder,
        daemon_probe=lambda path: True,
        daemon_search=lambda *args, **kwargs: None,
        local_search=local_search,
    ) as service:
        result = service.search(SearchRequest("alpha", mode="hybrid"))
        assert result.hits
        assert local_calls and local_calls[-1]["use_daemon"] is False

        local_calls.clear()
        keyword = service.search(SearchRequest("alpha", mode="keyword"))
        assert keyword.hits
        assert local_calls == []


@legacy_daemon_only
def test_daemon_failure_and_live_index_refusal_and_root_warning(tmp_path: Path) -> None:
    root = _root(tmp_path)
    config_path = _write_config(tmp_path, roots=[root])
    from seekr.client import DaemonError

    def failing(*args, **kwargs):
        raise DaemonError("daemon protocol failed")

    with _service(
        config_path,
        tmp_path,
        daemon_probe=lambda path: True,
        daemon_search=failing,
    ) as service:
        with pytest.raises(ApiError) as raised:
            service.search(SearchRequest("alpha", mode="keyword"))
        assert _error_code(raised.value) == "daemon_failure"

        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert _error_code(raised.value) == "daemon_active"

        added = service.add_root(str(_root(tmp_path, "second")), service.roots().revision)
        assert added.daemon_restart_required is False
        assert added.warning is None


@legacy_daemon_only
def test_daemon_socket_liveness_is_a_real_unix_probe(tmp_path: Path) -> None:
    root = _root(tmp_path)
    config_path = _write_config(tmp_path, roots=[root])
    sock_path = tmp_path / "live.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock_path))
    listener.listen(1)
    try:
        with _service(
            config_path,
            tmp_path,
            daemon_socket_path=sock_path,
            daemon_probe=_socket_is_live,
            daemon_search=lambda *args, **kwargs: [],
        ) as service:
            assert service.daemon_present() is True
            result = service.search(SearchRequest("alpha", mode="hybrid"))
            assert result.hits == ()
    finally:
        listener.close()


def test_socket_is_live_returns_false_for_long_path(tmp_path: Path) -> None:
    root = _root(tmp_path)
    config_path = _write_config(tmp_path, roots=[root])
    deep = tmp_path / ("d" * 120)
    deep.mkdir()
    long_sock = deep / "seekr.sock"
    assert _socket_is_live(long_sock) is False
    with _service(
        config_path,
        tmp_path,
        daemon_socket_path=long_sock,
        daemon_probe=_socket_is_live,
    ) as service:
        assert service.daemon_present() is False
        assert service.status().daemon["present"] is False
        assert service.search(SearchRequest("alpha", mode="keyword")).hits == ()


def test_search_corrupt_mismatched_and_older_schema_codes(tmp_path: Path) -> None:
    corrupt_db = tmp_path / "corrupt.db"
    corrupt_db.write_bytes(b"not sqlite")
    with _service(
        _write_config(tmp_path, db_path=corrupt_db), tmp_path
    ) as service:
        with pytest.raises(ApiError) as raised:
            service.search(SearchRequest("alpha", mode="keyword"))
        assert _error_code(raised.value) == "index_unavailable"
        assert service.status().index["state"] == "error"

    mismatch_db = tmp_path / "mismatch.db"
    _add_indexed_document(mismatch_db, model="other/model")
    with _service(
        _write_config(tmp_path, db_path=mismatch_db), tmp_path
    ) as service:
        with pytest.raises(ApiError) as raised:
            service.search(SearchRequest("alpha", mode="keyword"))
        assert _error_code(raised.value) == "incompatible_index"

    for older_version in ("1", "2"):
        older_db = tmp_path / f"older-v{older_version}.db"
        _add_indexed_document(older_db)
        connection = sqlite3.connect(older_db)
        try:
            connection.execute(
                "UPDATE meta SET value = ? WHERE key = 'schema_version'",
                (older_version,),
            )
            connection.commit()
        finally:
            connection.close()
        with _service(_write_config(tmp_path, db_path=older_db), tmp_path) as service:
            with pytest.raises(ApiError) as raised:
                service.search(SearchRequest("alpha", mode="keyword"))
            assert _error_code(raised.value) == "incompatible_index"
            assert "run `seekr index` once to upgrade it" in raised.value.message
            assert "reindex" not in raised.value.message.lower()
            status = service.status()
            assert status.index["state"] == "incompatible"
            assert "run `seekr index` once to upgrade it" in str(status.index["message"])
            assert "reindex" not in str(status.index["message"]).lower()


@pytest.mark.parametrize("table", ["chunks_fts", "chunks_vec"])
def test_populated_index_missing_required_table_is_unavailable(
    tmp_path: Path, table: str
) -> None:
    db_path = tmp_path / "index.db"
    _add_indexed_document(db_path)
    _drop_index_table(db_path, table)
    docs = _root(tmp_path, f"docs-{table}")
    config_path = _write_config(tmp_path, roots=[docs], db_path=db_path)
    opened: list[dict[str, object]] = []

    def store_factory(*args, **kwargs):
        opened.append(dict(kwargs))
        return Store(*args, **kwargs)

    with _service(
        config_path, tmp_path, store_factory=store_factory
    ) as service:
        status = service.status()
        assert status.index["state"] == "error"
        message = str(status.index["message"])
        assert "missing required table" in message
        assert table in message

        for mode in ("keyword", "hybrid"):
            with pytest.raises(ApiError) as raised:
                service.search(SearchRequest("alpha", mode=mode))
            assert raised.value.status == 503
            assert raised.value.code == "index_unavailable"

        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert raised.value.status == 503
        assert raised.value.code == "index_unavailable"
        assert opened == []


def test_populated_index_missing_meta_is_incompatible(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "index.db"
    _add_indexed_document(db_path)
    _drop_index_table(db_path, "meta")
    docs = _root(tmp_path, "docs-meta")
    config_path = _write_config(tmp_path, roots=[docs], db_path=db_path)
    opened: list[dict[str, object]] = []

    def store_factory(*args, **kwargs):
        opened.append(dict(kwargs))
        return Store(*args, **kwargs)

    with _service(
        config_path, tmp_path, store_factory=store_factory
    ) as service:
        status = service.status()
        assert status.index["state"] == "incompatible"
        assert "no embedding metadata" in str(status.index["message"])

        for mode in ("keyword", "hybrid"):
            with pytest.raises(ApiError) as raised:
                service.search(SearchRequest("alpha", mode=mode))
            assert raised.value.status == 409
            assert raised.value.code == "incompatible_index"

        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert raised.value.status == 409
        assert raised.value.code == "incompatible_index"
        assert opened == []


def test_partial_initialized_index_without_primary_tables_is_error(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "index.db"
    _add_indexed_document(db_path)
    _drop_primary_tables(db_path)
    docs = _root(tmp_path, "docs-partial")
    config_path = _write_config(tmp_path, roots=[docs], db_path=db_path)
    opened: list[dict[str, object]] = []

    def store_factory(*args, **kwargs):
        opened.append(dict(kwargs))
        return Store(*args, **kwargs)

    with _service(
        config_path, tmp_path, store_factory=store_factory
    ) as service:
        status = service.status()
        assert status.index["state"] == "error"
        message = str(status.index["message"])
        assert "missing required table" in message
        assert "documents" in message
        assert "chunks" in message

        for mode in ("keyword", "hybrid"):
            with pytest.raises(ApiError) as raised:
                service.search(SearchRequest("alpha", mode=mode))
            assert raised.value.status == 503
            assert raised.value.code == "index_unavailable"

        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert raised.value.status == 503
        assert raised.value.code == "index_unavailable"
        assert opened == []


def test_empty_uninitialized_database_is_empty_and_index_allowed(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "blank.db"
    db_path.touch()
    docs = _root(tmp_path, "docs-empty")
    config_path = _write_config(tmp_path, roots=[docs], db_path=db_path)

    with _service(config_path, tmp_path) as service:
        status = service.status()
        assert status.index["state"] == "empty"

        job = _wait_job(service, service.start_index())
        assert job.state == "succeeded"
        assert service.status().index["state"] == "empty"


def test_start_index_rejects_quant_mismatch_without_opening_store(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "float32.db"
    _add_indexed_document(db_path, vector_quant="float32")
    docs = _root(tmp_path, "quant-docs")
    config_path = _write_config(tmp_path, roots=[docs], db_path=db_path)
    opened: list[dict[str, object]] = []
    built: list[str] = []

    def store_factory(*args, **kwargs):
        opened.append(dict(kwargs))
        return Store(*args, **kwargs)

    def embedder_factory(cfg: Config):
        built.append(cfg.embed_model)
        return FakeEmbedder(dim=64)

    with _service(
        config_path,
        tmp_path,
        store_factory=store_factory,
        embedder_factory=embedder_factory,
    ) as service:
        status = service.status()
        assert status.index["state"] == "ready"
        assert status.index["vector_quant"] == "float32"

        keyword = service.search(SearchRequest("alpha", mode="keyword"))
        hybrid = service.search(SearchRequest("alpha", mode="hybrid"))
        assert keyword.hits
        assert hybrid.hits
        built_after_search = list(built)
        opened.clear()

        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert raised.value.status == 409
        assert raised.value.code == "incompatible_index"
        assert opened == []
        assert built == built_after_search
        assert service.status().activity == {"lane": "idle", "job_id": None}


def test_index_preflight_order(tmp_path: Path) -> None:
    invalid_config = tmp_path / "config.toml"
    invalid_config.write_text("roots = [broken\n", encoding="utf-8")
    with _service(invalid_config, tmp_path) as service:
        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert _error_code(raised.value) == "config_invalid"

    rx = tmp_path / "rx"
    rx.mkdir()
    no_roots_config = _write_config(rx, roots=None)
    with _service(no_roots_config, rx, daemon_probe=lambda path: True) as service:
        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert _error_code(raised.value) == "no_roots"

    if LEGACY_DAEMON:  # the daemon only exists on Linux
        daemon_root = _root(tmp_path, "daemon-root")
        daemon_config = _write_config(tmp_path, roots=[daemon_root])
        with _service(
            daemon_config, tmp_path, daemon_probe=lambda path: True
        ) as service:
            with pytest.raises(ApiError) as raised:
                service.start_index()
            assert _error_code(raised.value) == "daemon_active"

    mismatch_root = _root(tmp_path, "mismatch-root")
    (mismatch_root / "a.txt").write_text("alpha\n", encoding="utf-8")
    mismatch_db = tmp_path / "mismatch-index.db"
    _add_indexed_document(mismatch_db, model="other/model")
    mismatch_config = _write_config(tmp_path, roots=[mismatch_root], db_path=mismatch_db)
    with _service(mismatch_config, tmp_path) as service:
        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert _error_code(raised.value) == "incompatible_index"

    error_root = _root(tmp_path, "error-root")
    error_db = tmp_path / "error-index.db"
    error_db.write_bytes(b"not sqlite")
    error_config = _write_config(tmp_path, roots=[error_root], db_path=error_db)
    with _service(error_config, tmp_path) as service:
        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert _error_code(raised.value) == "index_unavailable"

    model_root = _root(tmp_path, "model-root")
    model_config = _write_config(tmp_path, roots=[model_root])

    def failing(cfg: Config):
        raise RuntimeError("model missing")

    with _service(model_config, tmp_path, embedder_factory=failing) as service:
        assert _wait_job(service, service.start_model_check()).state == "failed"
        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert _error_code(raised.value) == "model_unavailable"


def test_job_retention_unknown_and_no_queued_state(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    jobs: list[str] = []
    with _service(config_path, tmp_path) as service:
        for _ in range(21):
            job_id = service.start_model_check()
            view = _wait_job(service, job_id)
            assert view.state in {"succeeded", "failed", "partial"}
            assert view.state != "queued"
            assert view.elapsed_ms >= 0
            jobs.append(job_id)

        with pytest.raises(ApiError) as raised:
            service.job(jobs[0])
        assert _error_code(raised.value) == "job_not_found"
        for job_id in jobs[-20:]:
            assert service.job(job_id).job_id == job_id
        with pytest.raises(ApiError) as raised:
            service.job("not-a-real-job")
        assert _error_code(raised.value) == "job_not_found"


def test_shutdown_waits_for_active_index_job_and_stops_new_work(tmp_path: Path) -> None:
    docs = _root(tmp_path)
    _write_docs(docs, {"a.txt": "alpha xylophone notes\n"})
    config_path = _write_config(tmp_path, roots=[docs])
    paused = _PausingIndexer()

    def factory(store, embedder, cfg):
        del store, embedder, cfg
        return paused

    service = WebService(
        config_path,
        embedder_factory=lambda cfg: FakeEmbedder(dim=64),
        indexer_factory=factory,
        daemon_socket_path=tmp_path / "seekr.sock",
        daemon_probe=lambda path: False,
    )
    try:
        job_id = service.start_index()
        assert paused.started.wait(5.0)

        shutdown_thread = threading.Thread(target=service.shutdown)
        shutdown_thread.start()
        time.sleep(0.1)
        assert shutdown_thread.is_alive()

        with pytest.raises(ApiError) as raised:
            service.search(SearchRequest("alpha", mode="keyword"))
        assert _error_code(raised.value) == "busy"
        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert _error_code(raised.value) == "busy"
        with pytest.raises(ApiError) as raised:
            service.add_root(str(_root(tmp_path, "extra")), service.roots().revision)
        assert _error_code(raised.value) == "busy"

        paused.release.set()
        shutdown_thread.join(10.0)
        assert not shutdown_thread.is_alive()
        assert service.job(job_id).state == "succeeded"
        assert not any(
            thread.name.startswith("seekr-web-") for thread in threading.enumerate()
        )
    finally:
        paused.release.set()
        service.shutdown()


def test_begin_shutdown_rejects_work_and_shutdown_joins_active_job(
    tmp_path: Path,
) -> None:
    docs = _root(tmp_path)
    _write_docs(docs, {"a.txt": "alpha xylophone notes\n"})
    config_path = _write_config(tmp_path, roots=[docs])
    paused = _PausingIndexer()

    def factory(store, embedder, cfg):
        del store, embedder, cfg
        return paused

    service = WebService(
        config_path,
        embedder_factory=lambda cfg: FakeEmbedder(dim=64),
        indexer_factory=factory,
        daemon_socket_path=tmp_path / "seekr.sock",
        daemon_probe=lambda path: False,
    )
    try:
        job_id = service.start_index()
        assert paused.started.wait(5.0)
        revision = service.roots().revision

        service.begin_shutdown()
        service.begin_shutdown()
        assert service.job(job_id).state == "running"

        with pytest.raises(ApiError) as raised:
            service.search(SearchRequest("alpha", mode="keyword"))
        assert _error_code(raised.value) == "busy"
        with pytest.raises(ApiError) as raised:
            service.start_model_check()
        assert _error_code(raised.value) == "busy"
        with pytest.raises(ApiError) as raised:
            service.start_index()
        assert _error_code(raised.value) == "busy"
        with pytest.raises(ApiError) as raised:
            service.add_root(str(_root(tmp_path, "extra")), revision)
        assert _error_code(raised.value) == "busy"

        shutdown_thread = threading.Thread(target=service.shutdown)
        shutdown_thread.start()
        time.sleep(0.1)
        assert shutdown_thread.is_alive()

        paused.release.set()
        shutdown_thread.join(10.0)
        assert not shutdown_thread.is_alive()
        assert service.job(job_id).state == "succeeded"
        assert not any(
            thread.name.startswith("seekr-web-") for thread in threading.enumerate()
        )
    finally:
        paused.release.set()
        service.shutdown()


@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid"])
@pytest.mark.parametrize("bad_week", ["5-3", "0", "abc", 123])
def test_invalid_week_is_400_invalid_request(
    tmp_path: Path, mode: str, bad_week: object
) -> None:
    config_path = _write_config(tmp_path)
    daemon_calls: list[object] = []

    def daemon(*args, **kwargs):
        daemon_calls.append((args, kwargs))
        raise AssertionError("daemon must not be called for invalid week")

    with (
        _service(
            config_path,
            tmp_path,
            daemon_probe=lambda _path: True,
            daemon_search=daemon,
        ) as service,
        pytest.raises(ApiError) as raised,
    ):
        service.search(SearchRequest("alpha", mode=mode, week=bad_week))  # type: ignore[arg-type]

    exc = raised.value
    assert exc.status == 400
    assert exc.code == "invalid_request"
    assert set(exc.fields or {}) == {"week"}
    assert daemon_calls == []


def test_module_and_week_filters_and_labels_in_web_service(tmp_path: Path) -> None:
    docs = _root(tmp_path)
    _write_docs(
        docs,
        {
            "machine-learning/week4/trees.md": "# Trees\n\ngradient boosted trees\n",
            "maths/week1/algebra.md": "# Algebra\n\ngradient vectors\n",
        },
    )
    config_path = _write_config(tmp_path, roots=[docs])
    embedder = FakeEmbedder(dim=64)

    with _service(config_path, tmp_path, embedder_factory=lambda _cfg: embedder) as service:
        job = _wait_job(service, service.start_index())
        assert job.state == "succeeded"

        for mode in ("keyword", "hybrid"):
            res = service.search(
                SearchRequest("gradient", mode=mode, module="Machine Learning", week="3-5")
            )
            assert len(res.hits) == 1
            assert res.hits[0].module == "machine-learning"
            assert res.hits[0].week == 4

        with pytest.raises(ApiError) as unknown_err:
            service.search(SearchRequest("gradient", mode="keyword", module="biology"))
        assert unknown_err.value.status == 400
        assert unknown_err.value.code == "invalid_request"
        assert "module" in (unknown_err.value.fields or {})

        with pytest.raises(ApiError) as ambig_err:
            service.search(SearchRequest("gradient", mode="hybrid", module="m"))
        assert ambig_err.value.status == 400
        assert ambig_err.value.code == "invalid_request"
        assert "module" in (ambig_err.value.fields or {})

    from seekr.client import DaemonError

    def bad_module_daemon(*args, **kwargs):
        raise DaemonError("unknown module 'biology'; known modules: machine-learning")

    with (
        _service(
            config_path,
            tmp_path,
            daemon_probe=lambda _path: True,
            daemon_search=bad_module_daemon,
        ) as service,
        pytest.raises(ApiError) as daemon_err,
    ):
        service.search(SearchRequest("gradient", mode="hybrid", module="biology"))
    assert daemon_err.value.status == 400
    assert daemon_err.value.code == "invalid_request"
    assert "module" in (daemon_err.value.fields or {})


def test_hit_dto_highlights(tmp_path: Path) -> None:
    docs = _root(tmp_path)
    _write_docs(
        docs,
        {"notes.md": "The model learned slowly. Learning rates matter.\n"},
    )
    config_path = _write_config(tmp_path, roots=[docs])
    embedder = FakeEmbedder(dim=64)

    with _service(config_path, tmp_path, embedder_factory=lambda _cfg: embedder) as service:
        job = _wait_job(service, service.start_index())
        assert job.state == "succeeded"

        res = service.search(SearchRequest("learning", mode="keyword"))
        assert len(res.hits) == 1
        hit = res.hits[0]
        assert hit.highlights == ((10, 17), (26, 34))
        assert hit.to_dict()["highlights"] == [[10, 17], [26, 34]]


def test_search_with_spread_returns_groups(tmp_path: Path) -> None:
    docs = _root(tmp_path)
    _write_docs(
        docs,
        {
            "machine-learning/week4/trees.md": "# Trees\n\nentropy split criteria in decision trees.\n",
            "machine-learning/week2/basics.md": "# Basics\n\nentropy information gain basics.\n",
            "machine-learning/overview.md": "# Overview\n\nentropy across the module.\n",
            "data-structures/week1/huffman.md": "# Huffman\n\nentropy coding and trees.\n",
            "misc.md": "entropy note outside modules.\n",
        },
    )
    config_path = _write_config(tmp_path, roots=[docs])
    embedder = FakeEmbedder(dim=64)
    opened_stores: list[dict[str, object]] = []

    def store_factory(*args, **kwargs):
        if kwargs.get("read_only"):
            opened_stores.append(dict(kwargs))
        return Store(*args, **kwargs)

    with _service(
        config_path,
        tmp_path,
        embedder_factory=lambda _cfg: embedder,
        store_factory=store_factory,
    ) as service:
        job = _wait_job(service, service.start_index())
        assert job.state == "succeeded"

        plain = service.search(SearchRequest("entropy", mode="keyword", k=1, spread=False))
        assert plain.spread is None
        assert "spread" not in plain.to_dict()
        assert len(opened_stores) == 1

        opened_stores.clear()
        for mode in ("keyword", "vector", "hybrid"):
            res = service.search(
                SearchRequest("entropy", mode=mode, k=1, per_file=1, spread=True)
            )
            assert len(res.hits) == 1
            assert res.spread is not None
            payload = res.to_dict()
            assert "spread" in payload
            if mode in ("keyword", "hybrid"):
                assert [(g["module"], g["week"], g["first"]) for g in payload["spread"]] == [
                    ("data-structures", 1, True),
                    ("machine-learning", 2, True),
                    ("machine-learning", 4, False),
                    ("machine-learning", None, False),
                    (None, None, False),
                ]
            assert all(g["chunk_ids"] for g in payload["spread"])

        assert len(opened_stores) == 1

        empty_query = service.search(SearchRequest("   ", mode="keyword", spread=True))
        assert empty_query.spread == ()
        assert empty_query.to_dict()["spread"] == []

        no_hits = service.search(SearchRequest("unmatchedtokenxyz", mode="keyword", spread=True))
        assert no_hits.hits == ()
        assert no_hits.spread == ()
        assert no_hits.to_dict()["spread"] == []

    if LEGACY_DAEMON:  # the daemon half only exists on Linux
        daemon_calls: list[dict[str, object]] = []
        daemon_main_hit = Hit(
            chunk_id=101,
            path="/course/ml/week4/a.md",
            locator="L1",
            loc_start=1.0,
            snippet="entropy",
            score=0.9,
            kind="markdown",
            module="ml",
            week=4,
        )
        daemon_spread_hits = [
            daemon_main_hit,
            Hit(
                chunk_id=102,
                path="/course/ml/week2/b.md",
                locator="L1",
                loc_start=1.0,
                snippet="entropy",
                score=0.8,
                kind="markdown",
                module="ml",
                week=2,
            ),
        ]

        def daemon_search(*args, **kwargs):
            daemon_calls.append({"query": args[0], **kwargs})
            return [daemon_main_hit] if len(daemon_calls) == 1 else daemon_spread_hits

        with _service(
            config_path,
            tmp_path,
            daemon_probe=lambda _path: True,
            daemon_search=daemon_search,
        ) as service:
            res = service.search(
                SearchRequest("entropy", mode="hybrid", k=1, per_file=1, spread=True)
            )
            assert len(daemon_calls) == 2
            assert daemon_calls[1]["k"] == 30
            assert daemon_calls[1]["per_file"] == 2
            assert res.spread == (
                {"module": "ml", "week": 2, "first": True, "chunk_ids": [102]},
                {"module": "ml", "week": 4, "first": False, "chunk_ids": [101]},
            )


def test_hit_dto_location_windows_on_wsl(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    hits = [
        Hit(
            chunk_id=1,
            path="/mnt/c/course/notes.md",
            locator="§A > B",
            loc_start=1.0,
            snippet="entropy notes",
            score=0.9,
            kind="markdown",
        ),
        Hit(
            chunk_id=2,
            path="/home/alice/code.py",
            locator="L10-20",
            loc_start=10.0,
            snippet="def entropy(): pass",
            score=0.8,
            kind="code",
        ),
        Hit(
            chunk_id=3,
            path="/mnt/c/course/plain.txt",
            locator="",
            loc_start=0.0,
            snippet="plain entropy",
            score=0.7,
            kind="text",
        ),
    ]
    mounts = [
        Mount(source="C:\\", mount_point="/mnt/c", fstype="9p"),
        Mount(source="/dev/sdd", mount_point="/", fstype="ext4"),
    ]

    with _service(
        config_path,
        tmp_path,
        daemon_probe=lambda _path: True,
        daemon_search=lambda *_args, **_kwargs: hits,
    ) as service:
        # The daemon hook only injects fixed WSL paths; the location mapping
        # under test is pure string work, so report a daemon on every host.
        service._daemon_status = lambda: (True, tmp_path / "seekr.sock")
        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=True),
            mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
        ):
            wsl_result = service.search(SearchRequest("entropy", mode="hybrid"))

        assert [(h.location, h.posix_location) for h in wsl_result.hits] == [
            (
                r"C:\course\notes.md (section A > B)",
                "/mnt/c/course/notes.md (section A > B)",
            ),
            (
                "/home/alice/code.py (lines 10-20)",
                "/home/alice/code.py (lines 10-20)",
            ),
            (
                r"C:\course\plain.txt",
                "/mnt/c/course/plain.txt",
            ),
        ]
        assert wsl_result.hits[0].to_dict()["location"] == r"C:\course\notes.md (section A > B)"
        assert wsl_result.hits[0].to_dict()["posix_location"] == "/mnt/c/course/notes.md (section A > B)"

        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=False),
            mock.patch(
                "seekr.platform_info.read_mounts",
                side_effect=AssertionError("read_mounts must not run off WSL"),
            ),
        ):
            linux_result = service.search(SearchRequest("entropy", mode="hybrid"))

        assert [(h.location, h.posix_location) for h in linux_result.hits] == [
            (
                "/mnt/c/course/notes.md (section A > B)",
                "/mnt/c/course/notes.md (section A > B)",
            ),
            (
                "/home/alice/code.py (lines 10-20)",
                "/home/alice/code.py (lines 10-20)",
            ),
            (
                "/mnt/c/course/plain.txt",
                "/mnt/c/course/plain.txt",
            ),
        ]


def _make_pdf(path: Path, pages: list[tuple[float, float, str]]) -> Path:
    import pymupdf

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    try:
        for width, height, text in pages:
            page = doc.new_page(width=width, height=height)
            page.insert_textbox(
                pymupdf.Rect(36, 36, max(72.0, width - 36), max(72.0, height - 36)),
                text,
            )
        doc.save(path)
    finally:
        doc.close()
    return path


def _chunk_id_for_path(db_path: Path, target_path: str, locator: str | None = None) -> int:
    connection = sqlite3.connect(db_path)
    try:
        if locator is None:
            row = connection.execute(
                "SELECT c.id FROM chunks AS c "
                "JOIN documents AS d ON d.id = c.doc_id "
                "WHERE d.path = ? ORDER BY c.id LIMIT 1",
                (target_path,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT c.id FROM chunks AS c "
                "JOIN documents AS d ON d.id = c.doc_id "
                "WHERE d.path = ? AND c.locator = ? ORDER BY c.id LIMIT 1",
                (target_path, locator),
            ).fetchone()
        assert row is not None, (target_path, locator)
        return int(row[0])
    finally:
        connection.close()


def test_course_returns_sorted_modules_without_model_and_handles_missing_empty(
    tmp_path: Path,
) -> None:
    def explode(cfg: Config):
        raise AssertionError("course() must never load a model")

    missing_cfg = _write_config(tmp_path, db_path=tmp_path / "missing.db")
    with _service(missing_cfg, tmp_path, embedder_factory=explode) as service:
        view = service.course()
        assert view == CourseView(modules=())
        assert view.to_dict() == {"modules": []}
        assert not (tmp_path / "missing.db").exists()

    empty_db = tmp_path / "empty.db"
    Store(empty_db, "fake", 64, vector_quant="int8").close()
    empty_cfg = _write_config(tmp_path, db_path=empty_db)
    with _service(empty_cfg, tmp_path, embedder_factory=explode) as service:
        assert service.course() == CourseView(modules=())

    docs = _root(tmp_path, "course-docs")
    _write_docs(
        docs,
        {
            "BetaMod/week3/b.md": "beta week 3 b\n",
            "BetaMod/week1/a.md": "beta week 1 a\n",
            "BetaMod/week3/c.md": "beta week 3 c\n",
            "BetaMod/overview.md": "beta overview without week\n",
            "alpha_mod/week2/d.md": "alpha week 2 d\n",
            "unassigned.md": "root level document without module\n",
        },
    )
    ready_db = tmp_path / "ready.db"
    ready_cfg = _write_config(tmp_path, roots=[docs], db_path=ready_db)
    with _service(ready_cfg, tmp_path) as service:
        assert _wait_job(service, service.start_index()).state == "succeeded"

    with _service(ready_cfg, tmp_path, embedder_factory=explode) as service:
        view = service.course()
        assert view == CourseView(
            modules=(
                CourseModuleDTO(
                    name="alpha_mod",
                    display="alpha mod",
                    weeks=(2,),
                    files=1,
                ),
                CourseModuleDTO(
                    name="BetaMod",
                    display="BetaMod",
                    weeks=(1, 3),
                    files=4,
                ),
            )
        )
        assert view.to_dict() == {
            "modules": [
                {"name": "alpha_mod", "display": "alpha mod", "weeks": [2], "files": 1},
                {"name": "BetaMod", "display": "BetaMod", "weeks": [1, 3], "files": 4},
            ]
        }

    mismatch_db = tmp_path / "mismatch.db"
    _add_indexed_document(mismatch_db, model="different/model")
    with (
        _service(_write_config(tmp_path, db_path=mismatch_db), tmp_path, embedder_factory=explode) as service,
        pytest.raises(ApiError) as raised,
    ):
        service.course()
    assert _error_code(raised.value) == "incompatible_index"

    corrupt_db = tmp_path / "corrupt.db"
    corrupt_db.write_bytes(b"not sqlite")
    with (
        _service(_write_config(tmp_path, db_path=corrupt_db), tmp_path, embedder_factory=explode) as service,
        pytest.raises(ApiError) as raised,
    ):
        service.course()
    assert _error_code(raised.value) == "index_unavailable"


def test_preview_and_preview_page_png_with_width_cap_and_close(
    tmp_path: Path,
) -> None:
    import pymupdf

    docs = _root(tmp_path, "preview-docs")
    page1_words = " ".join(f"alpha{i}" for i in range(30))
    page2_words = " ".join(f"beta{i}" for i in range(30))
    pdf_path = _make_pdf(
        docs / "slides.pdf",
        [
            (400, 300, page1_words),
            (2000, 600, page2_words),
        ],
    )
    _write_docs(docs, {"notes.md": "# Heading\n\nFull markdown chunk text.\n"})
    db_path = tmp_path / "preview.db"
    config_path = _write_config(tmp_path, roots=[docs], db_path=db_path)

    with _service(config_path, tmp_path) as service:
        assert _wait_job(service, service.start_index()).state == "succeeded"

        pdf_p1_id = _chunk_id_for_path(db_path, str(pdf_path), "p.1")
        pdf_p2_id = _chunk_id_for_path(db_path, str(pdf_path), "p.2")
        md_id = _chunk_id_for_path(db_path, str(docs / "notes.md"))

        pdf_preview = service.preview(pdf_p2_id)
        assert isinstance(pdf_preview, PreviewView)
        assert pdf_preview.chunk == pdf_p2_id
        assert pdf_preview.kind == "pdf"
        assert pdf_preview.path == str(pdf_path)
        assert pdf_preview.locator == "p.2"
        assert pdf_preview.location == f"{pdf_path} (page 2)"
        assert "beta0" in pdf_preview.text
        assert pdf_preview.page_image is True
        assert set(pdf_preview.to_dict()) == {
            "chunk",
            "kind",
            "path",
            "locator",
            "location",
            "text",
            "page_image",
        }

        md_preview = service.preview(md_id)
        assert md_preview.kind == "markdown"
        assert md_preview.page_image is False
        assert "Full markdown chunk text." in md_preview.text

        # Normal page width (400pt) zooms by 1.5 -> 600px wide.
        png1 = service.preview_page_png(pdf_p1_id)
        assert png1.startswith(b"\x89PNG\r\n\x1a\n")
        pix1 = pymupdf.Pixmap(png1)
        assert pix1.width == 600

        # Wide page (2000pt) scales down so rendered PNG width is <= 2000px.
        png2 = service.preview_page_png(pdf_p2_id)
        assert png2.startswith(b"\x89PNG\r\n\x1a\n")
        pix2 = pymupdf.Pixmap(png2)
        assert pix2.width <= 2000

        # Verify PDF document is closed cleanly.
        real_open = pymupdf.open
        closed_flags: list[bool] = []

        def tracked_open(*args, **kwargs):
            doc = real_open(*args, **kwargs)
            orig_close = doc.close

            def wrapped_close():
                closed_flags.append(True)
                return orig_close()

            doc.close = wrapped_close  # type: ignore[method-assign]
            return doc

        with mock.patch("pymupdf.open", side_effect=tracked_open):
            service.preview_page_png(pdf_p1_id)
        assert closed_flags == [True]

        # Non-PDF chunk and unknown chunk -> 404 not_found.
        for bad_id in (md_id, 999999, 2**63):
            with pytest.raises(ApiError) as raised:
                service.preview_page_png(bad_id)
            assert _error_code(raised.value) == "not_found"

        for unknown_id in (999999, 2**63):
            with pytest.raises(ApiError) as raised:
                service.preview(unknown_id)
            assert _error_code(raised.value) == "not_found"

        # Deleted PDF file -> preview_page_png returns 404 not_found.
        pdf_path.unlink()
        with pytest.raises(ApiError) as raised:
            service.preview_page_png(pdf_p1_id)
        assert _error_code(raised.value) == "not_found"


def test_preview_page_png_out_of_range_page_is_404(tmp_path: Path) -> None:
    docs = _root(tmp_path, "oob-docs")
    pdf_path = _make_pdf(docs / "onepage.pdf", [(300, 400, "only one page")])
    db_path = tmp_path / "oob.db"
    embedder = FakeEmbedder(dim=64)
    store = Store(db_path, "fake", 64, vector_quant="int8")
    try:
        doc_id = store.upsert_document(str(pdf_path), 1, 10, "a" * 64, "pdf")
        store.replace_chunks(
            doc_id,
            [
                StoredChunk(
                    spec=ChunkSpec(text="bad page", locator="p.5", loc_start=5.0),
                    text_sha=text_sha("bad page"),
                    vector=embedder.embed_documents(["bad page"])[0],
                )
            ],
        )
    finally:
        store.close()
    chunk_id = _chunk_id_for_path(db_path, str(pdf_path), "p.5")
    config_path = _write_config(tmp_path, roots=[docs], db_path=db_path)

    with (
        _service(config_path, tmp_path) as service,
        pytest.raises(ApiError) as raised,
    ):
        service.preview_page_png(chunk_id)
    assert _error_code(raised.value) == "not_found"


def test_open_chunk_runs_opener_without_shell_and_maps_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import functools
    import types

    from seekr import opener

    # These are the Linux and WSL opener rules (Windows and macOS hosts would
    # otherwise use their native opener).
    monkeypatch.setattr(
        opener, "open_target", functools.partial(opener.open_target, platform="linux")
    )

    docs = _root(tmp_path, "open-docs")
    pdf_path = _make_pdf(
        docs / "handout.pdf",
        [(300, 400, " ".join(f"word{i}" for i in range(30)))],
    )
    _write_docs(docs, {"notes.md": "# Title\n\nsome body text\n"})
    db_path = tmp_path / "open.db"
    config_path = _write_config(tmp_path, roots=[docs], db_path=db_path)

    with _service(config_path, tmp_path) as service:
        assert _wait_job(service, service.start_index()).state == "succeeded"
        pdf_chunk = _chunk_id_for_path(db_path, str(pdf_path), "p.1")

        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=False),
            mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "zathura" else None),
            mock.patch(
                "subprocess.run",
                return_value=types.SimpleNamespace(returncode=0),
            ) as run_mock,
        ):
            service.open_chunk(pdf_chunk)
        run_mock.assert_called_once_with(
            ["zathura", "-P", "1", str(pdf_path)], check=False
        )

        mounts = [Mount(source="C:\\", mount_point="/mnt/c", fstype="9p")]
        store = Store(db_path, "fake", 64, vector_quant="int8")
        try:
            doc_id = store.upsert_document("/mnt/c/course/wsl.pdf", 1, 10, "b" * 64, "pdf")
            store.replace_chunks(
                doc_id,
                [
                    StoredChunk(
                        spec=ChunkSpec(text="wsl pdf", locator="p.3", loc_start=3.0),
                        text_sha=text_sha("wsl pdf"),
                        vector=FakeEmbedder(dim=64).embed_documents(["wsl pdf"])[0],
                    )
                ],
            )
        finally:
            store.close()
        wsl_chunk = _chunk_id_for_path(db_path, "/mnt/c/course/wsl.pdf", "p.3")

        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=True),
            mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
            mock.patch("shutil.which", return_value=None),
            mock.patch(
                "subprocess.run",
                return_value=types.SimpleNamespace(returncode=1),
            ) as run_mock,
        ):
            service.open_chunk(wsl_chunk)
        run_mock.assert_called_once_with(
            ["explorer.exe", r"C:\course\wsl.pdf"], check=False
        )

        # Non-success exit code -> 500 internal_error
        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=True),
            mock.patch("seekr.platform_info.read_mounts", return_value=mounts),
            mock.patch("shutil.which", return_value=None),
            mock.patch(
                "subprocess.run",
                return_value=types.SimpleNamespace(returncode=2),
            ),
            pytest.raises(ApiError) as raised,
        ):
            service.open_chunk(wsl_chunk)
        assert _error_code(raised.value) == "internal_error"

        # Missing executable -> 500 internal_error
        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=False),
            mock.patch("shutil.which", return_value=None),
            mock.patch("subprocess.run", side_effect=FileNotFoundError("xdg-open")),
            pytest.raises(ApiError) as raised,
        ):
            service.open_chunk(pdf_chunk)
        assert _error_code(raised.value) == "internal_error"

        # Unconvertible WSL path -> 500 internal_error
        with (
            mock.patch("seekr.platform_info.is_wsl", return_value=True),
            mock.patch("seekr.platform_info.read_mounts", return_value=[]),
            mock.patch("seekr.platform_info.wslpath_windows", return_value=None),
            mock.patch("shutil.which", return_value=None),
            pytest.raises(ApiError) as raised,
        ):
            service.open_chunk(pdf_chunk)
        assert _error_code(raised.value) == "internal_error"

        # Unknown chunk -> 404 not_found
        with pytest.raises(ApiError) as raised:
            service.open_chunk(999999)
        assert _error_code(raised.value) == "not_found"


def test_remove_root_purges_config_and_db_respects_prefix_lane_and_daemon(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, "course")
    root2 = _root(tmp_path, "course2")
    _write_docs(root, {"a.txt": "alpha uniquewordrootone\n"})
    _write_docs(root2, {"b.txt": "beta uniquewordroottwo\n"})
    db_path = tmp_path / "remove.db"
    config_path = _write_config(tmp_path, roots=[root, root2], db_path=db_path)

    if LEGACY_DAEMON:  # the daemon only exists on Linux
        with _service(config_path, tmp_path, daemon_probe=lambda _p: True) as daemon_service:
            rev = daemon_service.roots().revision
            with pytest.raises(ApiError) as raised:
                daemon_service.remove_root(str(root), rev)
            assert _error_code(raised.value) == "daemon_active"

    paused = _PausingIndexer()
    with _service(
        config_path, tmp_path, indexer_factory=lambda s, e, c: paused
    ) as busy_service:
        job_id = busy_service.start_index()
        assert paused.started.wait(5.0)
        rev = busy_service.roots().revision
        with pytest.raises(ApiError) as raised:
            busy_service.remove_root(str(root), rev)
        assert _error_code(raised.value) == "busy"
        paused.release.set()
        _wait_job(busy_service, job_id)

    with _service(config_path, tmp_path) as service:
        assert _wait_job(service, service.start_index()).state == "succeeded"
        rev = service.roots().revision

        with pytest.raises(ApiError) as raised:
            service.remove_root(str(root), "0" * 64)
        assert _error_code(raised.value) == "stale_config"

        with pytest.raises(ApiError) as raised:
            service.remove_root(str(tmp_path / "not-configured"), rev)
        assert _error_code(raised.value) == "invalid_root"

        for bad_path in ("bad\x00path", "\ud800", "", "   "):
            with pytest.raises(ApiError) as raised:
                service.remove_root(bad_path, rev)
            assert raised.value.status == 400
            assert _error_code(raised.value) == "invalid_root"
            assert load_config(config_path).roots == (root, root2)
            assert service.search(SearchRequest("uniquewordrootone", mode="keyword")).hits

        updated = service.remove_root(str(root), rev)
        assert updated.roots == (str(root2),)
        assert updated.revision != rev
        assert load_config(config_path).roots == (root2,)

        # root's docs are purged, root2's docs remain searchable.
        assert service.search(SearchRequest("uniquewordrootone", mode="keyword")).hits == ()
        remaining = service.search(SearchRequest("uniquewordroottwo", mode="keyword"))
        assert [hit.path for hit in remaining.hits] == [str(root2 / "b.txt")]


def test_start_download_job_scrubs_offline_env_keeps_last_50_lines_and_handles_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    config_path = _write_config(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    captured_calls: list[dict[str, object]] = []
    many_lines = "\n".join(f"line-{i}" for i in range(65)) + "\n"

    def fake_run(cmd, **kwargs):
        captured_calls.append({"cmd": list(cmd), "env": dict(kwargs.get("env") or {})})
        return types.SimpleNamespace(returncode=0, stdout=many_lines, stderr="")

    with (
        mock.patch("subprocess.run", side_effect=fake_run),
        _service(config_path, tmp_path) as service,
    ):
        job_id = service.start_download(rerank=False)
        view = _wait_job(service, job_id)
        assert view.operation == "download"
        assert view.state == "succeeded"
        assert view.error is None
        assert view.report is not None
        assert len(view.report["lines"]) == 50  # type: ignore[arg-type]
        assert view.report["lines"][0] == "line-15"  # type: ignore[index]
        assert view.report["lines"][-1] == "line-64"  # type: ignore[index]
        assert captured_calls[0]["cmd"] == [
            sys.executable,
            "-m",
            "seekr",
            "models",
            "pull",
            "--yes",
            "--grammars",
        ]
        child_env = captured_calls[0]["env"]
        assert isinstance(child_env, dict)
        assert child_env.get("SEEKR_CONFIG") == str(config_path)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            assert key not in child_env

        # With rerank=True
        rerank_job_id = service.start_download(rerank=True)
        rerank_view = _wait_job(service, rerank_job_id)
        assert rerank_view.state == "succeeded"
        assert captured_calls[1]["cmd"] == [
            sys.executable,
            "-m",
            "seekr",
            "models",
            "pull",
            "--yes",
            "--grammars",
            "--rerank",
        ]

    # Non-zero exit code -> failed job, and busy lane during download
    entered = threading.Event()
    release = threading.Event()

    def failing_run(cmd, **kwargs):
        entered.set()
        assert release.wait(5.0)
        return types.SimpleNamespace(returncode=1, stdout="", stderr="network unreachable\n")

    with (
        mock.patch("subprocess.run", side_effect=failing_run),
        _service(config_path, tmp_path) as service,
    ):
        job_id = service.start_download()
        assert entered.wait(5.0)
        assert service.status().activity == {"lane": "download", "job_id": job_id}
        with pytest.raises(ApiError) as raised:
            service.start_download()
        assert _error_code(raised.value) == "busy"
        release.set()
        failed = _wait_job(service, job_id)
        assert failed.state == "failed"
        assert "network unreachable" in (failed.error or "")
        assert service.status().activity == {"lane": "idle", "job_id": None}


def test_start_download_success_resets_unavailable_model_state(tmp_path: Path) -> None:
    import types

    config_path = _write_config(tmp_path)

    def failing_embedder(cfg: Config):
        raise RuntimeError("model missing")

    with _service(config_path, tmp_path, embedder_factory=failing_embedder) as service:
        check_view = _wait_job(service, service.start_model_check())
        assert check_view.state == "failed"
        assert service.status().model["state"] == "unavailable"

        with mock.patch(
            "subprocess.run",
            return_value=types.SimpleNamespace(returncode=0, stdout="Done.\n", stderr=""),
        ):
            dl_view = _wait_job(service, service.start_download())

        assert dl_view.state == "succeeded"
        model_status = service.status().model
        assert model_status["state"] == "unchecked"
        assert model_status["message"] is None


def test_preview_page_png_locator_variants_and_encrypted_pdf(tmp_path: Path) -> None:
    import pymupdf

    docs = _root(tmp_path, "loc-docs")
    pdf_path = _make_pdf(
        docs / "multi.pdf",
        [(300, 400, "page one text"), (400, 400, "page two text")],
    )
    db_path = tmp_path / "loc.db"
    embedder = FakeEmbedder(dim=64)
    store = Store(db_path, "fake", 64, vector_quant="int8")
    try:
        doc_id = store.upsert_document(str(pdf_path), 1, 10, "a" * 64, "pdf")
        store.replace_chunks(
            doc_id,
            [
                StoredChunk(
                    spec=ChunkSpec(text="empty loc", locator="", loc_start=1.0),
                    text_sha=text_sha("empty loc"),
                    vector=embedder.embed_documents(["empty loc"])[0],
                ),
                StoredChunk(
                    spec=ChunkSpec(text="range loc", locator="p.1-2", loc_start=1.0),
                    text_sha=text_sha("range loc"),
                    vector=embedder.embed_documents(["range loc"])[0],
                ),
                StoredChunk(
                    spec=ChunkSpec(text="zero loc", locator="p.0", loc_start=0.0),
                    text_sha=text_sha("zero loc"),
                    vector=embedder.embed_documents(["zero loc"])[0],
                ),
                StoredChunk(
                    spec=ChunkSpec(text="line loc", locator="L1", loc_start=1.0),
                    text_sha=text_sha("line loc"),
                    vector=embedder.embed_documents(["line loc"])[0],
                ),
            ],
        )
    finally:
        store.close()

    empty_id = _chunk_id_for_path(db_path, str(pdf_path), "")
    range_id = _chunk_id_for_path(db_path, str(pdf_path), "p.1-2")
    zero_id = _chunk_id_for_path(db_path, str(pdf_path), "p.0")
    line_id = _chunk_id_for_path(db_path, str(pdf_path), "L1")
    config_path = _write_config(tmp_path, roots=[docs], db_path=db_path)

    with _service(config_path, tmp_path) as service:
        for ok_id in (empty_id, range_id):
            png = service.preview_page_png(ok_id)
            assert png.startswith(b"\x89PNG\r\n\x1a\n")
            assert pymupdf.Pixmap(png).width == 450

        for bad_id in (zero_id, line_id):
            with pytest.raises(ApiError) as raised:
                service.preview_page_png(bad_id)
            assert _error_code(raised.value) == "not_found"

        closed: list[bool] = []

        class _EncryptedDoc:
            needs_pass = True

            def close(self) -> None:
                closed.append(True)

        with (
            mock.patch("pymupdf.open", return_value=_EncryptedDoc()),
            pytest.raises(ApiError) as raised,
        ):
            service.preview_page_png(empty_id)
        assert _error_code(raised.value) == "not_found"
        assert closed == [True]


# -- service-client surface: rerank/explain, similar, shutdown, path indexing --


def test_search_passes_rerank_and_explain_and_maps_explain(tmp_path: Path) -> None:
    root = _root(tmp_path)
    config_path = _write_config(tmp_path, roots=[root])
    _add_indexed_document(tmp_path / "index.db")
    seen: list[dict[str, object]] = []
    explained = Hit(
        chunk_id=1,
        path="/notes/a.txt",
        locator="L1",
        loc_start=1.0,
        snippet="alpha",
        score=0.5,
        kind="text",
        explain=HitExplain(keyword_rank=1, keyword_bm25=-2.5, rrf_score=0.03),
    )

    def local(query, **kwargs):
        seen.append(kwargs)
        return [explained]

    with _service(config_path, tmp_path, local_search=local) as service:
        plain = service.search(SearchRequest("alpha", mode="hybrid"))
        assert seen[-1]["rerank"] is False
        assert seen[-1]["explain"] is False
        assert "explain" not in plain.to_dict()["hits"][0]

        rich = service.search(
            SearchRequest("alpha", mode="hybrid", rerank=True, explain=True)
        )
        assert seen[-1]["rerank"] is True
        assert seen[-1]["explain"] is True
        payload = rich.to_dict()["hits"][0]["explain"]
        assert payload["keyword_rank"] == 1
        assert payload["keyword_bm25"] == -2.5
        assert payload["rrf_score"] == 0.03
        assert payload["vector_rank"] is None
        json.dumps(payload, allow_nan=False)


@legacy_daemon_only
def test_daemon_search_passes_rerank_and_explain(tmp_path: Path) -> None:
    root = _root(tmp_path)
    config_path = _write_config(tmp_path, roots=[root])
    seen: list[dict[str, object]] = []

    def daemon(query, **kwargs):
        seen.append(kwargs)
        return []

    with _service(
        config_path,
        tmp_path,
        daemon_probe=lambda path: True,
        daemon_search=daemon,
    ) as service:
        service.search(SearchRequest("alpha", rerank=True, explain=True))
        assert seen[-1]["rerank"] is True
        assert seen[-1]["explain"] is True


def test_search_rejects_non_boolean_rerank_and_explain(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, roots=[_root(tmp_path)])
    with _service(config_path, tmp_path) as service:
        for field in ("rerank", "explain"):
            with pytest.raises(ApiError) as raised:
                service.search(SearchRequest("alpha", **{field: "yes"}))
            assert _error_code(raised.value) == "invalid_request"


def test_similar_calls_shared_similarity_with_filters(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, roots=[_root(tmp_path)])
    _add_indexed_document(tmp_path / "index.db")
    seen: list[tuple[object, dict[str, object]]] = []
    notes = str(tmp_path / "notes")  # absolute on every host
    target_path = str(tmp_path / "notes" / "a.txt")
    hit = Hit(
        chunk_id=7,
        path="/notes/b.txt",
        locator="L2",
        loc_start=2.0,
        snippet="beta",
        score=0.8,
        kind="text",
    )

    def fake_similar(target, **kwargs):
        seen.append((target, kwargs))
        return [hit]

    with _service(config_path, tmp_path) as service:
        with mock.patch("seekr.cli_search.get_similar_hits", fake_similar):
            result = service.similar(
                SimilarRequest(
                    path=target_path,
                    k=3,
                    per_file=2,
                    path_prefix=notes,
                    kind="text",
                    since="7d",
                    module="ml",
                    week="1-2",
                )
            )
        assert [h.chunk_id for h in result.hits] == [7]
        assert result.took_ms >= 0.0
        target, kwargs = seen[0]
        assert str(target) == target_path
        assert kwargs["k"] == 3
        assert kwargs["per_file"] == 2
        assert kwargs["path_prefix"] == notes
        assert kwargs["kind"] == "text"
        assert kwargs["since"] == "7d"
        assert kwargs["module"] == "ml"
        assert kwargs["week"] == "1-2"
        assert service.status().activity["lane"] == "idle"


def test_similar_validation_and_error_mapping(tmp_path: Path) -> None:
    from seekr.cli_search import SearchError

    config_path = _write_config(tmp_path, roots=[_root(tmp_path)])
    with _service(config_path, tmp_path) as service:
        for bad in (
            SimilarRequest(path=""),
            SimilarRequest(path=str(tmp_path / "a"), k=0),
            SimilarRequest(path=str(tmp_path / "a"), k=51),
            SimilarRequest(path=str(tmp_path / "a"), per_file=11),
            SimilarRequest(path=str(tmp_path / "a"), kind="nonsense-kind"),
            SimilarRequest(path=str(tmp_path / "a"), since="not-a-date"),
        ):
            with pytest.raises(ApiError) as raised:
                service.similar(bad)
            assert _error_code(raised.value) == "invalid_request", bad

        def failing(target, **kwargs):
            raise SearchError("file not found and not in index")

        with (
            mock.patch("seekr.cli_search.get_similar_hits", failing),
            pytest.raises(ApiError) as raised,
        ):
            service.similar(SimilarRequest(path=str(tmp_path / "missing")))
        assert _error_code(raised.value) == "index_unavailable"

        service.begin_shutdown()
        with pytest.raises(ApiError):
            service.similar(SimilarRequest(path=str(tmp_path / "a")))


def test_similar_runs_during_index_but_not_during_download(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, roots=[_root(tmp_path)])
    with (
        _service(config_path, tmp_path) as service,
        mock.patch("seekr.cli_search.get_similar_hits", lambda *a, **k: []),
    ):
        service._begin_lane("index")
        try:
            assert service.similar(SimilarRequest(path=str(tmp_path / "a"))).hits == ()
        finally:
            service._end_lane()
        service._begin_lane("download")
        try:
            with pytest.raises(ApiError) as raised:
                service.similar(SimilarRequest(path=str(tmp_path / "a")))
            assert _error_code(raised.value) == "busy"
        finally:
            service._end_lane()


def test_search_is_refused_during_model_download(tmp_path: Path) -> None:
    docs = _root(tmp_path)
    _write_docs(docs, {"a.txt": "alpha xylophone notes\n"})
    config_path = _write_config(tmp_path, roots=[docs])
    with _service(config_path, tmp_path) as service:
        assert _wait_job(service, service.start_index()).state == "succeeded"
        service._begin_lane("download")
        try:
            with pytest.raises(ApiError) as raised:
                service.search(SearchRequest("xylophone", mode="hybrid"))
            assert _error_code(raised.value) == "busy"
            assert service.search(SearchRequest("xylophone", mode="keyword")).hits
        finally:
            service._end_lane()


def test_status_is_refused_once_shutdown_begins(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, roots=[_root(tmp_path)])
    with _service(config_path, tmp_path) as service:
        service.begin_shutdown()
        with pytest.raises(ApiError) as raised:
            service.status()
        assert _error_code(raised.value) == "busy"


def test_request_shutdown_invokes_callback_once_and_is_noop_without_one(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path, roots=[_root(tmp_path)])
    calls: list[str] = []
    with _service(config_path, tmp_path, on_shutdown=lambda: calls.append("stop")) as service:
        service.request_shutdown()
        service.request_shutdown()
        assert calls == ["stop"]
    with _service(config_path, tmp_path) as service:
        service.request_shutdown()


def test_start_index_paths_indexes_only_given_paths(tmp_path: Path) -> None:
    root = _root(tmp_path)
    other = _root(tmp_path, "other")
    _write_docs(root, {"keep.md": "# Keep\n\nzebra notes here.\n"})
    _write_docs(other, {"extra.md": "# Extra\n\nokapi notes here.\n"})
    config_path = _write_config(tmp_path, roots=[root])
    embedder = FakeEmbedder(dim=64)
    with _service(config_path, tmp_path, embedder_factory=lambda _cfg: embedder) as service:
        job = _wait_job(service, service.start_index_paths([str(other)]))
        assert job.operation == "index"
        assert job.state == "succeeded", job
        found = service.search(SearchRequest("okapi", mode="keyword"))
        assert [Path(h.path).name for h in found.hits] == ["extra.md"]
        assert service.search(SearchRequest("zebra", mode="keyword")).hits == ()


def test_start_index_paths_validation_and_daemon_refusal(tmp_path: Path) -> None:
    root = _root(tmp_path)
    config_path = _write_config(tmp_path, roots=[root])
    with _service(config_path, tmp_path) as service:
        for bad in ([], [""], [str(tmp_path / "does-not-exist")]):
            with pytest.raises(ApiError) as raised:
                service.start_index_paths(bad)
            assert _error_code(raised.value) == "invalid_request", bad
    if LEGACY_DAEMON:  # the daemon only exists on Linux
        with _service(config_path, tmp_path, daemon_probe=lambda path: True) as service:
            with pytest.raises(ApiError) as raised:
                service.start_index_paths([str(root)])
            assert _error_code(raised.value) == "daemon_active"


def test_keyword_rerank_and_relative_paths_are_invalid_requests(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, roots=[_root(tmp_path)])
    with _service(config_path, tmp_path) as service:
        for call in (
            lambda: service.search(SearchRequest("alpha", mode="keyword", rerank=True)),
            lambda: service.similar(SimilarRequest(path="rel/a.md")),
            lambda: service.start_index_paths(["rel"]),
        ):
            with pytest.raises(ApiError) as raised:
                call()
            assert _error_code(raised.value) == "invalid_request"


def test_removed_root_prefixes_match_case_and_unicode_variants_on_darwin() -> None:
    configured = [Path("/vol/MO\u0301DULO")]
    with mock.patch("seekr.pathnorm._PLATFORM", "darwin"):
        prefixes = WebService._removed_root_prefixes("/vol/m\u00f3dulo", configured)
    # The resolved and as-given spellings (which differ on Windows, where
    # resolving adds a drive), then the configured variant spelling.
    removed = Path("/vol/m\u00f3dulo")
    expected = [str(removed.resolve()), str(removed), str(configured[0])]
    assert prefixes == list(dict.fromkeys(expected))
    with mock.patch("seekr.pathnorm._PLATFORM", "linux"):
        prefixes = WebService._removed_root_prefixes("/vol/m\u00f3dulo", configured)
    assert prefixes == list(dict.fromkeys(expected[:2]))
