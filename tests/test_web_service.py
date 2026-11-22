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

from seekr.cli_search import get_hits
from seekr.config import Config, load_config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer, SyncReport
from seekr.lifecycle import LifecycleLock
from seekr.models import ChunkSpec, Hit, text_sha
from seekr.store import SCHEMA_VERSION, Store, StoredChunk
from seekr.web_contract import ApiError, HitDTO, SearchRequest
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
        with pytest.raises(ApiError) as raised:
            service.search(SearchRequest("xylophone", mode="hybrid"))
        assert _error_code(raised.value) == "busy"

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
        assert added.daemon_restart_required is True
        assert added.warning


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
