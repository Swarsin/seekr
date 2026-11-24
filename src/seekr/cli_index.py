"""Index-related CLI commands, registered into the main Typer app.

This module creates the ``index``, ``reindex`` and ``status`` commands without
editing ``seekr.cli``; the integrated CLI calls :func:`register` once::

    from seekr.cli_index import register
    register(app)

A ``reindex`` with per-file indexing errors or one that would replace a
non-empty live index with an empty rebuild is refused by default.  ``--force``
overrides only the error guard; ``--allow-empty`` overrides only the
empty-rebuild guard.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import typer

from . import platform_info
from .config import Config, load_config
from .course import display_module
from .embed import Embedder, build_embedder
from .indexer import Indexer, SyncReport, _error_text
from .lifecycle import LifecycleError, lifecycle_lock, rebuild_index
from .service_client import ServiceClient, ServiceError, ServiceUnavailable
from .store import Store

__all__ = ["register"]


def _load_config() -> Config:
    return load_config()


def _build_embedder(cfg: Config) -> Embedder:
    return build_embedder(cfg)


_USE_CONFIG = object()


def _open_store(
    cfg: Config,
    dim: int,
    *,
    vector_quant: object = _USE_CONFIG,
    record_backend: bool = True,
) -> Store:
    quant = cfg.vector_quant if vector_quant is _USE_CONFIG else vector_quant
    backend = cfg.embed_backend if record_backend else None
    if quant is None:
        return Store(cfg.db_path, cfg.embed_model, dim, embed_backend=backend)
    return Store(
        cfg.db_path, cfg.embed_model, dim, vector_quant=str(quant), embed_backend=backend
    )


def _stored_dim(cfg: Config) -> int | None:
    """Read the vector dimension recorded in an existing database."""
    db_path = Path(cfg.db_path)
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    try:
        value = int(row[0])
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _fatal(message: str) -> None:
    typer.echo(f"seekr: {message}", err=True)
    raise typer.Exit(code=1)


_JOB_POLL_SECONDS = 0.5


def _echo_report(
    counts: Sequence[tuple[str, object]],
    errors: Sequence[tuple[object, object]],
    warnings: Sequence[tuple[object, object]],
    error_total: int,
    warning_total: int,
) -> None:
    for name, value in counts:
        typer.echo(f"{name}: {value}")
    typer.echo(f"errors: {error_total}")
    typer.echo(f"warnings: {warning_total}")
    for label, items, total in (
        ("error", errors, error_total),
        ("warning", warnings, warning_total),
    ):
        for path, message in items:
            typer.echo(f"{label}: {path}: {message}")
        if total > len(items):
            typer.echo(f"{label}: ... {total - len(items)} more not shown")


_COUNT_NAMES = ("added", "updated", "unchanged", "removed", "skipped")


def _print_report(report: SyncReport) -> None:
    _echo_report(
        [(name, getattr(report, name)) for name in _COUNT_NAMES],
        report.errors,
        report.warnings,
        len(report.errors),
        len(report.warnings),
    )


def _print_report_dict(report: dict[str, Any]) -> None:
    """Print a service index-job report exactly like :func:`_print_report`."""

    def diagnostics(key: str) -> list[tuple[object, object]]:
        items = report.get(key) or []
        return [(item.get("path"), item.get("message")) for item in items]

    errors = diagnostics("errors")
    warnings = diagnostics("warnings")
    _echo_report(
        [(name, report.get(name, 0)) for name in _COUNT_NAMES],
        errors,
        warnings,
        int(report.get("error_total", len(errors))),
        int(report.get("warning_total", len(warnings))),
    )


def _split_index_paths(paths: list[Path]) -> tuple[list[str], list[tuple[str, str]]]:
    """Resolve ``paths`` and split off the ones the service would refuse.

    Unusable paths get the same per-path error text a local ``Indexer.sync``
    reports, so a delegated run prints the same report.
    """
    usable: list[str] = []
    errors: list[tuple[str, str]] = []
    for raw in paths:
        resolved, issue = Indexer._resolve(Path(raw))
        if resolved is None:
            errors.append((str(raw), issue or "cannot resolve root"))
            continue
        try:
            resolved.stat()
        except FileNotFoundError as exc:
            errors.append((str(resolved), f"root is missing or offline: {_error_text(exc)}"))
            continue
        except OSError as exc:
            errors.append((str(resolved), f"cannot stat root: {_error_text(exc)}"))
            continue
        usable.append(str(resolved))
    return usable, errors


def _with_path_errors(
    report: dict[str, Any], errors: list[tuple[str, str]]
) -> dict[str, Any]:
    if not errors:
        return report
    merged = dict(report)
    merged["errors"] = [
        {"path": path, "message": message} for path, message in errors
    ] + list(report.get("errors") or [])
    merged["error_total"] = int(report.get("error_total", 0)) + len(errors)
    return merged


def _wait_for_job(client: ServiceClient, job_id: str, activity: str) -> dict[str, Any]:
    """Poll a service job until it ends; return its report or raise."""
    while True:
        time.sleep(_JOB_POLL_SECONDS)
        try:
            job = client.job(job_id)
        except ServiceUnavailable as exc:
            raise ServiceError(
                f"lost contact with the seekr service while {activity}: {exc}"
            ) from exc
        state = job.get("state")
        if state == "running":
            continue
        report = job.get("report")
        if state in ("succeeded", "partial") and isinstance(report, dict):
            return report
        raise ServiceError(str(job.get("error") or f"index job ended in state {state!r}"))


def _index_via_service(client: ServiceClient, paths: list[Path] | None) -> dict[str, Any]:
    """Run an index job in the service and return its report.

    :class:`ServiceUnavailable` before the request is sent means "index
    locally"; once the service may have started the job, any failure is an
    error so two writers never run.
    """
    path_errors: list[tuple[str, str]] = []
    if paths:
        usable, path_errors = _split_index_paths(paths)
        if not usable:
            return _with_path_errors({name: 0 for name in _COUNT_NAMES}, path_errors)
        job_id = client.index_paths(usable)
    else:
        job_id = client.index_roots()
    return _with_path_errors(_wait_for_job(client, job_id, "indexing"), path_errors)


def _reindex_via_service(
    client: ServiceClient, *, force: bool, allow_empty: bool
) -> dict[str, Any]:
    """Run a reindex job in the service (same fallback rules as indexing)."""
    job_id = client.reindex(force=force, allow_empty=allow_empty)
    return _wait_for_job(client, job_id, "reindexing")


def index(
    paths: list[Path] = typer.Argument(
        None,
        metavar="PATHS",
        help="Files or directories to index (defaults to configured roots).",
    ),
) -> None:
    """Index files or directories, defaulting to the configured roots."""
    try:
        cfg = _load_config()
        raw_paths = paths if isinstance(paths, (list, tuple)) else None
        targets = [Path(item) for item in raw_paths] if raw_paths else list(cfg.roots)
        if not targets:
            _fatal(
                "no index roots configured; add `roots = [...]` to config.toml "
                "or pass PATH to `seekr index PATH`"
            )

        client = ServiceClient.from_discovery()
        if client is not None:
            try:
                service_report = _index_via_service(client, list(raw_paths or []))
            except ServiceUnavailable:
                pass
            else:
                _print_report_dict(service_report)
                return

        embedder = _build_embedder(cfg)
        with lifecycle_lock(cfg.db_path, shared=True):
            store = _open_store(cfg, embedder.dim)
            try:
                report = Indexer(store, embedder, cfg).sync(targets)
                store.set_last_error_count(len(report.errors))
            finally:
                store.close()
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fatal(str(exc))

    _print_report(report)


def reindex(
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace the live index even if indexing errors occurred.",
    ),
    allow_empty: bool = typer.Option(
        False,
        "--allow-empty",
        help="Allow replacing a non-empty live index with an empty rebuild.",
    ),
) -> None:
    """Rebuild the index in a fresh sibling database and swap it in safely.

    Per-file indexing errors and empty rebuilds are refused unless the matching
    override flag is passed.  With a running service the rebuild runs there,
    with the watcher paused and the service's index handles closed.
    """
    force_flag = force if isinstance(force, bool) else False
    allow_empty_flag = allow_empty if isinstance(allow_empty, bool) else False
    try:
        cfg = _load_config()
        client = ServiceClient.from_discovery()
        if client is not None:
            try:
                service_report = _reindex_via_service(
                    client, force=force_flag, allow_empty=allow_empty_flag
                )
            except ServiceUnavailable:
                pass
            else:
                _print_report_dict(service_report)
                return
        report = rebuild_index(cfg, force=force_flag, allow_empty=allow_empty_flag)
    except typer.Exit:
        raise
    except LifecycleError as exc:
        _fatal(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fatal(str(exc))

    _print_report(report)


def _daemon_status() -> tuple[bool, Path]:
    from .daemon import _socket_is_active, legacy_daemon_supported, socket_path

    if not legacy_daemon_supported():
        return False, Path("/unavailable")
    sock = socket_path()
    try:
        return _socket_is_active(sock), sock
    except OSError:
        return False, sock


def _service_status() -> dict[str, Any]:
    """Report the unified service (liveness-probed) for ``seekr status``."""
    try:
        from .service import find_running_service

        info = find_running_service(timeout=1.0)
    except Exception:  # noqa: BLE001 - status must never fail on discovery
        info = None
    if info is None:
        return {"running": False, "port": None, "pid": None, "version": None}
    return {
        "running": True,
        "port": info.port,
        "pid": info.pid,
        "version": info.version,
    }


def _format_service_status(service: dict[str, Any]) -> str:
    if service.get("running"):
        return (
            f"service: running at 127.0.0.1:{service['port']} "
            f"(pid {service['pid']}, version {service['version']})"
        )
    return "service: not running"


def _polled_watch_info(
    daemon_running: bool,
    sock: Path,
    cfg: Config,
    service_running: bool = False,
) -> tuple[list[Path], int]:
    from .daemon import legacy_daemon_supported

    daemon_running = daemon_running and legacy_daemon_supported()
    if not daemon_running and not service_running:
        return [], cfg.poll_interval_seconds
    payload = None
    if daemon_running:
        from .client import DaemonError, try_daemon_status

        try:
            payload = try_daemon_status(sock_path=sock, timeout=0.5)
        except DaemonError:
            payload = None

    if payload is not None and "polled_roots" in payload:
        raw_roots = payload["polled_roots"]
        raw_interval = payload.get(
            "poll_interval_seconds", cfg.poll_interval_seconds
        )
        if isinstance(raw_roots, list) and all(
            isinstance(item, str) for item in raw_roots
        ):
            interval = (
                raw_interval
                if isinstance(raw_interval, int)
                and not isinstance(raw_interval, bool)
                and raw_interval >= 1
                else cfg.poll_interval_seconds
            )
            return [Path(item) for item in raw_roots], interval

    mounts = platform_info.read_mounts()
    _, polled = platform_info.split_roots(cfg.roots, mounts)
    return polled, cfg.poll_interval_seconds


def _echo_daemon_status(
    daemon_running: bool, sock: Path, cfg: Config, service: dict[str, Any]
) -> None:
    typer.echo(_format_service_status(service))
    polled_roots, poll_interval = _polled_watch_info(
        daemon_running, sock, cfg, bool(service.get("running"))
    )
    for root in polled_roots:
        typer.echo(f"watch: {root} (polling every {poll_interval}s)")


def _db_size_bytes(db_path: Path) -> int:
    total = 0
    for candidate in (
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        try:
            total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _format_course_lines(
    course_rows: list[tuple[str | None, int | None, int]],
) -> list[str]:
    if not course_rows:
        return []
    grouped: dict[str | None, list[tuple[int | None, int]]] = {}
    for mod, wk, count in course_rows:
        grouped.setdefault(mod, []).append((wk, count))
    lines = ["course:"]
    for mod, items in grouped.items():
        display = display_module(mod) if mod is not None else "other"
        if not display:
            display = "other"
        parts = [
            f"week {wk} ({count})" if wk is not None else f"other ({count})"
            for wk, count in items
        ]
        lines.append(f"  {display}: {', '.join(parts)}")
    return lines


def status(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit index and service status as JSON.",
    ),
) -> None:
    """Print index counts, model metadata, disk size and service status."""
    json_flag = as_json if isinstance(as_json, bool) else False
    try:
        cfg = _load_config()
        db_path = Path(cfg.db_path)
        with lifecycle_lock(cfg.db_path, shared=True):
            dim = _stored_dim(cfg)
            if dim is None:
                if not db_path.exists():
                    daemon_running, sock = _daemon_status()
                    service = _service_status()
                    if json_flag:
                        typer.echo(
                            json.dumps(
                                {
                                    "exists": False,
                                    "database": str(db_path),
                                    "size_bytes": 0,
                                    "documents": 0,
                                    "chunks": 0,
                                    "last_error_count": 0,
                                    "kinds": {},
                                    "course": [],
                                    "embed_model": cfg.embed_model,
                                    "dim": None,
                                    "vector_quant": cfg.vector_quant,
                                    "embed_backend": cfg.embed_backend,
                                    "schema_version": None,
                                    "daemon": {
                                        "running": daemon_running,
                                        "socket": str(sock),
                                    },
                                    "service": service,
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                        return
                    typer.echo("documents: 0")
                    typer.echo("chunks: 0")
                    typer.echo("last error count: 0")
                    typer.echo(f"database: {db_path} (not initialized)")
                    typer.echo(
                        f"model: {cfg.embed_model} "
                        f"(quant={cfg.vector_quant}, backend={cfg.embed_backend})"
                    )
                    _echo_daemon_status(daemon_running, sock, cfg, service)
                    return
                dim = _build_embedder(cfg).dim

            store = _open_store(cfg, dim, vector_quant=None, record_backend=False)
            try:
                stats = store.stats()
                last_errors = store.get_last_error_count()
                detailed_fn = getattr(store, "detailed_stats", None)
                details: dict[str, Any] = dict(detailed_fn()) if callable(detailed_fn) else {}
                summary_fn = getattr(store, "course_summary", None)
                course_rows: list[tuple[str | None, int | None, int]] = (
                    list(summary_fn()) if callable(summary_fn) else []
                )
            finally:
                store.close()
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fatal(str(exc))

    size_bytes = _db_size_bytes(db_path)
    daemon_running, sock = _daemon_status()
    service = _service_status()
    if json_flag:
        typer.echo(
            json.dumps(
                {
                    "exists": True,
                    "database": str(db_path),
                    "size_bytes": size_bytes,
                    "size_mib": round(size_bytes / (1024 * 1024), 2),
                    "documents": stats["documents"],
                    "chunks": stats["chunks"],
                    "last_error_count": last_errors,
                    "kinds": details.get("kinds", {}),
                    "course": [
                        {"module": mod, "week": wk, "files": count}
                        for mod, wk, count in course_rows
                    ],
                    "embed_model": details.get("embed_model", cfg.embed_model),
                    "dim": details.get("dim", dim),
                    "vector_quant": details.get("vector_quant", cfg.vector_quant),
                    "schema_version": details.get("schema_version"),
                    "embed_backend": cfg.embed_backend,
                    "daemon": {
                        "running": daemon_running,
                        "socket": str(sock),
                    },
                    "service": service,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    typer.echo(f"documents: {stats['documents']}")
    typer.echo(f"chunks: {stats['chunks']}")
    typer.echo(f"last error count: {last_errors}")
    typer.echo(f"database: {db_path} ({size_bytes / (1024 * 1024):.2f} MiB)")
    if "embed_model" in details:
        typer.echo(
            f"model: {details['embed_model']} "
            f"(dim={details['dim']}, quant={details['vector_quant']}, "
            f"schema=v{details['schema_version']}, backend={cfg.embed_backend})"
        )
    kinds = details.get("kinds")
    if isinstance(kinds, dict) and kinds:
        summary = ", ".join(f"{name}={count}" for name, count in kinds.items())
        typer.echo(f"by kind: {summary}")
    for line in _format_course_lines(course_rows):
        typer.echo(line)
    _echo_daemon_status(daemon_running, sock, cfg, service)


def register(app: typer.Typer) -> None:
    """Add the index, reindex and status commands to *app*."""
    app.command("index")(index)
    app.command("reindex")(reindex)
    app.command("status")(status)
