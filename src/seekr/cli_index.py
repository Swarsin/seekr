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
from pathlib import Path
from typing import Any

import typer

from .config import Config, load_config
from .embed import Embedder, build_embedder
from .indexer import Indexer, SyncReport
from .lifecycle import LifecycleError, lifecycle_lock, rebuild_index
from .store import Store

__all__ = ["register"]


def _load_config() -> Config:
    return load_config()


def _build_embedder(cfg: Config) -> Embedder:
    return build_embedder(cfg)


_USE_CONFIG = object()


def _open_store(
    cfg: Config, dim: int, *, vector_quant: object = _USE_CONFIG
) -> Store:
    quant = cfg.vector_quant if vector_quant is _USE_CONFIG else vector_quant
    if quant is None:
        return Store(cfg.db_path, cfg.embed_model, dim)
    return Store(cfg.db_path, cfg.embed_model, dim, vector_quant=str(quant))


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


def _print_report(report: SyncReport) -> None:
    typer.echo(f"added: {report.added}")
    typer.echo(f"updated: {report.updated}")
    typer.echo(f"unchanged: {report.unchanged}")
    typer.echo(f"removed: {report.removed}")
    typer.echo(f"skipped: {report.skipped}")
    typer.echo(f"errors: {len(report.errors)}")
    typer.echo(f"warnings: {len(report.warnings)}")
    for path, message in report.errors:
        typer.echo(f"error: {path}: {message}")
    for path, message in report.warnings:
        typer.echo(f"warning: {path}: {message}")


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
    override flag is passed.
    """
    force_flag = force if isinstance(force, bool) else False
    allow_empty_flag = allow_empty if isinstance(allow_empty, bool) else False
    try:
        cfg = _load_config()
        report = rebuild_index(cfg, force=force_flag, allow_empty=allow_empty_flag)
    except typer.Exit:
        raise
    except LifecycleError as exc:
        _fatal(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fatal(str(exc))

    _print_report(report)


def _daemon_status() -> tuple[bool, Path]:
    from .daemon import _socket_is_active, socket_path

    sock = socket_path()
    try:
        return _socket_is_active(sock), sock
    except OSError:
        return False, sock


def _format_daemon_status(daemon_running: bool, sock: Path) -> str:
    if daemon_running:
        return f"daemon: running ({sock})"
    from .daemon import socket_path_problem

    if socket_path_problem(sock) is not None:
        return f"daemon: stopped (socket path too long: {sock})"
    return f"daemon: stopped ({sock})"


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


def status(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit index and daemon status as JSON.",
    ),
) -> None:
    """Print index counts, model metadata, disk size and daemon status."""
    json_flag = as_json if isinstance(as_json, bool) else False
    try:
        cfg = _load_config()
        db_path = Path(cfg.db_path)
        with lifecycle_lock(cfg.db_path, shared=True):
            dim = _stored_dim(cfg)
            if dim is None:
                if not db_path.exists():
                    daemon_running, sock = _daemon_status()
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
                                    "embed_model": cfg.embed_model,
                                    "dim": None,
                                    "vector_quant": cfg.vector_quant,
                                    "schema_version": None,
                                    "daemon": {
                                        "running": daemon_running,
                                        "socket": str(sock),
                                    },
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
                    typer.echo(f"model: {cfg.embed_model} (quant={cfg.vector_quant})")
                    typer.echo(_format_daemon_status(daemon_running, sock))
                    return
                dim = _build_embedder(cfg).dim

            store = _open_store(cfg, dim, vector_quant=None)
            try:
                stats = store.stats()
                last_errors = store.get_last_error_count()
                detailed_fn = getattr(store, "detailed_stats", None)
                details: dict[str, Any] = dict(detailed_fn()) if callable(detailed_fn) else {}
            finally:
                store.close()
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fatal(str(exc))

    size_bytes = _db_size_bytes(db_path)
    daemon_running, sock = _daemon_status()
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
                    "embed_model": details.get("embed_model", cfg.embed_model),
                    "dim": details.get("dim", dim),
                    "vector_quant": details.get("vector_quant", cfg.vector_quant),
                    "schema_version": details.get("schema_version"),
                    "daemon": {
                        "running": daemon_running,
                        "socket": str(sock),
                    },
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
            f"schema=v{details['schema_version']})"
        )
    kinds = details.get("kinds")
    if isinstance(kinds, dict) and kinds:
        summary = ", ".join(f"{name}={count}" for name, count in kinds.items())
        typer.echo(f"by kind: {summary}")
    typer.echo(_format_daemon_status(daemon_running, sock))


def register(app: typer.Typer) -> None:
    """Add the index, reindex and status commands to *app*."""
    app.command("index")(index)
    app.command("reindex")(reindex)
    app.command("status")(status)
