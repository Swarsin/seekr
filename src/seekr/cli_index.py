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

import sqlite3
from pathlib import Path

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
    typer.echo(f"seekr: {message}")
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
            _fatal("no index roots configured; set roots in config or pass PATH")

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


def status() -> None:
    """Print document/chunk counts and the last recorded error count."""
    try:
        cfg = _load_config()
        with lifecycle_lock(cfg.db_path, shared=True):
            dim = _stored_dim(cfg)
            if dim is None:
                if not Path(cfg.db_path).exists():
                    typer.echo("documents: 0")
                    typer.echo("chunks: 0")
                    typer.echo("last error count: 0")
                    return
                dim = _build_embedder(cfg).dim

            store = _open_store(cfg, dim, vector_quant=None)
            try:
                stats = store.stats()
                last_errors = store.get_last_error_count()
            finally:
                store.close()
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fatal(str(exc))

    typer.echo(f"documents: {stats['documents']}")
    typer.echo(f"chunks: {stats['chunks']}")
    typer.echo(f"last error count: {last_errors}")


def register(app: typer.Typer) -> None:
    """Add the index, reindex and status commands to *app*."""
    app.command("index")(index)
    app.command("reindex")(reindex)
    app.command("status")(status)
