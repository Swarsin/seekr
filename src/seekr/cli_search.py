"""The ``seekr search`` command and its shared standalone query entry point.

The main CLI module only needs to call ``cli_search.register(app)``; the
``get_hits`` function is deliberately pure (no printing, no Typer) so later
commands such as ``pick`` can reuse exactly the same parameter set, and the
daemon fast path lives inside ``get_hits`` without touching
:class:`seekr.search.Searcher`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from .client import DaemonError, DaemonMismatchError, try_daemon_search
from .config import Config, load_config
from .embed import build_embedder
from .lifecycle import LifecycleError, lifecycle_lock
from .models import Hit
from .search import SEARCH_MODES, Searcher, SearchMode
from .store import (
    VECTOR_QUANTS,
    ModelMismatchError,
    SchemaVersionError,
    Store,
    validate_schema_version,
)

if TYPE_CHECKING:
    import typer

    from .embed import Embedder
    from .rerank import Reranker


class SearchError(RuntimeError):
    """A user-facing search setup error."""


def _read_only_connection(db_path: Path) -> sqlite3.Connection:
    try:
        return sqlite3.connect(
            f"{db_path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=5.0,
        )
    except sqlite3.Error as exc:
        raise SearchError(f"cannot open index database {db_path}: {exc}") from exc


def _inspect_db(db_path: Path) -> tuple[dict[str, str] | None, bool]:
    """Read metadata and data-presence on a single read-only SQLite connection."""
    if not db_path.exists():
        return None, False

    conn = _read_only_connection(db_path)
    try:
        try:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
            meta: dict[str, str] | None = {str(key): str(value) for key, value in rows}
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                meta = None
            else:
                raise SearchError(f"cannot read index metadata from {db_path}: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            raise SearchError(f"cannot read index metadata from {db_path}: {exc}") from exc

        try:
            row = conn.execute(
                "SELECT EXISTS(SELECT 1 FROM documents) OR EXISTS(SELECT 1 FROM chunks)"
            ).fetchone()
            has_data = bool(row[0]) if row else False
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                has_data = False
            else:
                raise SearchError(f"cannot read index database {db_path}: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            raise SearchError(f"cannot read index database {db_path}: {exc}") from exc
    finally:
        conn.close()
    return meta, has_data


def _read_meta(db_path: Path) -> dict[str, str] | None:
    """Read embedding metadata without opening the Store or loading a model."""
    meta, _ = _inspect_db(db_path)
    return meta


def _has_indexed_data(db_path: Path) -> bool:
    _, has_data = _inspect_db(db_path)
    return has_data


def _embedder_dim(embedder: Embedder) -> int:
    dim = int(embedder.dim)
    if dim <= 0:
        raise SearchError("embedding model reported a non-positive dimension")
    return dim


def _build_reranker(cfg: Config) -> Reranker:
    try:
        from .rerank import build_reranker
    except ImportError as exc:
        raise SearchError(
            "reranking is not available in this build; omit --rerank"
        ) from exc
    return build_reranker(cfg)


def _build_searcher(cfg: Config, mode: str, rerank: bool) -> Searcher | None:
    """Open exactly the backends this query needs.

    A missing or empty index returns ``None`` immediately in every mode, so no
    embedding model or Store is built just to report no results.  A recorded
    model mismatch is reported before ``build_embedder`` can load any model.
    Vector/hybrid queries need the real dimension, so only those validate and
    load the embedding model during setup.
    """
    db_path = Path(cfg.db_path)
    meta, has_data = _inspect_db(db_path)

    if not meta:
        if has_data:
            raise ModelMismatchError(
                "index database has no recorded embedding model; "
                "run `seekr reindex` to rebuild it"
            )
        return None

    validate_schema_version(meta)

    stored_model_raw = meta.get("embed_model")
    stored_dim_raw = meta.get("dim")
    if stored_model_raw is None and stored_dim_raw is None:
        if has_data:
            raise ModelMismatchError(
                "index database has no recorded embedding model; "
                "run `seekr reindex` to rebuild it"
            )
        return None
    if stored_model_raw is None or stored_dim_raw is None:
        raise ModelMismatchError(
            "index database has incomplete embedding metadata; "
            "run `seekr reindex` to rebuild it"
        )

    stored_model = stored_model_raw.strip()
    if not stored_model:
        raise ModelMismatchError(
            "index database has invalid embedding metadata; "
            "run `seekr reindex` to rebuild it"
        )
    try:
        stored_dim = int(stored_dim_raw)
    except (TypeError, ValueError) as exc:
        raise ModelMismatchError(
            "index database has an invalid embedding dimension; "
            "run `seekr reindex` to rebuild it"
        ) from exc
    if stored_dim <= 0:
        raise ModelMismatchError(
            "index database has an invalid embedding dimension; "
            "run `seekr reindex` to rebuild it"
        )
    if stored_model != cfg.embed_model:
        raise ModelMismatchError(
            "index database was built with embedding model "
            f"{stored_model!r} (dim {stored_dim}), but this configuration uses "
            f"{cfg.embed_model!r}; run `seekr reindex` to rebuild it"
        )
    stored_quant = meta.get("vector_quant", "float32")
    if stored_quant not in VECTOR_QUANTS:
        raise ModelMismatchError(
            f"index database has unsupported vector quantization {stored_quant!r}; "
            "run `seekr reindex` to rebuild it"
        )

    # An empty index cannot produce hits in any mode.  Returning before
    # ``build_embedder`` keeps vector/hybrid queries from loading a model (or
    # creating a Store) just to answer "No results.".
    if not has_data:
        return None

    embedder = build_embedder(cfg)
    if mode in ("vector", "hybrid"):
        actual_dim = _embedder_dim(embedder)
        if actual_dim != stored_dim:
            raise ModelMismatchError(
                f"embedding model {cfg.embed_model!r} produces {actual_dim}-dimensional "
                f"vectors, but the index stores dim {stored_dim}; "
                "run `seekr reindex` to rebuild it"
            )

    reranker = _build_reranker(cfg) if rerank else None
    store = Store(db_path, cfg.embed_model, stored_dim, vector_quant=stored_quant)
    return Searcher(store, embedder, reranker)


def _daemon_search(
    cfg: Config,
    query: str,
    k: int,
    mode: SearchMode,
    per_file: int,
    rerank: bool,
    path_prefix: str | None,
) -> list[Hit] | None:
    """Try the warm daemon without opening a Store or loading any model.

    A daemon whose database or embedding model differs from the current config
    raises :class:`DaemonMismatchError`.  That is a deliberate invalidation and
    falls back to the local configured index; timeouts and other server errors
    propagate so a broken daemon is never silently ignored.
    """
    expected = {
        "db_path": str(
            Path(cfg.db_path).expanduser().resolve(strict=False)
        ),
        "embed_model": cfg.embed_model,
    }
    if rerank:
        expected["rerank_model"] = cfg.rerank_model
    params = {
        "op": "search",
        "query": query,
        "k": k,
        "mode": mode,
        "per_file": per_file,
        "rerank": rerank,
        "path_prefix": path_prefix,
    }
    try:
        return try_daemon_search(params, expected=expected)
    except DaemonMismatchError:
        return None


def get_hits(
    query: str,
    k: int = 10,
    mode: SearchMode = "hybrid",
    per_file: int = 2,
    rerank: bool = False,
    path_prefix: str | None = None,
    *,
    cfg: Config | None = None,
    searcher: Searcher | None = None,
) -> list[Hit]:
    """Return hits for a standalone query, preferring a warm daemon.

    This is the pure shared point for ``seekr search`` and ``seekr pick``.
    ``cfg`` and ``searcher`` are injection hooks for tests and for a caller
    that already owns a live search stack; ordinary CLI calls leave both unset
    and try the daemon before opening the configured database.
    """
    if mode not in SEARCH_MODES:
        raise SearchError(
            f"unknown search mode {mode!r}; expected 'keyword', 'vector' or 'hybrid'"
        )
    if k <= 0:
        raise SearchError("k must be a positive integer")
    if per_file <= 0:
        raise SearchError("per-file must be a positive integer")
    if not query.strip():
        # An empty or whitespace-only query can never match.  Short-circuit
        # before the daemon, config load or any embedder/store construction so
        # `seekr search ""` prints "No results." even on a model-less host.
        return []

    if searcher is not None:
        return searcher.get_hits(
            query,
            k=k,
            mode=mode,
            per_file=per_file,
            rerank=rerank,
            path_prefix=path_prefix,
        )

    selected_cfg = cfg if cfg is not None else load_config()
    daemon_hits = _daemon_search(
        selected_cfg,
        query,
        k,
        mode,
        per_file,
        rerank,
        path_prefix,
    )
    if daemon_hits is not None:
        return daemon_hits

    # The lock covers the whole Store lifetime, so a concurrent reindex cannot
    # swap the database out from under an in-process search.
    with lifecycle_lock(selected_cfg.db_path, shared=True):
        selected = _build_searcher(selected_cfg, mode, rerank)
        if selected is None:
            return []
        try:
            return selected.get_hits(
                query,
                k=k,
                mode=mode,
                per_file=per_file,
                rerank=rerank,
                path_prefix=path_prefix,
            )
        finally:
            selected.store.close()


def _exit_error(message: str) -> NoReturn:
    import typer

    typer.echo(f"seekr search: {message}", err=True)
    raise typer.Exit(code=1)


def _build_search_command(typer_module: Any) -> Any:
    def search(
        query: str = typer_module.Argument(
            ..., metavar="QUERY", help="Words to search for."
        ),
        k: int = typer_module.Option(
            10, "-k", "--k", min=1, help="Maximum number of results."
        ),
        mode: str = typer_module.Option(
            "hybrid", "--mode", help="Search mode: keyword, vector or hybrid."
        ),
        per_file: int = typer_module.Option(
            2, "--per-file", min=1, help="Maximum number of hits per file."
        ),
        rerank: bool = typer_module.Option(
            False, "--rerank", help="Rerank the top candidates."
        ),
        path_prefix: str | None = typer_module.Option(
            None, "--path", help="Only search under this path prefix."
        ),
    ) -> None:
        """Search the locally indexed files."""
        try:
            hits = get_hits(
                query,
                k=k,
                mode=mode,
                per_file=per_file,
                rerank=rerank,
                path_prefix=path_prefix,
            )
        except (
            SearchError,
            ModelMismatchError,
            SchemaVersionError,
            LifecycleError,
            DaemonError,
            ValueError,
        ) as exc:
            _exit_error(str(exc))
        except ImportError as exc:
            _exit_error(f"search backend is unavailable: {exc}; run `seekr doctor`")
        except (OSError, sqlite3.Error) as exc:
            _exit_error(f"cannot read the index: {exc}")

        if not hits:
            typer_module.echo("No results.")
            return

        for hit in hits:
            typer_module.echo(f"{hit.path}  ({hit.locator})  {hit.score:.4f}")
            typer_module.echo(hit.snippet)
            typer_module.echo()

    search.__doc__ = "Search the locally indexed files."
    return search


def register(app: typer.Typer) -> None:
    """Register the ``search`` command on the main Typer application."""
    import typer

    app.command("search")(_build_search_command(typer))


__all__ = [
    "SearchError",
    "get_hits",
    "register",
]
