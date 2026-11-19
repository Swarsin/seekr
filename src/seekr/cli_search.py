"""The ``seekr search`` command and its shared standalone query entry point.

The main CLI module only needs to call ``cli_search.register(app)``; the
``get_hits`` function is deliberately pure (no printing, no Typer) so later
commands such as ``pick`` can reuse exactly the same parameter set, and the
daemon fast path lives inside ``get_hits`` without touching
:class:`seekr.search.Searcher`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from .client import DaemonError, DaemonMismatchError, try_daemon_search
from .config import Config, load_config
from .embed import build_embedder
from .filters import normalise_kind, parse_since
from .lifecycle import LifecycleError, lifecycle_lock
from .logging_config import log_event
from .models import Hit, HitExplain
from .search import SEARCH_MODES, Searcher, SearchMode
from .store import (
    VECTOR_QUANTS,
    ModelMismatchError,
    SchemaVersionError,
    Store,
    validate_schema_version,
)

_LOGGER = logging.getLogger(__name__)

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
        raise SearchError(
            f"cannot open index database {db_path}: {exc}; verify the file is readable "
            "or run `seekr reindex` to rebuild it"
        ) from exc


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
                raise SearchError(
                    f"cannot read index metadata from {db_path}: {exc}; verify the file "
                    "is readable or run `seekr reindex` to rebuild it"
                ) from exc
        except sqlite3.DatabaseError as exc:
            raise SearchError(
                f"cannot read index metadata from {db_path}: {exc}; verify the file "
                "is readable or run `seekr reindex` to rebuild it"
            ) from exc

        try:
            row = conn.execute(
                "SELECT EXISTS(SELECT 1 FROM documents) OR EXISTS(SELECT 1 FROM chunks)"
            ).fetchone()
            has_data = bool(row[0]) if row else False
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                has_data = False
            else:
                raise SearchError(
                    f"cannot read index database {db_path}: {exc}; verify the file "
                    "is readable or run `seekr reindex` to rebuild it"
                ) from exc
        except sqlite3.DatabaseError as exc:
            raise SearchError(
                f"cannot read index database {db_path}: {exc}; verify the file "
                "is readable or run `seekr reindex` to rebuild it"
            ) from exc
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
            "reranking is not available in this build; omit --rerank or install "
            "sentence-transformers"
        ) from exc
    return build_reranker(cfg)


def _validated_store_info(cfg: Config) -> tuple[Path, int, str] | None:
    """Validate index metadata and return ``(db_path, dim, vector_quant)`` if non-empty."""
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
    return db_path, stored_dim, stored_quant


def _build_searcher(cfg: Config, mode: str, rerank: bool) -> Searcher | None:
    """Open exactly the backends this query needs."""
    info = _validated_store_info(cfg)
    if info is None:
        return None
    db_path, stored_dim, stored_quant = info

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
    *,
    explain: bool = False,
    kind: str | None = None,
    since_mtime_ns: int | None = None,
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
    params: dict[str, Any] = {
        "op": "search",
        "query": query,
        "k": k,
        "mode": mode,
        "per_file": per_file,
        "rerank": rerank,
        "path_prefix": path_prefix,
    }
    if explain:
        params["explain"] = True
    if kind is not None:
        params["kind"] = kind
    if since_mtime_ns is not None:
        params["since_mtime_ns"] = since_mtime_ns
    try:
        hits = try_daemon_search(params, expected=expected)
    except DaemonMismatchError:
        log_event(
            _LOGGER,
            logging.INFO,
            "cli_search.daemon_mismatch",
            db_path=cfg.db_path,
            embed_model=cfg.embed_model,
        )
        return None
    if hits is not None:
        log_event(
            _LOGGER,
            logging.DEBUG,
            "cli_search.daemon_hit",
            mode=mode,
            k=k,
            hits=len(hits),
        )
    return hits


def get_hits(
    query: str,
    k: int = 10,
    mode: SearchMode = "hybrid",
    per_file: int = 2,
    rerank: bool = False,
    path_prefix: str | None = None,
    *,
    explain: bool = False,
    kind: str | None = None,
    since: str | None = None,
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
    kind_val = normalise_kind(kind)
    since_mtime_ns = parse_since(since)
    if not query.strip():
        # An empty or whitespace-only query can never match.  Short-circuit
        # before the daemon, config load or any embedder/store construction so
        # `seekr search ""` prints "No results." even on a model-less host.
        return []

    searcher_kwargs: dict[str, Any] = {}
    if explain:
        searcher_kwargs["explain"] = True
    if kind_val is not None:
        searcher_kwargs["kind"] = kind_val
    if since_mtime_ns is not None:
        searcher_kwargs["since_mtime_ns"] = since_mtime_ns
    if searcher is not None:
        return searcher.get_hits(
            query,
            k=k,
            mode=mode,
            per_file=per_file,
            rerank=rerank,
            path_prefix=path_prefix,
            **searcher_kwargs,
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
        explain=explain,
        kind=kind_val,
        since_mtime_ns=since_mtime_ns,
    )
    if daemon_hits is not None:
        return daemon_hits

    # The lock covers the whole Store lifetime, so a concurrent reindex cannot
    # swap the database out from under an in-process search.
    log_event(
        _LOGGER,
        logging.DEBUG,
        "cli_search.local_open",
        db_path=selected_cfg.db_path,
        mode=mode,
    )
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
                **searcher_kwargs,
            )
        finally:
            selected.store.close()


def _vectors_for_target(
    target_path: Path,
    store: Store,
    embedder: Embedder | None,
    cfg: Config,
    stored_dim: int | None = None,
) -> list[Any]:
    vectors = store.get_document_vectors(str(target_path))
    if vectors:
        return vectors
    if not target_path.is_file():
        raise SearchError(
            f"file not found and not in index: {target_path}; "
            "check the path or run `seekr index`"
        )
    from .chunking import chunk
    from .extract import ExtractionError, extract, kind_for, language_for

    file_kind = kind_for(target_path) or "text"
    try:
        extracted = extract(target_path, file_kind)
    except ExtractionError as exc:
        raise SearchError(str(exc)) from exc
    language = language_for(target_path) if file_kind == "code" else None
    specs = chunk(file_kind, extracted.units, cfg, language=language)
    if not specs:
        return []

    active_embedder = embedder if embedder is not None else build_embedder(cfg)
    actual_dim = _embedder_dim(active_embedder)
    if stored_dim is not None and actual_dim != stored_dim:
        raise ModelMismatchError(
            f"embedding model {cfg.embed_model!r} produces {actual_dim}-dimensional "
            f"vectors, but the index stores dim {stored_dim}; "
            "run `seekr reindex` to rebuild it"
        )
    embedded = active_embedder.embed_documents([spec.text for spec in specs])
    return list(embedded)


def get_similar_hits(
    target: Path | str,
    k: int = 10,
    per_file: int = 1,
    path_prefix: str | None = None,
    kind: str | None = None,
    since: str | None = None,
    *,
    cfg: Config | None = None,
    searcher: Searcher | None = None,
) -> list[Hit]:
    """Return indexed hits semantically similar to ``target``."""
    if k <= 0:
        raise SearchError("k must be a positive integer")
    if per_file <= 0:
        raise SearchError("per-file must be a positive integer")
    kind_val = normalise_kind(kind)
    since_mtime_ns = parse_since(since)
    target_path = Path(target).expanduser().resolve(strict=False)
    selected_cfg = cfg if cfg is not None else load_config()

    if searcher is not None:
        vectors = _vectors_for_target(
            target_path, searcher.store, searcher.embedder, selected_cfg
        )
        return searcher.similar(
            vectors,
            k=k,
            per_file=per_file,
            path_prefix=path_prefix,
            kind=kind_val,
            since_mtime_ns=since_mtime_ns,
            exclude_path=str(target_path),
        )

    log_event(
        _LOGGER,
        logging.DEBUG,
        "cli_search.similar_open",
        db_path=selected_cfg.db_path,
        target=target_path,
    )
    with lifecycle_lock(selected_cfg.db_path, shared=True):
        info = _validated_store_info(selected_cfg)
        if info is None:
            if not target_path.is_file():
                raise SearchError(
                    f"file not found and not in index: {target_path}; "
                    "check the path or run `seekr index`"
                )
            return []
        db_path, stored_dim, stored_quant = info
        store = Store(db_path, selected_cfg.embed_model, stored_dim, vector_quant=stored_quant)
        try:
            vectors = _vectors_for_target(
                target_path, store, None, selected_cfg, stored_dim=stored_dim
            )
            if not vectors:
                return []
            local_searcher = Searcher(store, None)  # type: ignore[arg-type]
            return local_searcher.similar(
                vectors,
                k=k,
                per_file=per_file,
                path_prefix=path_prefix,
                kind=kind_val,
                since_mtime_ns=since_mtime_ns,
                exclude_path=str(target_path),
            )
        finally:
            store.close()


def _hit_to_json_dict(hit: Hit) -> dict[str, Any]:
    data = dataclasses.asdict(hit)
    if data.get("explain") is None:
        data.pop("explain", None)
    return data


def _format_explain_line(info: HitExplain) -> str:
    parts: list[str] = []
    if info.keyword_rank is not None:
        bm25 = f" (bm25={info.keyword_bm25:.4f})" if info.keyword_bm25 is not None else ""
        parts.append(f"keyword=#{info.keyword_rank}{bm25}")
    if info.vector_rank is not None:
        dist = f" (dist={info.vector_distance:.4f})" if info.vector_distance is not None else ""
        parts.append(f"vector=#{info.vector_rank}{dist}")
    if info.rrf_score is not None:
        parts.append(f"rrf={info.rrf_score:.4f}")
    if info.rerank_rank is not None:
        rscore = f" (score={info.rerank_score:.4f})" if info.rerank_score is not None else ""
        parts.append(f"rerank=#{info.rerank_rank}{rscore}")
    return "  explain: " + ("  ".join(parts) if parts else "none")


def _echo_hits(
    typer_module: Any,
    hits: list[Hit],
    *,
    as_json: bool = False,
    explain: bool = False,
) -> None:
    if as_json:
        payload = [_hit_to_json_dict(hit) for hit in hits]
        typer_module.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not hits:
        typer_module.echo("No results.")
        return

    for hit in hits:
        typer_module.echo(f"{hit.path}  ({hit.locator})  {hit.score:.4f}")
        if explain and hit.explain is not None:
            typer_module.echo(_format_explain_line(hit.explain))
        typer_module.echo(hit.snippet)
        typer_module.echo()


def _exit_error(message: str, *, command: str = "search") -> NoReturn:
    import typer

    typer.echo(f"seekr {command}: {message}", err=True)
    raise typer.Exit(code=1)


_CLI_SEARCH_ERRORS = (
    SearchError,
    ModelMismatchError,
    SchemaVersionError,
    LifecycleError,
    DaemonError,
    ValueError,
)


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
        kind: str | None = typer_module.Option(
            None, "--kind", help="Filter by kind: pdf, code, markdown, transcript or text."
        ),
        since: str | None = typer_module.Option(
            None,
            "--since",
            help="Only include files modified since duration (e.g. 7d, 12h) or ISO date.",
        ),
        as_json: bool = typer_module.Option(
            False, "--json", help="Emit results as JSON."
        ),
        explain: bool = typer_module.Option(
            False, "--explain", help="Include retriever rank and score attribution."
        ),
    ) -> None:
        """Search the locally indexed files."""
        extra: dict[str, Any] = {}
        if explain:
            extra["explain"] = True
        if kind is not None:
            extra["kind"] = kind
        if since is not None:
            extra["since"] = since
        try:
            hits = get_hits(
                query,
                k=k,
                mode=mode,
                per_file=per_file,
                rerank=rerank,
                path_prefix=path_prefix,
                **extra,
            )
        except _CLI_SEARCH_ERRORS as exc:
            _exit_error(str(exc), command="search")
        except ImportError as exc:
            _exit_error(
                f"search backend is unavailable: {exc}; run `seekr doctor`",
                command="search",
            )
        except (OSError, sqlite3.Error) as exc:
            _exit_error(
                f"cannot read the index: {exc}; run `seekr doctor` or `seekr reindex`",
                command="search",
            )

        _echo_hits(typer_module, hits, as_json=as_json, explain=explain)

    search.__doc__ = "Search the locally indexed files."
    return search


def _build_similar_command(typer_module: Any) -> Any:
    def similar(
        target: Path = typer_module.Argument(
            ..., metavar="PATH", help="Indexed file or document on disk to match against."
        ),
        k: int = typer_module.Option(
            10, "-k", "--k", min=1, help="Maximum number of results."
        ),
        per_file: int = typer_module.Option(
            1, "--per-file", min=1, help="Maximum number of hits per file."
        ),
        path_prefix: str | None = typer_module.Option(
            None, "--path", help="Only search under this path prefix."
        ),
        kind: str | None = typer_module.Option(
            None, "--kind", help="Filter by kind: pdf, code, markdown, transcript or text."
        ),
        since: str | None = typer_module.Option(
            None,
            "--since",
            help="Only include files modified since duration (e.g. 7d, 12h) or ISO date.",
        ),
        as_json: bool = typer_module.Option(
            False, "--json", help="Emit results as JSON."
        ),
    ) -> None:
        """Find indexed files semantically related to PATH."""
        try:
            hits = get_similar_hits(
                target,
                k=k,
                per_file=per_file,
                path_prefix=path_prefix,
                kind=kind,
                since=since,
            )
        except _CLI_SEARCH_ERRORS as exc:
            _exit_error(str(exc), command="similar")
        except ImportError as exc:
            _exit_error(
                f"search backend is unavailable: {exc}; run `seekr doctor`",
                command="similar",
            )
        except (OSError, sqlite3.Error) as exc:
            _exit_error(
                f"cannot read the index: {exc}; run `seekr doctor` or `seekr reindex`",
                command="similar",
            )

        _echo_hits(typer_module, hits, as_json=as_json)

    similar.__doc__ = "Find indexed files semantically related to PATH."
    return similar


def register(app: typer.Typer) -> None:
    """Register the ``search`` and ``similar`` commands on the main Typer app."""
    import typer

    app.command("search")(_build_search_command(typer))
    app.command("similar")(_build_similar_command(typer))


__all__ = [
    "SearchError",
    "get_hits",
    "get_similar_hits",
    "register",
]
