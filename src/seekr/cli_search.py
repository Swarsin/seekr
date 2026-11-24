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
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from .client import DaemonError, DaemonMismatchError, try_daemon_search
from .config import Config, load_config
from .course import display_module
from .embed import build_embedder
from .filters import normalise_kind, parse_since, parse_week
from .lifecycle import LifecycleError, lifecycle_lock
from .logging_config import log_event
from .models import Hit, HitExplain
from .search import SEARCH_MODES, Searcher, SearchMode
from .service_client import ServiceClient, ServiceError, ServiceUnavailable
from .spread import (
    SPREAD_CANDIDATES,
    SpreadGroup,
    compute_spread,
    format_spread,
    spread_to_dict,
)
from .store import (
    VECTOR_QUANTS,
    ModelMismatchError,
    SchemaVersionError,
    Store,
    validate_schema_version,
)
from .web_contract import LIMITS

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


def _build_searcher(
    cfg: Config,
    mode: str,
    rerank: bool,
    *,
    embedder: Embedder | None = None,
) -> Searcher | None:
    """Open exactly the backends this query needs.

    ``embedder`` is an optional injection point for a caller that already owns
    a warm model.  It must belong to the active ``cfg.embed_model``; the
    keyword path never reads ``.dim`` from it.
    """
    if embedder is not None and embedder.model_name != cfg.embed_model:
        raise SearchError(
            f"injected embedding model {embedder.model_name!r} does not match the "
            f"configured model {cfg.embed_model!r}"
        )

    info = _validated_store_info(cfg)
    if info is None:
        return None
    db_path, stored_dim, stored_quant = info

    active_embedder = embedder if embedder is not None else build_embedder(cfg)
    if mode in ("vector", "hybrid"):
        actual_dim = _embedder_dim(active_embedder)
        if actual_dim != stored_dim:
            raise ModelMismatchError(
                f"embedding model {cfg.embed_model!r} produces {actual_dim}-dimensional "
                f"vectors, but the index stores dim {stored_dim}; "
                "run `seekr reindex` to rebuild it"
            )

    reranker = _build_reranker(cfg) if rerank else None
    store = Store(db_path, cfg.embed_model, stored_dim, vector_quant=stored_quant)
    return Searcher(store, active_embedder, reranker)


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
    module: str | None = None,
    week_range: tuple[int, int] | None = None,
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
    if module is not None:
        params["module"] = module
    if week_range is not None:
        params["week_range"] = [week_range[0], week_range[1]]
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


def _normalise_module(module: str | None) -> str | None:
    if module is None:
        return None
    if not isinstance(module, str):
        raise TypeError("module filter must be a string")
    stripped = module.strip()
    return stripped or None


def _validate_query(
    query: str,
    k: int,
    mode: SearchMode,
    per_file: int,
    kind: str | None,
    since: str | None,
    module: str | None = None,
    week: str | None = None,
) -> tuple[str | None, int | None, str | None, tuple[int, int] | None]:
    """Apply the shared query/filter validation used by both entry points."""
    if mode not in SEARCH_MODES:
        raise SearchError(
            f"unknown search mode {mode!r}; expected 'keyword', 'vector' or 'hybrid'"
        )
    if k <= 0:
        raise SearchError("k must be a positive integer")
    if per_file <= 0:
        raise SearchError("per-file must be a positive integer")
    return (
        normalise_kind(kind),
        parse_since(since),
        _normalise_module(module),
        parse_week(week),
    )


def _service_hits(
    call: str, request: dict[str, Any]
) -> list[Hit] | None:
    """Send ``request`` to a running service, or return ``None`` to go local.

    An unreachable service (or none at all) returns ``None``; any error the
    service reports surfaces as :class:`SearchError` with its message.
    """
    client = ServiceClient.from_discovery()
    if client is None:
        return None
    try:
        return client.search(request) if call == "search" else client.similar(request)
    except ServiceUnavailable as exc:
        log_event(_LOGGER, logging.DEBUG, "cli_search.service_unavailable", error=exc)
        return None
    except ServiceError as exc:
        raise SearchError(str(exc)) from exc


def _wire_search_request(
    query: str,
    k: int,
    mode: SearchMode,
    per_file: int,
    rerank: bool,
    path_prefix: str | None,
    *,
    explain: bool,
    kind: str | None,
    since: str | None,
    module: str | None,
    week: str | None,
) -> dict[str, Any] | None:
    """Build a ``POST /api/search`` body, or ``None`` if the wire can't express it."""
    if (
        k > LIMITS["k_max"]
        or per_file > LIMITS["per_file_max"]
        or len(query) > LIMITS["query_chars"]
        or (rerank and mode == "keyword")
    ):
        return None
    request: dict[str, Any] = {
        "query": query,
        "k": k,
        "mode": mode,
        "per_file": per_file,
        "rerank": rerank,
        "explain": explain,
    }
    for name, value in (
        ("path", path_prefix),
        ("kind", kind),
        ("since", since),
        ("module", module),
        ("week", week),
    ):
        if value is not None:
            request[name] = value
    return request


_daemon_deprecation_logged = False


def _log_daemon_deprecation() -> None:
    global _daemon_deprecation_logged
    if _daemon_deprecation_logged:
        return
    _daemon_deprecation_logged = True
    log_event(
        _LOGGER,
        logging.INFO,
        "cli_search.daemon_deprecated",
        hint="`seekr daemon` is deprecated; use `seekr service start`",
    )


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
    module: str | None = None,
    week: str | None = None,
    cfg: Config | None = None,
    searcher: Searcher | None = None,
    embedder: Embedder | None = None,
    use_daemon: bool = True,
    use_service: bool = True,
) -> list[Hit]:
    """Return hits for a standalone query, preferring a running service.

    This is the pure shared point for ``seekr search``, ``seekr pick`` and MCP.
    The order is: the local service, then the legacy daemon, then a local
    search.  ``cfg``, ``searcher`` and ``embedder`` are injection hooks for
    tests and for a caller that already owns a live search stack (the service
    itself); any of them, ``use_service=False`` or ``use_daemon=False`` skips
    the service.  ``searcher`` wins over ``embedder``; ordinary CLI calls leave
    all of them unset.  ``use_daemon=False`` forces the local configured index
    even when a service or daemon is listening.
    """
    kind_val, since_mtime_ns, module_val, week_range = _validate_query(
        query, k, mode, per_file, kind, since, module, week
    )
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
    if module_val is not None:
        searcher_kwargs["module"] = module_val
    if week_range is not None:
        searcher_kwargs["week_range"] = week_range
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

    if use_service and use_daemon and cfg is None and embedder is None:
        wire = _wire_search_request(
            query,
            k,
            mode,
            per_file,
            rerank,
            path_prefix,
            explain=explain,
            kind=kind,
            since=since,
            module=module,
            week=week,
        )
        if wire is not None:
            service_hits = _service_hits("search", wire)
            if service_hits is not None:
                return service_hits

    selected_cfg = cfg if cfg is not None else load_config()
    if use_daemon:
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
            module=module_val,
            week_range=week_range,
        )
        if daemon_hits is not None:
            _log_daemon_deprecation()
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
        selected = (
            _build_searcher(selected_cfg, mode, rerank)
            if embedder is None
            else _build_searcher(selected_cfg, mode, rerank, embedder=embedder)
        )
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


def try_daemon_hits(
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
    module: str | None = None,
    week: str | None = None,
    cfg: Config | None = None,
) -> list[Hit] | None:
    """Ask the daemon for hits, or return ``None`` when it cannot be used.

    This shares the query/filter validation with :func:`get_hits` and never
    opens a Store or embedder.  A missing listener and a
    :class:`DaemonMismatchError` both return ``None``; every other
    :class:`DaemonError` propagates so a reachable broken daemon is never
    silently replaced by a local search.
    """
    kind_val, since_mtime_ns, module_val, week_range = _validate_query(
        query, k, mode, per_file, kind, since, module, week
    )
    if not query.strip():
        return []
    selected_cfg = cfg if cfg is not None else load_config()
    return _daemon_search(
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
        module=module_val,
        week_range=week_range,
    )


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
    module: str | None = None,
    week: str | None = None,
    cfg: Config | None = None,
    searcher: Searcher | None = None,
    use_service: bool = True,
) -> list[Hit]:
    """Return indexed hits semantically similar to ``target``.

    A running service answers first unless ``cfg`` or ``searcher`` is injected
    or ``use_service`` is false; otherwise the configured index is read here.
    """
    if k <= 0:
        raise SearchError("k must be a positive integer")
    if per_file <= 0:
        raise SearchError("per-file must be a positive integer")
    kind_val = normalise_kind(kind)
    since_mtime_ns = parse_since(since)
    module_val = _normalise_module(module)
    week_range = parse_week(week)
    similar_kwargs: dict[str, Any] = {}
    if module_val is not None:
        similar_kwargs["module"] = module_val
    if week_range is not None:
        similar_kwargs["week_range"] = week_range
    target_path = Path(target).expanduser().resolve(strict=False)
    if (
        use_service
        and cfg is None
        and searcher is None
        and k <= LIMITS["k_max"]
        and per_file <= LIMITS["per_file_max"]
    ):
        wire: dict[str, Any] = {"path": str(target_path), "k": k, "per_file": per_file}
        for name, value in (
            ("path_prefix", path_prefix),
            ("kind", kind),
            ("since", since),
            ("module", module),
            ("week", week),
        ):
            if value is not None:
                wire[name] = value
        service_hits = _service_hits("similar", wire)
        if service_hits is not None:
            return service_hits
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
            **similar_kwargs,
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
                **similar_kwargs,
            )
        finally:
            store.close()


def _hit_to_json_dict(hit: Hit) -> dict[str, Any]:
    data = dataclasses.asdict(hit)
    if data.get("explain") is None:
        data.pop("explain", None)
    data["highlights"] = [[int(a), int(b)] for a, b in hit.highlights]
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


def _hit_label_prefix(hit: Hit) -> str:
    parts: list[str] = []
    if hit.module:
        display = display_module(hit.module)
        if display:
            parts.append(display)
    if hit.week is not None:
        parts.append(f"Week {hit.week}")
    return f"{' · '.join(parts)} · " if parts else ""


def _format_snippet(typer_module: Any, hit: Hit) -> str:
    snippet = hit.snippet
    if not hit.highlights or not getattr(sys.stdout, "isatty", lambda: False)():
        return snippet
    parts: list[str] = []
    cursor = 0
    length = len(snippet)
    for start, end in sorted(hit.highlights):
        if start < cursor or start >= end or end > length:
            continue
        if start > cursor:
            parts.append(snippet[cursor:start])
        parts.append(typer_module.style(snippet[start:end], bold=True))
        cursor = end
    if cursor < length:
        parts.append(snippet[cursor:])
    return "".join(parts)


def _echo_hits(
    typer_module: Any,
    hits: list[Hit],
    *,
    as_json: bool = False,
    explain: bool = False,
    spread_groups: Sequence[SpreadGroup] | None = None,
) -> None:
    if as_json:
        hit_dicts = [_hit_to_json_dict(hit) for hit in hits]
        payload: Any = (
            hit_dicts
            if spread_groups is None
            else {
                "hits": hit_dicts,
                "spread": [spread_to_dict(group) for group in spread_groups],
            }
        )
        typer_module.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not hits:
        typer_module.echo("No results.")
        return

    for hit in hits:
        label = _hit_label_prefix(hit)
        typer_module.echo(f"{label}{hit.path}  ({hit.locator})  {hit.score:.4f}")
        if explain and hit.explain is not None:
            typer_module.echo(_format_explain_line(hit.explain))
        typer_module.echo(_format_snippet(typer_module, hit))
        typer_module.echo()

    if spread_groups:
        typer_module.echo(format_spread(spread_groups))


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
            None,
            "--kind",
            help="Filter by kind: pdf, code, markdown, word, slides, notebook, transcript or text.",
        ),
        since: str | None = typer_module.Option(
            None,
            "--since",
            help="Only include files modified since duration (e.g. 7d, 12h) or ISO date.",
        ),
        module: str | None = typer_module.Option(
            None, "--module", help="Filter by module name or unique prefix."
        ),
        week: str | None = typer_module.Option(
            None, "--week", help="Filter by week N or inclusive range N-M."
        ),
        as_json: bool = typer_module.Option(
            False, "--json", help="Emit results as JSON."
        ),
        explain: bool = typer_module.Option(
            False, "--explain", help="Include retriever rank and score attribution."
        ),
        spread: bool = typer_module.Option(
            False,
            "--spread",
            help="Show where the topic appears across modules and weeks.",
        ),
    ) -> None:
        """Search the locally indexed files."""
        filter_kwargs: dict[str, Any] = {}
        if kind is not None:
            filter_kwargs["kind"] = kind
        if since is not None:
            filter_kwargs["since"] = since
        if module is not None:
            filter_kwargs["module"] = module
        if week is not None:
            filter_kwargs["week"] = week
        extra = dict(filter_kwargs)
        if explain:
            extra["explain"] = True
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
            spread_groups: list[SpreadGroup] | None = None
            if spread:
                if query.strip() and hits:
                    spread_hits = get_hits(
                        query,
                        k=SPREAD_CANDIDATES,
                        mode=mode,
                        per_file=2,
                        rerank=False,
                        path_prefix=path_prefix,
                        **filter_kwargs,
                    )
                    spread_groups = compute_spread(spread_hits)
                else:
                    spread_groups = []
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

        _echo_hits(
            typer_module,
            hits,
            as_json=as_json,
            explain=explain,
            spread_groups=spread_groups,
        )

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
            None,
            "--kind",
            help="Filter by kind: pdf, code, markdown, word, slides, notebook, transcript or text.",
        ),
        since: str | None = typer_module.Option(
            None,
            "--since",
            help="Only include files modified since duration (e.g. 7d, 12h) or ISO date.",
        ),
        module: str | None = typer_module.Option(
            None, "--module", help="Filter by module name or unique prefix."
        ),
        week: str | None = typer_module.Option(
            None, "--week", help="Filter by week N or inclusive range N-M."
        ),
        as_json: bool = typer_module.Option(
            False, "--json", help="Emit results as JSON."
        ),
    ) -> None:
        """Find indexed files semantically related to PATH."""
        extra: dict[str, Any] = {}
        if module is not None:
            extra["module"] = module
        if week is not None:
            extra["week"] = week
        try:
            hits = get_similar_hits(
                target,
                k=k,
                per_file=per_file,
                path_prefix=path_prefix,
                kind=kind,
                since=since,
                **extra,
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
    "try_daemon_hits",
]
