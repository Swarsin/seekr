"""Hybrid keyword/vector retrieval with reciprocal rank fusion.

The public surface is small on purpose: :func:`fts_query` turns user text into
a safe FTS5 MATCH expression, :func:`rrf` fuses ranked id lists, and
:class:`Searcher` owns the read snapshot used for retrieval and row fetching.
Embedding happens before the snapshot is opened and reranking happens after it
is released, so neither model can block a concurrent writer.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

from .logging_config import log_event
from .models import Hit, HitExplain

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

    from .embed import Embedder
    from .rerank import Reranker
    from .store import ChunkRow, Store

SearchMode = Literal["keyword", "vector", "hybrid"]
SEARCH_MODES = frozenset({"keyword", "vector", "hybrid"})

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SNIPPET_CHARS = 240
_SNIPPET_CONTEXT_CHARS = 40
_MIN_CANDIDATES = 50
_CANDIDATES_PER_K = 5
# One candidate request per retriever.  Path scoping happens inside the
# retrievers before this limit is applied, so no global-depth growth is needed.
_RERANK_CANDIDATES = 30


def fts_query(raw: str) -> str:
    """Quote each word of ``raw`` and join the words with ``OR``.

    Only ``\\w`` runs survive, so FTS5 syntax characters cannot reach SQLite as
    operators.  An input without any token becomes ``""``.
    """
    if not isinstance(raw, str):
        return ""
    tokens = _TOKEN_RE.findall(raw)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse 1-based ranked id lists with ``score = sum(1 / (k + rank))``.

    Results are ordered by descending score.  Equal scores are broken by the
    best rank seen for the id and then by chunk id, so the output is stable
    regardless of dict/set iteration order.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")

    scores: dict[int, float] = {}
    best_ranks: dict[int, int] = {}
    for ranked in rank_lists:
        seen: set[int] = set()
        for rank, raw_id in enumerate(ranked, start=1):
            chunk_id = int(raw_id)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            previous = best_ranks.get(chunk_id)
            if previous is None or rank < previous:
                best_ranks[chunk_id] = rank

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], best_ranks[item[0]], item[0]),
    )


def _best_snippet(text: str, raw_query: str) -> str:
    """Return the 240-character window with the most distinct query terms."""
    if len(text) <= _SNIPPET_CHARS:
        return text

    wanted = {match.group(0).lower() for match in _TOKEN_RE.finditer(raw_query)}
    if not wanted:
        return text[:_SNIPPET_CHARS]

    starts = {0}
    for match in _TOKEN_RE.finditer(text):
        if match.group(0).lower() not in wanted:
            continue
        position = match.start()
        # Anchor a full-size window on the term, and a near-window for context.
        starts.add(max(0, min(position, len(text) - _SNIPPET_CHARS)))
        starts.add(max(0, position - _SNIPPET_CONTEXT_CHARS))

    best_start = 0
    best_coverage = -1
    for start in sorted(starts):
        window = text[start : start + _SNIPPET_CHARS]
        window_terms = {match.group(0).lower() for match in _TOKEN_RE.finditer(window)}
        coverage = len(wanted & window_terms)
        if coverage > best_coverage:
            best_coverage = coverage
            best_start = start

    if best_coverage <= 0:
        return text[:_SNIPPET_CHARS]
    return text[best_start : best_start + _SNIPPET_CHARS]


def _normalise_path_prefix(prefix: str | None) -> str | None:
    if prefix is None:
        return None
    value = str(prefix).rstrip("/")
    return value or None


def _store_has_chunks(store: Store) -> bool:
    has_chunks_fn = getattr(store, "has_chunks", None)
    if callable(has_chunks_fn):
        return bool(has_chunks_fn())
    stats_fn = getattr(store, "stats", None)
    if stats_fn is None:
        return True
    stats = stats_fn()
    try:
        return int(stats.get("chunks", 0)) > 0
    except (AttributeError, TypeError, ValueError):
        return True


def _snapshot(store: Store):
    """Use the store's read transaction when it has one (the real Store does)."""
    transaction = getattr(store, "transaction", None)
    if callable(transaction):
        return transaction()
    return nullcontext()


def _keyword_search(
    store: Store,
    match_query: str,
    limit: int,
    path_prefix: str | None,
) -> list[tuple[int, float]]:
    if path_prefix is None:
        return list(store.keyword_search(match_query, limit))
    return list(store.keyword_search(match_query, limit, path_prefix=path_prefix))


def _vector_search(
    store: Store,
    query_vector: np.ndarray,
    limit: int,
    path_prefix: str | None,
) -> list[tuple[int, float]]:
    if path_prefix is None:
        return list(store.vector_search(query_vector, limit))
    return list(store.vector_search(query_vector, limit, path_prefix=path_prefix))


def _ranked_candidates(
    store: Store,
    mode: SearchMode,
    match_query: str,
    query_vector: np.ndarray | None,
    limit: int,
    path_prefix: str | None,
) -> tuple[
    list[tuple[int, float]],
    dict[int, tuple[int, float]],
    dict[int, tuple[int, float]],
    dict[int, float],
]:
    """Return candidates plus per-retriever rank/score metadata.

    Non-null ``path_prefix`` is forwarded to both store retrievers, which apply
    exact component-aware filtering before their limit.  The Searcher therefore
    never re-scopes or grows a global candidate window itself.
    """
    keyword_pairs: list[tuple[int, float]] = []
    vector_pairs: list[tuple[int, float]] = []

    if mode in ("keyword", "hybrid") and match_query:
        keyword_pairs = _keyword_search(
            store, match_query, limit, path_prefix
        )

    if mode in ("vector", "hybrid"):
        assert query_vector is not None
        vector_pairs = _vector_search(store, query_vector, limit, path_prefix)

    keyword_info: dict[int, tuple[int, float]] = {}
    for rank, (raw_id, raw_score) in enumerate(keyword_pairs, start=1):
        chunk_id = int(raw_id)
        if chunk_id not in keyword_info:
            keyword_info[chunk_id] = (rank, float(raw_score))

    vector_info: dict[int, tuple[int, float]] = {}
    for rank, (raw_id, raw_score) in enumerate(vector_pairs, start=1):
        chunk_id = int(raw_id)
        if chunk_id not in vector_info:
            vector_info[chunk_id] = (rank, float(raw_score))

    if mode == "keyword":
        return (
            [(int(chunk_id), float(score)) for chunk_id, score in keyword_pairs],
            keyword_info,
            vector_info,
            {},
        )
    if mode == "vector":
        return (
            [(int(chunk_id), float(score)) for chunk_id, score in vector_pairs],
            keyword_info,
            vector_info,
            {},
        )

    keyword_rank = [int(chunk_id) for chunk_id, _ in keyword_pairs]
    vector_rank = [int(chunk_id) for chunk_id, _ in vector_pairs]
    fused = rrf([keyword_rank, vector_rank])
    rrf_scores = {chunk_id: float(score) for chunk_id, score in fused}
    return fused, keyword_info, vector_info, rrf_scores


def _cap_per_file(
    candidates: list[tuple[int, float]],
    chunks: dict[int, ChunkRow],
    per_file: int,
    k: int | None = None,
) -> list[tuple[int, float]]:
    counts: dict[str, int] = {}
    capped: list[tuple[int, float]] = []
    for chunk_id, score in candidates:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            continue
        count = counts.get(chunk.path, 0)
        if count >= per_file:
            continue
        counts[chunk.path] = count + 1
        capped.append((chunk_id, score))
        if k is not None and len(capped) >= k:
            break
    return capped


def _rerank_candidates(
    reranker: Reranker,
    query: str,
    candidates: list[tuple[int, float]],
    chunks: dict[int, ChunkRow],
) -> tuple[list[tuple[int, float]], dict[int, tuple[int, float]]]:
    selected = [
        (chunk_id, score)
        for chunk_id, score in candidates[:_RERANK_CANDIDATES]
        if chunk_id in chunks
    ]
    if not selected:
        return [], {}

    passages = [chunks[chunk_id].text for chunk_id, _ in selected]
    raw_scores = list(reranker.score(query, passages))
    if len(raw_scores) != len(selected):
        raise ValueError(
            f"reranker returned {len(raw_scores)} scores for {len(selected)} passages"
        )

    ranked = [
        (chunk_id, float(score), rank)
        for rank, ((chunk_id, _), score) in enumerate(zip(selected, raw_scores))
    ]
    ranked.sort(key=lambda item: (-item[1], item[2]))
    candidates_out = [(chunk_id, score) for chunk_id, score, _ in ranked]
    rerank_info = {
        chunk_id: (final_rank, score)
        for final_rank, (chunk_id, score, _) in enumerate(ranked, start=1)
    }
    return candidates_out, rerank_info


def _build_hits(
    candidates: list[tuple[int, float]],
    chunks: dict[int, ChunkRow],
    query: str,
    per_file: int,
    k: int,
    *,
    explain: bool = False,
    keyword_info: dict[int, tuple[int, float]] | None = None,
    vector_info: dict[int, tuple[int, float]] | None = None,
    rrf_scores: dict[int, float] | None = None,
    rerank_info: dict[int, tuple[int, float]] | None = None,
) -> list[Hit]:
    hits: list[Hit] = []
    for chunk_id, score in _cap_per_file(candidates, chunks, per_file, k):
        chunk = chunks[chunk_id]
        hit_explain: HitExplain | None = None
        if explain:
            kw = keyword_info.get(chunk_id) if keyword_info is not None else None
            vec = vector_info.get(chunk_id) if vector_info is not None else None
            rr = rerank_info.get(chunk_id) if rerank_info is not None else None
            hit_explain = HitExplain(
                keyword_rank=kw[0] if kw is not None else None,
                keyword_bm25=kw[1] if kw is not None else None,
                vector_rank=vec[0] if vec is not None else None,
                vector_distance=vec[1] if vec is not None else None,
                rrf_score=rrf_scores.get(chunk_id) if rrf_scores is not None else None,
                rerank_rank=rr[0] if rr is not None else None,
                rerank_score=rr[1] if rr is not None else None,
            )
        hits.append(
            Hit(
                chunk_id=chunk.id,
                path=chunk.path,
                locator=chunk.locator,
                loc_start=chunk.loc_start,
                snippet=_best_snippet(chunk.text, query),
                score=float(score),
                kind=chunk.kind,
                explain=hit_explain,
            )
        )
    return hits


class Searcher:
    """Run keyword, vector or hybrid retrieval against one store."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def search(
        self,
        query: str,
        k: int = 10,
        mode: SearchMode = "hybrid",
        per_file: int = 2,
        rerank: bool = False,
        path_prefix: str | None = None,
        *,
        explain: bool = False,
    ) -> list[Hit]:
        """Return up to ``k`` hits, capped per file and optionally path-scoped."""
        return self.get_hits(
            query,
            k=k,
            mode=mode,
            per_file=per_file,
            rerank=rerank,
            path_prefix=path_prefix,
            explain=explain,
        )

    def get_hits(
        self,
        query: str,
        k: int = 10,
        mode: SearchMode = "hybrid",
        per_file: int = 2,
        rerank: bool = False,
        path_prefix: str | None = None,
        *,
        explain: bool = False,
    ) -> list[Hit]:
        """Pure retrieval entry point shared by the CLI and future ``pick``."""
        if mode not in SEARCH_MODES:
            raise ValueError(
                f"unknown search mode {mode!r}; "
                "expected 'keyword', 'vector' or 'hybrid'"
            )
        if rerank and self.reranker is None:
            raise ValueError("rerank=True requires a reranker")
        started = time.monotonic()
        if k <= 0 or per_file <= 0:
            log_event(
                _LOGGER,
                logging.DEBUG,
                "search.query",
                mode=mode,
                k=k,
                per_file=per_file,
                rerank=rerank,
                hits=0,
                elapsed_ms=0.0,
            )
            return []
        if not isinstance(query, str) or not query.strip():
            log_event(
                _LOGGER,
                logging.DEBUG,
                "search.query",
                mode=mode,
                k=k,
                per_file=per_file,
                rerank=rerank,
                hits=0,
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 2),
            )
            return []
        if not _store_has_chunks(self.store):
            log_event(
                _LOGGER,
                logging.DEBUG,
                "search.query",
                mode=mode,
                k=k,
                per_file=per_file,
                rerank=rerank,
                hits=0,
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 2),
            )
            return []

        prefix = _normalise_path_prefix(path_prefix)

        # Model work happens before the store snapshot: a slow query embedding
        # must not hold the read lock that writers need.
        query_vector = None
        if mode in ("vector", "hybrid"):
            query_vector = self.embedder.embed_query(query)

        match_query = fts_query(query) if mode in ("keyword", "hybrid") else ""
        if mode == "keyword" and not match_query:
            log_event(
                _LOGGER,
                logging.DEBUG,
                "search.query",
                mode=mode,
                k=k,
                per_file=per_file,
                rerank=rerank,
                hits=0,
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 2),
            )
            return []

        limit = max(_MIN_CANDIDATES, _CANDIDATES_PER_K * k)

        with _snapshot(self.store):
            candidates, keyword_info, vector_info, rrf_scores = _ranked_candidates(
                self.store, mode, match_query, query_vector, limit, prefix
            )
            chunks = self.store.fetch_chunks(
                [chunk_id for chunk_id, _ in candidates]
            )

        selected = [
            (chunk_id, score)
            for chunk_id, score in candidates
            if chunk_id in chunks
        ]

        rerank_info: dict[int, tuple[int, float]] | None = None
        if rerank:
            assert self.reranker is not None
            selected, rerank_info = _rerank_candidates(
                self.reranker, query, selected, chunks
            )

        hits = _build_hits(
            selected,
            chunks,
            query,
            per_file,
            k,
            explain=explain,
            keyword_info=keyword_info,
            vector_info=vector_info,
            rrf_scores=rrf_scores,
            rerank_info=rerank_info,
        )
        log_event(
            _LOGGER,
            logging.DEBUG,
            "search.query",
            mode=mode,
            k=k,
            per_file=per_file,
            rerank=rerank,
            hits=len(hits),
            elapsed_ms=round((time.monotonic() - started) * 1000.0, 2),
        )
        return hits


__all__ = [
    "SEARCH_MODES",
    "SearchMode",
    "Searcher",
    "fts_query",
    "rrf",
]
