"""Offline, file-level evaluation for seekr retrieval modes.

Rankings are reduced to unique relative POSIX paths before any metric is
computed.  The harness indexes ``root`` into a throwaway SQLite database, asks
each mode for at least ten candidates per query (``per_file=1``) and measures
the complete ``Searcher.get_hits`` call, so query embedding and optional
reranking are part of the honest search timing.  The one untimed warm-up call
per mode keeps cold model loading and first-touch store setup out of the
reported latency, just like indexing itself.
"""

from __future__ import annotations

import json
import math
import statistics
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .config import Config
from .indexer import Indexer
from .search import Searcher
from .store import Store

if TYPE_CHECKING:
    from .embed import Embedder
    from .models import Hit
    from .rerank import Reranker

__all__ = [
    "EVAL_MODES",
    "EvalQuery",
    "load_queries",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "render_table",
    "run_eval",
]

EVAL_MODES = ("keyword", "vector", "hybrid", "hybrid+rerank")
_FIXED_METRIC_K = 10
_DEFAULT_STORE = Store


@dataclass(frozen=True)
class EvalQuery:
    """One query and the files that are relevant to it."""

    id: str
    query: str
    relevant: frozenset[str]


def _path_key(value: str) -> str:
    """Return a relative POSIX path key with leading ``./`` removed."""
    text = str(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return PurePosixPath(text).as_posix()


def _unique_top(ranked: Iterable[str], k: int) -> list[str]:
    if k <= 0:
        return []
    top: list[str] = []
    seen: set[str] = set()
    for raw_path in ranked:
        path = _path_key(raw_path)
        if path in seen:
            continue
        seen.add(path)
        top.append(path)
        if len(top) >= k:
            break
    return top


def _relevant_set(relevant: Iterable[str]) -> set[str]:
    return {_path_key(path) for path in relevant}


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Return the fraction of relevant files found in the unique top *k*."""
    expected = _relevant_set(relevant)
    if not expected or k <= 0:
        return 0.0
    found = expected.intersection(_unique_top(ranked, k))
    return len(found) / len(expected)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Return reciprocal rank of the first relevant file in the unique top *k*."""
    expected = _relevant_set(relevant)
    if not expected or k <= 0:
        return 0.0
    for rank, path in enumerate(_unique_top(ranked, k), start=1):
        if path in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Return binary-gain nDCG with a log2 rank discount in the unique top *k*."""
    expected = _relevant_set(relevant)
    if not expected or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, path in enumerate(_unique_top(ranked, k), start=1):
        if path in expected:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _malformed(path: Path, line_number: int, detail: str) -> ValueError:
    return ValueError(f"invalid query in {path} at line {line_number}: {detail}")


def load_queries(path: Path) -> list[EvalQuery]:
    """Read JSONL ``{"id", "query", "relevant": [...]}`` evaluation queries."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read queries file {source}: {exc}") from exc

    queries: list[EvalQuery] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _malformed(source, line_number, f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise _malformed(source, line_number, "expected a JSON object")

        query_id = payload.get("id")
        query_text = payload.get("query")
        relevant = payload.get("relevant")
        if not isinstance(query_id, str) or not query_id.strip():
            raise _malformed(source, line_number, "'id' must be a non-empty string")
        if not isinstance(query_text, str) or not query_text.strip():
            raise _malformed(source, line_number, "'query' must be a non-empty string")
        if not isinstance(relevant, list):
            raise _malformed(source, line_number, "'relevant' must be a list of paths")
        paths: set[str] = set()
        for item in relevant:
            if not isinstance(item, str) or not item.strip():
                raise _malformed(
                    source,
                    line_number,
                    "'relevant' must contain only non-empty path strings",
                )
            paths.add(_path_key(item))

        queries.append(
            EvalQuery(
                id=query_id,
                query=query_text,
                relevant=frozenset(paths),
            )
        )
    return queries


def render_table(results: dict[str, dict[str, float]]) -> str:
    """Render one markdown row per mode, preserving the results order."""
    header = ("mode", "recall@5", "recall@10", "mrr@10", "ndcg@10", "p50_ms")
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for mode, metrics in results.items():
        values = []
        for name in header[1:]:
            value = float(metrics.get(name, 0.0))
            values.append(f"{value:.2f}" if name == "p50_ms" else f"{value:.4f}")
        lines.append("| " + " | ".join((mode, *values)) + " |")
    return "\n".join(lines) + "\n"


def _relative_hit_path(root: Path, raw_path: str) -> str | None:
    path = Path(raw_path)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(root)
        except (OSError, ValueError):
            return None
    key = _path_key(path.as_posix())
    if key in {"", ".", ".."} or key.startswith("../"):
        return None
    return key


def _ranking(root: Path, hits: Iterable[Hit]) -> list[str]:
    ranking: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        path = _relative_hit_path(root, hit.path)
        if path is None or path in seen:
            continue
        seen.add(path)
        ranking.append(path)
    return ranking


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _internal_mode(mode: str) -> tuple[str, bool]:
    if mode == "hybrid+rerank":
        return "hybrid", True
    return mode, False


def _validate_modes(modes: Sequence[str]) -> list[str]:
    selected: list[str] = []
    for mode in modes:
        if not isinstance(mode, str):
            # Public evaluation API reports invalid modes as ValueError.
            raise ValueError(  # noqa: TRY004
                f"evaluation modes must be strings, got {mode!r}"
            )
        if mode not in EVAL_MODES:
            choices = ", ".join(EVAL_MODES)
            raise ValueError(f"unknown evaluation mode {mode!r}; expected one of {choices}")
        if mode not in selected:
            selected.append(mode)
    return selected


def run_eval(
    root: Path,
    queries: list[EvalQuery],
    modes: list[str],
    embedder: Embedder,
    reranker: Reranker | None,
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Index *root* into a temp store and evaluate every query per mode.

    ``k`` sets the minimum candidate depth requested from the searcher.  The
    reported metrics themselves are fixed at ``@5``/``@10``, so at least ten
    candidates are always requested even when a smaller ``k`` is supplied.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    selected = _validate_modes(modes)
    if not selected:
        return {}
    if "hybrid+rerank" in selected and reranker is None:
        raise ValueError("mode 'hybrid+rerank' requires a reranker")

    root_path = Path(root).resolve()
    query_list = list(queries)
    request_k = max(k, _FIXED_METRIC_K)

    with tempfile.TemporaryDirectory(prefix="seekr-eval-") as temp_dir:
        db_path = Path(temp_dir) / "index.db"
        cfg = Config(
            roots=(root_path,),
            db_path=db_path,
            embed_model=embedder.model_name,
        )
        if Store is _DEFAULT_STORE:
            store = Store(
                db_path,
                embedder.model_name,
                int(embedder.dim),
                vector_quant=cfg.vector_quant,
            )
        else:
            store = Store(db_path, embedder.model_name, int(embedder.dim))
        try:
            report = Indexer(store, embedder, cfg).sync([root_path])
            errors = list(getattr(report, "errors", []))
            if errors:
                first_path, first_error = errors[0]
                raise RuntimeError(
                    f"indexing failed for {first_path}: {first_error}"
                )
            searcher = Searcher(store, embedder, reranker)
            results: dict[str, dict[str, float]] = {}

            for mode in selected:
                internal_mode, rerank = _internal_mode(mode)
                if query_list:
                    # Untimed warm-up: load lazy models, populate SQLite page
                    # cache and exercise reranking before measuring queries.
                    searcher.get_hits(
                        query_list[0].query,
                        k=request_k,
                        mode=internal_mode,  # type: ignore[arg-type]
                        per_file=1,
                        rerank=rerank,
                    )

                rankings: list[list[str]] = []
                timings_ms: list[float] = []
                for item in query_list:
                    started = time.perf_counter()
                    hits = searcher.get_hits(
                        item.query,
                        k=request_k,
                        mode=internal_mode,  # type: ignore[arg-type]
                        per_file=1,
                        rerank=rerank,
                    )
                    timings_ms.append((time.perf_counter() - started) * 1000.0)
                    rankings.append(_ranking(root_path, hits))

                recall5 = _mean(
                    [
                        recall_at_k(ranking, set(item.relevant), 5)
                        for ranking, item in zip(rankings, query_list)
                    ]
                )
                recall10 = _mean(
                    [
                        recall_at_k(ranking, set(item.relevant), 10)
                        for ranking, item in zip(rankings, query_list)
                    ]
                )
                mrr10 = _mean(
                    [
                        mrr_at_k(ranking, set(item.relevant), 10)
                        for ranking, item in zip(rankings, query_list)
                    ]
                )
                ndcg10 = _mean(
                    [
                        ndcg_at_k(ranking, set(item.relevant), 10)
                        for ranking, item in zip(rankings, query_list)
                    ]
                )
                results[mode] = {
                    "recall@5": recall5,
                    "recall@10": recall10,
                    "mrr@10": mrr10,
                    "ndcg@10": ndcg10,
                    "p50_ms": (
                        float(statistics.median(timings_ms)) if timings_ms else 0.0
                    ),
                }
            return results
        finally:
            store.close()
