#!/usr/bin/env python3
"""Check that the ONNX Runtime backend matches the PyTorch backend.

Embeddings: the corpus is chunked with seekr's default chunker, the first
``--chunks`` chunks are embedded with both backends, and the row-wise cosine
similarity is reported. Reranking: for ``--rerank-queries`` queries, the
hybrid top-30 candidates from an index of the corpus are scored by both
cross-encoders and the two orders are compared with Kendall's tau.

Both models must already be cached locally with their PyTorch and ONNX files
(``seekr models pull`` with each backend). Nothing is downloaded.

    python bench/onnx_equivalence.py --corpus eval/data/scifact/corpus \\
        --chunks 1000 --rerank-queries 50

Gate: mean cosine >= 0.999, min cosine >= 0.995, mean Kendall tau >= 0.95.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

MEAN_COSINE_GATE = 0.999
MIN_COSINE_GATE = 0.995
KENDALL_TAU_GATE = 0.95
RERANK_DEPTH = 30


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's tau-b between two score lists over the same items (O(n^2))."""
    if len(a) != len(b):
        raise ValueError("score lists must have the same length")
    concordant = discordant = ties_a = ties_b = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    denom = ((concordant + discordant + ties_a) * (concordant + discordant + ties_b)) ** 0.5
    return 1.0 if denom == 0 else (concordant - discordant) / denom


def _corpus_chunks(corpus: Path, limit: int) -> list[str]:
    from seekr.chunking import chunk
    from seekr.config import Config
    from seekr.extract import extract, kind_for

    cfg = Config()
    texts: list[str] = []
    for path in sorted(corpus.iterdir()):
        kind = kind_for(path)
        if kind is None or not path.is_file():
            continue
        for spec in chunk(kind, extract(path, kind).units, cfg):
            texts.append(spec.text)
            if len(texts) >= limit:
                return texts
    return texts


def _queries(path: Path, limit: int) -> list[str]:
    out: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line)["query"])
            if len(out) >= limit:
                break
    return out


def _rerank_candidates(corpus: Path, queries: list[str], embedder: object) -> list[list[str]]:
    from seekr.config import Config
    from seekr.indexer import Indexer
    from seekr.search import Searcher
    from seekr.store import Store

    with tempfile.TemporaryDirectory(prefix="seekr-equiv-") as tmp:
        db_path = Path(tmp) / "index.db"
        cfg = Config(roots=(corpus.resolve(),), db_path=db_path)
        store = Store(db_path, cfg.embed_model, int(embedder.dim), vector_quant=cfg.vector_quant)  # type: ignore[attr-defined]
        try:
            Indexer(store, embedder, cfg).sync([corpus.resolve()])  # type: ignore[arg-type]
            searcher = Searcher(store, embedder)  # type: ignore[arg-type]
            candidates: list[list[str]] = []
            for query in queries:
                hits = searcher.get_hits(query, k=RERANK_DEPTH, mode="hybrid", per_file=RERANK_DEPTH)
                rows = store.fetch_chunks([hit.chunk_id for hit in hits])
                candidates.append([rows[hit.chunk_id].text for hit in hits if hit.chunk_id in rows])
            return candidates
        finally:
            store.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=None,
                        help="queries.jsonl (default: <corpus>/../queries.jsonl)")
    parser.add_argument("--chunks", type=int, default=1000)
    parser.add_argument("--rerank-queries", type=int, default=50)
    args = parser.parse_args(argv)

    import numpy as np

    from seekr.config import Config
    from seekr.embed import SentenceTransformerEmbedder
    from seekr.onnx_backend import OnnxEmbedder, OnnxReranker
    from seekr.rerank import CrossEncoderReranker

    cfg = Config()
    texts = _corpus_chunks(args.corpus, args.chunks)
    print(f"chunks: {len(texts)}")

    torch_embedder = SentenceTransformerEmbedder(cfg.embed_model, device="cpu")
    onnx_embedder = OnnxEmbedder(cfg.embed_model)

    started = time.perf_counter()
    torch_vecs = torch_embedder.embed_documents(texts)
    torch_s = time.perf_counter() - started
    started = time.perf_counter()
    onnx_vecs = onnx_embedder.embed_documents(texts)
    onnx_s = time.perf_counter() - started
    cosines = (torch_vecs * onnx_vecs).sum(axis=1)
    mean_cosine = float(np.mean(cosines))
    min_cosine = float(np.min(cosines))
    print(f"embed_seconds_torch_cpu: {torch_s:.2f}")
    print(f"embed_seconds_onnx_cpu: {onnx_s:.2f}")
    print(f"mean_cosine: {mean_cosine:.6f}")
    print(f"min_cosine: {min_cosine:.6f}")

    taus: list[float] = []
    if args.rerank_queries > 0:
        queries_path = args.queries or args.corpus.parent / "queries.jsonl"
        queries = _queries(queries_path, args.rerank_queries)
        candidates = _rerank_candidates(args.corpus, queries, onnx_embedder)
        torch_reranker = CrossEncoderReranker(cfg.rerank_model)
        onnx_reranker = OnnxReranker(cfg.rerank_model)
        for query, passages in zip(queries, candidates, strict=True):
            if len(passages) < 2:
                continue
            taus.append(
                kendall_tau(
                    torch_reranker.score(query, passages), onnx_reranker.score(query, passages)
                )
            )
        print(f"rerank_queries: {len(taus)}")
        print(f"kendall_tau_mean: {sum(taus) / len(taus):.6f}")
        print(f"kendall_tau_min: {min(taus):.6f}")

    passed = (
        mean_cosine >= MEAN_COSINE_GATE
        and min_cosine >= MIN_COSINE_GATE
        and (not taus or sum(taus) / len(taus) >= KENDALL_TAU_GATE)
    )
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
