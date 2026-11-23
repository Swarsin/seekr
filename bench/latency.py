#!/usr/bin/env python3
"""Measure warm retrieval latency on a synthetic 20k-chunk seekr index.

The benchmark drives the production :class:`seekr.store.Store` and
:class:`seekr.search.Searcher` code paths with normalised 384-dimensional
synthetic vectors and FTS-friendly synthetic text.  Retrieval timings exclude
embedding-model inference: the searcher receives precomputed query vectors
through a lookup embedder.  The real embedding model is timed separately, and
only when it is already available in the local cache.  Nothing is downloaded
and no latency value is invented.

Typical use::

    python bench/latency.py --chunks 20000

Writes ``bench/latency-results.md`` (or the path passed to ``--out``) after all
retrieval timings succeed.  Missing runtime dependencies fail with an
actionable error before any results file is written.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import platform
import sqlite3
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DIM = 384
DEFAULT_CHUNKS = 20_000
DEFAULT_QUERIES = 50
DEFAULT_K = 10
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SEED = 202_611_18
DEFAULT_QUANT = "int8"
DEFAULT_BACKEND = "onnx"
DEFAULT_DEVICE = "cpu"
EMBED_BACKENDS = ("onnx", "torch")
DEVICES = ("cpu", "cuda")
VECTOR_QUANTS = ("float32", "int8")
MODES = ("keyword", "vector", "hybrid")
RESULTS_PATH = Path(__file__).resolve().parents[1] / "bench" / "latency-results.md"

_VOCAB_SIZE = 2_000
_WORDS_PER_CHUNK = 40
_CHUNKS_PER_DOC = 40
_BENCH_DB_MODEL = "synthetic-benchmark-384"


class BenchmarkError(RuntimeError):
    """A benchmark precondition failed or a required backend is unavailable."""


@dataclass(frozen=True)
class Query:
    """One synthetic query and its precomputed normalised vector."""

    text: str
    vector: Any


class _StaticEmbedder:
    """Look up a precomputed query vector instead of running a model."""

    def __init__(self, queries: Sequence[Query], dim: int) -> None:
        self.dim = int(dim)
        self._vectors = {query.text: query.vector for query in queries}

    def embed_documents(self, texts: list[str]) -> Any:
        raise RuntimeError("the latency benchmark only embeds queries")

    def embed_query(self, text: str) -> Any:
        try:
            return self._vectors[text]
        except KeyError as exc:
            raise RuntimeError("benchmark query was not precomputed") from exc


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a percentile using linear interpolation between closest ranks."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _bootstrap_src_path() -> None:
    """Let a checkout run this script without installing the package first."""
    package_root = Path(__file__).resolve().parents[1] / "src"
    if (package_root / "seekr" / "__init__.py").is_file():
        value = str(package_root)
        if value not in sys.path:
            sys.path.insert(0, value)


def _require_runtime() -> tuple[Any, Any, Any]:
    """Import and probe the real backends, or raise a clear BenchmarkError."""
    _bootstrap_src_path()
    missing = [
        name
        for name in ("numpy", "sqlite_vec")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise BenchmarkError(
            "missing runtime dependencies: "
            + ", ".join(missing)
            + "; install seekr first (for example `pip install -e .` or "
            "`pipx install .`) and retry"
        )

    try:
        import numpy as np
        import sqlite_vec

        from seekr.search import Searcher
        from seekr.store import Store
    except ImportError as exc:
        raise BenchmarkError(
            f"cannot import seekr and its runtime dependencies: {exc}; "
            "set PYTHONPATH=src when running from a checkout or install the package"
        ) from exc

    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        raise BenchmarkError(
            "this Python's sqlite3 module cannot load SQLite extensions; "
            "use a build with loadable extension support (run `seekr doctor`)"
        )

    conn = sqlite3.connect(":memory:")
    try:
        options = [str(row[0]) for row in conn.execute("PRAGMA compile_options")]
        if "ENABLE_FTS5" not in options:
            raise BenchmarkError(
                "this SQLite build lacks FTS5; use a Python build whose SQLite "
                "includes FTS5 (run `seekr doctor`)"
            )
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.execute("SELECT vec_version()").fetchone()
        except (ImportError, OSError, RuntimeError, sqlite3.Error) as exc:
            raise BenchmarkError(
                f"sqlite-vec is installed but could not be loaded: {exc}; "
                "run `seekr doctor` for diagnostics"
            ) from exc
        finally:
            try:
                conn.enable_load_extension(False)
            except (AttributeError, sqlite3.Error):
                pass
    finally:
        conn.close()

    return np, Store, Searcher


def _vocabulary() -> list[str]:
    return [f"term{index:05d}" for index in range(_VOCAB_SIZE)]


def populate_store(
    store: Any,
    chunk_count: int,
    *,
    dim: int = DIM,
    seed: int = DEFAULT_SEED,
) -> None:
    """Fill *store* with deterministic synthetic documents and unit vectors."""
    import numpy as np

    from seekr.models import ChunkSpec, text_sha
    from seekr.store import StoredChunk

    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    vocabulary = _vocabulary()
    word_rng = np.random.default_rng(seed)
    vector_rng = np.random.default_rng(seed + 1)
    remaining = int(chunk_count)
    doc_index = 0

    while remaining:
        size = min(_CHUNKS_PER_DOC, remaining)
        path = f"synthetic/doc-{doc_index:05d}.txt"
        doc_id = store.upsert_document(
            path,
            mtime_ns=1_700_000_000_000_000_000 + doc_index,
            size=size * 512,
            sha256=text_sha(path),
            kind="text",
        )

        word_indexes = word_rng.integers(
            0, len(vocabulary), size=(size, _WORDS_PER_CHUNK)
        )
        vectors = vector_rng.standard_normal((size, dim), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.asarray(vectors / norms, dtype=np.float32)

        chunks: list[Any] = []
        for local_index in range(size):
            text = " ".join(
                vocabulary[int(index)] for index in word_indexes[local_index]
            )
            start_line = local_index * _WORDS_PER_CHUNK + 1
            chunks.append(
                StoredChunk(
                    spec=ChunkSpec(
                        text=text,
                        locator=f"L{start_line}-{start_line + _WORDS_PER_CHUNK - 1}",
                        loc_start=float(start_line),
                    ),
                    text_sha=text_sha(text),
                    vector=np.asarray(vectors[local_index], dtype=np.float32),
                )
            )
        store.replace_chunks(doc_id, chunks)

        remaining -= size
        doc_index += 1


def make_queries(
    count: int,
    *,
    dim: int = DIM,
    seed: int = DEFAULT_SEED + 2,
) -> list[Query]:
    """Return *count* deterministic FTS-term queries with unit query vectors."""
    import numpy as np

    if count <= 0:
        raise ValueError("count must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    vocabulary = _vocabulary()
    word_rng = np.random.default_rng(seed)
    vector_rng = np.random.default_rng(seed + 1)
    vectors = vector_rng.standard_normal((count, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.asarray(vectors / norms, dtype=np.float32)

    queries: list[Query] = []
    for index in range(count):
        terms = word_rng.integers(0, len(vocabulary), size=2)
        # The final token makes every query string unique without affecting
        # the FTS match materially: it simply has no posting.
        text = " ".join(
            (vocabulary[int(terms[0])], vocabulary[int(terms[1])], f"q{index:04d}")
        )
        queries.append(Query(text=text, vector=np.asarray(vectors[index], dtype=np.float32)))
    return queries


def _searcher(store: Any, queries: Sequence[Query], dim: int) -> Any:
    from seekr.search import Searcher

    return Searcher(store, _StaticEmbedder(queries, dim))


def measure_mode(
    store: Any,
    queries: Sequence[Query],
    mode: str,
    *,
    k: int = DEFAULT_K,
    repetitions: int = 1,
    dim: int = DIM,
) -> list[float]:
    """Time warm ``Searcher.get_hits`` calls without model inference."""
    if mode not in MODES:
        raise ValueError(f"unknown benchmark mode {mode!r}")
    if not queries:
        raise ValueError("queries must not be empty")
    if k <= 0 or repetitions <= 0:
        raise ValueError("k and repetitions must be positive")

    searcher = _searcher(store, queries, dim)
    warmup_hits = searcher.get_hits(
        queries[0].text, k=k, mode=mode, per_file=2
    )
    if not warmup_hits:
        raise BenchmarkError(
            f"warm-up query returned no hits in {mode!r}; the synthetic corpus "
            "or query generator is broken"
        )

    samples: list[float] = []
    for _ in range(repetitions):
        for query in queries:
            started = time.perf_counter()
            searcher.get_hits(query.text, k=k, mode=mode, per_file=2)
            samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def measure_query_embedding(
    queries: Sequence[Query],
    model_name: str,
    *,
    repetitions: int = 1,
    embed_backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
) -> tuple[list[float] | None, str | None]:
    """Time the real embedding model if it is already available locally.

    ``embed_backend`` selects ONNX Runtime (CPU) or PyTorch on ``device``.
    """
    if not queries or repetitions <= 0:
        raise ValueError("queries and repetitions must be positive")
    if embed_backend == "onnx" and device != "cpu":
        return None, "the ONNX backend runs on CPU only; use --embed-backend torch for CUDA"
    try:
        if embed_backend == "onnx":
            from seekr import onnx_backend

            embedder = onnx_backend.OnnxEmbedder(model_name)
        else:
            from seekr import embed

            embedder = embed.SentenceTransformerEmbedder(model_name, device=device)
        embedder.embed_query(queries[0].text)
        samples: list[float] = []
        for _ in range(repetitions):
            for query in queries:
                started = time.perf_counter()
                embedder.embed_query(query.text)
                samples.append((time.perf_counter() - started) * 1000.0)
        return samples, None
    except Exception as exc:  # noqa: BLE001 - report unavailability, never guess
        return None, f"{type(exc).__name__}: {exc}"


def _cpu_description() -> str:
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except (OSError, UnicodeDecodeError):
        model = ""
    if not model:
        model = platform.processor().strip() or platform.machine()
    logical = os.cpu_count()
    return f"{model} ({logical} logical CPUs)" if logical else model


def _vector_backend_version() -> str:
    conn = sqlite3.connect(":memory:")
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        row = conn.execute("SELECT vec_version()").fetchone()
    except (ImportError, OSError, RuntimeError, sqlite3.Error) as exc:
        return f"unknown ({exc})"
    finally:
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass
        conn.close()
    return str(row[0]) if row and row[0] is not None else "unknown"


def _peak_rss_mib() -> float | None:
    """Return process peak resident set size in MiB when available."""
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss <= 0:
        return None
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def _accelerator_note() -> tuple[str, str] | None:
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return None
    try:
        if torch_mod.cuda.is_available():
            return ("Accelerator", f"CUDA ({torch_mod.cuda.get_device_name(0)})")
    except Exception:  # noqa: BLE001 - informational environment note only
        return None
    return ("Accelerator", "CPU")


def environment_notes(np_module: Any) -> list[tuple[str, str]]:
    """Return measured environment facts for the results file."""
    notes = [
        ("Python", platform.python_version()),
        ("Platform", platform.platform()),
        ("CPU", _cpu_description()),
        ("SQLite", sqlite3.sqlite_version),
        ("sqlite-vec", _vector_backend_version()),
        ("NumPy", str(np_module.__version__)),
    ]
    accel = _accelerator_note()
    if accel is not None:
        notes.append(accel)
    return notes


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _doc_count(chunk_count: int) -> int:
    return math.ceil(chunk_count / _CHUNKS_PER_DOC)


def _db_size_mib(db_path: Path) -> float:
    total_bytes = 0
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            total_bytes += candidate.stat().st_size
    return float(total_bytes) / (1024.0 * 1024.0)


def render_results(
    *,
    chunk_count: int,
    query_count: int,
    repetitions: int,
    mode_samples: dict[str, list[float]],
    embedding_samples: list[float] | None,
    embedding_note: str | None,
    model_name: str,
    notes: Sequence[tuple[str, str]],
    index_seconds: float | None = None,
    peak_rss_mib: float | None = None,
    vector_quant: str = "float32",
    db_size_mib: float | None = None,
) -> str:
    """Render measured samples as Markdown, refusing incomplete measurements."""
    for mode in MODES:
        if mode not in mode_samples or not mode_samples[mode]:
            raise BenchmarkError(f"no measured samples for mode {mode!r}")

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# seekr latency benchmark",
        "",
        f"Measured {timestamp} by `bench/latency.py`. Only measured values appear here.",
        "",
        "## Setup",
        "",
        (
            f"- Synthetic corpus: {chunk_count:,} chunks in "
            f"{_doc_count(chunk_count):,} documents, normalised {DIM}-dimensional "
            f"{vector_quant} vectors."
        ),
        f"- Queries: {query_count} per mode, {repetitions} timed round(s) over the set.",
        "- Each mode received one untimed warm-up after the index was fully written.",
        (
            "- Retrieval timing covers `Searcher.get_hits`: SQLite FTS5/vec0 "
            "retrieval, RRF, per-file capping and hit construction."
        ),
        (
            "- Query vectors were precomputed; no embedding-model inference is "
            "included in the retrieval timings."
        ),
        "",
        "| mode | p50 (ms) | p95 (ms) | samples |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        samples = mode_samples[mode]
        lines.append(
            f"| {mode} | {percentile(samples, 0.50):.2f} | "
            f"{percentile(samples, 0.95):.2f} | {len(samples)} |"
        )

    lines.extend(["", "## Query embedding", ""])
    if embedding_samples:
        lines.extend(
            [
                (
                    f"Model: `{model_name}` (real local model, measured separately "
                    "from retrieval)."
                ),
                "",
                "| statistic | value |",
                "| --- | ---: |",
                f"| p50 | {percentile(embedding_samples, 0.50):.2f} ms |",
                f"| p95 | {percentile(embedding_samples, 0.95):.2f} ms |",
                f"| samples | {len(embedding_samples)} |",
            ]
        )
    else:
        detail = embedding_note or "no reason was reported"
        lines.append(
            f"Not available on this run: {_md_cell(detail)}. "
            "No embedding latency is reported."
        )

    if index_seconds is not None or peak_rss_mib is not None or db_size_mib is not None:
        lines.extend(
            [
                "",
                "## Indexing and memory",
                "",
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
        if index_seconds is not None and index_seconds > 0:
            doc_count = _doc_count(chunk_count)
            lines.extend(
                [
                    f"| Store populate time | {index_seconds:.2f} s |",
                    f"| Indexing throughput (docs/sec) | {doc_count / index_seconds:.1f} |",
                    f"| Indexing throughput (chunks/sec) | {chunk_count / index_seconds:.1f} |",
                ]
            )
        if db_size_mib is not None:
            lines.append(f"| Database size (MiB) | {db_size_mib:.2f} |")
        if peak_rss_mib is not None:
            lines.append(f"| Peak process RSS | {peak_rss_mib:.1f} MiB |")

    lines.extend(
        [
            "",
            "## Environment",
            "",
            "| component | value |",
            "| --- | --- |",
        ]
    )
    for name, value in notes:
        lines.append(f"| {_md_cell(name)} | {_md_cell(value)} |")

    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    chunks: int = DEFAULT_CHUNKS,
    db_path: Path | None = None,
    queries: int = DEFAULT_QUERIES,
    repetitions: int = 1,
    model: str = DEFAULT_MODEL,
    out: Path = RESULTS_PATH,
    seed: int = DEFAULT_SEED,
    dim: int = DIM,
    k: int = DEFAULT_K,
    quant: str = DEFAULT_QUANT,
    embed_backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
) -> Path:
    """Run the benchmark and write measured results to *out*."""
    np_module, store_class, _ = _require_runtime()
    if chunks <= 0 or queries <= 0 or repetitions <= 0 or dim <= 0 or k <= 0:
        raise BenchmarkError("chunks, queries, repetitions, dim and k must be positive")
    if quant not in VECTOR_QUANTS:
        raise BenchmarkError(f"unsupported vector quantization {quant!r}")

    query_list = make_queries(queries, dim=dim, seed=seed + 2)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if db_path is None:
        temporary = tempfile.TemporaryDirectory(prefix="seekr-latency-")
        database = Path(temporary.name) / "index.db"
    else:
        database = Path(db_path).expanduser()
        if database.exists() and database.stat().st_size > 0:
            raise BenchmarkError(
                f"benchmark database {database} already exists and is not empty; "
                "pass a new path or remove it yourself"
            )

    try:
        store = store_class(database, _BENCH_DB_MODEL, dim, vector_quant=quant)
        try:
            index_started = time.perf_counter()
            populate_store(store, chunks, dim=dim, seed=seed)
            index_seconds = time.perf_counter() - index_started
            actual_chunks = int(store.stats()["chunks"])
            if actual_chunks != chunks:
                raise BenchmarkError(
                    f"populated {actual_chunks} chunks but requested {chunks}; "
                    "refusing to report results for an unexpected corpus"
                )
            mode_samples = {
                mode: measure_mode(
                    store,
                    query_list,
                    mode,
                    k=k,
                    repetitions=repetitions,
                    dim=dim,
                )
                for mode in MODES
            }
        finally:
            store.close()
        db_size_mib = _db_size_mib(database)
    finally:
        if temporary is not None:
            temporary.cleanup()

    embedding_samples, embedding_note = measure_query_embedding(
        query_list, model, repetitions=repetitions, embed_backend=embed_backend, device=device
    )
    peak_rss_mib = _peak_rss_mib()
    text = render_results(
        chunk_count=actual_chunks,
        query_count=len(query_list),
        repetitions=repetitions,
        mode_samples=mode_samples,
        embedding_samples=embedding_samples,
        embedding_note=embedding_note,
        model_name=f"{model} ({embed_backend}, {device})",
        notes=environment_notes(np_module),
        index_seconds=index_seconds,
        peak_rss_mib=peak_rss_mib,
        vector_quant=quant,
        db_size_mib=db_size_mib,
    )

    output = Path(out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing any benchmark backend."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure warm keyword, vector and hybrid retrieval latency on a "
            "synthetic seekr index."
        )
    )
    parser.add_argument(
        "--chunks",
        type=_positive_int,
        default=DEFAULT_CHUNKS,
        help=f"number of synthetic chunks to index (default: {DEFAULT_CHUNKS})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="use this empty/new database path instead of a temporary store",
    )
    parser.add_argument(
        "--queries",
        type=_positive_int,
        default=DEFAULT_QUERIES,
        help=f"queries per mode (default: {DEFAULT_QUERIES})",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_int,
        default=1,
        help="timed rounds over the query set per mode (default: 1)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"embedding model to time if locally available (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_PATH,
        help=f"markdown output path (default: {RESULTS_PATH})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"deterministic synthetic-data seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--k",
        type=_positive_int,
        default=DEFAULT_K,
        help=f"hits requested per query (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--embed-backend",
        choices=EMBED_BACKENDS,
        default=DEFAULT_BACKEND,
        help=f"runtime for the query-embedding timing (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default=DEFAULT_DEVICE,
        help=f"device for the PyTorch backend (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--quant",
        choices=VECTOR_QUANTS,
        default=DEFAULT_QUANT,
        help=f"vector quantization mode in sqlite-vec (default: {DEFAULT_QUANT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning an exit status instead of raising on bad input."""
    args = build_parser().parse_args(argv)
    try:
        output = run_benchmark(
            chunks=args.chunks,
            db_path=args.db,
            queries=args.queries,
            repetitions=args.repetitions,
            model=args.model,
            out=args.out,
            seed=args.seed,
            k=args.k,
            quant=args.quant,
            embed_backend=args.embed_backend,
            device=args.device,
        )
    except BenchmarkError as exc:
        print(f"seekr latency benchmark: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports a clean error
        print(
            f"seekr latency benchmark failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(output.read_text(encoding="utf-8"), end="")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
