"""Performance smoke test for warm hybrid retrieval.

The real vector backend is required: numpy, sqlite-vec, FTS5 and loadable
SQLite extensions.  The test skips with a specific reason when that backend is
missing; it never replaces the measurement with a fake pass.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

try:  # pragma: no cover - pytest is an optional development dependency
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

_BENCH_PATH = Path(__file__).resolve().parents[1] / "bench" / "latency.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("seekr_bench_latency", _BENCH_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark module from {_BENCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


def _slow(test_class: type[unittest.TestCase]) -> type[unittest.TestCase]:
    if pytest is None:
        return test_class
    return pytest.mark.slow(test_class)


def _vector_backend_reason() -> str | None:
    if importlib.util.find_spec("numpy") is None:
        return "real vector backend unavailable: numpy is not installed"
    if importlib.util.find_spec("sqlite_vec") is None:
        return "real vector backend unavailable: sqlite-vec is not installed"
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        return (
            "real vector backend unavailable: this Python cannot load SQLite "
            "extensions"
        )

    conn = sqlite3.connect(":memory:")
    try:
        options = [str(row[0]) for row in conn.execute("PRAGMA compile_options")]
        if "ENABLE_FTS5" not in options:
            return "real vector backend unavailable: SQLite lacks FTS5"
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.execute("SELECT vec_version()").fetchone()
        except (ImportError, OSError, RuntimeError, sqlite3.Error) as exc:
            return f"real vector backend unavailable: sqlite-vec cannot load: {exc}"
        finally:
            try:
                conn.enable_load_extension(False)
            except (AttributeError, sqlite3.Error):
                pass
    finally:
        conn.close()
    return None


_SKIP_REASON = _vector_backend_reason()


class PercentileTests(unittest.TestCase):
    """Fast checks for the benchmark's own statistics helper."""

    def test_percentile_interpolates(self) -> None:
        self.assertAlmostEqual(bench.percentile([1.0, 2.0, 3.0, 4.0], 0.50), 2.5)
        self.assertAlmostEqual(bench.percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)

    def test_percentile_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            bench.percentile([], 0.95)

    def test_parser_accepts_backend_and_device(self) -> None:
        args = bench.build_parser().parse_args(["--embed-backend", "torch", "--device", "cuda"])
        self.assertEqual((args.embed_backend, args.device), ("torch", "cuda"))
        defaults = bench.build_parser().parse_args([])
        self.assertEqual((defaults.embed_backend, defaults.device), ("onnx", "cpu"))
        with self.assertRaises(SystemExit):
            bench.build_parser().parse_args(["--embed-backend", "tf"])

    def test_query_embedding_uses_selected_backend(self) -> None:
        from unittest import mock

        built: list[tuple[str, dict[str, object]]] = []

        class _Recorder:
            def __init__(self, name: str, **kwargs: object) -> None:
                built.append((name, kwargs))

            def embed_query(self, _text: str) -> list[float]:
                return [0.0]

        queries = [bench.Query(text="alpha", vector=None)]
        with (
            mock.patch("seekr.onnx_backend.OnnxEmbedder", _Recorder),
            mock.patch("seekr.embed.SentenceTransformerEmbedder", _Recorder),
        ):
            samples, note = bench.measure_query_embedding(
                queries, "m", embed_backend="onnx", device="cpu"
            )
            self.assertIsNone(note)
            self.assertEqual(len(samples), 1)
            bench.measure_query_embedding(queries, "m", embed_backend="torch", device="cuda")
        self.assertEqual(built, [("m", {}), ("m", {"device": "cuda"})])

    def test_render_results_includes_optional_indexing_and_memory(self) -> None:
        samples = {"keyword": [1.0, 2.0], "vector": [3.0, 4.0], "hybrid": [5.0, 6.0]}
        without = bench.render_results(
            chunk_count=400,
            query_count=2,
            repetitions=1,
            mode_samples=samples,
            embedding_samples=None,
            embedding_note="not measured",
            model_name="fake",
            notes=[("Python", "3.12")],
        )
        self.assertNotIn("## Indexing and memory", without)

        rendered = bench.render_results(
            chunk_count=400,
            query_count=2,
            repetitions=1,
            mode_samples=samples,
            embedding_samples=None,
            embedding_note="not measured",
            model_name="fake",
            notes=[("Python", "3.12")],
            index_seconds=2.0,
            peak_rss_mib=128.5,
        )
        self.assertIn("## Indexing and memory", rendered)
        self.assertIn("Indexing throughput (docs/sec) | 5.0", rendered)
        self.assertIn("Indexing throughput (chunks/sec) | 200.0", rendered)
        self.assertIn("Peak process RSS | 128.5 MiB", rendered)


@_slow
@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class PerfSmokeTests(unittest.TestCase):
    """A loose guard, not the 100 ms product goal."""

    def test_perf_smoke(self) -> None:
        from seekr.store import Store

        with tempfile.TemporaryDirectory(prefix="seekr-perf-") as temp_dir:
            store = Store(
                Path(temp_dir) / "index.db",
                "perf-smoke",
                bench.DIM,
            )
            try:
                bench.populate_store(store, 5_000, dim=bench.DIM, seed=42)
                self.assertEqual(store.stats()["chunks"], 5_000)

                queries = bench.make_queries(50, dim=bench.DIM, seed=43)
                samples = bench.measure_mode(
                    store,
                    queries,
                    "hybrid",
                    k=10,
                    repetitions=1,
                    dim=bench.DIM,
                )
            finally:
                store.close()

        self.assertEqual(len(samples), 50)
        p95_ms = bench.percentile(samples, 0.95)
        self.assertLess(
            p95_ms,
            250.0,
            (
                f"hybrid retrieval p95 was {p95_ms:.1f} ms over {len(samples)} "
                "queries at 5,000 chunks; the smoke guard is 250 ms"
            ),
        )
