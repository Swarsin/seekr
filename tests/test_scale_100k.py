"""Slow 100k-chunk scale test for real sqlite-vec int8 storage and retrieval."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

try:  # pragma: no cover - pytest is an optional development dependency
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

if pytest is not None:  # pragma: no cover
    pytestmark = pytest.mark.slow

_BENCH_PATH = Path(__file__).resolve().parents[1] / "bench" / "latency.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("seekr_bench_latency_100k", _BENCH_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark module from {_BENCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _vector_backend_reason() -> str | None:
    if importlib.util.find_spec("numpy") is None:
        return "real vector backend unavailable: numpy is not installed"
    if importlib.util.find_spec("sqlite_vec") is None:
        return "real vector backend unavailable: sqlite-vec is not installed"
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        return "real vector backend unavailable: this Python cannot load SQLite extensions"

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


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class Scale100kTests(unittest.TestCase):
    """Exercise indexing size and warm query latency at 100,000 chunks."""

    def test_100k_chunks_int8_latency_and_size(self) -> None:
        from seekr.store import Store

        bench = _load_benchmark_module()
        chunk_count = 100_000
        query_count = 30

        with tempfile.TemporaryDirectory(prefix="seekr-scale-100k-") as temp_dir:
            db_path = Path(temp_dir) / "index.db"
            store = Store(
                db_path,
                "scale-100k",
                bench.DIM,
                vector_quant="int8",
            )
            try:
                started = time.perf_counter()
                with store.transaction(write=True):
                    bench.populate_store(store, chunk_count, dim=bench.DIM, seed=101)
                populate_s = time.perf_counter() - started
                stats = store.stats()
                self.assertEqual(stats, {"documents": 2_500, "chunks": chunk_count})

                queries = bench.make_queries(query_count, dim=bench.DIM, seed=103)
                results = {
                    mode: bench.measure_mode(
                        store,
                        queries,
                        mode,
                        k=10,
                        repetitions=1,
                        dim=bench.DIM,
                    )
                    for mode in bench.MODES
                }
            finally:
                store.close()

            total_bytes = sum(
                Path(str(db_path) + suffix).stat().st_size
                for suffix in ("", "-wal", "-shm")
                if Path(str(db_path) + suffix).exists()
            )
            db_mib = total_bytes / (1024.0 * 1024.0)

        summary = {
            mode: (bench.percentile(samples, 0.50), bench.percentile(samples, 0.95))
            for mode, samples in results.items()
        }
        print(
            f"\n[scale-100k int8] populate={populate_s:.2f}s db={db_mib:.2f} MiB "
            + " ".join(
                f"{mode}={p50:.2f}/{p95:.2f}ms"
                for mode, (p50, p95) in summary.items()
            )
        )

        self.assertLess(db_mib, 250.0)
        self.assertLess(summary["keyword"][1], 100.0)
        self.assertLess(summary["hybrid"][1], 500.0)


if __name__ == "__main__":
    unittest.main()
