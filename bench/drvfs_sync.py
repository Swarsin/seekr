#!/usr/bin/env python3
"""Measure initial sync and no-op re-sync timing on a target directory.

Copies the packaged demo course N times into ``--target`` and benchmarks
:meth:`seekr.indexer.Indexer.sync` with ``FakeEmbedder(dim=64)`` while storing
the temporary SQLite database on Linux ``/tmp`` (mirroring a WSL setup where
``index.db`` lives on the Linux filesystem even when files live on ``/mnt/c``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def _bootstrap_src_path() -> None:
    """Allow running directly from a checkout without prior installation."""
    package_root = Path(__file__).resolve().parents[1] / "src"
    if (package_root / "seekr" / "__init__.py").is_file():
        value = str(package_root)
        if value not in sys.path:
            sys.path.insert(0, value)


@dataclass(frozen=True)
class SyncBenchmarkResult:
    """Measured file counts and timings from one sync benchmark run."""

    target: Path
    fstype: str
    copies: int
    files_added: int
    files_skipped: int
    files_unchanged_resync: int
    chunks: int
    initial_sync_s: float
    noop_resync_s: float


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _prepare_target(target: Path, copies: int) -> Path:
    import seekr

    resolved = target.expanduser().resolve()
    protected = {Path("/"), Path.home().resolve(), Path("/tmp").resolve()}
    if resolved in protected:
        raise ValueError(f"refusing to replace protected directory: {resolved}")

    if resolved.is_symlink():
        resolved.unlink()
    elif resolved.exists():
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()

    resolved.mkdir(parents=True, exist_ok=True)
    demo_src = seekr.demo_course_path()
    width = max(2, len(str(copies - 1)))
    for index in range(copies):
        shutil.copytree(demo_src, resolved / f"copy-{index:0{width}d}")
    return resolved


def run_sync_benchmark(target: Path, *, copies: int = 20) -> SyncBenchmarkResult:
    """Populate *target* with *copies* demo courses and time initial + no-op sync."""
    _bootstrap_src_path()
    from seekr import platform_info
    from seekr.config import Config
    from seekr.embed import FakeEmbedder
    from seekr.indexer import Indexer
    from seekr.store import Store

    if copies <= 0:
        raise ValueError("copies must be a positive integer")

    resolved_target = _prepare_target(target, copies)
    mount = platform_info.mount_for(resolved_target, platform_info.read_mounts())
    fstype = mount.fstype if mount is not None else "unknown"

    embedder = FakeEmbedder(dim=64)
    with tempfile.TemporaryDirectory(prefix="seekr-drvfs-sync-") as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        cfg = Config(roots=(resolved_target,), db_path=db_path, embed_model="fake")
        store = Store(
            db_path,
            embedder.model_name,
            embedder.dim,
            vector_quant=cfg.vector_quant,
        )
        try:
            initial_started = time.perf_counter()
            initial_report = Indexer(store, embedder, cfg).sync([resolved_target])
            initial_sync_s = time.perf_counter() - initial_started
            if initial_report.errors:
                raise RuntimeError(
                    f"initial sync reported errors: {initial_report.errors[:3]}"
                )

            stats = store.stats()

            resync_started = time.perf_counter()
            resync_report = Indexer(store, embedder, cfg).sync([resolved_target])
            noop_resync_s = time.perf_counter() - resync_started
            if resync_report.errors:
                raise RuntimeError(
                    f"no-op resync reported errors: {resync_report.errors[:3]}"
                )
        finally:
            store.close()

    return SyncBenchmarkResult(
        target=resolved_target,
        fstype=fstype,
        copies=copies,
        files_added=initial_report.added,
        files_skipped=initial_report.skipped,
        files_unchanged_resync=resync_report.unchanged,
        chunks=int(stats["chunks"]),
        initial_sync_s=initial_sync_s,
        noop_resync_s=noop_resync_s,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ``drvfs_sync.py``."""
    parser = argparse.ArgumentParser(
        description="Measure Indexer.sync initial and no-op resync latency on a target directory."
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Target directory to populate with copied demo courses.",
    )
    parser.add_argument(
        "--copies",
        type=_positive_int,
        default=20,
        help="Number of copies of the packaged demo course to create (default: 20).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sync benchmark CLI."""
    args = build_parser().parse_args(argv)
    try:
        result = run_sync_benchmark(args.target, copies=args.copies)
    except Exception as exc:  # noqa: BLE001 - clean CLI error reporting
        print(f"drvfs_sync: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"target: {result.target}")
    print(f"fstype: {result.fstype}")
    print(f"copies: {result.copies}")
    print(f"files_added: {result.files_added}")
    print(f"files_skipped: {result.files_skipped}")
    print(f"files_unchanged_resync: {result.files_unchanged_resync}")
    print(f"chunks: {result.chunks}")
    print(f"initial_sync_s: {result.initial_sync_s:.4f}")
    print(f"noop_resync_s: {result.noop_resync_s:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
