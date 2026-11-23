#!/usr/bin/env python3
"""Measure the on-disk size of a clean seekr-search install.

Builds a wheel from this checkout (or uses ``--wheel``), installs it with its
default dependencies only into a fresh virtual environment, and prints the
size of that environment's site-packages in MB. Pass ``--extra torch`` to
measure an install with the PyTorch extra. Requires ``uv`` or the standard
``venv`` + ``pip`` tooling and network access to PyPI.

    python bench/install_size.py
    python bench/install_size.py --keep /tmp/seekr-minimal   # keep the venv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def _build_wheel(out_dir: Path) -> Path:
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "build", "--wheel", "--out-dir", str(out_dir), str(ROOT)], check=True)
    else:
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir), str(ROOT)],
            check=True,
        )
    wheels = sorted(out_dir.glob("seekr_search-*.whl"))
    if not wheels:
        raise SystemExit("wheel build produced no seekr_search wheel")
    return wheels[-1]


def _install(venv: Path, wheel: Path, extra: str | None) -> Path:
    target = f"{wheel}[{extra}]" if extra else str(wheel)
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "venv", "--quiet", "--python", sys.executable, str(venv)], check=True)
        python = venv / "bin" / "python"
        subprocess.run([uv, "pip", "install", "--quiet", "--python", str(python), target], check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / "bin" / "python"
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", target], check=True)
    purelib = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(purelib)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wheel", type=Path, default=None)
    parser.add_argument("--extra", default=None, help="optional extra to include, e.g. torch")
    parser.add_argument("--keep", type=Path, default=None, help="create the venv here and keep it")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="seekr-size-") as tmp:
        wheel = args.wheel or _build_wheel(Path(tmp) / "dist")
        venv = args.keep or Path(tmp) / "venv"
        site_packages = _install(venv, wheel, args.extra)
        size = _dir_size(site_packages)

    print(f"python: {sysconfig.get_python_version()}")
    print(f"wheel: {wheel.name}")
    print(f"extra: {args.extra or 'none'}")
    print(f"site_packages_mb: {size / 1_000_000:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
