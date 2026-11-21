#!/usr/bin/env python3
"""Convert the public BEIR FiQA dataset into seekr's evaluation layout.

Data source
-----------
The primary archive is the public BEIR FiQA dataset published by the
UKP Lab at TU Darmstadt::

    https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip

Downloading is opt-in: importing this module has no side effects, and network
access happens only when this script is explicitly run without an existing
local archive/source.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from eval import beir_common
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval import beir_common

FIQA_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
)
FIQA_HF_URL = "https://huggingface.co/datasets/BeIR/fiqa/tree/main"

DATASET = beir_common.BeirDataset(
    name="fiqa",
    label="FiQA",
    url=FIQA_URL,
    hf_url=FIQA_HF_URL,
    script_name="beir_fiqa.py",
)

convert_beir = beir_common.convert_beir


def download_fiqa_archive(
    destination: Path,
    *,
    url: str = FIQA_URL,
    timeout: float = 60.0,
) -> Path:
    """Return a valid local FiQA archive, downloading it only if needed."""
    return beir_common.download_archive(
        DATASET, destination, url=url, timeout=timeout
    )


def extract_fiqa_zip(
    archive: Path,
    destination: Path,
    *,
    max_total_bytes: int = beir_common.MAX_EXTRACT_BYTES,
) -> None:
    """Selectively extract FiQA corpus/queries/test qrels."""
    beir_common.extract_zip(
        DATASET, archive, destination, max_total_bytes=max_total_bytes
    )


def prepare_fiqa(
    data_dir: Path,
    *,
    allow_download: bool = True,
    timeout: float = 60.0,
) -> Path:
    """Ensure converted FiQA data exists, downloading only when allowed."""
    return beir_common.prepare_dataset(
        DATASET, data_dir, allow_download=allow_download, timeout=timeout
    )


def main(argv: list[str] | None = None) -> int:
    """Download (opt-in), extract and convert FiQA for ``seekr eval``."""
    return beir_common.run_cli(DATASET, argv)


if __name__ == "__main__":
    raise SystemExit(main())
