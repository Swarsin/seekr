# seekr latency benchmark

Measured 2026-11-19 02:23:05 UTC by `bench/latency.py`. Only measured values appear here.

## Setup

- Synthetic corpus: 20,000 chunks in 500 documents, normalised 384-dimensional float32 vectors.
- Queries: 50 per mode, 1 timed round(s) over the set.
- Each mode received one untimed warm-up after the index was fully written.
- Retrieval timing covers `Searcher.get_hits`: SQLite FTS5/vec0 retrieval, RRF, per-file capping and hit construction.
- Query vectors were precomputed; no embedding-model inference is included in the retrieval timings.

| mode | p50 (ms) | p95 (ms) | samples |
| --- | ---: | ---: | ---: |
| keyword | 1.41 | 1.54 | 50 |
| vector | 12.04 | 13.05 | 50 |
| hybrid | 14.12 | 14.53 | 50 |

## Query embedding

Model: `BAAI/bge-small-en-v1.5` (real local model, measured separately from retrieval).

| statistic | value |
| --- | ---: |
| p50 | 10.17 ms |
| p95 | 11.02 ms |
| samples | 50 |

## Indexing and memory

| metric | value |
| --- | ---: |
| Store populate time | 2.09 s |
| Indexing throughput (docs/sec) | 239.6 |
| Indexing throughput (chunks/sec) | 9582.5 |
| Peak process RSS | 1327.5 MiB |

## Environment

| component | value |
| --- | --- |
| Python | 3.12.15 |
| Platform | Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.43 |
| CPU | AMD Ryzen 9 3900X 12-Core Processor (24 logical CPUs) |
| SQLite | 3.53.1 |
| sqlite-vec | v0.1.9 |
| NumPy | 2.5.5 |
| Accelerator | CUDA (NVIDIA GeForce RTX 3070) |
