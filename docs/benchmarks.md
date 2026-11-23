# Benchmarks & Ablations

All numbers below were measured on an AMD Ryzen 9 3900X (24 logical CPUs), 31 GiB RAM, NVIDIA GeForce RTX 3070 (8 GiB, CUDA), Python 3.12.15, SQLite 3.53.1, `sqlite-vec v0.1.9`, PyTorch 2.15.0+cu132, and `sentence-transformers 6.3.0`. Full experiment logs live in [`experiments.md`](experiments.md).

## Retrieval quality (BEIR SciFact, NFCorpus, and FiQA)

Reproduce the dataset evaluations (converters download only when invoked without `--no-download`):

```bash
python eval/beir_scifact.py && seekr eval --root eval/data/scifact/corpus --queries eval/data/scifact/queries.jsonl --out bench/scifact-results.md
python eval/beir_nfcorpus.py && seekr eval --root eval/data/nfcorpus/corpus --queries eval/data/nfcorpus/queries.jsonl --out bench/nfcorpus-results.md
python eval/beir_fiqa.py && seekr eval --root eval/data/fiqa/corpus --queries eval/data/fiqa/queries.jsonl --out bench/fiqa-results.md
```

Measured with the current `int8` default (`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2` on top 30):

| Dataset | Mode | recall@5 | recall@10 | mrr@10 | ndcg@10 | p50_ms | Wall Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact (`5,183` docs, `300` q) | keyword | 0.7387 | 0.8078 | 0.6547 | 0.6866 | 14.20 | 91.38 s |
| SciFact | vector | 0.7623 | 0.8262 | 0.6822 | 0.7109 | 23.77 | 91.38 s |
| SciFact | **hybrid** | **0.7872** | **0.8712** | **0.6960** | **0.7320** | **31.26** | 91.38 s |
| SciFact | hybrid+rerank | 0.7532 | 0.8356 | 0.6673 | 0.7020 | 118.78 | 91.38 s |
| NFCorpus (`3,633` docs, `323` q) | keyword | 0.1228 | 0.1553 | 0.5224 | 0.3214 | 3.28 | 76.59 s |
| NFCorpus | vector | 0.1266 | 0.1626 | 0.5253 | 0.3410 | 20.87 | 76.59 s |
| NFCorpus | **hybrid** | **0.1372** | **0.1744** | **0.5666** | **0.3632** | **17.45** | 76.59 s |
| NFCorpus | hybrid+rerank | 0.1393 | 0.1693 | 0.5842 | 0.3570 | 105.20 | 76.59 s |
| FiQA (`57,638` docs, `648` q) | keyword | 0.2384 | 0.3044 | 0.3000 | 0.2447 | 72.13 | 501.65 s |
| FiQA | **vector** | **0.3771** | **0.4536** | **0.4818** | **0.3963** | **46.83** | 501.65 s |
| FiQA | hybrid | 0.3513 | 0.4412 | 0.4285 | 0.3589 | 118.78 | 501.65 s |
| FiQA | hybrid+rerank | 0.3780 | 0.4482 | 0.4446 | 0.3761 | 187.93 | 501.65 s |

3-dataset macro-average `ndcg@10`: `keyword 0.4176`, `vector 0.4827`, `hybrid 0.4847`, `hybrid+rerank 0.4784`. To evaluate your own workspace files, see [`eval.md`](eval.md) and the sample fixture under `eval/example/`.

## Retrieval ablations (single-variable)

Controlled single-variable sweeps on SciFact + NFCorpus (`hybrid` baseline: `0.7318` / `0.3642`, macro-avg `0.5480` on `float32`; keep threshold: `+1.00` pt macro-avg gain without `> 0.50` pt loss on either dataset):

| Variable | Variant | SciFact ndcg@10 | NFCorpus ndcg@10 | Macro-Avg | ΔAvg (pts) | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | `bge-small-en-v1.5`, 300/50, RRF k=60, `porter unicode61` | 0.7318 | 0.3642 | 0.5480 | 0.00 | Kept default |
| Chunking | 150 words / 50 overlap | 0.7223 | 0.3686 | 0.5455 | -0.25 | Reverted |
| Chunking | 300 words / 0 overlap | 0.7349 | 0.3655 | 0.5502 | +0.22 | Not adopted (< +1.0 pt) |
| Chunking | 450 words / 50 overlap | 0.7305 | 0.3649 | 0.5477 | -0.03 | Reverted |
| Headers | Prepend filename + heading path (BEIR IDs) | 0.7232 | 0.3635 | 0.5433 | -0.47 | Reverted |
| Fusion | RRF `k=10` | 0.7362 | 0.3662 | 0.5512 | +0.32 | Not adopted (< +1.0 pt) |
| Fusion | Min-max score linear fusion (`alpha_kw=0.5`) | 0.7357 | 0.3640 | 0.5499 | +0.19 | Not adopted (< +1.0 pt) |
| Rerank | Direct cross-encoder replacement (top 30) | 0.7010 | 0.3569 | 0.5289 | -1.91 | Opt-in (`--rerank`) |
| Rerank | RRF blend (first-stage hybrid + CE, top 30) | 0.7297 | 0.3641 | 0.5469 | -0.11 | Logged |
| Prefix | BGE query prefix OFF | 0.7414 | 0.3626 | 0.5520 | +0.40 | Reverted (NF vec -0.53) |
| FTS5 | `unicode61` without Porter stemmer | 0.7107 | 0.3555 | 0.5331 | -1.49 | Reverted |
| BM25 cols | `(header, body)` weights `2.0, 1.0` | 0.7340 | 0.3643 | 0.5492 | +0.12 | Not adopted (< +1.0 pt) |
| Model | `BAAI/bge-base-en-v1.5` (768-d, 102.9 chunks/s) | 0.7411 | 0.3745 | 0.5578 | +0.98 | Opt-in config |
| Model | `sentence-transformers/all-MiniLM-L6-v2` (384-d) | 0.7185 | 0.3530 | 0.5357 | -1.23 | Rejected |
| Model | `intfloat/e5-small-v2` (`query:`/`passage:`) | 0.7321 | 0.3510 | 0.5415 | -0.65 | Rejected |

## Latency, indexing throughput, quantization, and 100k scale

```bash
python bench/latency.py --chunks 20000
```

**Before vs. after performance engineering & `int8` quantization:**

| Benchmark | Baseline (`float32`) | Current (`int8`, batched) | Change |
| --- | ---: | ---: | --- |
| SciFact fresh `Indexer.sync` (5,183 docs / 5,820 chunks, CUDA) | 86.555 s (67.2 chunks/s) | **26.432 s** (220.2 chunks/s) | **3.27x faster** |
| SciFact unchanged re-sync (5,183 files) | 2.269 s | **0.122 s** | **18.6x faster** |
| Cold `seekr daemon` time to `ping` ready | 7.282 s | **0.349 s** | **20.9x faster** |
| SciFact index DB size | 24.17 MiB | **17.41 MiB** | **-28.0%** |
| 20k synthetic `keyword` p50 / p95 (excl. embed) | 1.41 / 1.54 ms | **1.35 / 1.51 ms** | — |
| 20k synthetic `vector` p50 / p95 (excl. embed) | 12.04 / 13.05 ms | **8.48 / 8.98 ms** | **1.42x / 1.45x faster** |
| 20k synthetic `hybrid` p50 / p95 (excl. embed) | 14.12 / 14.53 ms | **10.96 / 11.39 ms** | **1.29x / 1.28x faster** |
| 20k synthetic populate time / DB size (`bench/latency.py`) | 2.09 s / 44.88 MiB (`float32` opt.) | **1.83 s / 22.35 MiB** | **2.01x smaller DB** |
| 20k peak process RSS (incl. CUDA model) | 1327.5 MiB | **1330.5 MiB** | Flat |

Query embedding latency (`BAAI/bge-small-en-v1.5`, CUDA, 50 warm queries) measured `10.17 / 11.02 ms` (`bench/latency-results.md` Phase 0) and `18.32 / 19.65 ms` (Phase 2 run; higher tail variance across isolated runs).

**100,000-chunk scale test (`tests/test_scale_100k.py`, `@pytest.mark.slow`, `int8`):** `2,500` documents × `40` chunks (`100,000` chunks) populated in `9.11 s` (`10,977 chunks/sec`), producing a `110.45 MiB` database with warm query latencies `keyword 4.35 / 5.00 ms`, `vector 41.39 / 42.55 ms`, and `hybrid 44.37 / 46.38 ms` (p50 / p95 over 30 queries).

## ONNX Runtime vs PyTorch backend

The default install runs both models with ONNX Runtime; PyTorch is the optional `seekr-search[torch]` extra. Equivalence was measured before switching the default (full tables in [`experiments.md`](experiments.md#onnx-runtime-backend-equivalence-default-since-020)):

| Check | PyTorch | ONNX Runtime |
| --- | ---: | ---: |
| Embedding cosine vs PyTorch, 1,000 chunks (mean / min) | — | 1.000000 / 1.000000 |
| Reranker Kendall τ vs PyTorch, top 30 × 50 queries (mean / min) | — | 1.000000 / 1.000000 |
| SciFact hybrid `ndcg@10` | 0.7320 | 0.7320 |
| NFCorpus hybrid `ndcg@10` | 0.3632 | 0.3632 |
| Embedding 1,000 chunks on CPU | 59.44 s | 46.31 s |
| Default install size (site-packages, `bench/install_size.py`) | several GB with CUDA PyTorch | 349.1 MB |
| Model download (`seekr models pull`, embedding / reranker) | safetensors + tokenizer files | 128 MB / 88 MB |

## Windows-drive sync overhead under WSL2 (`bench/drvfs_sync.py`)

Measured with `bench/drvfs_sync.py` across `20` copies of the packaged demo course (`500` indexed files, `1,500` chunks, `20` skipped `queries.jsonl` files) using `FakeEmbedder(dim=64)` and an isolated SQLite database under Linux `/tmp` (matching real WSL setups where `index.db` stays in `~/.local/share/seekr` even when course folders live on `/mnt/c`):

```bash
python bench/drvfs_sync.py --target /mnt/c/Users/<you>/seekr-bench --copies 20
python bench/drvfs_sync.py --target /tmp/seekr-bench --copies 20
```

| Target | Filesystem (`mount_for`) | Copies | Indexed files | Chunks | Initial sync | No-op re-sync |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `/mnt/c/Users/<you>/seekr-bench` | `9p` (Windows drive / DrvFS) | 20 | 500 | 1,500 | 22.1450 s | 4.2414 s |
| `/tmp/seekr-bench` | `tmpfs` (Linux `/tmp`) | 20 | 500 | 1,500 | 1.0075 s | 0.0174 s |

Using `FakeEmbedder(dim=64)` isolates filesystem walk/stat/extract overhead from neural model inference. Cross-OS `9p`/DrvFS metadata and read calls make initial extraction ~`22x` slower and no-op `stat` re-syncs ~`244x` slower than Linux `/tmp`, while keeping `index.db` on the Linux filesystem keeps no-op re-syncs for `500` files under `5 s`. Because `inotify` events do not propagate reliably across `/mnt/c`, `seekr daemon` automatically selects polling (`PollingObserver`, default `poll_interval_seconds = 15`) for paths on `9p`, `drvfs`, and network mounts.

## Design Decisions & What Did Not Work

### What works well
- **RRF (`k=60`) over FTS5 + `sqlite-vec`:** Calibrates unbounded BM25 scores and L2 vector distances without score-scale fitting, lifting macro-average `ndcg@10` on SciFact+NFCorpus from `0.5040` (keyword) / `0.5293` (vector) to `0.5480`.
- **`int8` vector quantization by default (`min(500, max(127, 25*sqrt(dim)))`):** Cuts 20k DB size in half (`44.88 -> 22.35 MiB`), speeds vector queries by ~`1.67x` over post-opt `float32`, and loses only `0.06` pts macro-avg `ndcg@10` on SciFact+NFCorpus (`0.5480 -> 0.5474`, and `0.5476` in full post-Phase-2 `seekr eval`).
- **Batched indexing + deferred daemon prewarm:** Batching GPU embedding and SQLite commits across 64 files / 128 chunks cut SciFact cold sync from `86.555 s` to `26.432 s`; precomputing stat/ignore fast paths cut no-op syncs by `18.6x`; binding the daemon socket before background embedder warmup cut cold daemon readiness by `20.9x` (`7.282 s -> 0.349 s`).
- **Chunk-hash vector reuse + zero-model `seekr similar`:** Identical chunk text never re-embeds across edits or renames, and `seekr similar` on an indexed file pools stored DB vectors without loading PyTorch.

### What did not work (and was reverted or kept opt-in)
- **Binary (`bit`) quantization:** Regressed hybrid `ndcg@10` by `-1.93` pts macro-avg (`0.5287`) and vector-only by `-7.41` pts (`0.4552`), failing the `< 1` pt loss gate.
- **Replacing RRF order directly with `ms-marco-MiniLM-L-6-v2`:** Hurt `ndcg@10` on scientific/medical corpora (SciFact `-3.08` pts, NFCorpus `-0.73` pts) due to domain mismatch, though it helped FiQA (`+1.72` pts over `hybrid`). Remains opt-in via `--rerank`.
- **Prepending filename/heading headers unconditionally:** Regressed SciFact+NFCorpus (`-0.47` pts macro-avg) because ID-style filenames add lexical noise and break path-independent `text_sha` deduplication.
- **Dropping Porter stemming in FTS5:** Dropped macro-avg `hybrid` `ndcg@10` by `-1.49` pts (`0.5480 -> 0.5331`).
- **Heavier defaults (`bge-base-en-v1.5`) or lighter models (`all-MiniLM-L6-v2`, `e5-small-v2`):** `bge-base-en-v1.5` gained `+0.98` pts hybrid at `2.5x` slower indexing and `+37%` DB size (kept configurable); `MiniLM` (`-1.23` pts) and `e5-small-v2` (`-0.65` pts) trailed `bge-small-en-v1.5`.
