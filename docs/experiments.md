# seekr Experiments & Benchmark Log

All measurements recorded in this file come from actual runs on this repository and machine. Negative results are logged alongside improvements. Unless noted otherwise, the keep threshold for retrieval-quality changes in Phase 1 is an average `ndcg@10` improvement of at least `+0.0100` (`+1.00` point) across the active evaluation datasets with no single-dataset regression exceeding `-0.0050` (`-0.50` points).

## Environment

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 9 3900X 12-Core Processor (24 logical CPUs) |
| GPU | NVIDIA GeForce RTX 3070, 8192 MiB (CUDA active) |
| System RAM | 31 GiB |
| Platform | Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.43 |
| Python | 3.12.15 |
| SQLite | 3.53.1 |
| sqlite-vec | v0.1.9 |
| PyTorch | 2.15.0+cu132 |
| sentence-transformers | 6.3.0 |
| NumPy | 2.5.5 |
| PyMuPDF | 2.1 |
| tree-sitter-language-pack | 1.25.2 |

## Datasets

| Dataset | Corpus Docs | Judged Queries | Source |
| --- | ---: | ---: | --- |
| BEIR SciFact | 5,183 | 300 | `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip` |
| BEIR NFCorpus | 3,633 | 323 | `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip` |
| BEIR FiQA | 57,638 | 648 | `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip` |

> **Note on FiQA:** `eval/beir_fiqa.py` was implemented and the FiQA dataset (57,638 documents, 648 judged queries) was downloaded and converted into `eval/data/fiqa/`. At the baseline `Indexer.sync` rate of 59.88 docs/sec (1 file per embedding call and SQLite commit), indexing FiQA alone extrapolates to ~16 minutes (~18–20 minutes including evaluation queries), so FiQA evaluation is deferred until Phase 2 indexing batching is implemented.

## Phase 0: Baseline Fixes & Verification

Before taking baseline measurements, four baseline implementation/test issues found in the initial run were resolved:

1. **CLI help argument metavars (`7b27053`)**: Typer 0.29.2 rendered lowercase placeholders (`{query}`, `[paths]...`, `{target}`), failing 4 subtests in `tests/test_cli_wiring.py`. Added explicit uppercase `metavar` (`PATHS`, `TARGET`, `QUERY`) in `src/seekr/cli_index.py`, `src/seekr/cli_open.py`, and `src/seekr/cli_search.py`.
2. **`sqlite-vec` extension loading in benchmark and perf smoke (`88b07b8`)**: `bench/latency.py` and `tests/test_perf_smoke.py` called `sqlite_vec.load(conn)` without `conn.enable_load_extension(True)`, causing `PerfSmokeTests::test_perf_smoke` to skip (`not authorized`) and blocking `bench/latency.py`. Added `enable_load_extension(True)` / `False` cleanup to match `src/seekr/store.py`.
3. **Embedding dimension deprecation warning (`2c68270`)**: Updated `src/seekr/embed.py` to prefer `model.get_embedding_dimension()` over `get_sentence_embedding_dimension()`, eliminating `FutureWarning` in `sentence-transformers 6.3.0`.
4. **Slow cross-encoder test fixture (`84eb3ea`)**: When `SEEKR_TEST_RERANK_MODEL_DIR` pointed to the real cached `cross-encoder/ms-marco-MiniLM-L-6-v2` model, `tests/test_rerank.py::RealCrossEncoderTests::test_cross_encoder_prefers_the_relevant_passage` failed (`1 != 0`). Direct probing showed `CrossEncoder.predict` works properly (`+8.6071` on MS MARCO Berlin population, `+7.1583` when query mentions backpropagation, `+4.1546` when passage mentions neural networks), whereas the original test query (`"How do neural networks learn from their mistakes?"`) shared no lexical/subword cues with passage 0 (`"Backpropagation computes gradients of the loss with respect to each weight."`), causing all 3 passages to receive saturated negative logits (`-11.3124`, `-11.3051`, `-11.3349`). Updated passage 0 to explicitly reference neural networks learning from errors via backpropagation while keeping the strict `argmax == 0` assertion.

Verification after Phase 0 fixes: `ruff check .` clean, `pytest -m "not slow"` 422 passed / 5 skipped (4 uncached `tree-sitter-language-pack 1.25.2` Python AST tests, 1 `MissingVectorBackendTests`), `pytest -m slow` 5 passed / 0 skipped.

## Phase 0: Baseline Retrieval Quality

Default config: `embed_model = "BAAI/bge-small-en-v1.5"` (384-d float32, L2-normalised, query prefix `"Represent this sentence for searching relevant passages: "`), `rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"` (top 30 fused candidates), `chunk_target_words = 300`, `chunk_max_words = 450`, `chunk_overlap_words = 50`, FTS5 `porter unicode61`, RRF `k = 60`, candidate depth `max(50, 5*k) = 50`.

### BEIR SciFact (`bench/scifact-results.md`, 5,183 docs, 300 queries, wall time 156.97 s)

| mode | recall@5 | recall@10 | mrr@10 | ndcg@10 | p50_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.7387 | 0.8078 | 0.6547 | 0.6866 | 14.21 |
| vector | 0.7590 | 0.8379 | 0.6853 | 0.7159 | 23.94 |
| hybrid | 0.7806 | 0.8701 | 0.6962 | 0.7318 | 32.54 |
| hybrid+rerank | 0.7499 | 0.8322 | 0.6669 | 0.7010 | 119.88 |

### BEIR NFCorpus (`bench/nfcorpus-results.md`, 3,633 docs, 323 queries, wall time 119.89 s)

| mode | recall@5 | recall@10 | mrr@10 | ndcg@10 | p50_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.1228 | 0.1553 | 0.5224 | 0.3214 | 3.26 |
| vector | 0.1267 | 0.1614 | 0.5316 | 0.3427 | 22.35 |
| hybrid | 0.1376 | 0.1733 | 0.5694 | 0.3642 | 24.48 |
| hybrid+rerank | 0.1389 | 0.1695 | 0.5842 | 0.3569 | 106.30 |

### Baseline Cross-Dataset Summary (Macro-Average `ndcg@10`)

| mode | SciFact ndcg@10 | NFCorpus ndcg@10 | Macro-Avg ndcg@10 |
| --- | ---: | ---: | ---: |
| keyword | 0.6866 | 0.3214 | 0.5040 |
| vector | 0.7159 | 0.3427 | 0.5293 |
| hybrid | 0.7318 | 0.3642 | 0.5480 |
| hybrid+rerank | 0.7010 | 0.3569 | 0.5290 |

Key baseline observation: `hybrid` outperforms both `keyword` (`+4.40` pts avg) and `vector` (`+1.87` pts avg) across both datasets, whereas replacing RRF scores directly with `cross-encoder/ms-marco-MiniLM-L-6-v2` logits (`hybrid+rerank`) degrades `ndcg@10` on both SciFact (`-3.08` pts) and NFCorpus (`-0.73` pts).

## Phase 0: Baseline Latency, Indexing Throughput & Peak RAM

### Synthetic 20,000-Chunk Benchmark (`bench/latency.py --chunks 20000`, `bench/latency-results.md`)

- Setup: 20,000 chunks across 500 synthetic documents, 384-d float32 vectors, 50 queries per mode, 1 untimed warm-up per mode. Retrieval timings exclude query embedding inference.

| Mode / Metric | p50 | p95 | Samples / Notes |
| --- | ---: | ---: | --- |
| keyword retrieval | 1.41 ms | 1.54 ms | 50 queries |
| vector retrieval | 12.04 ms | 13.05 ms | 50 queries |
| hybrid retrieval | 14.12 ms | 14.53 ms | 50 queries |
| query embedding (`bge-small-en-v1.5`, CUDA) | 10.17 ms | 11.02 ms | 50 queries |

| Store & Memory Metric | Measured Value |
| --- | ---: |
| Synthetic store populate time (20,000 chunks / 500 docs) | 2.09 s |
| Synthetic store throughput (docs/sec) | 239.6 |
| Synthetic store throughput (chunks/sec) | 9582.5 |
| Peak process RSS (`bench/latency.py` incl. CUDA model) | 1327.5 MiB |

### Real End-to-End Indexing (BEIR SciFact, `BAAI/bge-small-en-v1.5` on RTX 3070 CUDA)

Measured in a fresh subprocess running `SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")` + `Store` + `Indexer.sync(...)` over `eval/data/scifact/corpus`:

| Metric | Measured Value |
| --- | ---: |
| Documents indexed | 5,183 (0 errors, 0 warnings) |
| Chunks produced | 5,820 |
| Model load time | 5.180 s |
| `Indexer.sync` time | 86.555 s |
| Total wall time (load + sync) | 91.740 s |
| `Indexer.sync` throughput (docs/sec) | 59.88 |
| `Indexer.sync` throughput (chunks/sec) | 67.24 |
| Total throughput (docs/sec) | 56.50 |
| SQLite DB size | 24.17 MiB |
| Peak process RSS (`ru_maxrss`) | 1411.4 MiB |
