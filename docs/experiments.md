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

## Phase 1: Retrieval Quality Experiments

All Phase 1 experiments were run as controlled single-variable comparisons on both BEIR SciFact (5,183 docs, 300 queries) and BEIR NFCorpus (3,633 docs, 323 queries) using the real `extract`, `chunk`, `Store`, `Searcher`, `SentenceTransformerEmbedder`, `CrossEncoderReranker`, and `evaluation` modules. A batch-indexed parity check matched Phase 0 `seekr eval` baselines to 4 decimal places on every metric and mode (`hybrid` `ndcg@10`: SciFact `0.7318`, NFCorpus `0.3642`, macro-avg `0.5480`).

Keep rule: retain a change only if it improves macro-average `ndcg@10` by at least `+1.00` point (`+0.0100`) without regressing either dataset by more than `0.50` points (`-0.0050`).

### Exp 1.1: Chunk Size & Overlap Sweep (`BAAI/bge-small-en-v1.5`, hybrid `ndcg@10`)

`chunk_max_words` was set to `max(450, chunk_target_words)`.

| Target / Overlap | SF Chunks | NF Chunks | SciFact ndcg@10 | NFCorpus ndcg@10 | Macro-Avg | ΔAvg (pts) | ΔMin (pts) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 150 / 0 | 10,066 | 7,349 | 0.7240 | 0.3658 | 0.5449 | -0.31 | -0.78 | Reverted |
| 150 / 50 | 11,292 | 8,631 | 0.7223 | 0.3686 | 0.5455 | -0.25 | -0.94 | Reverted |
| 150 / 100 | 14,804 | 11,772 | 0.7225 | 0.3669 | 0.5447 | -0.33 | -0.92 | Reverted |
| 300 / 0 | 5,811 | 4,127 | 0.7349 | 0.3655 | 0.5502 | +0.22 | +0.13 | Not adopted (< +1.00 pt) |
| **300 / 50 (baseline)** | **5,820** | **4,134** | **0.7318** | **0.3642** | **0.5480** | **0.00** | **0.00** | **Kept** |
| 300 / 100 | 5,842 | 4,147 | 0.7292 | 0.3647 | 0.5470 | -0.10 | -0.25 | Reverted |
| 450 / 0 | 5,256 | 3,674 | 0.7274 | 0.3653 | 0.5464 | -0.16 | -0.43 | Reverted |
| 450 / 50 | 5,257 | 3,674 | 0.7305 | 0.3649 | 0.5477 | -0.03 | -0.12 | Reverted |
| 450 / 100 | 5,260 | 3,675 | 0.7276 | 0.3650 | 0.5463 | -0.17 | -0.42 | Reverted |

### Exp 1.2: Contextual Chunk Headers (`BAAI/bge-small-en-v1.5`, hybrid `ndcg@10`)

Prepended `{filename}` (or `{stem}`) plus heading locator when present (`§A > B`) to the chunk text embedded and FTS-indexed. Also verified on `eval/example/` (8 Markdown files with headings), where all 3 variants scored `1.0000` across `keyword`, `vector`, and `hybrid`.

| Variant | SciFact ndcg@10 | NFCorpus ndcg@10 | Macro-Avg | ΔAvg (pts) | ΔMin (pts) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **none (baseline)** | **0.7318** | **0.3642** | **0.5480** | **0.00** | **0.00** | **Kept** |
| filename + heading | 0.7232 | 0.3635 | 0.5433 | -0.47 | -0.86 | Reverted |
| stem + heading | 0.7281 | 0.3630 | 0.5456 | -0.24 | -0.36 | Reverted |

Why it regressed on BEIR: BEIR corpus documents use opaque numeric/ID filenames (`10004123.txt`, `MED-1234.txt`) with no Markdown headings; prepending opaque IDs adds token noise and would also break path-independent chunk `text_sha` reuse across file renames.

### Exp 1.3: Fusion, Candidate Depth & Rerank Depth

#### RRF Constant `k` (limit=50, equal weights)

| Variant | SciFact ndcg@10 | NFCorpus ndcg@10 | Macro-Avg | ΔAvg (pts) | ΔMin (pts) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| k=10 | 0.7362 | 0.3662 | 0.5512 | +0.32 | +0.19 | Not adopted (< +1.00 pt) |
| k=20 | 0.7338 | 0.3643 | 0.5491 | +0.11 | +0.01 | Reverted |
| k=40 | 0.7307 | 0.3647 | 0.5477 | -0.03 | -0.10 | Reverted |
| **k=60 (baseline)** | **0.7318** | **0.3642** | **0.5480** | **0.00** | **0.00** | **Kept** |
| k=80 | 0.7313 | 0.3642 | 0.5477 | -0.02 | -0.05 | Reverted |
| k=100 | 0.7313 | 0.3641 | 0.5477 | -0.03 | -0.05 | Reverted |

#### Weighted RRF (limit=50)

| Variant | SciFact ndcg@10 | NFCorpus ndcg@10 | Macro-Avg | ΔAvg (pts) | ΔMin (pts) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| k=60, w_kw=0.2, w_vec=0.8 | 0.7318 | 0.3593 | 0.5456 | -0.24 | -0.50 | Reverted |
| k=60, w_kw=0.3, w_vec=0.7 | 0.7305 | 0.3608 | 0.5456 | -0.24 | -0.35 | Reverted |
| k=60, w_kw=0.4, w_vec=0.6 | 0.7288 | 0.3602 | 0.5445 | -0.35 | -0.40 | Reverted |
| **k=60, w_kw=0.5, w_vec=0.5** | **0.7318** | **0.3642** | **0.5480** | **0.00** | **0.00** | **Kept** |
| k=60, w_kw=0.6, w_vec=0.4 | 0.7241 | 0.3635 | 0.5438 | -0.42 | -0.76 | Reverted |
| k=60, w_kw=0.7, w_vec=0.3 | 0.7223 | 0.3625 | 0.5424 | -0.56 | -0.95 | Reverted |
| k=20, w_kw=0.2, w_vec=0.8 | 0.7302 | 0.3577 | 0.5440 | -0.40 | -0.65 | Reverted |
| k=20, w_kw=0.3, w_vec=0.7 | 0.7323 | 0.3645 | 0.5484 | +0.04 | +0.02 | Reverted |
| k=20, w_kw=0.4, w_vec=0.6 | 0.7335 | 0.3647 | 0.5491 | +0.11 | +0.05 | Reverted |
| k=20, w_kw=0.5, w_vec=0.5 | 0.7338 | 0.3643 | 0.5491 | +0.11 | +0.01 | Reverted |
| k=20, w_kw=0.6, w_vec=0.4 | 0.7266 | 0.3630 | 0.5448 | -0.32 | -0.51 | Reverted |
| k=20, w_kw=0.7, w_vec=0.3 | 0.7187 | 0.3564 | 0.5376 | -1.04 | -1.31 | Reverted |

#### Score-Normalized Linear Fusion (per-query min-max `-bm25` and cosine similarity `1 - 0.5*d^2`, limit=50)

| Variant | SciFact ndcg@10 | NFCorpus ndcg@10 | Macro-Avg | ΔAvg (pts) | ΔMin (pts) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| alpha_kw=0.2 | 0.7298 | 0.3536 | 0.5417 | -0.63 | -1.07 | Reverted |
| alpha_kw=0.3 | 0.7306 | 0.3596 | 0.5451 | -0.29 | -0.46 | Reverted |
| alpha_kw=0.4 | 0.7356 | 0.3629 | 0.5492 | +0.12 | -0.14 | Reverted |
| alpha_kw=0.5 | 0.7357 | 0.3640 | 0.5499 | +0.19 | -0.02 | Reverted |
| alpha_kw=0.6 | 0.7284 | 0.3592 | 0.5438 | -0.42 | -0.50 | Reverted |
| alpha_kw=0.7 | 0.7200 | 0.3529 | 0.5365 | -1.15 | -1.17 | Reverted |

#### Candidate Depth Per Retriever (`limit`, RRF k=60)

| Limit | SciFact ndcg@10 | NFCorpus ndcg@10 | Macro-Avg | ΔAvg (pts) | ΔMin (pts) | Outcome |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 0.7326 | 0.3632 | 0.5479 | -0.01 | -0.11 | Reverted |
| **50 (baseline)** | **0.7318** | **0.3642** | **0.5480** | **0.00** | **0.00** | **Kept** |
| 100 | 0.7305 | 0.3651 | 0.5478 | -0.02 | -0.13 | Reverted |
| 150 | 0.7316 | 0.3653 | 0.5484 | +0.04 | -0.02 | Reverted |
| 200 | 0.7316 | 0.3658 | 0.5487 | +0.07 | -0.02 | Reverted |

#### Rerank Depth (`20/30/50`) and Rerank Fusion (`cross-encoder/ms-marco-MiniLM-L-6-v2` vs `hybrid` baseline)

| Variant | SciFact ndcg@10 | NFCorpus ndcg@10 | Macro-Avg | ΔAvg vs hybrid (pts) | SF p50 (ms) | NF p50 (ms) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| replace, depth=20 | 0.7046 | 0.3615 | 0.5331 | -1.49 | 64.75 | 61.94 | Reverted |
| **replace, depth=30 (baseline)** | **0.7010** | **0.3569** | **0.5289** | **-1.91** | **91.82** | **88.90** | **Kept (opt-in `--rerank`)** |
| replace, depth=50 | 0.7008 | 0.3516 | 0.5262 | -2.18 | 125.72 | 123.51 | Reverted |
| rrf(hybrid, ce), depth=20 | 0.7297 | 0.3629 | 0.5463 | -0.17 | 65.27 | 62.36 | Logged (recovers +1.74 pts vs replace@30) |
| rrf(hybrid, ce), depth=30 | 0.7297 | 0.3641 | 0.5469 | -0.11 | 92.78 | 89.64 | Logged (recovers +1.80 pts vs replace@30) |
| rrf(hybrid, ce), depth=50 | 0.7286 | 0.3610 | 0.5448 | -0.32 | 125.48 | 123.33 | Logged |

### Exp 1.4: Query Prefix, FTS5 Tokenizer & BM25 Column Weighting

#### BGE Query Prefix (`BAAI/bge-small-en-v1.5`)

| Variant | SF vec | NF vec | SF hyb | NF hyb | Macro-Avg hyb | ΔAvg hyb (pts) | ΔMin hyb (pts) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **prefix_on (baseline)** | **0.7159** | **0.3427** | **0.7318** | **0.3642** | **0.5480** | **0.00** | **0.00** | **Kept** |
| prefix_off | 0.7205 | 0.3374 | 0.7414 | 0.3626 | 0.5520 | +0.40 | -0.17 | Reverted (< +1.00 pt; vector NF -0.53) |

#### FTS5 Tokenizer

| Tokenizer | SF kw | NF kw | SF hyb | NF hyb | Macro-Avg hyb | ΔAvg hyb (pts) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **porter unicode61 (baseline)** | **0.6866** | **0.3214** | **0.7318** | **0.3642** | **0.5480** | **0.00** | **Kept** |
| unicode61 (no stemming) | 0.6639 | 0.3052 | 0.7107 | 0.3555 | 0.5331 | -1.49 | Reverted (proves Porter stemming helps) |
| porter unicode61 remove_diacritics 2 | 0.6866 | 0.3214 | 0.7318 | 0.3642 | 0.5480 | 0.00 | Not adopted (identical) |
| porter unicode61 tokenchars '_' | 0.6866 | 0.3214 | 0.7318 | 0.3642 | 0.5480 | 0.00 | Not adopted (identical) |

#### BM25 2-Column Weighting (`header` = title/first paragraph, `body` = remainder)

| Weights `(header, body)` | SF kw | NF kw | SF hyb | NF hyb | Macro-Avg hyb | ΔAvg hyb (pts) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **1.0, 1.0 (baseline)** | **0.6866** | **0.3214** | **0.7318** | **0.3642** | **0.5480** | **0.00** | **Kept** |
| 2.0, 1.0 | 0.6832 | 0.3169 | 0.7340 | 0.3643 | 0.5492 | +0.12 | Reverted (< +1.00 pt) |
| 3.0, 1.0 | 0.6726 | 0.3081 | 0.7318 | 0.3622 | 0.5470 | -0.10 | Reverted |
| 5.0, 1.0 | 0.6607 | 0.2976 | 0.7260 | 0.3596 | 0.5428 | -0.52 | Reverted |

### Exp 1.5: Embedding Model Comparison (Isolated Subprocesses, `trust_remote_code=False`)

| Model | Dim | SF vec | NF vec | SF hyb | NF hyb | Macro-Avg hyb | ΔAvg hyb (pts) | SF emb (chunks/s) | Query p50/p95 (ms) | SF DB (MiB) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **BAAI/bge-small-en-v1.5 (baseline)** | **384** | **0.7159** | **0.3427** | **0.7318** | **0.3642** | **0.5480** | **0.00** | **258.5** | **10.95 / 20.74** | **24.23** | **1562.6** |
| BAAI/bge-base-en-v1.5 | 768 | 0.7439 | 0.3740 | 0.7411 | 0.3745 | 0.5578 | +0.98 | 102.9 | 17.84 / 20.05 | 33.27 | 1585.9 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.6518 | 0.3168 | 0.7185 | 0.3530 | 0.5357 | -1.23 | 878.8 | 13.66 / 14.52 | 24.23 | 1562.2 |
| intfloat/e5-small-v2 (`query:`/`passage:`) | 384 | 0.6837 | 0.3243 | 0.7321 | 0.3510 | 0.5415 | -0.65 | 256.4 | 19.79 / 26.47 | 24.23 | 1568.3 |
| intfloat/e5-small-v2 (no prefix) | 384 | 0.6823 | 0.3183 | 0.7362 | 0.3487 | 0.5425 | -0.55 | 258.1 | 10.57 / 11.11 | 24.23 | 1564.1 |

Decision: keep `BAAI/bge-small-en-v1.5` as the default embedding model (`bge-base-en-v1.5` gained `+0.98` pts hybrid and `+2.97` pts vector-only, just below the `+1.00` pt threshold, while cutting indexing throughput by 2.5x and increasing DB size by 37%; users wanting higher semantic recall can select `BAAI/bge-base-en-v1.5` via `embed_model` in `config.toml`).

## Phase 2: Performance Profiling, Hotspot Fixes, `int8` Quantization & 100k Scale Test

### Profiling Baseline (Pre-Optimization)

1. **Fresh BEIR SciFact indexing (`91.91 s` instrumented total, `86.555 s` uninstrumented `Indexer.sync`)**:
   - **`embed_documents`**: `85.14 s` (**92.6%**) across `5,183` single-file calls (1 file / ~1.12 chunks per GPU encode call, amortising PyTorch kernel launch + tokenization overhead per file).
   - `store_replace_chunks`: `1.86 s` across `5,183` separate SQLite write transactions.
   - `sha256_file`: `0.25 s`, `chunk`: `0.22 s`, `store_upsert_document`: `0.19 s`, `walk_root`: `0.19 s`, `extract`: `0.17 s`, `store_get_document`: `0.085 s`, `store_find_vec_by_sha`: `0.061 s`.
2. **Unchanged SciFact re-sync (`2.269 s` for `5,183` files)**: `walk_root` `0.034 s`, `store_get_document` `0.073 s`, ~`2.16 s` in per-file `Path.resolve()`, `_is_ignored()` (`PurePosixPath.match` + `fnmatch`), and duplicate `stat()`.
3. **20k-chunk `Searcher.get_hits` hybrid breakdown (`14.74 ms` p50 / `16.02 ms` p95)**:
   - `vector_search`: `14.19 ms` p50 / `14.96 ms` p95 (**96%**)
   - `keyword_search`: `0.77 ms` p50 / `0.94 ms` p95
   - `fetch_chunks`: `0.55 ms` p50, `build_hits`: `0.42 ms` p50, `rrf`: `0.078 ms` p50, `stats_check` (`2x COUNT(*)`): `0.029 ms` p50.
4. **Cold `seekr daemon` startup (`7.282 s` to `ping` ready on existing SciFact DB)**: forced synchronous `embedder.dim` (~5 s model load) + `2.27 s` unchanged `indexer.sync` BEFORE binding the Unix socket.

### Optimization Results (Before vs. After Phase 2 Commits `db03cb9`, `290d0df`, `da667dc`)

Changes implemented:
- Cross-file sync batching (`64` files / `128` chunks per batch): one `find_vectors_by_text_shas` lookup, one `embed_documents` call, and one `store.transaction(write=True)` commit (with per-file fallback preserving isolation on errors).
- Per-root metadata prefetch (`Store.get_documents_metadata`), `os.lstat` canonical fast path on non-symlink regular files, and precompiled `ignore_globs` exact-part `frozenset`.
- WAL performance pragmas (`synchronous=NORMAL`, `temp_store=MEMORY`, `cache_size=-32000`), direct `np.ascontiguousarray(..., dtype="<f4").tobytes()` vector serialization, skipping empty chunk deletes on new docs, `Store.has_chunks()` (`SELECT EXISTS(...)`), single-connection `cli_search._inspect_db`, and deferred background embedder prewarm in `cli_daemon`.
- Default `vector_quant = "int8"` in `Config` (`sqlite-vec` `int8[dim]` with dimension-aware scaling `min(500, max(127, 25 * sqrt(dim)))`; `Store(...)` direct constructor keeps `"float32"` default for backwards compatibility and legacy DBs default to `"float32"`).

| Benchmark / Metric | Phase 0 / Baseline (`float32`) | Phase 2 (`int8`, batched) | Speedup / Delta |
| --- | ---: | ---: | ---: |
| SciFact fresh `Indexer.sync` (5,183 docs / 5,820 chunks) | 86.555 s (59.88 docs/s, 67.24 chunks/s) | 26.432 s (196.08 docs/s, 220.18 chunks/s) | **3.27x faster** |
| SciFact end-to-end DB size | 24.17 MiB | 17.41 MiB | **-28.0%** |
| SciFact unchanged re-sync (5,183 files) | 2.269 s | 0.122 s | **18.6x faster** |
| Cold `seekr daemon` time to `ping` ready (SciFact DB) | 7.282 s | 0.349 s | **20.9x faster** |
| SciFact `seekr eval` wall time (all 4 modes) | 156.97 s | 91.38 s | **1.72x faster** |
| NFCorpus `seekr eval` wall time (all 4 modes) | 119.89 s | 76.59 s | **1.57x faster** |
| 20k synthetic DB size (`float32` vs `int8`) | 44.88 MiB | 22.35 MiB | **2.01x smaller** |
| 20k synthetic `vector` latency p50 / p95 | 14.17 / 15.72 ms (`float32` post-opt) | 8.48 / 8.98 ms | **1.67x / 1.75x faster** |
| 20k synthetic `hybrid` latency p50 / p95 | 16.27 / 16.81 ms (`float32` post-opt) | 10.96 / 11.39 ms | **1.48x / 1.48x faster** |

> **Test note (`tests/test_indexer.py:308`):** `SyncTests.test_sync_adds_files` previously asserted `len(self.embedder.calls) == 3` when syncing 3 single-chunk files, which explicitly enforced the unbatched 1-file-per-call bottleneck. It was updated to assert `sum(len(call) for call in self.embedder.calls) == 3` and `len(self.embedder.calls) == 1`, alongside new batch deduplication and partial-failure isolation tests.

### Vector Quantization Trade-Off Study (`sqlite-vec v0.1.9`, 384-d `bge-small-en-v1.5`)

Standalone quantization sweep (fixed corpus/query vectors across variants; adopt only if quality loss < 1.00 `ndcg@10` point):

| Variant | SF vec | NF vec | SF hyb | NF hyb | Macro-Avg hyb | ΔAvg hyb (pts) | 20k vec p50/p95 (ms) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `float32` L2 | 0.7159 | 0.3427 | 0.7318 | 0.3642 | 0.5480 | 0.00 | 13.78 / 15.51 | Legacy / opt-in |
| **`int8` L2 (scale 500 -> adopted `min(500, 25*sqrt(dim))`)** | **0.7148** | **0.3424** | **0.7312** | **0.3636** | **0.5474** | **-0.06** | **8.67 / 8.88** | **Adopted (default)** |
| `int8` L2 (scale 127) | 0.7138 | 0.3424 | 0.7292 | 0.3641 | 0.5466 | -0.14 | — | Superseded by higher scale |
| `int8` cosine (maxabs) | 0.7171 | 0.3419 | 0.7317 | 0.3643 | 0.5480 | 0.00 | 12.82 / 13.93 | Reverted (slower in sqlite-vec) |
| `bit` (binary Hamming) | 0.6296 | 0.2808 | 0.7087 | 0.3488 | 0.5287 | -1.93 | 1.89 / 2.05 | Rejected (> 1 pt loss) |
| `bit` top-200 + `int8` rescore | 0.7098 | 0.3412 | 0.7177 | 0.3654 | 0.5416 | -0.64 | — | Rejected (SF -1.40 pts) |

### 100,000-Chunk Scale Test (`tests/test_scale_100k.py`, `@pytest.mark.slow`)

Measured on `100,000` synthetic chunks (`2,500` documents × `40` chunks/doc, 384-d `int8` vectors, single write transaction, 30 warm queries after 1 warm-up per mode):

| Metric | Measured Value |
| --- | ---: |
| Store populate time | 9.11 s (10,977 chunks/sec) |
| Database size on disk | 110.45 MiB |
| `keyword` p50 / p95 latency | 4.35 ms / 5.00 ms |
| `vector` p50 / p95 latency | 41.39 ms / 42.55 ms |
| `hybrid` p50 / p95 latency | 44.37 ms / 46.38 ms |

### Post-Phase 2 Full BEIR `seekr eval` Results (SciFact, NFCorpus & FiQA, `int8` Default)

| Dataset | Mode | recall@5 | recall@10 | mrr@10 | ndcg@10 | p50_ms | Eval Wall |
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

3-Dataset Macro-Average `ndcg@10` (`int8` default): `keyword` = `0.4176`, `vector` = `0.4827`, `hybrid` = `0.4847`, `hybrid+rerank` = `0.4784`. SciFact+NFCorpus `hybrid` macro-avg moved only `-0.04` pts (`0.5480 -> 0.5476`: SciFact `+0.02`, NFCorpus `-0.10`). On FiQA (conversational personal-finance Q&A), `OR`-joined FTS5 queries match many weak lexical postings (72.13 ms `p50`), so pure `vector` (`0.3963`) beats equal-weight `hybrid` (`0.3589`) while `--rerank` recovers `+1.72` pts over `hybrid` (`0.3761`).

## ONNX Runtime Backend Equivalence (default since 0.2.0)

The default backend runs the same two models with ONNX Runtime (`onnxruntime 1.32.0`, `tokenizers 0.23.3`) using each repository's published `onnx/model.onnx`, instead of PyTorch + sentence-transformers. The switch was gated on equivalence to the PyTorch backend: mean cosine `>= 0.999` and min cosine `>= 0.995` over 1,000 chunks, mean reranker Kendall τ `>= 0.95` over the hybrid top 30 for 50 queries, and hybrid `ndcg@10` within `0.50` pts on SciFact and NFCorpus.

### Embeddings and reranker order (`bench/onnx_equivalence.py`, CPU only)

```bash
CUDA_VISIBLE_DEVICES= python bench/onnx_equivalence.py --corpus eval/data/scifact/corpus --chunks 1000 --rerank-queries 50
```

| Check | Result | Gate |
| --- | ---: | --- |
| Mean cosine, 1,000 SciFact chunks (PyTorch vs ONNX) | 1.000000 | `>= 0.999` — pass |
| Min cosine, 1,000 SciFact chunks | 1.000000 | `>= 0.995` — pass |
| Reranker Kendall τ, hybrid top 30, mean over 50 queries | 1.000000 | `>= 0.95` — pass |
| Reranker Kendall τ, minimum over 50 queries | 1.000000 | — |
| Embedding time, 1,000 chunks, PyTorch CPU | 59.44 s | — |
| Embedding time, 1,000 chunks, ONNX CPU | 46.31 s | — |

Cosines are identical to six decimal places, and the two cross-encoders rank every candidate list identically. ONNX scores are raw logits; the PyTorch backend may apply an activation, so `--explain` rerank score values can differ in scale while the order matches.

### Retrieval quality (`seekr eval`, configs differing only in `embed_backend`)

| Dataset | Mode | ndcg@10 (PyTorch) | ndcg@10 (ONNX) | Δ (pts) |
| --- | --- | ---: | ---: | ---: |
| SciFact | keyword | 0.6866 | 0.6866 | 0.00 |
| SciFact | vector | 0.7109 | 0.7109 | 0.00 |
| SciFact | **hybrid** | **0.7320** | **0.7320** | **0.00** |
| SciFact | hybrid+rerank | 0.7020 | 0.7020 | 0.00 |
| NFCorpus | keyword | 0.3214 | 0.3214 | 0.00 |
| NFCorpus | vector | 0.3410 | 0.3410 | 0.00 |
| NFCorpus | **hybrid** | **0.3632** | **0.3632** | **0.00** |
| NFCorpus | hybrid+rerank | 0.3570 | 0.3570 | 0.00 |

recall@5, recall@10 and mrr@10 also match to four decimal places in every row. Gate: pass. (These evaluations ran before the batch-size change below; that change only alters padding, which is masked, and the re-run embedding check above still gives cosine 1.000000.)

### CPU batching

Small, length-sorted batches are fastest for ONNX Runtime on CPU (less padding, better cache use). Measured on 400 SciFact chunks and 30-passage rerank calls (median of 5):

| Batch size | Embedding (chunks/s) | Rerank 30 passages |
| ---: | ---: | ---: |
| 32 | 14.7 | 1,329 ms |
| 16 | 15.5 | 1,137 ms |
| 8 | 16.7 | 1,206 ms |
| **4 (adopted)** | **21.0** | **840 ms** |
| 2 | 22.2 | 832 ms |
| 1 | 22.4 | 771 ms |

Batch size 4 keeps nearly all of the gain while still batching. The PyTorch backend keeps batches of 32, which suit CUDA.
