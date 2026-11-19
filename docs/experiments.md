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
