# seekr

seekr is a local-first search tool for your own notes, PDFs, code, and lecture transcripts. It combines exact keyword matching with semantic retrieval: FTS5 finds error codes, function names, and identifiers, while local sentence embeddings find passages that mean the same thing without sharing words. All text, vectors, and rankings live in one SQLite file on your machine. After one-time local model caching, indexing and querying make no network calls, and seekr ships with a warm Unix-socket query daemon plus a stdio MCP server (`seekr mcp`) for coding agents.

## Quickstart

Requires Python 3.11+ on Linux with SQLite loadable extension support.

```bash
# Install the CLI and its runtime dependencies.
pipx install .

# One-time model provisioning, outside seekr. seekr itself never downloads
# models implicitly; this populates the local Hugging Face cache.
~/.local/pipx/venvs/seekr/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-small-en-v1.5")'

# Configure your roots in ~/.config/seekr/config.toml.
mkdir -p ~/.config/seekr
cat > ~/.config/seekr/config.toml <<'EOF'
roots = ["~/notes", "~/Documents/coursework"]
EOF

seekr doctor
seekr index
seekr status
seekr search "how do I checkpoint a WAL database?"
```

Useful search, inspection, and discovery workflows:

```bash
# Filter by mode, path prefix, document kind, or modification recency
seekr search "migration" --kind code --since 7d --path ~/projects/app

# Machine-readable JSON output + per-retriever rank/score attribution
seekr search "checkpoint" --json --explain

# Find indexed files semantically related to a file (reuses stored vectors if indexed)
seekr similar ~/notes/sqlite-wal.md -k 5 --kind markdown

# Rich index/model/daemon status (or --json) and structured stderr diagnostics
seekr status --json
seekr --verbose search "wal"

# Interactive picker/opener (PDFs open in zathura; code/Markdown open in $EDITOR)
seekr pick "database locking"
seekr open "/home/me/notes/sqlite.md:L42"

# Shell completions
seekr completion bash   # also: zsh, fish
```

The optional cross-encoder reranker requires its own cached model and is invoked with `--rerank` or in `seekr eval`:

```bash
~/.local/pipx/venvs/seekr/bin/python -c 'from sentence_transformers import CrossEncoder; CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")'
seekr search "gradient descent" --rerank
```

## Architecture

```text
PDF / Markdown / text / code / transcripts (UTF-8, UTF-8/UTF-16 BOM, latin-1 fallback)
                  │
                  ▼
        extract text units + locators
                  │
                  ▼
  chunk: word windows (300/450/50), heading sections,
  tree-sitter AST chunks, transcript cues
                  │
                  ▼
  batched embed locally (BAAI/bge-small-en-v1.5, 384-d L2-normalised,
  int8 quantized by default in sqlite-vec)
                  │
                  ▼
        one SQLite WAL index (schema v2)
        ├── documents (indexed on kind, mtime_ns) + chunks + meta
        ├── chunks_fts (external-content FTS5, porter unicode61)
        └── chunks_vec (sqlite-vec vec0 KNN + scalar filters)
                  ▲
                  │
query ──► BGE query prefix + embed locally
          ├──► FTS5 BM25 (path/kind/since SQL filters) ──┐
          └──► vec0 KNN or filtered scalar L2 search ────┴──► RRF (k=60)
                                                                 │
                                                                 ▼
                                                      optional cross-encoder rerank
                                                                 │
                                                                 ▼
                                         CLI (text / --json / --explain),
                                         warm Unix daemon, or `seekr mcp`
```

Indexing is incremental and batched (up to 64 files / 128 chunks per transaction): each chunk is keyed by a SHA-256 of its normalised text, so unchanged chunks reuse stored vectors across edits and renames. Schema version tracking (`schema_version = 2`) migrates legacy `float32` databases safely inside an atomic transaction.

## Configuration

The config file lives at `$XDG_CONFIG_HOME/seekr/config.toml` (usually `~/.config/seekr/config.toml`). Missing keys use these defaults:

```toml
roots = ["~/notes", "~/Documents/coursework"]
# db_path = "~/.local/share/seekr/index.db"
# ignore_globs = [".git", "node_modules", "__pycache__", ".venv", "venv"]
# max_file_bytes = 20000000
# embed_model = "BAAI/bge-small-en-v1.5"
# rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# vector_quant = "int8"          # or "float32"
# chunk_target_words = 300
# chunk_max_words = 450
# chunk_overlap_words = 50
```

Existing `float32` databases remain readable after upgrading; switching `vector_quant` on an existing DB prompts you to run `seekr reindex`.

## MCP Server for Coding Agents (`seekr mcp`)

`seekr mcp` runs a newline-delimited JSON-RPC 2.0 MCP server over `stdin`/`stdout` with zero model downloads and clean stderr-only diagnostics. It exposes two tools:

- `search` — `query` (required), `k`, `mode` (`keyword`/`vector`/`hybrid`), `per_file`, `rerank`, `path`, `kind`, `since`, `explain`
- `similar` — `path` (required), `k`, `per_file`, `path_prefix`, `kind`, `since`

Example MCP client configuration (Claude Desktop, Cursor, or any stdio MCP host):

```json
{
  "mcpServers": {
    "seekr": {
      "command": "seekr",
      "args": ["mcp"]
    }
  }
}
```

If seekr is installed inside a dedicated virtualenv (such as pipx), set `"command"` to `~/.local/bin/seekr` or that environment's `seekr` binary.

## Benchmarks & Ablations

All numbers below were measured on an AMD Ryzen 9 3900X (24 logical CPUs), 31 GiB RAM, NVIDIA GeForce RTX 3070 (8 GiB, CUDA), Python 3.12.15, SQLite 3.53.1, `sqlite-vec v0.1.9`, PyTorch 2.15.0+cu132, and `sentence-transformers 6.3.0`. Full experiment logs live in [`docs/experiments.md`](docs/experiments.md).

### Retrieval quality (BEIR SciFact, NFCorpus, and FiQA)

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

3-dataset macro-average `ndcg@10`: `keyword 0.4176`, `vector 0.4827`, `hybrid 0.4847`, `hybrid+rerank 0.4784`. To evaluate your own workspace files, see [`docs/eval.md`](docs/eval.md) and the sample fixture under `eval/example/`.

### Retrieval ablations (single-variable)

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

### Latency, indexing throughput, quantization, and 100k scale

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

## Daemon and Desktop Integration

```bash
seekr daemon
# in another terminal
seekr search "lecture 4 backpropagation"
```

The CLI uses `$XDG_RUNTIME_DIR/seekr.sock` (or `/tmp/seekr-<uid>.sock`, mode `0600`, bound atomically via sibling temp socket + `link`) when listening and falls back to one-shot local search otherwise. Configured roots are watched for creates, edits, deletes, and renames with per-path debounce.

### systemd user unit

```bash
mkdir -p ~/.config/systemd/user
cp contrib/seekr.service ~/.config/systemd/user/seekr.service
systemctl --user daemon-reload
systemctl --user enable --now seekr.service
journalctl --user -u seekr -f
```

### Hyprland rofi popup

```bash
install -m 755 contrib/seekr-popup.sh ~/.local/bin/seekr-popup
```

Add `bind = $mainMod, S, exec, ~/.local/bin/seekr-popup` to `~/.config/hypr/hyprland.conf` (requires `rofi`; opens PDFs in `zathura`, code/Markdown in `$EDITOR`, other files via `xdg-open`).

## Offline Guarantee

After models are cached, seekr never accesses the network at index, query, daemon, or MCP time. `seekr.offline` sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, disables telemetry, passes `local_files_only=True`, and rejects `trust_remote_code`. Missing local caches fail with explicit remediation steps. Network access is strictly limited to the explicit BEIR downloaders in `eval/beir_*.py` (which accept `--no-download`).

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -m "not slow"
pytest -m slow          # real cached models, sqlite-vec int8, and 100k scale test
```

CI (`.github/workflows/ci.yml`) runs `ruff check .` and `pytest -m "not slow"` on Python 3.11 and 3.12 on Ubuntu.

## Limitations

- No OCR for scanned image PDFs (extracts embedded text only).
- Linux only (`AF_UNIX`, `SO_PEERCRED`, and `fcntl.flock` lifecycle locks); Windows and macOS are not supported.
- No LLM answer generation — seekr retrieves and locates source passages rather than synthesizing prose answers.
- `OR`-joined multi-term FTS5 queries scan long posting lists on large conversational corpora (e.g., FiQA `keyword p50 = 72.13 ms`).
- Filtered vector queries (`--path`, `--kind`, `--since`, `seekr similar`) use sqlite-vec's exact `vec_distance_L2` scalar scan so filters apply before top-k selection, scaling linearly with matching chunk count.
- Transcript locators format timestamps in `open` links as labels without seeking external media players.

## License

MIT. See [`LICENSE`](LICENSE).
