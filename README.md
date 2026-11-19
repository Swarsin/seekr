# seekr

seekr is a local-first search tool for your own notes, PDFs, code, and lecture
transcripts. It combines exact keyword matching with semantic retrieval: FTS5
finds error codes, function names, and module codes, while local sentence
embeddings find passages that mean the same thing without sharing words. All
text, vectors, and rankings stay in one SQLite file on your machine. After the
one-time model download, indexing and querying make no network calls, and an
optional warm daemon keeps the models loaded so repeated searches stay fast.

## Quickstart

Requires Python 3.11+ on Linux with SQLite extension support.

```bash
# Install the CLI and its runtime dependencies.
pipx install .

# One-time model provisioning, outside seekr. seekr itself never downloads
# models implicitly; this step needs network access and populates the Hugging
# Face cache used by the local loader. This uses the interpreter in pipx's app
# venv; adjust the path if PIPX_HOME is set.
~/.local/pipx/venvs/seekr/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-small-en-v1.5")'

# Tell seekr where your files live.
mkdir -p ~/.config/seekr
cat > ~/.config/seekr/config.toml <<'EOF'
roots = ["~/notes", "~/Documents/coursework"]
EOF

seekr doctor
seekr index
seekr search "how do I checkpoint a WAL database?"
```

`seekr doctor` checks FTS5 and sqlite-vec before you build an index. `seekr
index` defaults to the configured `roots`; you can also pass paths explicitly.
`seekr search` prints the path, locator, score, and a short snippet for each
hit.

The optional cross-encoder reranker needs its own cached model. It is only used
with `--rerank` or in the `hybrid+rerank` evaluation mode:

```bash
~/.local/pipx/venvs/seekr/bin/python -c 'from sentence_transformers import CrossEncoder; CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")'
seekr search "gradient descent" --rerank
```

## Architecture

```text
PDF / Markdown / text / code / transcripts
                  │
                  ▼
        extract text units + locators
                  │
                  ▼
  chunk: word windows, heading sections,
  tree-sitter AST chunks, transcript cues
                  │
                  ▼
  embed chunks locally (BAAI/bge-small-en-v1.5,
  384-d float32, L2-normalised)
                  │
                  ▼
        one SQLite index file
        ├── documents + chunks + metadata
        ├── chunks_fts (FTS5, porter unicode61)
        └── chunks_vec (sqlite-vec vec0 KNN)
                  ▲
                  │
query ──► add query prefix, embed locally
          ├──► FTS5 BM25 top-k ──────┐
          └──► vec0 L2 KNN top-k ────┴──► reciprocal rank fusion
                                             │
                                             ▼
                                  optional cross-encoder rerank
                                             │
                                             ▼
                                  hits with page / line / heading
```

The index is incremental. Each chunk is keyed by a hash of its normalised text,
so unchanged chunks are never re-embedded. A relocated file keeps its chunk
IDs and vectors when its content is unchanged. The watcher and daemon reuse the
same `Store`/`Searcher` code as the CLI.

## Configuration

The config file is `$XDG_CONFIG_HOME/seekr/config.toml` (usually
`~/.config/seekr/config.toml`). Missing keys use the defaults shown here:

```toml
roots = ["~/notes", "~/Documents/coursework"]
# db_path = "~/.local/share/seekr/index.db"
# ignore_globs = [".git", "node_modules", "__pycache__", ".venv", "venv"]
# max_file_bytes = 20000000
# embed_model = "BAAI/bge-small-en-v1.5"
# rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# chunk_target_words = 300
# chunk_max_words = 450
# chunk_overlap_words = 50
```

`seekr index PATH...` indexes specific paths; `seekr status` reports document
and chunk counts. Search modes are `keyword`, `vector`, and `hybrid` (the
default). Useful flags include `--mode`, `--k`, `--per-file`, `--path`, and
`--rerank`.

To open a result at its location, use `pick` and `open`:

```bash
seekr pick "database locking"
seekr open "/home/me/notes/sqlite.md:L42"
```

PDFs open in zathura at the page when available; code and Markdown open in
`$EDITOR` at the line (nvim by default); other results use `xdg-open`.

## Evaluation

All benchmark numbers below were measured on an AMD Ryzen 9 3900X (24 logical CPUs), 31 GiB RAM, and NVIDIA GeForce RTX 3070 (CUDA, PyTorch 2.15.0+cu132, Python 3.12.15, SQLite 3.53.1, sqlite-vec v0.1.9).

### Retrieval quality (BEIR SciFact & NFCorpus)

Download and convert the datasets once, then run all four retrieval modes:

```bash
python eval/beir_scifact.py
seekr eval \
  --root eval/data/scifact/corpus \
  --queries eval/data/scifact/queries.jsonl \
  --modes keyword,vector,hybrid,hybrid+rerank \
  --out bench/scifact-results.md

python eval/beir_nfcorpus.py
seekr eval \
  --root eval/data/nfcorpus/corpus \
  --queries eval/data/nfcorpus/queries.jsonl \
  --modes keyword,vector,hybrid,hybrid+rerank \
  --out bench/nfcorpus-results.md
```

**BEIR SciFact** (5,183 documents, 300 queries; `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`):

| mode | recall@5 | recall@10 | mrr@10 | ndcg@10 | p50_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.7387 | 0.8078 | 0.6547 | 0.6866 | 14.21 |
| vector | 0.7590 | 0.8379 | 0.6853 | 0.7159 | 23.94 |
| **hybrid** | **0.7806** | **0.8701** | **0.6962** | **0.7318** | **32.54** |
| hybrid+rerank | 0.7499 | 0.8322 | 0.6669 | 0.7010 | 119.88 |

**BEIR NFCorpus** (3,633 documents, 323 queries; same models):

| mode | recall@5 | recall@10 | mrr@10 | ndcg@10 | p50_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.1228 | 0.1553 | 0.5224 | 0.3214 | 3.26 |
| vector | 0.1267 | 0.1614 | 0.5316 | 0.3427 | 22.35 |
| **hybrid** | **0.1376** | **0.1733** | **0.5694** | **0.3642** | **24.48** |
| hybrid+rerank | 0.1389 | 0.1695 | 0.5842 | 0.3569 | 106.30 |

### Retrieval ablations

Controlled single-variable ablations against the default `hybrid` baseline (`0.7318` SciFact, `0.3642` NFCorpus, macro-avg `0.5480`; full tables in [`docs/experiments.md`](docs/experiments.md)):

| Experiment | Variant | SciFact ndcg@10 | NFCorpus ndcg@10 | Avg ndcg@10 | ΔAvg (pts) | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | `bge-small-en-v1.5`, 300/50 chunks, RRF k=60, `porter unicode61` | 0.7318 | 0.3642 | 0.5480 | 0.00 | Default |
| Chunking | 150 words / 50 overlap | 0.7223 | 0.3686 | 0.5455 | -0.25 | Reverted |
| Chunking | 300 words / 0 overlap | 0.7349 | 0.3655 | 0.5502 | +0.22 | Reverted (< +1.0 pt) |
| Chunking | 450 words / 50 overlap | 0.7305 | 0.3649 | 0.5477 | -0.03 | Reverted |
| Headers | Prepend filename + heading path (BEIR IDs) | 0.7232 | 0.3635 | 0.5433 | -0.47 | Reverted |
| Fusion | RRF `k=10` | 0.7362 | 0.3662 | 0.5512 | +0.32 | Reverted (< +1.0 pt) |
| Fusion | Min-max score linear fusion (`alpha_kw=0.5`) | 0.7357 | 0.3640 | 0.5499 | +0.19 | Reverted (< +1.0 pt) |
| Rerank | Direct cross-encoder replacement (top 30) | 0.7010 | 0.3569 | 0.5289 | -1.91 | Opt-in only (`--rerank`) |
| Rerank | RRF blend (first-stage hybrid + CE, top 30) | 0.7297 | 0.3641 | 0.5469 | -0.11 | Logged |
| Prefix | BGE query prefix OFF | 0.7414 | 0.3626 | 0.5520 | +0.40 | Reverted (vector NF -0.53) |
| FTS5 | `unicode61` without Porter stemmer | 0.7107 | 0.3555 | 0.5331 | -1.49 | Reverted |
| Model | `BAAI/bge-base-en-v1.5` (768-d, 102.9 chunks/s) | 0.7411 | 0.3745 | 0.5578 | +0.98 | Opt-in config |
| Model | `sentence-transformers/all-MiniLM-L6-v2` (384-d, 878.8 chunks/s) | 0.7185 | 0.3530 | 0.5357 | -1.23 | Rejected |
| Model | `intfloat/e5-small-v2` (`query:`/`passage:`, 256.4 chunks/s) | 0.7321 | 0.3510 | 0.5415 | -0.65 | Rejected |

To evaluate your own notes instead, see [`docs/eval.md`](docs/eval.md) for the query JSONL format and guidance on writing roughly 50 file-level queries. A small versioned example corpus lives at `eval/example/`.

### Retrieval latency

```bash
python bench/latency.py --chunks 20000
```

Measured on 20,000 synthetic chunks (500 documents, 384-d float32, 50 queries per mode after warm-up; [`bench/latency-results.md`](bench/latency-results.md)):

| Measurement | p50 | p95 | Notes |
| --- | ---: | ---: | --- |
| Keyword retrieval (FTS5 BM25) | 1.41 ms | 1.54 ms | Excludes embedding |
| Vector retrieval (sqlite-vec float32) | 12.04 ms | 13.05 ms | Excludes embedding |
| Hybrid retrieval (RRF k=60) | 14.12 ms | 14.53 ms | Excludes embedding |
| Query embedding (`BAAI/bge-small-en-v1.5`, CUDA) | 10.17 ms | 11.02 ms | Measured separately |

Real end-to-end indexing of BEIR SciFact (5,183 files / 5,820 chunks) with `BAAI/bge-small-en-v1.5` on CUDA took 86.555 s (`59.88 docs/sec`, `67.24 chunks/sec`), producing a 24.17 MiB SQLite index at 1411.4 MiB peak RSS.

## Design decisions

- **Hybrid retrieval with reciprocal rank fusion (RRF).** Keyword search gives
  exactness for identifiers, error strings, and code names; vector search gives
  recall for paraphrases. Fusing ranked lists with `1 / (60 + rank)` avoids
  calibrating BM25 and L2 scores against each other and is stable when one
  retriever returns a shorter list. The reranker is optional and sees the top
  30 fused candidates, so it cannot dominate normal query latency.
- **SQLite with FTS5 and sqlite-vec.** One owned file replaces a database
  server, a keyword engine, and a vector service. Transactions keep documents,
  chunks, FTS rows, and vectors consistent; WAL allows the daemon and a reindex
  to coordinate through the lifecycle lock. FTS5 is an external-content table,
  so chunk text is stored once. sqlite-vec's `vec0` table provides the KNN
  path. There is intentionally no silent vector fallback: `seekr doctor`
  reports a missing extension before indexing.
- **Chunk-level embedding reuse.** A SHA-256 of normalised chunk text identifies
  unchanged content. Reindexing only embeds new or changed chunks; renames and
  moves reuse the existing vectors. Chunk IDs (and their vectors) survive
  explicit `move_path` calls and live watcher move events; a rename discovered
  only by a bulk scan reuses content-addressed vectors but may assign new
  document and chunk IDs. This is the difference between a useful incremental
  tool and re-embedding the entire corpus after every edit.
- **Asymmetric query prefix.** Documents are embedded as-is; queries get the
  exact prefix `Represent this sentence for searching relevant passages: `.
  The model and dimension are recorded in the index metadata. A mismatch stops
  the search and tells you to run `seekr reindex`, so vectors from different
  models are never mixed.
- **Chunking.** Prose uses a 300-word target, 450-word maximum, and 50-word
  overlap so passages remain focused without losing cross-boundary context.
  Markdown chunks carry heading paths (`§A > B`); code uses tree-sitter
  function/class chunks with a 60-line fallback; transcripts group cues and
  keep the first timestamp. Every hit carries a locator suitable for opening
  the source at the right place.

## Daemon and desktop integration

A warm daemon keeps the embedding model and SQLite page cache loaded and
watches the configured roots:

```bash
seekr daemon
# in another terminal
seekr search "lecture 4 backpropagation"
```

The CLI tries the daemon first and falls back to a one-shot local search when no
daemon is listening. The socket is `$XDG_RUNTIME_DIR/seekr.sock` (or
`/tmp/seekr-<uid>.sock`), mode `0600`. Configured roots are watched for
creates, edits, deletes, and renames with per-path debounce.

### systemd user unit

```bash
mkdir -p ~/.config/systemd/user
cp contrib/seekr.service ~/.config/systemd/user/seekr.service
# Edit ExecStart if pipx installed seekr somewhere other than ~/.local/bin.
systemctl --user daemon-reload
systemctl --user enable --now seekr.service
journalctl --user -u seekr -f
```

### Hyprland rofi popup

`contrib/seekr-popup.sh` asks rofi for a query, asks rofi to choose one pick line,
and opens the result at its locator:

```bash
install -m 755 contrib/seekr-popup.sh ~/.local/bin/seekr-popup
```

Add a binding to `~/.config/hypr/hyprland.conf`:

```ini
bind = $mainMod, S, exec, ~/.local/bin/seekr-popup
```

It requires `rofi`; PDF page opening uses `zathura` when present, code and
Markdown use `$EDITOR`, and other files use `xdg-open`. The popup script's
interactive rofi flow is verified manually; automated tests cover the pick/open
wiring and locator parsing.

## Offline guarantee

After the models are cached, seekr has no reason and no path to the network at
index or query time. Model loading is centralised in `seekr.offline`, which sets
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, disables telemetry, and passes
`local_files_only=True`; a missing cache entry raises an actionable error rather
than downloading. Indexing, searching, the daemon, and the latency benchmark
only read local files and the local SQLite database. The one intentional
network operation is the explicit, opt-in `eval/beir_scifact.py` dataset
download; `--no-download` forbids even that and fails with instructions if the
source files are absent.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -m "not slow"
pytest -m slow          # real models, sqlite-vec, and longer benchmarks
```

The default suite is offline and uses fake embedders/rerankers. CI runs Python
3.11 and 3.12 on Ubuntu with the same lint and non-slow test commands. The
performance smoke test uses a real sqlite-vec index and skips with an explicit
reason when the vector backend is unavailable rather than passing without
measuring.

## Limitations

These are deliberate v1 non-goals:

- No OCR for scanned PDFs.
- No web UI, cloud sync, or multi-user mode.
- Linux only; Windows and macOS are not supported.
- No LLM answer generation: seekr finds passages, it does not write answers.
- No image or screenshot search.
- No MCP server (planned post-v1).

Also note that a Python/SQLite build with loadable extension support is
required. If `seekr doctor` reports a missing extension feature, use a Python
build that includes it (for example a uv-managed CPython). Transcript search
does not yet jump to video timestamps; that and the other post-v1 ideas are
tracked in the implementation plan.

## License

MIT. See [`LICENSE`](LICENSE).
