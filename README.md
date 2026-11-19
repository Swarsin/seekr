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

The repository deliberately ships **no pre-filled benchmark numbers**. Results
must come from real runs on a stated corpus and machine.

### Retrieval quality (BEIR SciFact)

Download and convert the public SciFact dataset once, then run the four
retrieval modes:

```bash
python eval/beir_scifact.py
seekr eval \
  --root eval/data/scifact/corpus \
  --queries eval/data/scifact/queries.jsonl \
  --modes keyword,vector,hybrid,hybrid+rerank \
  --out bench/scifact-results.md
```

This produces a markdown table of `recall@5`, `recall@10`, `mrr@10`, `ndcg@10`,
and `p50_ms` for each mode. `eval/data/` is git-ignored. To evaluate your own
notes instead, see [`docs/eval.md`](docs/eval.md) for the query JSONL format and
guidance on writing roughly 50 file-level queries. A small versioned example
corpus is available at `eval/example/`.

### Retrieval latency

```bash
python bench/latency.py --chunks 20000
```

The script builds a temporary synthetic index with real `Store`/`Searcher`
paths, 20,000 chunks, normalised 384-dimensional vectors, and FTS-friendly
text. It reports p50/p95 for keyword, vector, and hybrid retrieval, excluding
embedding-model inference, and reports real query-embedding p50/p95 separately
when the embedding model is already cached. It writes
`bench/latency-results.md` only after all retrieval measurements succeed. If
the model is unavailable, the file says so instead of inventing a number.

### Current status

| Measurement | Status |
| --- | --- |
| BEIR SciFact retrieval quality | To be measured: run the commands above and commit only the generated, real table. |
| Synthetic 20k-chunk retrieval latency | To be measured: run `python bench/latency.py --chunks 20000`. |
| Warm-daemon versus cold-CLI search time | To be measured: compare `time seekr search ...` with `seekr daemon` running and stopped. |
| Personal-notes evaluation | To be measured on the owner's corpus. |

No row above is a claim about achieved performance. Results are added only when
a real run has produced them.

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
