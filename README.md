# seekr

Private, offline hybrid keyword and semantic search for your notes, PDFs, Word documents, slides, Jupyter notebooks, code, and lecture transcripts — stored in a single SQLite file on your machine.

[![CI](https://github.com/Swarsin/seekr/actions/workflows/ci.yml/badge.svg)](https://github.com/Swarsin/seekr/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/seekr-search?logo=pypi&logoColor=white)](https://pypi.org/project/seekr-search/)
[![Python](https://img.shields.io/pypi/pyversions/seekr-search?logo=python&logoColor=white)](https://pypi.org/project/seekr-search/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Swarsin/seekr/blob/main/LICENSE)

![seekr CLI demo](https://raw.githubusercontent.com/Swarsin/seekr/main/docs/img/demo.gif)

## Quickstart

Requires Python 3.11+ on Linux or WSL2 with SQLite loadable extension support. Note that the PyPI distribution name is `seekr-search`, while the command, Python import name, and config/data directories stay `seekr`.

```bash
pipx install seekr-search
seekr init ~/Documents/ai-course
seekr search "bias variance tradeoff" --module machine-learning
```

`seekr init` writes `~/.config/seekr/config.toml`, explicitly prompts to download `BAAI/bge-small-en-v1.5` (~130 MB of ONNX model files) and tree-sitter code grammars (via `seekr models pull`), and builds your initial index. The default install runs the models with ONNX Runtime on the CPU and is about 350 MB on disk — no PyTorch required. Want to try it before pointing at your own folders? Run `seekr init --demo` to copy and index the packaged sample course, or use `seekr models pull --rerank` / `--all`, `seekr models status`, and `seekr doctor` to inspect local readiness.

### GPU / PyTorch backend

The default ONNX Runtime backend gives the same results as PyTorch (mean cosine similarity 1.000 on 1,000 chunks, identical reranker order; see [`docs/benchmarks.md`](https://github.com/Swarsin/seekr/blob/main/docs/benchmarks.md)). To run the models with PyTorch instead — for example on a CUDA GPU — install the extra and switch the backend; existing indexes keep working without a reindex:

```bash
pipx install 'seekr-search[torch]'     # or: pipx inject seekr-search sentence-transformers
echo 'embed_backend = "torch"' >> ~/.config/seekr/config.toml
seekr models pull
```

## What it does

- **CLI (`seekr`)** — one-command setup (`seekr init`, `seekr init --demo`), explicit offline asset provisioning (`seekr models pull`, `seekr models status`), incremental indexing (`seekr index`, `seekr reindex`, `seekr status`), hybrid/keyword/vector queries with `--module`, `--week`, `--kind`, `--since`, `--path`, `--rerank`, `--spread` (cross-module/week appearance summary), `--json`, `--explain`, and stemmed term highlights, related-file lookup (`seekr similar`), interactive opening (`seekr pick`, `seekr open` with Windows app launch via `wslview` / `explorer.exe` under WSL), and shell completions (`seekr completion bash|zsh|fish`).
- **Web UI (`seekr web`)** — local-only browser workspace (`http://127.0.0.1:8000`) with first-run setup and one-click model download, folder add/remove management, **Module** / **Week** / file-type / date filters, **Appears in** cross-course spread chips, highlighted snippets, inline passage and PDF page preview, and one-click file opening or location copying (with WSL Windows/POSIX path toggle). See [`docs/web.md`](https://github.com/Swarsin/seekr/blob/main/docs/web.md) and [`docs/web-api.md`](https://github.com/Swarsin/seekr/blob/main/docs/web-api.md).
- **Daemon (`seekr daemon`)** — keeps the embedding model warm in memory, answers CLI/web queries over `$XDG_RUNTIME_DIR/seekr.sock` (falling back automatically to local search when stopped), and watches configured roots for file changes—automatically switching to polling (`poll_interval_seconds`, default `15`s) on Windows drives (`9p`/`drvfs`) and network mounts (`cifs`, `smb3`, `nfs`, `fuse.sshfs`); includes [`contrib/seekr.service`](https://github.com/Swarsin/seekr/blob/main/contrib/seekr.service) (systemd user unit) and [`contrib/seekr-popup.sh`](https://github.com/Swarsin/seekr/blob/main/contrib/seekr-popup.sh) (Hyprland/rofi launcher).
- **MCP (`seekr mcp`)** — newline-delimited JSON-RPC 2.0 stdio server exposing `search` and `similar` tools (including `module`, `week`, and `spread` support) for coding agents with zero implicit network calls and stderr-only diagnostics.

![seekr web UI](https://raw.githubusercontent.com/Swarsin/seekr/main/docs/img/web.png)

## Use with Claude Code / Cursor

Register `seekr` in Claude Code with one command:

```bash
claude mcp add seekr -- seekr mcp
```

Or add the stdio MCP server block to Claude Desktop, Cursor, or any MCP host configuration:

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

If `seekr` is installed inside a dedicated virtualenv (such as `pipx`), set `"command"` to `~/.local/bin/seekr` or that environment's `seekr` binary. The server exposes `search` (`query`, `k`, `mode`, `per_file`, `rerank`, `path`, `kind`, `since`, `module`, `week`, `explain`, `spread`) and `similar` (`path`, `k`, `per_file`, `path_prefix`, `kind`, `since`, `module`, `week`).

## How it works

```text
PDF / Markdown / Word (.docx) / PowerPoint (.pptx) / Jupyter (.ipynb) /
code / transcripts / text (UTF-8, UTF-8/UTF-16 BOM, latin-1 fallback)
                  │
                  ▼
        extract text units + locators + module/week labels
                  │
                  ▼
  chunk: word windows (300/450/50), heading sections,
  tree-sitter AST chunks, transcript cues, slide/cell blocks
                  │
                  ▼
  batched embed locally with ONNX Runtime (BAAI/bge-small-en-v1.5,
  384-d L2-normalised, int8 quantized by default in sqlite-vec)
                  │
                  ▼
        one SQLite WAL index (schema v3)
        ├── documents (indexed on kind/mtime_ns and module/week) + chunks + meta
        ├── chunks_fts (external-content FTS5, porter unicode61)
        └── chunks_vec (sqlite-vec vec0 KNN + scalar filters)
                  ▲
                  │
query ──► BGE query prefix + embed locally
          ├──► FTS5 BM25 (path/kind/since/module/week SQL filters) ──┐
          └──► vec0 KNN or filtered scalar L2 search ────────────────┴──► RRF (k=60)
                                                                               │
                                                                               ▼
                                                                    optional cross-encoder rerank
                                                                               │
                                                                               ▼
                                                        stemmed highlights + optional --spread
                                                                               │
                                                                               ▼
                                                     CLI, web preview, warm Unix daemon, or `seekr mcp`
```

Indexing runs in batches (up to 64 files / 128 chunks per transaction) and keys each chunk by the SHA-256 of its normalised text, so unchanged passages reuse stored vectors across edits and renames. Configuration lives at `$XDG_CONFIG_HOME/seekr/config.toml` (`roots`, `db_path`, `ignore_globs`, `embed_model`, `rerank_model`, `embed_backend`, `vector_quant`, `course_structure`, `week_pattern`, `poll_interval_seconds`), and `schema_version = 3` migrates older v1/v2 and `float32` databases inside an atomic transaction. Once models are cached locally, `seekr.offline` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and disables telemetry; the ONNX backend reads model files straight from the local cache, and the optional PyTorch backend passes `local_files_only=True` and rejects `trust_remote_code`; network access happens only during explicit `seekr models pull` (or the setup download button) and BEIR dataset converters in `eval/beir_*.py` unless `--no-download` is passed.

## Benchmarks

- Hybrid retrieval achieves the best 3-dataset macro-average `ndcg@10` across BEIR SciFact, NFCorpus, and FiQA (`0.4847` vs `0.4176` keyword and `0.4827` vector).
- Default `int8` quantization halves index DB size (`22.35 MiB` vs `44.88 MiB` at 20k chunks) with negligible quality change (`-0.06` pts macro-avg on SciFact+NFCorpus).
- Warm hybrid search runs at ~`11 ms` p95 (`11.39 ms` at 20k chunks, excluding query embedding).

See [`docs/benchmarks.md`](https://github.com/Swarsin/seekr/blob/main/docs/benchmarks.md) for full BEIR tables, single-variable ablations, latency/throughput numbers, 100k-chunk scale results, Windows-drive (`9p`/DrvFS) sync timings, and design notes, and [`docs/eval.md`](https://github.com/Swarsin/seekr/blob/main/docs/eval.md) to evaluate your own corpus.

## Limitations

- No OCR for scanned image PDFs (extracts embedded text only).
- Linux and WSL2 only (`AF_UNIX`, `SO_PEERCRED`, and `fcntl.flock` lifecycle locks); native Windows and macOS are not supported.
- Legacy binary Office files (`.doc`, `.ppt`) are skipped with a warning; convert them to `.docx` or `.pptx` to index them.
- No LLM answer generation — seekr retrieves and locates source passages rather than synthesizing prose answers.
- `OR`-joined multi-term FTS5 queries scan long posting lists on large conversational corpora (e.g., FiQA `keyword p50 = 72.13 ms`).
- Filtered vector queries (`--path`, `--kind`, `--since`, `--module`, `--week`, `seekr similar`) use sqlite-vec's exact `vec_distance_L2` scalar scan so filters apply before top-k selection, scaling linearly with matching chunk count.
- Transcript locators format timestamps in `open` links as labels without seeking external media players.

## Development

```bash
pip install -e ".[dev]"          # add ,torch to also test the PyTorch backend
ruff check .
pytest -m "not slow"
pytest -m slow          # real cached models, sqlite-vec int8, and 100k scale test
```

CI ([`.github/workflows/ci.yml`](https://github.com/Swarsin/seekr/blob/main/.github/workflows/ci.yml)) runs `ruff check .` and `pytest -m "not slow"` on Python 3.11–3.13 on Ubuntu, the default suite again in a wheel install without PyTorch, and wheel installation checks.

## License

MIT. See [`LICENSE`](https://github.com/Swarsin/seekr/blob/main/LICENSE).
