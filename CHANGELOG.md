# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-24

### Added
- Native Windows and macOS support alongside Linux, including Windows-safe folder management, UTF-8 output when redirected to a file or pipe, drive-path handling in `seekr open` and the web UI, PowerShell syntax in the `seekr init --demo` config hint, and no re-embedding of a renamed file on macOS. See [`docs/install-windows-macos.md`](docs/install-windows-macos.md) for installing, registering the MCP server and starting the service at login (Task Scheduler script `contrib/seekr-task.ps1` and launchd agent `contrib/com.seekr.service.plist`; the systemd unit is unchanged).
- One background service, controlled with `seekr service start`, `seekr service stop` and `seekr service status`. It keeps the model warm, serves the web UI, watches the configured folders, and runs index updates and folder changes so they never race with the watcher.
- `seekr web` starts the service (or reuses a running one), prints its address and returns; `--foreground` keeps the old serve-in-the-terminal behaviour.
- CLI searches, `seekr similar`, indexing and `seekr mcp` route through the service when it is running and fall back to local work when it is not. The service API gained `/api/reindex`, `/api/similar`, `/api/shutdown`, path indexing and rerank/explain options, documented in `docs/web-api.md`.
- `seekr status` reports the service state.

### Changed
- Config, data and state directories follow the platform conventions through `platformdirs`: `~/Library/Application Support/seekr` on macOS and `%LOCALAPPDATA%\seekr` on Windows. Linux paths (`$XDG_CONFIG_HOME/seekr`, `$XDG_DATA_HOME/seekr`) are unchanged.
- On macOS, seekr needs a Python whose `sqlite3` can load extensions (the python.org build cannot); use Homebrew Python or a uv-managed CPython and check with `seekr doctor`.
- Lifecycle locking uses portable shared/exclusive file locks, and the index swap in a reindex is Windows-safe.

### Deprecated
- `seekr daemon` is a deprecated alias for `seekr service start` and will be removed in 0.4.0. The old Unix-socket query fast path remains Linux-only and is disabled on Windows and macOS.

### Fixed
- An edit that keeps a file's size and lands within one filesystem timestamp tick of indexing it is now caught when the file is checked again soon after, before the index records other writes more than 2 s later (for example, a quick re-save picked up by the next sync or watcher event). Files modified within 2 s of the latest index write are hashed instead of trusting an unchanged mtime and size. Before, such an edit was always missed.
- The watcher indexes a file created where another was just moved away (A→B, new A, B→C) instead of leaving it for the next full sync.

## [0.2.0] - 2026-11-23

### Added
- ONNX Runtime backend for the embedding model and cross-encoder reranker (`embed_backend = "onnx"`, the new default), reading `onnx/model.onnx` and `tokenizer.json` from the local model cache; verified equivalent to PyTorch (mean/min cosine 1.000 over 1,000 chunks, identical reranker order on 50 queries).
- `embed_backend` config key (`onnx` or `torch`), shown by `seekr status` and `seekr doctor`; switching backends for the same model keeps the existing index.
- Optional `seekr-search[torch]` extra for the PyTorch (sentence-transformers) backend, e.g. on CUDA GPUs.
- One-command setup via `seekr init` (with `--demo`, `--yes`, `--no-download`, and `--force`) that scans folders, writes `config.toml`, downloads models and code grammars explicitly, and builds the initial index.
- Explicit offline provisioning CLI (`seekr models pull` with `--rerank`, `--grammars`, `--all`, `--yes`; `seekr models status`) and expanded `seekr doctor` checks for cached models, code grammars, WSL status, slow mounts, and daemon readiness.
- Extraction and chunking support for Word documents (`.docx`), PowerPoint slides (`.pptx`), and Jupyter notebooks (`.ipynb`), with warnings for legacy `.doc`/`.ppt` binary files.
- Automatic module and week labeling derived from course folder hierarchies (`course_structure` and `week_pattern` in `config.toml`), with `--module` and `--week` filtering across the CLI, daemon, MCP tools, and web interface.
- Cross-course topic spread summaries (`--spread` on `seekr search`, `spread` on MCP `search`, and **Appears in** chips in the web UI) plus word-aligned snippets with stemmed query highlights.
- Web UI course endpoints (`/api/course`), inline passage and PDF page preview (`/api/preview`, `/api/preview/page`), native/WSL file opening (`/api/open`), folder removal (`/api/roots/remove`), and one-click setup model download (`/api/setup/download`, tracked via `/api/jobs/{job_id}`).
- WSL2 support: automatic polling (`PollingObserver`, configurable via `poll_interval_seconds`) on `9p`, `drvfs`, and network mounts, Windows path/location formatting, and launching results in Windows apps via `wslview` or `explorer.exe`.
- Personal course evaluation guidance in `docs/eval.md` (with `eval/personal/` git-ignored), real-model demo course evaluation test (`tests/test_demo_eval_slow.py`), and Windows-drive sync benchmark (`bench/drvfs_sync.py`) recorded in `docs/benchmarks.md`.

### Changed
- **Breaking for the PyTorch backend:** the default install no longer includes PyTorch or sentence-transformers (about 350 MB on disk instead of several GB). Users who set `embed_backend = "torch"` must install the `torch` extra.
- `seekr models pull` downloads only the files the active backend needs (about 130 MB for the embedding model and 90 MB for the reranker with ONNX).
- **Upgrading from 0.1.x:** run `seekr models pull` once (add `--rerank` if you use reranking) to fetch the ONNX model files; until then searches that need the model fail with a message naming that command. Existing indexes keep working without a reindex. `seekr models pull` now also stops with an explanation when a configured model publishes no ONNX export.
- Upgraded SQLite index schema to `schema_version = 3` (with automatic transactional migration from v1/v2) to store and index `module` and `week` document metadata.
- Updated `README.md` quickstart to use `seekr init`, refreshed feature/architecture docs, and re-recorded `docs/img/demo.gif`.

### Fixed
- Handled `python-docx` XML schema validation errors cleanly as `ExtractionError` instead of crashing indexing.
- Checked `tree-sitter-language-pack` `downloaded_languages` when `LanguageRegistry` lacks a preloaded parser so downloaded grammars are recognized offline.
- Validated `remove_root` paths strictly and propagated `SEEKR_CONFIG` to background web model-download subprocesses.

## [0.1.1] - 2026-11-22

### Added
- Built-in demo course sample corpus and quickstart demo support.
- Packaging and release tools (`build>=1.2`, `twine>=5`) in `dev` optional dependencies.

### Changed
- Renamed the PyPI distribution from `seekr` to `seekr-search` while keeping the `seekr` CLI command, Python package, and config/data directories.
- Updated project metadata URLs to `https://github.com/Swarsin/seekr`.
- Reworked `README.md` for PyPI installation and moved detailed benchmarks to `docs/benchmarks.md`.

### Fixed
- Handled Unix domain socket paths exceeding the platform `AF_UNIX` length limit with clean fallback to local search and `stopped` status reporting.
- Consolidated missing tree-sitter grammar warnings into a single summary per `seekr index` run.

## [0.1.0] - 2026-11-21

### Added
- Hybrid keyword (SQLite FTS5 with Porter stemming) and semantic vector (`sqlite-vec`) search fused with Reciprocal Rank Fusion (`k=60`) and optional local cross-encoder reranking (`--rerank`).
- Default `int8` vector quantization (with `float32` legacy DB compatibility) and transactional schema migrations (`schema_version = 2`).
- Incremental, batched indexing across PDF, Markdown, plain text, code (tree-sitter AST chunking with line fallback), and transcripts with chunk-hash vector reuse.
- Warm Unix-socket query daemon (`seekr daemon`) with filesystem watcher (`watchdog`) and automatic local search fallback.
- Stdio MCP server (`seekr mcp`) exposing `search` and `similar` tools for coding agents.
- Local browser web interface (`seekr web`) served on `127.0.0.1` for folder setup, indexing, and search.
- `seekr similar` command for finding semantically related files from stored document vectors.
- CLI search and inspection features: `--json`, `--explain`, `--kind`, `--since`, interactive `seekr pick`, `seekr open`, `seekr status`, `seekr doctor`, `--verbose` structured logging, and `seekr completion` (`bash`, `zsh`, `fish`).
- Offline-only model loading enforcement (`seekr.offline`) and BEIR evaluation harness (`seekr eval` with SciFact, NFCorpus, and FiQA converters).
