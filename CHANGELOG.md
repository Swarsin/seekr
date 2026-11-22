# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
