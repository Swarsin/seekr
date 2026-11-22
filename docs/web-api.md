# Seekr local web API contract

This document is the contract for the local Seekr web interface. Any change to
the contract must update this file and `src/seekr/web_contract.py` together.

The service is a single-user, loopback-only tool served at
`http://127.0.0.1:<actual-bound-port>`. It is not a public server, a multi-user
authentication boundary, or a defence against malware already running as the
same user. Python's standard-library HTTP server, a small JSON API, and
packaged plain HTML/CSS/JavaScript are sufficient. No Node build step, web
framework, CDN, external asset, or new runtime dependency is part of the
contract.

**Deliberate 403 note:** headerless `curl` requests to `/api/session` and the
static asset routes are intentionally rejected with `403 forbidden`. Their
message must hint that browser-origin headers are required. Normal browser
navigation and subresources send the required `Sec-Fetch-Site`/`Origin` values;
non-session API calls additionally need the session token. This is a
diagnosable security rule, not a missing-token accident.

## Authentication and request rules

- `GET /api/session` is the only `/api/*` exemption from the token. It requires
  `X-Seekr-Client: 1` and returns `{token, limits}`. It returns no config
  contents.
- Every other `/api/*` call requires `X-Seekr-Token` with the ephemeral token
  from session. Compare it with `hmac.compare_digest`, never `==`. A missing,
  duplicate, or wrong token is `403 forbidden`.
- The token is memory-only, never in URLs, logs, HTML interpolation, or
  persistent browser storage. It is one process-lifetime value returned by
  every successful session bootstrap; it is not rotated per request, so a
  refresh and multiple tabs can reuse it. Restarting the server invalidates it.
- Host must be exactly `127.0.0.1:<actual bound port>`. Do not accept
  `localhost`, a different port, a missing Host, or a forwarded-host alias.
  `--port 0` records the actual bound port before any request is validated.
- If `Origin` is present, it must equal `http://127.0.0.1:<actual-port>`.
  `Sec-Fetch-Site`, when present, must be `same-origin` or `none`; `cross-site`
  and `same-site` are `403 forbidden`.
- A request with both `Origin` and `Sec-Fetch-Site` absent (for example curl)
  is allowed only with the token. `/api/session` therefore requires at least
  one of a matching `Origin` or `Sec-Fetch-Site` in
  `{same-origin, none}`. Static assets have no token, so they also require at
  least one of a matching `Origin` or an allowed `Sec-Fetch-Site`; normal
  browser navigation sends `none` and same-origin subresources send
  `same-origin`.
- No CORS headers are ever emitted. Do not answer preflight. `OPTIONS` returns
  `405 method_not_allowed` immediately after Host validation, before token or
  fetch-metadata checks, with no `Access-Control-Allow-*` headers. A cross-site
  preflight therefore fails without a usable CORS response.
- POST bodies require `Content-Type: application/json` (an optional
  `charset=utf-8` is allowed; other media types/characters are `415
  unsupported_media_type`). `Content-Length` is required and must be an
  unambiguous integer. `Transfer-Encoding`/chunked requests are `415
  unsupported_media_type`. A body over 64 KiB is `413 payload_too_large`.
- GET routes take no request body. `Content-Length: 0` is allowed, but a
  positive length, a body, or `Transfer-Encoding` on a GET is
  `400 invalid_request`.
- Reject unknown fields, wrong JSON types, duplicate security-sensitive
  headers, duplicate `Content-Length`, booleans used as integers, and
  malformed JSON with `400 invalid_request`. Do not coerce values.
- Host/origin/token checks run before reading a request body. For non-`OPTIONS`
  requests, method is checked after the route is resolved; `OPTIONS` is
  rejected before routing. API/session responses are `Cache-Control: no-store`.
- Query strings are accepted **only** on `GET /api/preview` and
  `GET /api/preview/page`, and **only** as an exact single parameter
  `chunk=<id>` matching `^[1-9]\d{0,18}$` (no empty keys, duplicate keys, or
  extra parameters). Missing or malformed preview query strings return
  `400 invalid_request` (after Host/Origin/Fetch-Metadata/token checks). All
  other known routes require exact request targets and reject any query string
  with `400 invalid_request`.
- Request bodies and document text are never logged. Error messages must be
  safe for display and must not include document text.

## Endpoint matrix

| Method and route | Request | Successful response | Error codes |
| --- | --- | --- | --- |
| `GET /api/session` | No body; no query; `X-Seekr-Client: 1`; exact Host; valid Origin/Sec-Fetch as above | `200 {token: string, limits: object}`; no config contents | `400 invalid_request`, `403 forbidden`, `405 method_not_allowed`, `500 internal_error` |
| `GET /api/status` | Token; no body; no query | `200` status object; never loads a model or takes the lifecycle lock | `400 invalid_request`, `403 forbidden`, `405 method_not_allowed`, `500 internal_error` |
| `GET /api/roots` | Token; no body; no query | `200 {roots: string[], config_path: string, revision: string}`; roots in stored order and form | `400 invalid_request`, `403 forbidden`, `405 method_not_allowed`, `409 config_invalid`, `500 internal_error` |
| `POST /api/roots` | Token; no query; JSON `{path: string, revision: string}` | `200 {roots, config_path, revision, daemon_restart_required: boolean, warning: string or null}` | `400 invalid_request`, `400 invalid_root`, `400 roots_limit`, `403 forbidden`, `405 method_not_allowed`, `409 busy`, `409 stale_config`, `409 duplicate_root`, `409 config_invalid`, `413 payload_too_large`, `415 unsupported_media_type`, `500 internal_error`, `500 config_write_failed` |
| `POST /api/roots/remove` | Token; no query; JSON `{path: string, revision: string}` | `200 {roots, config_path, revision, daemon_restart_required: boolean, warning: string or null}` | `400 invalid_request`, `400 invalid_root`, `403 forbidden`, `405 method_not_allowed`, `409 busy`, `409 stale_config`, `409 daemon_active`, `409 config_invalid`, `413 payload_too_large`, `415 unsupported_media_type`, `500 internal_error`, `500 config_write_failed` |
| `GET /api/course` | Token; no body; no query | `200 {modules: [{name, display, weeks, files}]}` | `400 invalid_request`, `403 forbidden`, `405 method_not_allowed`, `409 busy`, `409 config_invalid`, `409 incompatible_index`, `500 internal_error`, `503 index_unavailable` |
| `GET /api/preview?chunk=<id>` | Token; no body; exact query `chunk=<^[1-9]\d{0,18}$>` | `200 {chunk, kind, path, locator, location, text, page_image}` | `400 invalid_request`, `403 forbidden`, `404 not_found`, `405 method_not_allowed`, `409 busy`, `409 config_invalid`, `409 incompatible_index`, `500 internal_error`, `503 index_unavailable` |
| `GET /api/preview/page?chunk=<id>` | Token; no body; exact query `chunk=<^[1-9]\d{0,18}$>` | `200` raw `image/png` bytes (`Content-Type: image/png`, `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, no CORS) | `400 invalid_request`, `403 forbidden`, `404 not_found`, `405 method_not_allowed`, `409 busy`, `409 config_invalid`, `409 incompatible_index`, `500 internal_error`, `503 index_unavailable` |
| `POST /api/open` | Token; no query; JSON `{chunk: int}` (`> 0`, strict integer, no unknown keys) | `200 {opened: true}` | `400 invalid_request`, `403 forbidden`, `404 not_found`, `405 method_not_allowed`, `409 busy`, `409 config_invalid`, `409 incompatible_index`, `413 payload_too_large`, `415 unsupported_media_type`, `500 internal_error`, `503 index_unavailable` |
| `POST /api/search` | Token; no query; JSON `{query, mode?, k?, per_file?, path?, kind?, since?, module?, week?, spread?}` | `200 {hits: Hit[], took_ms: number, spread?: SpreadGroup[]}` | `400 invalid_request`, `403 forbidden`, `405 method_not_allowed`, `409 busy`, `409 incompatible_index`, `409 config_invalid`, `413 payload_too_large`, `415 unsupported_media_type`, `500 internal_error`, `503 index_unavailable`, `503 model_unavailable`, `503 daemon_failure` |
| `POST /api/model/check` | Token; no query; JSON exactly `{}` | `202 {job_id: string}`; job result arrives through the job endpoint | `400 invalid_request`, `403 forbidden`, `405 method_not_allowed`, `409 busy`, `409 config_invalid`, `413 payload_too_large`, `415 unsupported_media_type`, `500 internal_error` |
| `POST /api/index` | Token; no query; JSON exactly `{}`; targets come only from validated config | `202 {job_id: string}` | `400 invalid_request`, `400 no_roots`, `403 forbidden`, `405 method_not_allowed`, `409 busy`, `409 daemon_active`, `409 incompatible_index`, `409 config_invalid`, `413 payload_too_large`, `415 unsupported_media_type`, `500 internal_error`, `503 index_unavailable`, `503 model_unavailable` |
| `POST /api/setup/download` | Token; no query; JSON `{}` or `{rerank?: boolean}` | `202 {job_id: string}`; runs model/grammar download job | `400 invalid_request`, `403 forbidden`, `405 method_not_allowed`, `409 busy`, `409 config_invalid`, `413 payload_too_large`, `415 unsupported_media_type`, `500 internal_error` |
| `GET /api/jobs/{job_id}` | Token; no body; no query; job ID is a server-issued opaque string | `200` job object with state, operation, timestamps, report/error, and truncation flags | `400 invalid_request`, `403 forbidden`, `404 job_not_found`, `405 method_not_allowed`, `500 internal_error` |
| `GET /`, `/index.html`, `/styles.css`, `/app.js` | No token; no query; exact Host; matching Origin or allowed `Sec-Fetch-Site` required | `200` packaged asset with the correct MIME type | `400 invalid_request` for query string; `404 not_found` for a missing allowlisted asset or unknown path; `405 method_not_allowed` for non-GET; `403 forbidden` for Host/Origin/Sec-Fetch |

Unknown routes return `404 not_found`. There is no directory listing and no
document-serving route. A missing allowlisted asset is `404 not_found`, not a
fallback to another file.

The asset allowlist is exactly `("/", "/index.html", "/styles.css",
"/app.js")`. The frozen MIME types are:

| Route | Content-Type |
| --- | --- |
| `/`, `/index.html` | `text/html; charset=utf-8` |
| `/styles.css` | `text/css; charset=utf-8` |
| `/app.js` | `text/javascript; charset=utf-8` |

Assets are loaded from package resources with
`importlib.resources.files("seekr") / "web_assets"`, never from the working
directory. `/` and `/index.html` serve the same packaged entry document.

`GET /api/roots` uses the shared `RootsView` shape. Its `to_dict()` always
includes `daemon_restart_required` and `warning`; for GET those are the defaults
`false` and `null`. The GET row above names the information-bearing fields; a
client must tolerate the two default-valued write-metadata fields.
`POST /api/roots` sets them when a live daemon requires a restart.

## State sets

- Index states: `missing`, `empty`, `ready`, `incompatible`, `error`.
- Model states: `unchecked`, `loading`, `ready`, `unavailable`. "Checking" in
  the UI corresponds to `loading`.
- Job states: `running`, `succeeded`, `partial`, `failed`. `queued` does not
  exist.
- Job operations: `model_check`, `index`, `download`.
- Activity lane values: `idle`, `search`, `model_check`, `index`, `download`.

`ready` must reflect a successful local check/load, not merely a directory that
happens to exist. `partial` is used if and only if a completed index report has
a non-empty `errors` list. Warnings alone remain `succeeded` and remain
visible.

## Limits

- Request body: 64 KiB.
- Query: 2,000 characters.
- Hits: `1 <= k <= 50`.
- Per file: `1 <= per_file <= 10`.
- Roots: 50.
- Submitted root path: 4,096 UTF-8 bytes.
- Retained jobs: 20; evict the oldest finished job first.
- Diagnostics: first 100 errors and first 100 warnings, with full totals and
  truncation flags.
- Handler concurrency: 8 request-handler threads; bounded accept backlog; no
  unbounded thread-per-connection.
- Socket read/request timeouts: 10 seconds; incomplete requests are closed.
- Config writes: serialized; revision rechecked immediately before replace.
- Search `took_ms`, job `elapsed_ms`, and status counts are server-computed.

`GET /api/session` returns these exact `limits` keys:

```json
{
  "request_body_bytes": 65536,
  "query_chars": 2000,
  "k_min": 1,
  "k_max": 50,
  "per_file_min": 1,
  "per_file_max": 10,
  "roots_max": 50,
  "root_path_bytes": 4096,
  "jobs_retained": 20,
  "errors_retained": 100,
  "warnings_retained": 100,
  "handler_threads": 8,
  "read_timeout_s": 10
}
```

Do not add undocumented keys to `limits`; the frontend may rely on these names.

## Status response

```json
{
  "index": {
    "state": "missing | empty | ready | incompatible | error",
    "documents": 0,
    "chunks": 0,
    "size_bytes": 0,
    "last_error_count": 0,
    "model": null,
    "dim": null,
    "vector_quant": null,
    "schema_version": null,
    "message": null
  },
  "model": {
    "state": "unchecked | loading | ready | unavailable",
    "model": "configured model name",
    "dim": null,
    "message": null
  },
  "daemon": {"present": false, "socket": "/run/user/…/seekr.sock"},
  "config": {
    "state": "ok | invalid",
    "path": "…/config.toml",
    "revision": "sha256-hex-or-null",
    "roots": 0,
    "message": null
  },
  "activity": {"lane": "idle | search | model_check | index | download", "job_id": null}
}
```

Counts and metadata are `null` when they cannot be read safely. `size_bytes` is
the main database plus `-wal`/`-shm` files when present. `daemon.present` is
only a socket-liveness probe; no activity or indexing claim is attached.
`config.state` is `ok` when the config is absent (defaults) or valid, and
`invalid` when `load_config` raises `ConfigError`; status still returns `200`
with `invalid`. `config.revision` is the SHA-256 of the raw config bytes
whenever the path is readable (including invalid TOML), or `null` when it is
not; `config.roots` is the stored root count when the file is parseable,
otherwise `null`, and is never a directory listing. Status must not initialize
a missing database, create tables, take the shared lifecycle lock, or trigger a
model load.

Index state mapping:

- `missing`: the configured database file does not exist.
- `empty`: the database exists but has no documents/chunks, or it has no model
  metadata and no data.
- `ready`: valid schema/model/quantization metadata and at least one document
  or chunk.
- `incompatible`: `ModelMismatchError`, `SchemaVersionError`, or data present
  without model metadata.
- `error`: any other `SearchError`, `sqlite3.Error`, or read-only inspection
  failure.

`error` remains a successful `200` status response; local search and index
preflight map the same inspection failure to `503 index_unavailable`.

Use the existing `_inspect_db`/`_validated_store_info` behavior (private helpers
may be imported explicitly or copied minimally) and a read-only count query. Do
not create tables.

## Roots response semantics

- `GET /api/roots` returns roots exactly as stored, in config order, plus the
  absolute config path and the raw-bytes revision. Missing config: empty roots,
  revision of empty bytes.
- `GET /api/roots` displays any `load_config`-valid array of strings.
  `POST /api/roots` edits only the two frozen TOML shapes and returns
  `409 config_invalid` for other valid-but-unsupported layouts.
- `POST /api/roots` accepts exactly `{path, revision}`. `path` must be an
  existing, readable directory after
  `Path.expanduser().resolve(strict=True)`. The stored value is the canonical
  absolute path.
- Deduplicate against existing roots after canonicalizing them. Existing
  entries may be files, relative paths, or `~` paths and stay valid and display
  as stored.
- A duplicate is `409 duplicate_root`. The 50-root limit is `400 roots_limit`.
  A path over 4,096 UTF-8 bytes, missing, not a directory, unreadable,
  containing NUL, or not valid UTF-8 is `400 invalid_root`.
- A stale or externally changed revision is `409 stale_config`. An unsupported
  or ambiguous TOML shape is `409 config_invalid` with the config path. A write
  failure after validation is `500 config_write_failed`.
- Adding a root while an engine operation is active is `409 busy`. Adding a
  root while the daemon socket is live succeeds but returns
  `daemon_restart_required: true` and a warning to restart the daemon.
- Saving a root never indexes, deletes, or modifies source files.

Config-write safety details are part of the contract:

- Support only two TOML shapes: (a) no top-level `roots` key, adding
  `roots = [...]` in the top-level portion, or (b) a top-level single-line
  `roots = [...]` value, including `[]`, optionally with a trailing comment.
  Replacing shape (b) changes only the array value span; every byte outside the
  inserted/replaced span is preserved, including key spelling, assignment
  whitespace, a trailing comment, and the line ending.
- In a flat file with no table headers, append an absent key at EOF. If a table
  header is present, refuse an absent top-level key as ambiguous rather than
  adding it inside a table.
- Refuse multiline arrays, comments inside the array, quoted or dotted keys,
  duplicate `roots` keys, table forms (`[roots]`, `[[roots]]`, or a `roots`
  assignment inside another table), and any ambiguous expression, all as
  `409 config_invalid` with the config path for manual correction.
- Use TOML-safe basic-string escaping for new paths: quotes, backslashes,
  Unicode, tabs, newlines, and other control characters must be escaped
  correctly. Reject NUL and invalid Unicode. JSON string escaping alone is not
  sufficient.
- Validate the full original and candidate with `tomllib`, then validate the
  candidate with `load_config` on the temp file, not just `tomllib`. Verify that
  the parsed candidate roots equal the expected list.
- Missing config file: create it privately with mode `0600` and only the new
  roots; create a missing parent directory with mode `0700` (do not chmod an
  existing parent). Existing file: preserve its mode. Refuse a symlink target
  as `409 config_invalid`.
- Write a sibling temp file, flush and `fsync` it, and `os.replace` it
  atomically; fsync the directory when practical. Serialize web writes.
  Revision is the SHA-256 hex digest of the raw config bytes (the empty bytes
  when the file is missing). Recheck the revision immediately before
  replacement. A mismatch or external edit returns `409 stale_config`.
- Do not claim atomic compare-and-swap against uncooperative external editors.
  The revision check narrows the race; it cannot eliminate it.
- Saving a root never indexes or deletes files. Existing indexed documents
  remain searchable as they do in the CLI.

## Search semantics

- `query` is required and is a string of at most 2,000 characters. Existing
  behavior for empty or whitespace-only queries (empty hit list without opening
  a database or model) is preserved. Unknown fields/types are
  `400 invalid_request`.
- `mode` is `keyword`, `vector`, or `hybrid`; default `hybrid`. `k` defaults to
  10 and is 1–50; `per_file` defaults to 2 and is 1–10; booleans are rejected.
- `path` is an optional path-prefix filter (default null), `kind` uses existing
  kind names (default null), `since` uses the existing duration/ISO parser
  (default null), `module` is an optional module name or unique prefix
  (default null), and `week` is an optional week number (`"4"`) or inclusive
  range (`"3-5"`, default null). Invalid filter values or unknown/ambiguous
  modules are `400 invalid_request`.
- `spread` is an optional boolean (default `false`; non-booleans including
  `null` are `400 invalid_request`). When `true`, the response includes a
  `spread` array of `{module, week, first, chunk_ids}` groups summarizing where
  the topic appears across modules and weeks, computed from a second retrieval
  with `k=30`, `per_file=2`, and the same mode and filters (or `[]` when the
  query is blank or main hits are empty). When `spread` is `false` or omitted,
  `spread` is omitted from the response.
- Preserve result ordering from `Searcher`/`get_hits`. Serialize `Hit` fields
  `chunk_id`, `path`, `locator`, `loc_start`, `snippet`, `score`, `kind`,
  `module`, `week`, `highlights` (`number[][]` of `[start, end]` character
  offsets into `snippet`), `location`, and `posix_location`; omit `explain`
  entirely (do not emit `explain: null`). Do not mislabel scores as confidence
  percentages. Rerank/explain controls are out of scope.
- `took_ms` is the server-measured search duration, excluding HTTP body parsing
  and response writing. It is a number, not a formatted string.
- Keyword search bypasses the engine lane with the true read-only path below
  and is available during a web index job. It must not use `_build_searcher` or
  a writable `Store`. For a local search (after the daemon attempt), a missing
  or empty index returns `200` with empty hits. An incompatible index returns
  `409 incompatible_index`; an index inspection error returns
  `503 index_unavailable`. Hybrid/vector local search uses the lane; a second
  engine operation is `409 busy`. If a hybrid/vector model is known
  unavailable, return `503 model_unavailable`; keyword never needs the model.
- After the daemon attempt, a missing or empty local index short-circuits to
  `200` with empty hits for every search mode, without loading the model or
  acquiring the lane. Search preflight precedence after
  authentication/body validation is: config validity →
  `409 incompatible_index` → `503 index_unavailable` →
  `503 model_unavailable` (hybrid/vector only) → lane `409 busy`. After config
  validity, the daemon attempt happens before this local index/model/lane
  preflight; a reachable broken daemon is `503 daemon_failure` and is not
  cold-retried.
- If a daemon socket is live, the service calls `try_daemon_hits` outside the
  lane. A returned list, including `[]`, is the response and bypasses the local
  embedder and lane. On `None` (no listener or daemon mismatch), keyword uses
  the read-only path and hybrid/vector acquire the lane and call
  `get_hits(..., use_daemon=False)`. If no daemon socket is live, keyword uses
  the read-only path and hybrid/vector acquire the lane and call
  `get_hits(..., use_daemon=False)`.
- Keyword lock/busy handling uses a 2-second deadline for shared
  lifecycle-lock acquisition and a 2-second SQLite busy timeout. A wait that
  cannot be resolved within its bound returns `409 busy`. Other read-only
  inspection failures map to `503 index_unavailable`, and incompatible
  metadata maps to `409 incompatible_index`. These bounds cover lock/busy
  waiting, not the duration of a successful FTS query.
- Parity: for the same ready index and filters, the web keyword path returns
  the same ordered hits as
  `get_hits(..., mode="keyword", use_daemon=False)`.

### Read-only keyword path

- Open and close Stores inside operations, under lifecycle locks. Do not retain
  a Store, lifecycle lock, or live connection while the web app is idle. Do not
  share live connections between request handlers.
- The engine lane serializes expensive work in this process; it does not
  replace the shared lifecycle lock that protects against `seekr reindex` or
  other processes.
- Keyword search acquires the shared lifecycle lock with a 2-second deadline
  (non-blocking retries), opens `Store(..., read_only=True)`, constructs
  `Searcher` with a lazy/cached embedder that keyword mode never loads, calls
  the existing keyword retrieval path, and closes the Store in a `finally`
  block. It never calls `_build_searcher` or opens a writable Store.
- The web keyword path obtains the stored `dim` and `vector_quant` from the
  same 2-second read-only inspection (a minimal equivalent of
  `_validated_store_info`) and passes them with `cfg.embed_model` to
  `Store(read_only=True)`; it never builds an embedder just to determine
  dimensions.
- There is a minimal, additive, keyword-only `read_only: bool = False` to
  `Store.__init__`; normal `Store(...)` behavior is unchanged and there is no
  schema or migration change. In read-only mode, `Store` opens
  `file:<absolute-db-path>?mode=ro` with `uri=True` and a 2-second busy
  timeout, sets `PRAGMA query_only=ON`, and skips parent-directory creation,
  `journal_mode`/`synchronous` writes, `executescript(_SCHEMA_SQL)`, schema and
  metadata writes/migrations, sqlite-vec extension loading, and vector-table
  creation. It validates the existing schema and metadata with SELECTs only; a
  read-only metadata check must not migrate an older schema. Missing, corrupt,
  or incompatible databases raise clear
  `sqlite3.Error`/`ModelMismatchError`/`SchemaVersionError` exceptions; the web
  layer maps an unexpected missing/corrupt read-only Store to
  `503 index_unavailable` and mismatch/schema to `409 incompatible_index`. The
  normal preflight handles a stable missing/empty index as empty hits. Every
  write path (`transaction(write=True)`, `upsert_document`, `move_document`,
  `replace_chunks`, `delete_document`, `set_last_error_count`) raises a clear
  `RuntimeError("store is read-only")` before executing SQL. Read methods and
  `close()` work normally. Read-only mode is for keyword/metadata reads, not
  vector queries; the keyword path never calls the unloaded vector extension
  or vector table.
- The web keyword path preflights the index state with a read-only inspection
  using the same 2-second busy timeout (not the status helper's 5-second
  default): `missing` or `empty` returns empty hits without constructing the
  read-only Store; only `ready` opens it. A race that makes a previously ready
  DB missing/corrupt maps to `503 index_unavailable`; incompatible metadata
  maps to `409 incompatible_index`.
- An unavailable lifecycle lock or SQLite busy wait that cannot be resolved
  within its 2-second bound maps to `409 busy`. These bounds cover waiting to
  start, not the duration of a successful read.
- Status bypasses the shared lifecycle lock and never waits behind the engine
  lane.

### Daemon-first resolution and local-only fallback

These small hooks sit on the shared search path:

```python
def get_hits(
    ...,
    *,
    searcher: Searcher | None = None,
    embedder: Embedder | None = None,
    use_daemon: bool = True,
) -> list[Hit]: ...

def try_daemon_hits(
    query: str,
    k: int = 10,
    mode: SearchMode = "hybrid",
    per_file: int = 2,
    rerank: bool = False,
    path_prefix: str | None = None,
    *,
    explain: bool = False,
    kind: str | None = None,
    since: str | None = None,
    module: str | None = None,
    week: str | None = None,
    cfg: Config | None = None,
) -> list[Hit] | None: ...

def _build_searcher(
    cfg: Config,
    mode: str,
    rerank: bool,
    *,
    embedder: Embedder | None = None,
) -> Searcher | None: ...
```

- `searcher` takes precedence. If both `searcher` and `embedder` are supplied,
  the injected `searcher` is used and `embedder` is ignored.
- If `embedder` is supplied, verify `embedder.model_name == cfg.embed_model`
  before using it and fail clearly with `SearchError` on a mismatch. Tests
  using `FakeEmbedder` must set `cfg.embed_model = "fake"`.
- When `embedder` is `None`, call `_build_searcher(cfg, mode, rerank)` exactly
  as today, so existing callers, `build_embedder` monkeypatches, and backward
  compatibility tests continue to hold. Only pass the injected embedder as a
  keyword argument.
- `use_daemon=False` skips `_daemon_search` and performs a local-only search;
  the default `True` preserves existing callers and daemon-preference behavior.
- `try_daemon_hits(...)` reuses the same query/filter validation as `get_hits`
  (extract a shared private validator rather than duplicating parsing), calls
  `_daemon_search`, and returns daemon hits, including an empty list, or `None`
  when no listener is present or `DaemonMismatchError` invalidates the daemon.
  Other `DaemonError` exceptions propagate. It preserves the
  empty/whitespace query short-circuit and never opens a Store or embedder.
- Web flow: if a daemon socket is live, call `try_daemon_hits` outside the
  lane. A returned list is the response and does not touch the lane. On `None`
  (no listener or daemon mismatch), keyword uses the read-only path;
  hybrid/vector acquire the lane and call `get_hits(..., use_daemon=False)`. A
  reachable broken daemon raises `DaemonError` and maps to
  `503 daemon_failure`; do not cold-retry.
- Vector and hybrid modes may access `.dim` to validate the stored dimension.
  Keyword mode must not access `.dim` or run inference; it remains model-free.
- Daemon results bypass injection entirely. The web service caches one
  embedder per active model identity and rebuilds it when the config changes; a
  changed `embed_model` always invalidates the cache. It never retains a Store
  while idle.
- Existing daemon semantics are unchanged: a daemon mismatch falls back to the
  local index; a reachable broken daemon raises and is not silently hidden.

### Exact search error mappings

After authentication and body/field validation:

| Condition | Result |
| --- | --- |
| Invalid config (`ConfigError`) | `409 config_invalid` |
| Live daemon result, including `[]` | `200` with daemon hits; no lane/embedder |
| Reachable daemon search/protocol failure | `503 daemon_failure`; no cold retry |
| No daemon listener or daemon mismatch | Continue to local preflight/fallback |
| Missing or empty local index | `200` with `hits: []`; no Store or model |
| `ModelMismatchError`, `SchemaVersionError`, data without model metadata | `409 incompatible_index` |
| Other `SearchError`, `sqlite3.Error`, read-only inspection failure | `503 index_unavailable` |
| Hybrid/vector with model state `unavailable` | `503 model_unavailable`; do not retry |
| Local keyword lifecycle lock or SQLite busy beyond 2 seconds | `409 busy` |
| Second engine operation while the lane is active | `409 busy` |
| Unexpected unclassified failure | `500 internal_error` |

The local missing/empty short-circuit happens before model loading and lane
acquisition for all modes. Hybrid/vector may build once inside the lane when
the model state is `unchecked`, transitioning through `loading` to
`ready`/`unavailable`; a successful load is cached. Keyword never loads the
model.

## Model check semantics

- `POST /api/model/check` accepts exactly `{}` and returns `202 {job_id}`.
- The job operation is `model_check`. It runs `build_embedder(cfg).dim` inside
  the engine lane. On success it caches the embedder, sets model state `ready`,
  and records the model name and dimension. On failure it sets `unavailable`
  with the error text and caches nothing. No automatic retry.
- A successful or failed check is reported through the job endpoint. Status
  never triggers a check or load.
- A second model check while one is running is `409 busy`. A new explicit check
  is allowed after `unavailable` and replaces the failed state on success.
- A model check job that succeeds has `report: {"model": "…", "dim": 384}`.

## Index and daemon conflict

- `POST /api/index` accepts exactly `{}` and targets only validated config
  roots. Empty roots is `400 no_roots`. An incompatible existing index is
  `409 incompatible_index`; an index inspection error is
  `503 index_unavailable`; a missing or empty index is allowed and may be
  created by the job. A known-unavailable model is `503 model_unavailable`.
- If a seekr daemon socket is live, return `409 daemon_active` before starting
  a job. The message explains that the daemon maintains the index and that the
  user must stop the running `seekr daemon` process before a web scan.
- Index preflight precedence is: config validity → `no_roots` →
  `daemon_active` → `incompatible_index` → `index_unavailable` →
  `model_unavailable` → lane `busy`. All of these are checked before the job is
  accepted; a live daemon therefore yields `daemon_active` even if the lane is
  also busy.
- Never claim to detect whether the daemon is actively indexing. A socket probe
  is the only check.
- The UI disables "Update index" and displays the daemon as present. It may
  still use daemon-backed search.
- A web-initiated index is incremental and does not replace the live database
  with a fresh one. Use `Indexer.sync` and its batching. `Indexer.sync` remains
  the owner of the persisted last-error count; the web job must not duplicate
  that write.
- Shutdown waits for an active `Indexer.sync`; there is no cancellation hook
  and a browser disconnect is not cancellation.

## Remove folder semantics

- `POST /api/roots/remove` accepts `{path, revision}` and runs inside the
  engine lane (`409 busy` when another engine operation is running).
- If a seekr daemon socket is live, refuse with `409 daemon_active` before
  editing config or touching the index database.
- Matches the submitted `path` canonically against stored config roots (allowing
  a previously deleted directory on disk to be removed cleanly); if no
  configured root matches, raises `400 invalid_root`.
- Enforces the same TOML-preserving editor and `409 stale_config` revision
  checks as `POST /api/roots`.
- When the index database exists and is ready, purges indexed documents under
  the removed root via `Store.delete_documents_under` (component-aware prefix,
  so removing `/tmp/root` never deletes `/tmp/root2`), along with their chunks,
  FTS rows, and vectors.

## Course overview, preview, and open semantics

- `GET /api/course` reads `Store.course_summary()` using a read-only Store
  without loading any embedding model. Missing or empty databases return
  `{"modules": []}`. Unassigned documents (`module is None`) are omitted;
  named modules are sorted case-insensitively by `name`, with `display`
  formatted by `course.display_module(name)`, unique sorted non-null integer
  `weeks`, and total `files` across all indexed files in that module.
- `GET /api/preview?chunk=<id>` looks up an indexed chunk in a read-only Store
  and returns `{chunk, kind, path, locator, location, text, page_image}`. Unknown
  chunk IDs return `404 not_found`. `location` uses WSL-aware Windows path
  formatting when running on WSL; `page_image` is `true` when `kind == "pdf"`
  and the PDF file exists on disk.
- `GET /api/preview/page?chunk=<id>` renders the 1-based page from `locator`
  (`p.<N>`, defaulting to page 1 when `locator` is empty) using PyMuPDF at zoom
  `1.5`, scaling down (`min(1.5, 2000.0 / page.rect.width)`) so the rendered PNG
  width is at most `2000` pixels. Unknown chunks, non-PDF chunks, missing or
  unreadable PDFs, and out-of-range page numbers return `404 not_found`. The
  response emits raw `image/png` bytes with `Content-Type: image/png`,
  `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, the baseline
  security headers, and no CORS headers.
- `POST /api/open` accepts `{chunk: int}` (strict integer `> 0`), resolves the
  indexed chunk, builds the argv via `seekr.opener.open_command` (WSL-aware),
  and executes it directly without a shell (`subprocess.run(argv, check=False)`),
  honoring `opener.OPEN_SUCCESS_CODES`. Unknown chunks return `404 not_found`;
  opener launch failures return `500 internal_error`.

## Model download setup semantics

- `POST /api/setup/download` accepts `{}` or `{"rerank": <bool>}` (strict
  boolean defaulting to `false`; unknown keys or non-booleans return
  `400 invalid_request`).
- Runs as a background job with operation `"download"` in the engine lane
  (`409 busy` if occupied).
- Spawns `[sys.executable, "-m", "seekr", "models", "pull", "--yes", "--grammars"]`
  (plus `"--rerank"` when `rerank` is `true`) in a subprocess with
  `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, and `HF_DATASETS_OFFLINE` scrubbed
  from the child environment so downloads succeed even when the web server
  configured an offline parent process.
- Captures stdout and stderr, retains the last `50` output lines in
  `report.lines`, and transitions the job to `"succeeded"` on exit `0` or
  `"failed"` with a safe error summary on non-zero exit.

## Job response and report

```json
{
  "job_id": "server-issued-opaque-string",
  "operation": "index | model_check | download",
  "state": "running | succeeded | partial | failed",
  "started_at": "2026-11-20T12:34:56Z",
  "finished_at": null,
  "elapsed_ms": 1234,
  "report": null,
  "error": null
}
```

Timestamps are ISO 8601 UTC strings; `finished_at` is `null` while running.
`elapsed_ms` is a server-computed number for both running and finished jobs.
`error` is a safe display string or `null`.

For a completed index, `report` is:

```json
{
  "added": 0,
  "updated": 0,
  "unchanged": 0,
  "removed": 0,
  "skipped": 0,
  "errors": [{"path": "…", "message": "…"}],
  "warnings": [{"path": "…", "message": "…"}],
  "error_total": 0,
  "warning_total": 0,
  "errors_truncated": false,
  "warnings_truncated": false
}
```

Only the first 100 errors and first 100 warnings are serialized; totals and
truncation flags always describe the full report. `partial` is used iff
`error_total > 0`. Warnings alone do not make a job partial. A failed job may
have `report: null` and an `error` string. For a successful `model_check`,
`report` is `{"model": "…", "dim": 384}`. For a `download` job, `report` is
`{"rerank": false, "returncode": 0, "lines": ["..."]}` containing at most the
last 50 captured output lines. `Indexer.sync` remains the owner of
the persisted last-error count.

Retain at most 20 jobs in memory, dropping the oldest finished job first.
Unknown, expired, or malformed job IDs return `404 job_not_found`. There is no
`queued` state.

## Error envelope and frozen matrix

```json
{"error": {"code": "invalid_request", "message": "…", "fields": {"path": "…"}}}
```

`fields` is optional and contains only safe per-field strings. Never include
request bodies, document text, tracebacks, or the token.

| HTTP | Code | Meaning and applicable endpoints |
| --- | --- | --- |
| 400 | `invalid_request` | Malformed JSON, unknown fields/types, duplicate sensitive headers, missing content length, invalid or disallowed query strings, invalid search filters/mode/k/per_file/spread, invalid chunk/rerank fields, bootstrap-header failure. All routes. |
| 400 | `invalid_root` | Submitted root is missing, not a readable directory, contains NUL, is not valid UTF-8, exceeds 4,096 bytes, or is not configured on remove. `POST /api/roots`, `POST /api/roots/remove`. |
| 400 | `roots_limit` | The 50-root limit would be exceeded. `POST /api/roots`. |
| 400 | `no_roots` | Index requested with no configured roots. `POST /api/index`. |
| 403 | `forbidden` | Bad Host, bad Origin, cross-site/same-site fetch metadata, missing/wrong token. All token-protected routes and static assets for Host/fetch violations. |
| 404 | `not_found` | Unknown route, unknown path, missing allowlisted asset, or unknown/missing chunk or PDF preview page. Static, unknown, preview, and open routes. |
| 404 | `job_not_found` | Unknown, expired, or malformed job ID. `GET /api/jobs/{job_id}`. |
| 405 | `method_not_allowed` | Wrong method, including `OPTIONS`; no CORS/preflight response. Every route. |
| 409 | `busy` | A second engine operation is active, a root write conflicts with the engine lane, or a read/purge cannot obtain its snapshot within the bounded wait. `POST /api/search`, `POST /api/roots`, `POST /api/roots/remove`, `POST /api/model/check`, `POST /api/index`, `POST /api/setup/download`, `GET /api/course`, `GET /api/preview`, `GET /api/preview/page`, `POST /api/open`. |
| 409 | `stale_config` | Revision mismatch or an external edit was detected before replace. `POST /api/roots`, `POST /api/roots/remove`. |
| 409 | `daemon_active` | A seekr daemon socket is live and a web index or folder removal was requested. `POST /api/index`, `POST /api/roots/remove`. |
| 409 | `incompatible_index` | Stored model/schema/quantization metadata is incompatible with the config. Local search, `POST /api/index`, `GET /api/course`, `GET /api/preview`, `GET /api/preview/page`, `POST /api/open`. |
| 409 | `duplicate_root` | The canonical path already exists in roots. `POST /api/roots`. |
| 409 | `config_invalid` | Config is malformed, unsupported, ambiguous, or unsafe to edit. `GET /api/roots`, `POST /api/roots`, `POST /api/roots/remove`, `GET /api/course`, `GET /api/preview`, `GET /api/preview/page`, `POST /api/open`, `POST /api/search`, `POST /api/model/check`, `POST /api/index`, `POST /api/setup/download`. Status reports invalid config in its `200` body instead. |
| 413 | `payload_too_large` | Body exceeds 64 KiB. Body-taking routes. |
| 415 | `unsupported_media_type` | Wrong Content-Type, unsupported charset, or chunked transfer encoding. Body-taking routes. |
| 500 | `internal_error` | Unexpected server failure or opener launch failure with no safe-specific code. Every route. |
| 500 | `config_write_failed` | Validated config could not be written/replaced. `POST /api/roots`, `POST /api/roots/remove`. |
| 503 | `index_unavailable` | Index inspection reports `error` (corrupt/unreadable database). Status still returns `200` with `index.state = "error"`. `POST /api/search`, `POST /api/index`, `GET /api/course`, `GET /api/preview`, `GET /api/preview/page`, `POST /api/open`. |
| 503 | `model_unavailable` | Hybrid/vector search or index requested while the model is known unavailable. `POST /api/search`, `POST /api/index`. |
| 503 | `daemon_failure` | A reachable daemon failed a search/protocol interaction; do not hide it with a cold local retry. `POST /api/search`. |

Error precedence: Host (403); route existence (404, except `OPTIONS`, which is
405 before routing and before token/fetch checks); known-route method (405);
Origin/fetch/auth (403; session bootstrap header failure is 400); media/length/
body size (415/400/413); JSON/shape/field validation (400); config/resource
preconditions (409/503); then unexpected internal failure (500). Do not read an
unauthenticated body before the earlier checks.

## Localhost security, headers, and caching

- Bind only to `127.0.0.1`. Validate the exact Host, including the actual bound
  port; reject `localhost`, alternate ports, missing Host, duplicate Host,
  spoofed forwarded hosts, and DNS-rebinding-style requests. `--port 0` must
  use the actual port in Host and Origin validation.
- Reject cross-origin requests. When present, `Origin` must equal
  `http://127.0.0.1:<actual-port>` and `Sec-Fetch-Site` must be `same-origin`
  or `none`. `cross-site` and `same-site` are `403`. Absent `Origin` and
  `Sec-Fetch-Site` (for example curl) is allowed only with the token; session
  bootstrap and static assets, which cannot use a token, require a matching
  Origin or an allowed `Sec-Fetch-Site`. Headerless curl is intentionally
  rejected for those routes; the `403 forbidden` message must hint that
  "browser-origin headers are required" so the failure is diagnosable.
- Require `X-Seekr-Client: 1` for the bootstrap and `X-Seekr-Token` for every
  other API call. Compare the token with `hmac.compare_digest`, never `==`. Do
  not enable CORS, answer preflights, or emit `Access-Control-Allow-*` headers.
  `OPTIONS` is `405` after Host validation and before token/fetch checks. The
  token stays in memory; restart invalidates it.
- Accept mutations only as JSON POSTs with the token. No state changes on GET,
  no form-encoded fallback, and no token embedded in HTML through unsafe string
  interpolation. Session bootstrap must not be cross-origin-readable.
- Reject chunked transfer encoding, invalid/ambiguous `Content-Length`,
  duplicate security-sensitive headers, malformed JSON, unsupported methods,
  and bodies over 64 KiB. Apply 10-second read/request timeouts, a bounded
  accept backlog, and at most eight concurrent handler threads.
- API and session responses are `Cache-Control: no-store`. The frozen response
  header baseline is:

```text
Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
Referrer-Policy: no-referrer
```

- Serve scripts/styles locally with no inline `<script>`, no inline styles or
  style attributes, no `eval`, no `document.write`, and no `innerHTML`. Render
  paths, locators, errors, and snippets as text nodes. Query-term highlighting
  may create text nodes and safe `<mark>` elements only.
- No external asset fetch, remote font/image/script URL, telemetry, arbitrary
  document-serving route, arbitrary filesystem read, shell endpoint, arbitrary
  executable launch, directory traversal, or directory listing. Preview and open
  routes accept only server-indexed positive integer chunk IDs (`/api/preview`,
  `/api/preview/page`, `/api/open`), and `/api/open` executes only the
  allowlisted argv produced by `seekr.opener.open_command` without a shell. The
  asset allowlist is exact: `/`, `/index.html`, `/styles.css`, `/app.js`; missing
  assets are `404`.

## Backend protocol and DTOs

`src/seekr/web_contract.py` contains the import-safe executable shape of this
section. The HTTP layer receives a constructor-injected backend, so tests and
frontend development can use synthetic fixtures without a real Store, model, or
config. The protocol is:

```python
class WebBackend(Protocol):
    def status(self) -> StatusView: ...
    def roots(self) -> RootsView: ...
    def add_root(self, path: str, revision: str) -> RootsView: ...
    def remove_root(self, path: str, revision: str) -> RootsView: ...
    def course(self) -> CourseView: ...
    def preview(self, chunk_id: int) -> PreviewView: ...
    def preview_page_png(self, chunk_id: int) -> bytes: ...
    def open_chunk(self, chunk_id: int) -> None: ...
    def search(self, request: SearchRequest) -> SearchResult: ...
    def start_model_check(self) -> str: ...
    def start_index(self) -> str: ...
    def start_download(self, rerank: bool = False) -> str: ...
    def job(self, job_id: str) -> JobView: ...
```

These methods take exactly the arguments shown; there are no additional
undocumented parameters. Implementations own all Store/model/config work and
must respect the lane, daemon, read-only, and lifecycle rules above.

DTOs are frozen dataclasses with `to_dict()` methods:

- `HitDTO(chunk_id, path, locator, loc_start, snippet, score, kind,
  module=None, week=None, highlights=(), location="", posix_location="")`;
  `highlights` is `tuple[tuple[int, int], ...]` and serializes to `number[][]`
  of `[start, end]` offsets; `location` is the human-readable display location
  (Windows drive path when on WSL and mounted from a drive letter, otherwise
  POSIX path) and `posix_location` always uses the POSIX path; no `explain`
  field, and `to_dict()` must not emit `explain`.
- `CourseModuleDTO(name, display, weeks=(), files=0)` and
  `CourseView(modules=())` serialize to
  `{"modules": [{"name": str, "display": str, "weeks": int[], "files": int}]}`.
- `PreviewView(chunk, kind, path, locator, location, text, page_image=False)`
  serializes to `{"chunk", "kind", "path", "locator", "location", "text", "page_image"}`.
- `SearchRequest(query, mode="hybrid", k=10, per_file=2, path=None, kind=None,
  since=None, module=None, week=None, spread=False)`. `to_query_kwargs()` maps
  wire field `path` to `get_hits(path_prefix=...)` and returns exactly `query`,
  `mode`, `k`, `per_file`, `path_prefix`, `kind`, `since`, `module`, and `week`.
  The caller chooses `use_daemon` and the read-only keyword path.
- `SearchResult(hits: tuple[HitDTO, ...], took_ms: float, spread: tuple[dict[str, object], ...] | None = None)`
  serializes to `{"hits": [...], "took_ms": number}`, including `"spread": [...]`
  only when `spread` is not `None`.
- `RootsView(roots: tuple[str, ...], config_path, revision,
  daemon_restart_required=False, warning=None)` serializes all five fields;
  GET uses the defaults noted above.
- `JobView(job_id, operation, state, started_at, finished_at, elapsed_ms,
  report, error)` validates that `operation` is in `JOB_OPERATIONS` and `state`
  is one of `running`, `succeeded`, `partial`, or `failed`. `report` is `null`
  while running or for a failed job without a report.
- `StatusView(index, model, daemon, config, activity)` serializes the five
  documented status objects.

`ApiError(status, code, message, fields=None)` validates that `code` exists in
`ERROR_STATUS` and that `status` matches its frozen value. Its `to_envelope()`
returns exactly `{"error": {"code", "message"}}`, adding `"fields"` only when
there are safe field strings. `LIMITS`, `ERROR_STATUS`, the state/operation tuples
(`INDEX_STATES`, `MODEL_STATES`, `JOB_OPERATIONS`, `JOB_STATES`, `LANES`),
`ASSET_ALLOWLIST`, and `ASSET_MIME_TYPES` in `src/seekr/web_contract.py` are the
executable copies of the frozen lists in this document. The contract module
imports only the standard library; it must not import `http.server`, numpy,
torch, sentence-transformers, sqlite-vec, or any backend module.
