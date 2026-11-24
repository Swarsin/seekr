# Seekr web interface

Seekr’s local web interface lets you add or remove folders, download the local search model, update the index, search your files with course filters and cross-course spread summaries, preview matching passages and PDF pages, and open source files directly from results. It is a single-user tool started by one local command (`seekr web`) and uses the same settings and index as the `seekr` CLI and MCP server; there is no separate web-only index. The [web API contract](web-api.md) documents the HTTP API.

**Words used here.** The **index** is Seekr’s local SQLite database of extracted text, passages, and vectors. The **embedding model** is a local model that turns text into vectors for meaning search; the default is `BAAI/bge-small-en-v1.5`. **Hybrid** search combines exact-word and meaning-based matches. The **service** (`seekr service start`) is a background process that serves this web interface, keeps the model warm, and keeps the index current. `seekr web` starts it for you.

## Start and stop

From a terminal where Seekr is installed, run:

```bash
seekr web
```

`seekr web` starts the background service if it is not already running, prints the address to open, and returns to your prompt:

```text
seekr is running at http://127.0.0.1:8000/
```

The service keeps running after the command exits. If one is already running, `seekr web` reuses it and prints its actual address (which can differ from `--port`). If the service does not come up within 10 seconds the command fails; run `seekr service start` to see the error. Stop the service with `seekr service stop`, and check it with `seekr service status` or `seekr status`. To serve in the terminal instead, with the older Ctrl+C behaviour, use `seekr web --foreground`.

Open the printed `http://127.0.0.1:<port>` address. Use it exactly: the server is loopback-only and rejects `http://localhost:<port>`, other host names, and other machines. There is no option to expose it on your LAN or the public internet.

```bash
# Same as the default port.
seekr web --port 8000

# Ask the OS for a free port; use the actual port printed at startup.
seekr web --port 0

# Also open the printed address in the default browser.
seekr web --open-browser
```

`--open-browser` is a convenience. On a headless machine, or if no browser can be launched, Seekr prints why and keeps the service running; open the printed address yourself. With `--foreground`, press Ctrl+C in the terminal running `seekr web` to stop. If an index update is active, Seekr waits for it to finish before exiting, so a large scan can delay shutdown. A second Ctrl+C warns but does not cancel the work. Closing the browser tab neither stops the server nor cancels an index job.

## First run: choose folders

When no folders are configured yet (or when the local embedding model is unavailable), the **Set up Seekr** panel appears at the top of the **Search** tab so you can add a folder directly or download the search model in one click. You can also manage folders at any time on the **Folders** tab.

Seekr searches only the folders listed there; it does not scan your whole home folder. Supported files include PDFs with embedded text, Markdown, plain text, source code, transcripts, Word documents (`.docx`), slides (`.pptx`), and Jupyter notebooks (`.ipynb`). Scanned image-only PDFs are not OCRed. A missing `roots` key, or `roots = []`, means no folders; Seekr does not assume the example paths in the README.

To add a folder:

1. Type the full local path, for example `~/notes` or `~/Documents/coursework`. A leading `~` means your home folder.
2. Choose **Add folder**. The folder must already exist, be a readable directory, and not already be listed.
3. The page shows the saved path. New paths are canonical absolute paths: `~` is expanded and symbolic links are resolved. Existing entries are shown as stored, so an older `~/notes` entry stays `~/notes`.
4. Saving only changes your settings. It does not read, change, delete, or index files. Choose **Update index** afterwards.

To remove a folder, open the **Folders** tab and choose **Remove** next to the folder path. Removing a folder updates `config.toml` and purges that folder’s indexed documents from Seekr’s local database without touching any source files on disk. Removal runs through the background service, so it does not race with the watcher.

The limits are 50 folders and 4,096 UTF-8 bytes per submitted path. The service picks up added folders automatically; see [The background service](#the-background-service).

## Make the local model available

When no folders are configured or the local model is unavailable, the **Set up Seekr** panel shows:

- **Download search model (about 130 MB) — the only time seekr uses the internet** — runs `seekr models pull --yes --grammars` in the background, streaming progress and output lines until the local model and grammars are cached.

You can also provision the configured embedding model and code grammars from a terminal with `seekr models pull` (or run `seekr init` for guided first-run setup):

```bash
seekr models pull --grammars
```

Downloading runs only once to fill the local Hugging Face cache; afterwards Seekr’s indexing and searching work offline. Add `--rerank` (or `--all`) if you also use `--rerank` scoring, and if you changed `embed_model`, `seekr models pull` caches that model instead.

The **Local model** panel in **Index status** shows:

- **Not checked** — no check has run. Exact-words search does not need the model; meaning search and index updates try to load it when needed.
- **Checking…** — Seekr is loading the model locally to confirm it works. The rest of the page stays responsive.
- **Ready** — the model loaded and is cached for this web process.
- **Unavailable** — loading failed; the error is shown. Download the model from the setup panel or run the provisioning command, then choose **Check model** again. There is no automatic retry.

**Check model** checks local availability only; it never downloads. Exact-words search keeps working while the model is missing or unavailable. If the model is unavailable, **Update index** is disabled until the model is downloaded or a check succeeds.

## Update the index

**Update index** starts an asynchronous, incremental scan of the configured folders. The page polls real status about once a second and shows elapsed time, then final counts: added, updated, unchanged, removed, and skipped files. It lists per-file errors and warnings, showing up to 100 of each with full totals and a truncation note.

If the completed scan reports any per-file errors, the job is **partial**: the page explains that some files could not be read and lists them. Warnings alone do not make a job partial; the job succeeds and still shows the warnings. A failed job shows its error message. There is no percentage or cancellation: a large folder can take a while, and Seekr does not invent progress. Exact-words search stays available during the scan; meaning modes do not (see [Search](#search)).

If you reload the page during a job, it recovers the current job from server status. If the server itself restarted, the old job record is gone; the page reports that the task is no longer available and shows the current index state. The web index is incremental and shared with the CLI: a compatible database already built by `seekr index` appears as **Ready** here, and **Update index** does not replace it with a fresh one. Seekr never rebuilds an incompatible index automatically.

Don’t start a separate `seekr index`, `seekr reindex`, or another web index job while a scan is already running. The lifecycle lock protects database replacement, not all concurrent writers, so overlapping manual scans should be avoided.

| Panel label | API state | Meaning |
| --- | --- | --- |
| Not created yet | `missing` | No database exists yet. **Update index** creates it from your folders. |
| Empty | `empty` | The database exists but has no documents yet. **Update index** fills it. |
| Ready | `ready` | The index is readable and searchable. |
| Needs rebuilding | `incompatible` | The index was built with a different model, schema, or vector format, or has data without readable metadata. Seekr leaves it untouched; rebuild it from a terminal with `seekr reindex`. |
| Can’t be read | `error` | The database is corrupt, unreadable, or could not be inspected. The page cannot repair it. Run `seekr doctor` to check SQLite/FTS5/sqlite-vec support, then rebuild from a terminal with `seekr reindex` if the database itself is damaged. Searches and index updates that need the local index fail until it is fixed. |

For a ready or empty index, the panel also shows document and passage counts, size on disk, and, when nonzero, the number of unreadable files from the last run. Status inspects the database read-only and never loads the model.

## Search

Open the **Search** tab, type a query, and press Enter. Queries are limited to 2,000 characters.

| Page label | API mode | What it does |
| --- | --- | --- |
| Words and meaning | `hybrid` | Default. Combines exact-word and meaning-based matches. Best general choice. |
| Exact words | `keyword` | Full-text search only. Good for error codes, function names, identifiers, and phrases. Needs no model. |
| Meaning only | `vector` | Meaning-based search only. Good when you remember the idea but not the words. Needs the model. |

While a web index job runs, **Exact words** stays available. The other two modes are disabled with a short explanation, and the page temporarily switches to **Exact words**, restoring your previous mode when the job ends. Results can change as the index completes. If a meaning search fails because the model is unavailable or the engine is busy, the error offers **Search by exact words**.

### Course and advanced filters

Directly under the match options, **Module**, **Week from**, and **Week to** dropdowns are populated automatically from the indexed course folders (`GET /api/course`). Select a module and an optional week range (setting either **Week from**, **Week to**, or both) to scope results to part of a course.

Expand **Filters** for additional filters:

- **Folder or path starts with** — a literal full-path prefix. Use a full path; `~` is not expanded in this filter.
- **File type** — Any type, PDF, Markdown, Plain text, Source code, Transcript, Word (`.docx`), Slides (`.pptx`), or Notebook (`.ipynb`).
- **Changed** — Any time, the last day, week, 30 days, or year, or a specific **Since date**.
- **Most results** — 1 to 50; default 10. **Passages per file** — 1 to 10; default 2.

**Clear filters** resets all filters (including **Module** and **Week from** / **Week to**) back to their defaults.

### Appears in, results, preview, and opening files

When a search matches across course modules or weeks, an **Appears in** strip displays clickable chips per module (such as `Week 1 (first)`, `Week 3`, or `other`). Clicking a chip filters the displayed results to that group; clicking it again or choosing **All results** clears the group filter.

Each result card shows a header (`Module · Week N · file · location`, omitting missing segments), a file-type tag, the full path, a human-readable locator in the margin, and the matching passage with stemmed query highlights rendered safely as `<mark>` text nodes (paths and snippets are never interpreted as HTML).

Click any result card (or its **Preview** button) to open the **Preview** panel:

- Shows the full passage text and, for PDF files on disk, renders the matching page image directly in the panel.
- **Open file** launches the source file at its page, line, or slide in your local desktop application (including WSL-to-Windows path translation when running under WSL).
- **Copy location** copies the formatted location (for example `/path/to/report.pdf (page 3)` or a Windows path on WSL). Turn on **POSIX path** to switch both the displayed location and **Copy location** output to the POSIX path. If the browser cannot access the clipboard, Seekr shows a selected read-only field; press Ctrl+C (or ⌘C) to copy.

### Keyboard shortcuts

When not typing inside a text field or dropdown:

- `/` — focus the search box
- `ArrowDown` / `ArrowUp` — move between result cards
- `Enter` — open the **Preview** panel for the active result
- `o` — open the active or previewed file in its local application
- `Escape` — close the **Preview** panel

If nothing is indexed, a search returns no hits; **Index status** explains whether the index is missing or empty, so add a folder and choose **Update index**. If the index is ready but nothing matches, the page says **No passages match** and suggests clearing filters and trying different words or a match mode. An incompatible index, an unreadable index, or an invalid settings file shows an error with the next step.

## The background service

The web interface is served by the Seekr service, one background process that also watches your folders, keeps the index current, and keeps the model warm for the CLI and MCP server. `seekr web` reuses a running service or starts one and waits up to 10 seconds for it; the CLI and MCP server find it automatically and fall back to local search when it is stopped.

- `seekr service status` shows the service's pid, port, version, and uptime; `seekr status` also prints a one-line `service:` summary.
- `seekr service stop` asks it to stop and waits for it to exit. Closing the browser tab does not stop it.
- `seekr service start` runs it in the terminal (Ctrl+C stops it), which is also how to see start-up errors.
- To run it at login, install the [`contrib/seekr.service`](../contrib/seekr.service) systemd user unit, which runs `seekr service start`; stop it with `systemctl --user stop seekr.service`.

Index updates from the page and folder changes go through the service, so they never race with its watcher. `seekr daemon` is a deprecated alias for `seekr service start` and will be removed in 0.4.0. See [What it does](../README.md#what-it-does).

## Settings file

Seekr reads `$XDG_CONFIG_HOME/seekr/config.toml`, usually `~/.config/seekr/config.toml`. The **Folders** tab shows the path in use under **Saved in**. A missing `roots` key means no folders; Seekr does not use the README example paths as defaults.

Adding or removing a folder edits the TOML file. For supported layouts, Seekr changes only the top-level `roots` array value and preserves comments, spacing, line endings, and unrelated settings. It supports a file with no table headers and no top-level `roots` key (Seekr appends the key), or one top-level single-line `roots = [...]` value, optionally with a trailing comment (Seekr edits only the array). The page can display roots from any valid TOML array shape, but editing folders from the UI supports only the two layouts described above: a flat file with no table headers and an absent top-level `roots` key, or a top-level single-line `roots = [...]` array; other valid shapes must be edited by hand and will be refused with the config path.

Seekr refuses to guess at ambiguous layouts and shows the settings path for manual correction. Examples include a multiline array, comments inside the array, a `[roots]` or `[[roots]]` table, quoted or dotted `roots` keys, duplicate `roots` keys, or table headers with no top-level `roots` key. After editing by hand, choose **Refresh**. Seekr validates the file before and after any write and creates a missing settings file privately.

Removing a folder with **Remove** updates `config.toml` and removes indexed entries under that folder from the database without altering or deleting source files on disk. If the settings file becomes invalid or is edited externally, the page reports the path and never rewrites it; fix it and refresh. A stale edit asks you to refresh and retry.

## Privacy and local-only security

- The server binds only to `127.0.0.1` and validates the exact `127.0.0.1:<port>` host and `http://127.0.0.1:<port>` origin. There is no LAN or public option.
- Pages, CSS, and JavaScript are packaged locally. Seekr emits no CORS headers and makes no CDN, remote-font, remote-image, or telemetry requests. Once the model is cached, indexing and searching are offline; the only optional network action from the page is the explicit **Download search model (about 130 MB) — the only time seekr uses the internet** setup button. See the [offline guarantee](../README.md#how-it-works).
- A session token and browser-origin checks stop other websites from driving the local server. The token stays in memory, is never in the URL, and is invalidated by a restart. This is CSRF protection, not user authentication, and not a boundary against malware or another process running as your user.
- Headerless `curl` to `/api/session` and the static assets gets a deliberate `403 forbidden` with a hint that browser-origin headers are required. This session request works (replace 8000 with the printed port if needed):

  ```bash
  curl -i http://127.0.0.1:8000/api/session \
    -H 'X-Seekr-Client: 1' \
    -H 'Origin: http://127.0.0.1:8000'
  ```

  Other `/api/*` calls also need the returned token in `X-Seekr-Token`. A matching `Sec-Fetch-Site: same-origin` or `none` can replace `Origin`. See the [web API contract](web-api.md) for all routes.
- As static assets, the server serves only `/`, `/index.html`, `/styles.css`, and `/app.js`. Preview (`/api/preview`, `/api/preview/page`) and file opening (`/api/open`) operate strictly on server-indexed positive chunk IDs; there is no arbitrary filesystem read, upload, directory-listing, or shell endpoint.
- Document text can contain hostile HTML, but paths, locators, errors, previews, and snippets are always rendered as text, never executed.

## Troubleshooting

- **Port 8000 is already in use** — use `seekr web --port 8001`, or `seekr web --port 0` and open the actual printed address.
- **Local model is Unavailable** — choose **Download search model (about 130 MB) — the only time seekr uses the internet** in the setup panel, or run `seekr models pull --grammars` (see [Make the local model available](#make-the-local-model-available)), then choose **Check model** again. Exact-words search still works. `seekr doctor` (and `seekr models status`) checks whether your configured models, code grammars, roots, and SQLite environment are ready locally.
- **Index says Needs rebuilding** — stop the service (`seekr service stop`), make the configured model available, and run `seekr reindex` in a terminal. The page leaves the old index untouched.
- **Index says Can’t be read** — the page cannot repair it. Run `seekr doctor` to rule out SQLite/FTS5/sqlite-vec problems; if the database file is damaged, stop the service and rebuild from a terminal with `seekr reindex`. Search and index updates cannot use the old index until it is fixed.
- **Settings file says Can’t be read** — open the path shown, fix the TOML by hand, and choose **Refresh**. If the file has table headers but no `roots` key, add a top-level `roots = [...]` above the first table header. For other unsupported layouts, correct the existing `roots` expression by hand.
- **Seekr is busy** — only one local model/index operation runs at a time; there is no queue. Wait, or use **Exact words**, which keeps working during an index update. Retry if an exact-words search is briefly busy.
- **Seekr isn’t responding / Reconnect** — check that `seekr web` is still running. Restart it and open the printed address; if the port changed, use the new address. Reloading during an index job recovers it while the server is alive.
- **Update index or Remove is disabled** — another index job is running, or a standalone `seekr reindex` holds the index. Wait, choose **Refresh**, then retry.
- **The browser did not open** — `--open-browser` failed or no browser is available; the server is still running. Open the printed URL manually.
- **The page or assets return 404** — the installed package may be missing its `web_assets` files, a packaging problem rather than a settings problem. Reinstall a build that includes them, or run `seekr web` from the source checkout’s environment. Unknown paths also return 404 by design.
- **The address returns 403** — use the exact `127.0.0.1` address printed by `seekr web`; `localhost` and other origins are rejected. For `curl`, send a matching `Origin` or an allowed `Sec-Fetch-Site`.

## Limits and deferred features

Limits: 2,000 query characters; 1–50 results; 1–10 passages per file; 50 folders; 4,096 UTF-8 bytes per submitted folder path; 64 KiB request bodies; 20 recent jobs in memory.

Not included in this release: OCR for scanned PDFs, uploads, remote file access, cancel/pause/queue for jobs, authentication accounts or multi-user access, LAN/public access, editing model or database settings, full rebuilds, force flags, daemon management, saved searches, chat/LLM answers, similarity browsing, and reranker or score-explanation controls. The web interface is a single-user tool; use the CLI for terminal workflows.

## Related documentation

- [README quickstart](../README.md#quickstart)
- [What it does (CLI, web UI, daemon, MCP)](../README.md#what-it-does)
- [How it works and offline guarantee](../README.md#how-it-works)
- [Web API contract](web-api.md)
