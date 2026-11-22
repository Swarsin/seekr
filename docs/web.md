# Seekr web interface

Seekr’s local web interface lets you add folders, update the index, and search your files in a browser. It is a single-user, Linux-only tool started by one local command. It uses the same settings and index as the `seekr` CLI and MCP server; there is no separate web-only index. The [web API contract](web-api.md) documents the HTTP API.

**Words used here.** The **index** is Seekr’s local SQLite database of extracted text, passages, and vectors. The **embedding model** is a local model that turns text into vectors for meaning search; the default is `BAAI/bge-small-en-v1.5`. **Hybrid** search combines exact-word and meaning-based matches. The optional **daemon** (`seekr daemon`) is a background process that keeps the model warm and maintains the index.

## Start and stop

From a terminal where Seekr is installed, run:

```bash
seekr web
```

By default it serves on port 8000 and prints the exact address to open:

```text
Seekr web interface: http://127.0.0.1:8000
Press Ctrl+C to stop.
```

Open the printed `http://127.0.0.1:<port>` address. Use it exactly: the server is loopback-only and rejects `http://localhost:<port>`, other host names, and other machines. There is no option to expose it on your LAN or the public internet.

```bash
# Same as the default port.
seekr web --port 8000

# Ask the OS for a free port; use the actual port printed at startup.
seekr web --port 0

# Also open the printed address in the default browser.
seekr web --open-browser
```

`--open-browser` is a convenience. On a headless machine, or if no browser can be launched, Seekr prints why and keeps the server running; open the printed address yourself. To stop, press Ctrl+C in the terminal running `seekr web`. If an index update is active, Seekr waits for it to finish before exiting, so a large scan can delay shutdown. A second Ctrl+C warns but does not cancel the work. Closing the browser tab neither stops the server nor cancels an index job.

## First run: choose folders

Open the **Folders** tab. Seekr searches only the folders listed there; it does not scan your whole home folder. Supported files include PDFs with embedded text, Markdown, plain text, source code, and transcripts. Scanned image-only PDFs are not OCRed. A missing `roots` key, or `roots = []`, means no folders; Seekr does not assume the example paths in the README.

To add a folder:

1. Type the full local path, for example `~/notes` or `~/Documents/coursework`. A leading `~` means your home folder.
2. Choose **Add folder**. The folder must already exist, be a readable directory, and not already be listed.
3. The page shows the saved path. New paths are canonical absolute paths: `~` is expanded and symbolic links are resolved. Existing entries are shown as stored, so an older `~/notes` entry stays `~/notes`.
4. Saving only changes your settings. It does not read, change, delete, or index files. Choose **Update index** afterwards.

The limits are 50 folders and 4,096 UTF-8 bytes per submitted path. If a `seekr daemon` is running, adding a folder succeeds but warns that the daemon must be restarted before daemon-backed searches see it; see [If a background daemon is running](#if-a-background-daemon-is-running).

## Make the local model available

The web page never downloads a model or executes an installation command. The model must already be on this computer; Seekr loads it locally with network access disabled. Run this one-time provisioning command in a terminal, using the Python interpreter from the environment where Seekr is installed (this is the README quickstart command):

```bash
~/.local/pipx/venvs/seekr-search/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-small-en-v1.5")'
```

If you installed Seekr differently, use that environment’s Python (for example `.venv/bin/python`). The command may use the network once to fill the local Hugging Face cache; afterwards Seekr’s indexing and searching are offline. If you changed `embed_model`, provision that model name instead.

The **Local model** panel in **Index status** shows:

- **Not checked** — no check has run. Exact-words search does not need the model; meaning search and index updates try to load it when needed.
- **Checking…** — Seekr is loading the model locally to confirm it works. The rest of the page stays responsive.
- **Ready** — the model loaded and is cached for this web process.
- **Unavailable** — loading failed; the error is shown. Run the provisioning command, then choose **Check model** again. There is no automatic retry.

**Check model** checks local availability only; it never downloads. Exact-words search keeps working while the model is missing or unavailable. If the model is unavailable, **Update index** is disabled until a check succeeds.

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

Expand **Filters** to narrow results:

- **Folder or path starts with** — a literal full-path prefix. Use a full path; `~` is not expanded in this filter.
- **File type** — Any type, PDF, Markdown, Plain text, Source code, or Transcript.
- **Changed** — Any time, the last day, week, 30 days, or year, or a specific **Since date**.
- **Most results** — 1 to 50; default 10. **Passages per file** — 1 to 10; default 2.

**Clear filters** resets the fields. Each result shows the file name, a file-type tag, the full path, a location when available (page, line, heading, or transcript timestamp), and the matching passage. Query matches are highlighted as text; paths and snippets are never interpreted as HTML. **Copy location** copies the full path plus locator, for example `/path/to/report.pdf (p.3)`. If the browser cannot copy, Seekr shows a selected read-only field; press Ctrl+C (or ⌘C) to copy. Use the location in your editor, file manager, or terminal. There is no **Open file** or preview button.

If nothing is indexed, a search returns no hits; **Index status** explains whether the index is missing or empty, so add a folder and choose **Update index**. If the index is ready but nothing matches, the page says **No passages match** and suggests clearing filters and trying different words or a match mode. An incompatible index, an unreadable index, or an invalid settings file shows an error with the next step.

## If a background daemon is running

`seekr daemon` is optional. It watches configured folders, maintains the index, and keeps a model warm. The page only probes whether the daemon socket is live; it never starts, stops, or reconfigures the daemon. When the daemon is present, **Background service** shows **Running** and **Update index** is disabled with an explanation. The page never claims to detect live daemon indexing activity; socket presence is the only signal.

To update from the page, stop the daemon first: press Ctrl+C in its terminal, or run `systemctl --user stop seekr.service` if you installed the [`contrib/seekr.service`](../contrib/seekr.service) systemd user unit. Then choose **Refresh**; **Update index** becomes available. Adding a folder while the daemon runs succeeds but warns that the daemon needs a restart; restart `seekr daemon` (or the systemd unit) so it sees the new folder for daemon-backed searches. The page never restarts it. If the daemon is reachable but fails a search, the page shows an error and suggests restarting it or stopping it so the page searches the local index directly. See [What it does](../README.md#what-it-does).

When a daemon answers searches, the **Local model** indicator still describes this web process; the daemon may use its own warm model.

## Settings file

Seekr reads `$XDG_CONFIG_HOME/seekr/config.toml`, usually `~/.config/seekr/config.toml`. The **Folders** tab shows the path in use under **Saved in**. A missing `roots` key means no folders; Seekr does not use the README example paths as defaults.

Adding a folder edits the TOML file. For supported layouts, Seekr changes only the top-level `roots` array value and preserves comments, spacing, line endings, and unrelated settings. It supports a file with no table headers and no top-level `roots` key (Seekr appends the key), or one top-level single-line `roots = [...]` value, optionally with a trailing comment (Seekr edits only the array). The page can display roots from any valid TOML array shape, but adding a folder supports only the two layouts described above: a flat file with no table headers and an absent top-level `roots` key, or a top-level single-line `roots = [...]` array; other valid shapes must be edited by hand and will be refused with the config path.

Seekr refuses to guess at ambiguous layouts and shows the settings path for manual correction. Examples include a multiline array, comments inside the array, a `[roots]` or `[[roots]]` table, quoted or dotted `roots` keys, duplicate `roots` keys, or table headers with no top-level `roots` key. After editing by hand, choose **Refresh**. Seekr validates the file before and after any write and creates a missing settings file privately.

There is no **Remove folder** button in this release. Edit the `roots` array by hand and choose **Refresh** (restart the daemon if you use one). Removing a folder only changes what future scans read; it does not delete source files, and documents already in the index can remain searchable. Saving a folder never indexes, deletes, or changes files. If the settings file becomes invalid or is edited externally, the page reports the path and never rewrites it; fix it and refresh. A stale edit asks you to refresh and retry.

## Privacy and local-only security

- The server binds only to `127.0.0.1` and validates the exact `127.0.0.1:<port>` host and `http://127.0.0.1:<port>` origin. There is no LAN or public option.
- Pages, CSS, and JavaScript are packaged locally. Seekr emits no CORS headers and makes no CDN, remote-font, remote-image, or telemetry requests. After the model is cached, indexing and searching are offline; see the [offline guarantee](../README.md#how-it-works).
- A session token and browser-origin checks stop other websites from driving the local server. The token stays in memory, is never in the URL, and is invalidated by a restart. This is CSRF protection, not user authentication, and not a boundary against malware or another process running as your user.
- Headerless `curl` to `/api/session` and the static assets gets a deliberate `403 forbidden` with a hint that browser-origin headers are required. This session request works (replace 8000 with the printed port if needed):

  ```bash
  curl -i http://127.0.0.1:8000/api/session \
    -H 'X-Seekr-Client: 1' \
    -H 'Origin: http://127.0.0.1:8000'
  ```

  Other `/api/*` calls also need the returned token in `X-Seekr-Token`. A matching `Sec-Fetch-Site: same-origin` or `none` can replace `Origin`. See the [web API contract](web-api.md) for all routes.
- As static assets, the server serves only `/`, `/index.html`, `/styles.css`, and `/app.js`. There is no document-serving, opening, upload, directory-listing, shell, or arbitrary-file endpoint.
- Document text can contain hostile HTML, but paths, locators, errors, and snippets are always rendered as text, never executed.

## Troubleshooting

- **Port 8000 is already in use** — use `seekr web --port 8001`, or `seekr web --port 0` and open the actual printed address.
- **Local model is Unavailable** — run the [provisioning command](#make-the-local-model-available), then choose **Check model** again. Exact-words search still works. `seekr doctor` checks the SQLite environment, not whether a model is cached.
- **Index says Needs rebuilding** — stop any daemon, make the configured model available, and run `seekr reindex` in a terminal. The page leaves the old index untouched.
- **Index says Can’t be read** — the page cannot repair it. Run `seekr doctor` to rule out SQLite/FTS5/sqlite-vec problems; if the database file is damaged, stop any daemon and rebuild from a terminal with `seekr reindex`. Search and index updates cannot use the old index until it is fixed.
- **Settings file says Can’t be read** — open the path shown, fix the TOML by hand, and choose **Refresh**. If the file has table headers but no `roots` key, add a top-level `roots = [...]` above the first table header. For other unsupported layouts, correct the existing `roots` expression by hand.
- **Seekr is busy** — only one local model/index operation runs at a time; there is no queue. Wait, or use **Exact words**, which keeps working during an index update. Retry if an exact-words search is briefly busy.
- **Seekr isn’t responding / Reconnect** — check that `seekr web` is still running. Restart it and open the printed address; if the port changed, use the new address. Reloading during an index job recovers it while the server is alive.
- **Update index is disabled and Background service is Running** — stop `seekr daemon` (or its systemd unit), choose **Refresh**, then update. Restart the daemon after adding a folder.
- **The browser did not open** — `--open-browser` failed or no browser is available; the server is still running. Open the printed URL manually.
- **The page or assets return 404** — the installed package may be missing its `web_assets` files, a packaging problem rather than a settings problem. Reinstall a build that includes them, or run `seekr web` from the source checkout’s environment. Unknown paths also return 404 by design.
- **The address returns 403** — use the exact `127.0.0.1` address printed by `seekr web`; `localhost` and other origins are rejected. For `curl`, send a matching `Origin` or an allowed `Sec-Fetch-Site`.

## Limits and deferred features

Limits: 2,000 query characters; 1–50 results; 1–10 passages per file; 50 folders; 4,096 UTF-8 bytes per submitted folder path; 64 KiB request bodies; 20 recent jobs in memory.

Not included in this release: OCR for scanned PDFs, uploads, remote file access, browser preview or file opening, root removal or content deletion from the page, cancel/pause/queue for jobs, authentication accounts or multi-user access, LAN/public access, Windows/macOS support, editing model or database settings, full rebuilds, force flags, daemon management, saved searches, chat/LLM answers, similarity browsing, and reranker or score-explanation controls. The web interface is a single-user Linux tool; use the CLI for terminal workflows.

## Related documentation

- [README quickstart](../README.md#quickstart)
- [What it does (CLI, web UI, daemon, MCP)](../README.md#what-it-does)
- [How it works and offline guarantee](../README.md#how-it-works)
- [Web API contract](web-api.md)
