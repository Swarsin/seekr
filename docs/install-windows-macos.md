# Install on Windows and macOS

seekr runs natively on Windows and macOS as well as Linux. The commands are the same everywhere; this page covers what differs: choosing a Python, where files live, registering the MCP server, and starting the service at login.

The PyPI package is `seekr-search`; the command is `seekr`.

## Python

seekr needs Python 3.11 or newer whose `sqlite3` module can load extensions, because the vector index uses the `sqlite-vec` extension.

**macOS.** The python.org installer and the builds `actions/setup-python` provides do not include `sqlite3.Connection.enable_load_extension`, so seekr cannot load `sqlite-vec` with them. Use a CPython that does, such as Homebrew Python or a uv-managed CPython:

```bash
# Homebrew Python
brew install python pipx
pipx install seekr-search

# or a uv-managed CPython
uv python install 3.12
uv tool install --python 3.12 --managed-python seekr-search
# or, with pipx:
pipx install --python "$(uv python find --managed-python 3.12)" seekr-search
```

**Windows.** Install Python 3.11 or newer from [python.org](https://www.python.org/downloads/windows/) (or `winget install Python.Python.3.12`), then:

```powershell
py -m pip install --user pipx
py -m pipx ensurepath
pipx install seekr-search
```

Open a new terminal afterwards so `seekr` is on your `PATH`. CI installs and runs seekr on Windows with the python.org CPython that `actions/setup-python` provides; other distributions (including the Microsoft Store build) are not tested.

Check the result on either OS:

```text
seekr doctor
```

`seekr doctor` reports whether SQLite can load extensions and whether `sqlite-vec` works. On a Python without extension support it fails with the hint "use a CPython build with extension support (for example a uv-managed CPython)"; reinstall seekr with one of the Pythons above.

## First run

```text
seekr init --demo        # index the packaged sample course, or:
seekr init C:\Users\you\Documents\notes
seekr search "bias variance tradeoff"
seekr web
```

`seekr init` downloads the embedding model (about 130 MB) and code grammars once, writes `config.toml`, and builds the index. `seekr web` starts the background service if it is not running and prints its address (`http://127.0.0.1:8000/` by default). On Windows, `seekr init --demo` prints the `SEEKR_CONFIG` hint in PowerShell syntax.

## Where files live

| | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Config (`config.toml`) | `%LOCALAPPDATA%\seekr` | `~/Library/Application Support/seekr` | `$XDG_CONFIG_HOME/seekr` (`~/.config/seekr`) |
| Index | `%LOCALAPPDATA%\seekr` | `~/Library/Application Support/seekr` | `$XDG_DATA_HOME/seekr` (`~/.local/share/seekr`) |

Set `SEEKR_CONFIG` to use a config file elsewhere. The service's runtime state file (`service.json`) is kept in the platform state directory.

## Use with an MCP client

Any MCP client that launches stdio servers can run `seekr mcp`. Use the command `seekr` with the argument `mcp`. For Claude Code:

```text
claude mcp add seekr -- seekr mcp
```

For clients configured with a JSON file (Claude Desktop, Cursor and others), add this server block to the `mcpServers` object; each client's documentation says where its file lives on your OS:

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

If the client does not inherit your `PATH`, set `"command"` to the full path of the executable: `where seekr` on Windows (pipx puts it in `%USERPROFILE%\.local\bin`), or `which seekr` on macOS.

## Start the service at login

`seekr web` starts the service on demand, and searches work without it. To have it running after you log in, register one of the examples below. Each runs `seekr service start --port 8000`, which keeps running in the foreground until stopped; `seekr web` and `seekr service status` print the address actually in use. Without `--port`, the service picks a random free port.

**Windows (Task Scheduler).** [`contrib/seekr-task.ps1`](https://github.com/Swarsin/seekr/blob/main/contrib/seekr-task.ps1) registers a per-user task named `seekr service` that runs at logon (it has no restart-on-failure setting, so `seekr service stop` stays stopped):

```powershell
powershell -ExecutionPolicy Bypass -File contrib\seekr-task.ps1
```

It looks for `seekr` on your `PATH`; pass `-SeekrPath C:\path\to\seekr.exe` if it is elsewhere. Remove it with `Unregister-ScheduledTask -TaskName "seekr service" -Confirm:$false`.

**macOS (launchd).** [`contrib/com.seekr.service.plist`](https://github.com/Swarsin/seekr/blob/main/contrib/com.seekr.service.plist) is a launch agent that starts the service at login. It does not restart it, so `seekr service stop` stays stopped. Edit the path to `seekr` in `ProgramArguments` (the example assumes `~/.local/bin/seekr`, where `pipx` and `uv tool` install it; launchd does not expand `~`, so write the full path), then:

```bash
cp contrib/com.seekr.service.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.seekr.service.plist
```

Stop the service with `seekr service stop`. To remove the agent so it no longer starts at login, run `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.seekr.service.plist` and delete the file.

**Linux.** See [`contrib/seekr.service`](https://github.com/Swarsin/seekr/blob/main/contrib/seekr.service) for a systemd user unit.

## Limits on Windows and macOS

- The old Unix-socket `seekr daemon` fast path is Linux-only and disabled on Windows and macOS. `seekr daemon` is a deprecated alias for `seekr service start` and will be removed in 0.4.0; use `seekr service start` instead.
- `seekr open` and the web UI **Open file** button open PDFs, Word and PowerPoint files in their default application. Code, text, Markdown, notebooks and transcripts open in `$EDITOR` when it is set (at the line, as `$EDITOR +N file`), otherwise in Notepad on Windows or the default text editor on macOS (`open -t`). They never go through the default "open" action, which can run a `.py` or `.js` file instead of showing it. Set `$SEEKR_OPENER` to a command (for example `code`) to open every result with it; the file path is appended. For the background service, set these variables in the environment that starts the service.

See also the [web interface guide](web.md) and the [README](../README.md).
