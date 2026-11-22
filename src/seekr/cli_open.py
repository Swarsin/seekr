"""The ``seekr pick`` and ``seekr open`` commands.

``pick`` deliberately reuses :func:`seekr.cli_search.get_hits`, the same query
entry point as ``seekr search``, so the daemon fast path there serves
both commands without another code path.  ``open`` executes the
argv from :func:`seekr.opener.open_command` directly, never through a shell.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from .cli_search import SearchError, get_hits
from .client import DaemonError
from .extract import kind_for
from .lifecycle import LifecycleError
from .opener import format_pick_line, open_command, parse_pick_line
from .store import ModelMismatchError, SchemaVersionError

if TYPE_CHECKING:
    import typer

__all__ = ["register"]

_SEARCH_ERRORS = (
    SearchError,
    ModelMismatchError,
    SchemaVersionError,
    LifecycleError,
    DaemonError,
    ValueError,
)


def _exit_error(command: str, message: str) -> NoReturn:
    import typer

    typer.echo(f"seekr {command}: {message}", err=True)
    raise typer.Exit(code=1)


def _build_pick_command(typer_module: Any) -> Any:
    def pick(
        query: str = typer_module.Argument(
            ..., metavar="QUERY", help="Words to search for."
        ),
        k: int = typer_module.Option(10, "-k", "--k", min=1, help="Maximum number of results."),
        mode: str = typer_module.Option(
            "hybrid", "--mode", help="Search mode: keyword, vector or hybrid."
        ),
        per_file: int = typer_module.Option(
            2, "--per-file", min=1, help="Maximum number of hits per file."
        ),
        rerank: bool = typer_module.Option(False, "--rerank", help="Rerank the top candidates."),
        path_prefix: str | None = typer_module.Option(
            None, "--path", help="Only search under this path prefix."
        ),
        kind: str | None = typer_module.Option(
            None,
            "--kind",
            help="Filter by kind: pdf, code, markdown, word, slides, notebook, transcript or text.",
        ),
        since: str | None = typer_module.Option(
            None,
            "--since",
            help="Only include files modified since duration (e.g. 7d, 12h) or ISO date.",
        ),
        module: str | None = typer_module.Option(
            None, "--module", help="Filter by module name or unique prefix."
        ),
        week: str | None = typer_module.Option(
            None, "--week", help="Filter by week N or inclusive range N-M."
        ),
    ) -> None:
        """Print one pick line per hit, ready for dmenu or rofi."""
        extra: dict[str, Any] = {}
        if kind is not None:
            extra["kind"] = kind
        if since is not None:
            extra["since"] = since
        if module is not None:
            extra["module"] = module
        if week is not None:
            extra["week"] = week
        try:
            hits = get_hits(
                query,
                k=k,
                mode=mode,
                per_file=per_file,
                rerank=rerank,
                path_prefix=path_prefix,
                **extra,
            )
        except _SEARCH_ERRORS as exc:
            _exit_error("pick", str(exc))
        except ImportError as exc:
            _exit_error("pick", f"search backend is unavailable: {exc}; run `seekr doctor`")
        except (OSError, sqlite3.Error) as exc:
            _exit_error(
                "pick",
                f"cannot read the index: {exc}; run `seekr doctor` or `seekr reindex`",
            )

        for hit in hits:
            typer_module.echo(format_pick_line(hit))

    pick.__doc__ = "Print one pick line per hit, ready for dmenu or rofi."
    return pick


def _build_open_command(typer_module: Any) -> Any:
    def open_target(
        target: str = typer_module.Argument(
            ..., metavar="TARGET", help="A pick line or an unambiguous path:locator."
        ),
    ) -> None:
        """Open a result at its page, line or Markdown heading."""
        path, locator = parse_pick_line(target)
        kind = kind_for(Path(path)) or ""
        argv = open_command(path, locator, kind, os.environ, shutil.which)
        try:
            result = subprocess.run(argv, check=False)
        except FileNotFoundError:
            _exit_error(
                "open",
                f"cannot run {argv[0]!r}: executable not found; install it or set $EDITOR / $SEEKR_OPENER",
            )
        except (OSError, ValueError) as exc:
            _exit_error("open", f"cannot run {argv[0]!r}: {exc}")
        if result.returncode:
            raise typer_module.Exit(code=result.returncode)

    open_target.__doc__ = "Open a result at its page, line or Markdown heading."
    return open_target


def register(app: typer.Typer) -> None:
    """Register the ``pick`` and ``open`` commands on the main Typer application."""
    import typer

    app.command("pick")(_build_pick_command(typer))
    app.command("open")(_build_open_command(typer))
