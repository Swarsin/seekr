"""Command-line interface for seekr."""

from __future__ import annotations

import typer

from . import (
    __version__,
    cli_daemon,
    cli_eval,
    cli_index,
    cli_open,
    cli_search,
)

app = typer.Typer(
    help="Local semantic search over your own files.",
    no_args_is_help=True,
)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"seekr {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_show_version,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Local semantic search over your own files."""


@app.command()
def doctor() -> None:
    """Check that SQLite, FTS5 and sqlite-vec are available."""
    from .doctor import run_doctor

    report = run_doctor()
    typer.echo(report.render())
    if not report.ok:
        raise typer.Exit(code=1)


# Registered explicitly so the root app owns every command while the command
# modules keep their model/backend imports behind the CLI call boundary.
cli_daemon.register(app)
cli_eval.register(app)
cli_index.register(app)
cli_open.register(app)
cli_search.register(app)


if __name__ == "__main__":
    app()
