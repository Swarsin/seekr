"""Command-line interface for seekr."""

from __future__ import annotations

import typer

from . import (
    __version__,
    cli_daemon,
    cli_eval,
    cli_index,
    cli_init,
    cli_models,
    cli_open,
    cli_search,
    cli_web,
)
from .logging_config import configure_logging

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
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable structured diagnostic logs on stderr.",
    ),
) -> None:
    """Local semantic search over your own files."""
    configure_logging(verbose=verbose)


_SUPPORTED_SHELLS = frozenset({"bash", "zsh", "fish"})


@app.command()
def doctor(
    env_only: bool = typer.Option(
        False,
        "--env-only",
        help="Only check SQLite, FTS5 and sqlite-vec without checking config, roots or models.",
    ),
) -> None:
    """Check that SQLite, FTS5, sqlite-vec, config, roots and models are ready."""
    from .doctor import run_doctor

    report = run_doctor(env_only=env_only)
    typer.echo(report.render())
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def completion(
    shell: str = typer.Argument(
        ...,
        metavar="SHELL",
        help="Target shell: bash, zsh or fish.",
    ),
) -> None:
    """Print a shell completion script for bash, zsh or fish."""
    import click.shell_completion as sc

    target = shell.strip().lower()
    if target not in _SUPPORTED_SHELLS:
        choices = ", ".join(sorted(_SUPPORTED_SHELLS))
        typer.echo(
            f"seekr completion: unsupported shell {shell!r}; choose one of: {choices}",
            err=True,
        )
        raise typer.Exit(code=1)

    cls = sc.get_completion_class(target)
    if cls is None:  # pragma: no cover - standard Click shells always resolve
        typer.echo(f"seekr completion: no completion generator for {target!r}", err=True)
        raise typer.Exit(code=1)
    click_app = typer.main.get_command(app)
    script = cls(click_app, {}, "seekr", "_SEEKR_COMPLETE").source()
    typer.echo(script.rstrip("\n"))


# Registered explicitly so the root app owns every command while the command
# modules keep their model/backend imports behind the CLI call boundary.
cli_daemon.register(app)
cli_eval.register(app)
cli_index.register(app)
cli_init.register(app)
cli_models.register(app)
cli_open.register(app)
cli_search.register(app)
cli_web.register(app)


if __name__ == "__main__":
    app()
