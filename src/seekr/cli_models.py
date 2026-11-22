"""The ``seekr models`` subcommand group for explicit asset provisioning."""

from __future__ import annotations

import typer

__all__ = ["models_app", "register"]

models_app = typer.Typer(
    help="Download and inspect local models and code grammars.",
    no_args_is_help=True,
)


def _approx_mb(total_bytes: int) -> int:
    if total_bytes <= 0:
        return 0
    return max(1, round(total_bytes / 1_000_000))


@models_app.command("pull")
def pull(
    rerank: bool = typer.Option(
        False,
        "--rerank",
        help="Also download the configured reranker model.",
    ),
    grammars: bool = typer.Option(
        False,
        "--grammars",
        help="Also download missing tree-sitter code grammars.",
    ),
    pull_all: bool = typer.Option(
        False,
        "--all",
        help="Download embedding model, reranker model and missing code grammars.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Download without asking for confirmation.",
    ),
) -> None:
    """Download required models (and optionally reranker/grammars) for offline use."""
    from . import provision
    from .config import ConfigError, load_config

    try:
        cfg = load_config()
    except ConfigError as exc:
        typer.echo(f"seekr models pull: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    include_rerank = rerank or pull_all
    include_grammars = grammars or pull_all

    assets = provision.assets_for(cfg, rerank=include_rerank)
    missing_assets = [asset for asset in assets if not provision.is_cached(asset)]
    missing_grams = provision.missing_grammars() if include_grammars else []

    if not missing_assets and not missing_grams:
        typer.echo("All requested models and grammars are already cached.")
        return

    total_bytes = 0
    has_known_size = False
    for asset in missing_assets:
        size = provision.estimate_bytes(asset)
        if size is not None:
            total_bytes += size
            has_known_size = True
            size_label = f"~{_approx_mb(size)} MB"
        else:
            size_label = "size unknown"
        typer.echo(f"  {asset.purpose}: {asset.repo_id} ({size_label})")

    if missing_grams:
        typer.echo(
            f"  code grammars ({len(missing_grams)}): {', '.join(missing_grams)} (size unknown)"
        )

    item_count = len(missing_assets) + (1 if missing_grams else 0)
    if has_known_size:
        prompt = f"Download {item_count} item(s), about {_approx_mb(total_bytes)} MB?"
    else:
        prompt = f"Download {item_count} item(s)?"

    if not yes and not typer.confirm(prompt, default=False):
        typer.echo("Aborted.")
        raise typer.Exit(code=1)

    try:
        for asset in missing_assets:
            typer.echo(f"Downloading {asset.purpose}: {asset.repo_id}...")
            provision.download(asset)
        if missing_grams:
            typer.echo(
                f"Downloading code grammars ({len(missing_grams)}): {', '.join(missing_grams)}..."
            )
            provision.download_grammars(missing_grams)
    except provision.ProvisionError as exc:
        typer.echo(f"seekr models pull: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Done.")


@models_app.command("status")
def status() -> None:
    """Show cached or missing status for configured models and code grammars."""
    from . import provision
    from .config import ConfigError, load_config

    try:
        cfg = load_config()
    except ConfigError as exc:
        typer.echo(f"seekr models status: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for asset in provision.assets_for(cfg, rerank=True):
        state = "cached" if provision.is_cached(asset) else "missing"
        typer.echo(f"{asset.purpose}: {asset.repo_id}: {state}")

    langs = provision.grammar_languages()
    missing = set(provision.missing_grammars())
    typer.echo("code grammars:")
    for lang in langs:
        state = "missing" if lang in missing else "cached"
        typer.echo(f"  {lang}: {state}")


def register(app: typer.Typer) -> None:
    """Register the ``models`` command group on the main Typer app."""
    app.add_typer(models_app, name="models")
