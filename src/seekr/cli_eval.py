"""The ``seekr eval`` command, registered into the root Typer application.

The command deliberately lives outside :mod:`seekr.cli`; the main module only
needs ``cli_eval.register(app)``.  Evaluation data and models are loaded only
after the user runs the command, never at CLI import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from .config import load_config
from .embed import build_embedder
from .evaluation import EVAL_MODES, load_queries, render_table, run_eval
from .rerank import build_reranker

if TYPE_CHECKING:
    import typer


def _exit_error(message: str) -> NoReturn:
    import typer

    typer.echo(f"seekr eval: {message}", err=True)
    raise typer.Exit(code=1)


def _parse_modes(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    modes: list[str] = []
    for value in raw:
        for part in str(value).split(","):
            mode = part.strip()
            if mode and mode not in modes:
                modes.append(mode)
    return modes


def _build_eval_command(typer_module: Any) -> Any:
    def eval_command(
        root: Path = typer_module.Option(
            ...,
            "--root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory of files to index into a throwaway evaluation store.",
        ),
        queries: Path = typer_module.Option(
            ...,
            "--queries",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="JSONL evaluation queries with relative file-level relevance.",
        ),
        modes: list[str] | None = typer_module.Option(
            None,
            "--modes",
            help=(
                "Comma-separated or repeated modes: keyword, vector, hybrid, "
                "hybrid+rerank. Defaults to all four."
            ),
        ),
        k: int = typer_module.Option(
            10,
            "--k",
            min=1,
            help="Minimum candidate depth; metrics are always reported at 5 and 10.",
        ),
        out: Path | None = typer_module.Option(
            None,
            "--out",
            help="Optional path to write the markdown results table.",
        ),
    ) -> None:
        """Evaluate retrieval modes on file-level queries and print a table."""
        try:
            selected = _parse_modes(modes) or list(EVAL_MODES)
            selected_queries = load_queries(queries)
            cfg = load_config()
            embedder = build_embedder(cfg)
            reranker = build_reranker(cfg) if "hybrid+rerank" in selected else None
            results = run_eval(
                root,
                selected_queries,
                selected,
                embedder,
                reranker,
                k=k,
            )
            table = render_table(results)
        except typer_module.Exit:
            raise
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            _exit_error(str(exc))

        typer_module.echo(table, nl=False)
        if out is not None:
            output_path = Path(out).expanduser()
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(table, encoding="utf-8")
            except OSError as exc:
                _exit_error(f"cannot write {output_path}: {exc}")
            typer_module.echo(f"wrote {output_path}")

    eval_command.__doc__ = (
        "Evaluate keyword, vector, hybrid and hybrid+rerank retrieval."
    )
    return eval_command


def register(app: typer.Typer) -> None:
    """Register the ``eval`` command on the main Typer application."""
    import typer

    app.command("eval")(_build_eval_command(typer))


__all__ = ["register"]
