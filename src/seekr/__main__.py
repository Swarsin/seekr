"""Module entry point for ``python -m seekr``."""

from __future__ import annotations

from .cli import app

if __name__ == "__main__":
    app(prog_name="seekr")
