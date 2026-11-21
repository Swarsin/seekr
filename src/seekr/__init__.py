"""seekr: local semantic search over your own files."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

__version__ = "0.1.1"

__all__ = ["__version__", "demo_course_path"]


def demo_course_path() -> Path:
    """Return the filesystem path to the packaged demo course directory."""
    return Path(str(importlib.resources.files("seekr") / "demo_course"))
