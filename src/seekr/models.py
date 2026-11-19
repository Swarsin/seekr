"""Core data models shared by the seekr pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedUnit:
    """A contiguous piece of text extracted from one file location."""

    text: str
    locator: str
    loc_start: float


@dataclass(frozen=True)
class ChunkSpec:
    """A chunk ready to be embedded and stored."""

    text: str
    locator: str
    loc_start: float


@dataclass(frozen=True)
class Hit:
    """One search result pointing back to a location in a file."""

    chunk_id: int
    path: str
    locator: str
    loc_start: float
    snippet: str
    score: float
    kind: str


def text_sha(text: str) -> str:
    """Return the SHA-256 hex digest of lowercased, whitespace-normalised text."""
    normalised = " ".join(text.split()).lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
