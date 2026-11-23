"""Cross-encoder reranking backends.

The real backend is deliberately lazy and local-only: importing this module
does not import sentence-transformers or numpy, and the model is loaded through
:func:`seekr.offline.load_local_model` with ``kind="cross_encoder"`` so seekr
never downloads a model implicitly.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Protocol, runtime_checkable

from . import offline
from .config import Config


@runtime_checkable
class Reranker(Protocol):
    """A backend that scores passages against one query."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one finite score per passage, ordered like ``passages``."""
        ...


class FakeReranker:
    """Deterministic substring reranker for tests and offline usage.

    A passage receives the value of the first configured substring it contains
    (in dictionary insertion order), or ``0.0`` when nothing matches.
    """

    def __init__(self, scores_by_substring: dict[str, float] | None = None) -> None:
        self.scores_by_substring = dict(scores_by_substring or {})
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        batch = list(passages)
        self.calls.append((query, batch))
        scores: list[float] = []
        for passage in batch:
            matched = 0.0
            for substring, configured_score in self.scores_by_substring.items():
                if substring in passage:
                    matched = float(configured_score)
                    break
            scores.append(matched)
        return scores


class CrossEncoderReranker:
    """Thread-safe, lazy, local-only sentence-transformers cross-encoder."""

    def __init__(self, model_name: str) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def _ensure_model(self) -> Any:
        model = self._model
        if model is not None:
            return model
        with self._load_lock:
            if self._model is None:
                self._model = offline.load_local_model(
                    self.model_name,
                    kind="cross_encoder",
                )
            return self._model

    @staticmethod
    def _convert_scores(raw_scores: Any, expected: int) -> list[float]:
        try:
            values = list(raw_scores)
        except TypeError as exc:
            raise ValueError(
                "cross-encoder must return a sequence of scores"
            ) from exc
        if len(values) != expected:
            raise ValueError(
                f"cross-encoder returned {len(values)} scores for "
                f"{expected} passages"
            )

        scores: list[float] = []
        for value in values:
            try:
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cross-encoder returned a non-numeric score: {value!r}"
                ) from exc
            if not math.isfinite(score):
                raise ValueError(
                    f"cross-encoder returned a non-finite score: {score!r}"
                )
            scores.append(score)
        return scores

    def score(self, query: str, passages: list[str]) -> list[float]:
        batch = list(passages)
        if not batch:
            return []

        model = self._ensure_model()
        pairs = [[query, passage] for passage in batch]
        with self._predict_lock:
            raw_scores = model.predict(pairs)
        return self._convert_scores(raw_scores, len(batch))


def build_reranker(cfg: Config) -> Reranker:
    """Build the configured lazy cross-encoder backend (``cfg.embed_backend``)."""
    if cfg.embed_backend == "onnx":
        from .onnx_backend import OnnxReranker

        return OnnxReranker(cfg.rerank_model)
    return CrossEncoderReranker(cfg.rerank_model)


__all__ = [
    "CrossEncoderReranker",
    "FakeReranker",
    "Reranker",
    "build_reranker",
]
