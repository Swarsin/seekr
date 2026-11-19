"""Embedding backends for seekr.

Numpy and sentence-transformers are imported lazily so importing this module
stays cheap and does not disturb offline/telemetry environment setup.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from . import offline
from .config import Config

if TYPE_CHECKING:
    import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """A backend that produces normalised document and query vectors."""

    model_name: str

    @property
    def dim(self) -> int:
        """Vector dimension."""
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents with shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query with shape ``(dim,)``."""
        ...


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, np.float32(1.0))
    return np.asarray(matrix / safe_norms, dtype=np.float32)


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder for offline tests.

    Hashing uses ``blake2b`` rather than Python's randomised ``hash()`` so a
    vector is identical in every process.  Shared words land in shared buckets,
    which makes texts with overlapping vocabulary more similar.
    """

    model_name = "fake"

    def __init__(self, dim: int = 64) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.calls: list[list[str]] = []

    def _vector(self, text: str) -> np.ndarray:
        import numpy as np

        tokens = _TOKEN_RE.findall(" ".join(text.split()).lower())
        if not tokens:
            # Keep even empty/whitespace-only text a valid unit vector.  Using
            # the normalised text keeps different punctuation-only inputs
            # distinct while remaining deterministic.
            tokens = ["\x00" + " ".join(text.split())]

        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dim
            vector[bucket] += np.float32(1.0)

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:  # Defensive: a non-zero token count always sets a bucket.
            vector[0] = np.float32(1.0)
            norm = 1.0
        return np.asarray(vector / np.float32(norm), dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        import numpy as np

        batch = list(texts)
        self.calls.append(batch)
        if not batch:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(text) for text in batch]).astype(
            np.float32, copy=False
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class SentenceTransformerEmbedder:
    """Lazy, local-only sentence-transformers embedding backend."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        self._device = device
        self._model: Any | None = None
        self._dim: int | None = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def dim(self) -> int:
        self._ensure_model()
        assert self._dim is not None
        return self._dim

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                model = offline.load_local_model(
                    self.model_name,
                    kind="embedding",
                    device=self._device,
                )
                get_dim = getattr(model, "get_embedding_dimension", None)
                if not callable(get_dim):
                    get_dim = getattr(model, "get_sentence_embedding_dimension", None)
                dimension = get_dim() if callable(get_dim) else None
                if dimension is None:
                    raise ValueError(
                        f"embedding model {self.model_name!r} did not report a dimension"
                    )
                dimension = int(dimension)
                if dimension <= 0:
                    raise ValueError(
                        f"embedding model {self.model_name!r} reported invalid dimension "
                        f"{dimension}"
                    )
                self._dim = dimension
                self._model = model
        return self._model

    def _encode_with_model(self, model: Any, texts: list[str]) -> Any:
        """Run one model inference call, serialising access to the model."""
        with self._inference_lock:
            return model.encode(
                texts,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def _encode(self, texts: list[str]) -> np.ndarray:
        # Load the model before importing numpy so missing runtime dependencies
        # surface as the documented local-model error (with its original cause).
        model = self._ensure_model()
        import numpy as np

        assert self._dim is not None
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        encoded = self._encode_with_model(model, texts)
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim == 1 and len(texts) == 1:
            matrix = matrix.reshape(1, -1)
        expected_shape = (len(texts), self._dim)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"embedding model {self.model_name!r} returned shape "
                f"{matrix.shape}, expected {expected_shape}"
            )
        return _normalise_rows(matrix)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([QUERY_PREFIX + text])[0]


def build_embedder(cfg: Config) -> Embedder:
    """Build the configured lazy embedding backend."""
    return SentenceTransformerEmbedder(cfg.embed_model)


__all__ = [
    "BATCH_SIZE",
    "QUERY_PREFIX",
    "Embedder",
    "FakeEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
]
