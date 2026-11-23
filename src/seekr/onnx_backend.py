"""ONNX Runtime embedding and reranking backends.

These backends run the same Hugging Face models as the sentence-transformers
backends, using the ``onnx/model.onnx`` and ``tokenizer.json`` files that the
model repositories publish. Model files are read from an explicit directory or
from the local Hugging Face cache only; nothing here downloads. Heavy imports
(numpy, onnxruntime, tokenizers, huggingface_hub) are deferred until a model
is first used.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .embed import BATCH_SIZE, QUERY_PREFIX
from .offline import ModelUnavailableError, configure_offline_environment

if TYPE_CHECKING:
    import numpy as np

ONNX_MODEL_FILE = "onnx/model.onnx"
ONNX_EMBED_FILES: tuple[str, ...] = (ONNX_MODEL_FILE, "tokenizer.json", "config.json")
ONNX_EMBED_OPTIONAL_FILES: tuple[str, ...] = ("1_Pooling/config.json",)
ONNX_RERANK_FILES: tuple[str, ...] = (ONNX_MODEL_FILE, "tokenizer.json", "config.json")
MAX_SEQ_LEN = 512
_FEED_NAMES = frozenset({"input_ids", "attention_mask", "token_type_ids"})
_DEFAULT_PROVIDERS = ("CPUExecutionProvider",)

PoolingMode = Literal["cls", "mean"]


def resolve_model_files(
    model_name: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
    *,
    model_dir: Path | None = None,
) -> dict[str, Path]:
    """Return local paths for a model's files without any network access.

    With ``model_dir`` (or a ``model_name`` that is an existing directory) the
    files are looked up under that directory. Otherwise each file is looked up
    in the local Hugging Face cache. Optional files that are absent are left
    out of the result; a missing required file raises
    :class:`ModelUnavailableError`.
    """
    if model_dir is None:
        candidate = Path(model_name).expanduser()
        if candidate.is_dir():
            model_dir = candidate

    found: dict[str, Path] = {}
    if model_dir is not None:
        for name in (*required, *optional):
            path = Path(model_dir) / name
            if path.is_file():
                found[name] = path
    else:
        configure_offline_environment()
        from huggingface_hub import try_to_load_from_cache

        for name in (*required, *optional):
            cached = try_to_load_from_cache(model_name, name)
            if isinstance(cached, str):
                found[name] = Path(cached)

    missing = [name for name in required if name not in found]
    if missing:
        where = f"in {model_dir}" if model_dir is not None else "in the local cache"
        raise ModelUnavailableError(
            f"model files for {model_name!r} are not available {where} "
            f"(missing: {', '.join(missing)}); run `seekr models pull` to download "
            "them. seekr never downloads models implicitly."
        )
    return found


def read_pooling_mode(path: Path | None) -> PoolingMode:
    """Read a sentence-transformers pooling config; mean pooling by default."""
    if path is None:
        return "mean"
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "mean"
    if isinstance(config, dict) and config.get("pooling_mode_cls_token") is True:
        return "cls"
    return "mean"


def _default_session_factory(providers: Sequence[str]) -> Callable[[Path], Any]:
    def factory(path: Path) -> Any:
        try:
            import onnxruntime
        except ImportError as exc:
            raise ModelUnavailableError(
                f"could not import onnxruntime: {exc}. Reinstall seekr-search to "
                "restore its runtime dependencies."
            ) from exc
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        return onnxruntime.InferenceSession(
            str(path), sess_options=options, providers=list(providers)
        )

    return factory


def _default_tokenizer_factory(path: Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ModelUnavailableError(
            f"could not import tokenizers: {exc}. Reinstall seekr-search to "
            "restore its runtime dependencies."
        ) from exc
    return Tokenizer.from_file(str(path))


class _OnnxModel:
    """Shared lazy loading, tokenisation and session feeding."""

    _required: tuple[str, ...] = ONNX_EMBED_FILES
    _optional: tuple[str, ...] = ()

    def __init__(
        self,
        model_name: str,
        *,
        model_dir: Path | None = None,
        providers: Sequence[str] | None = None,
        session_factory: Callable[[Path], Any] | None = None,
        tokenizer_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        self._model_dir = Path(model_dir) if model_dir is not None else None
        self._providers = tuple(providers) if providers else _DEFAULT_PROVIDERS
        self._session_factory = session_factory or _default_session_factory(self._providers)
        self._tokenizer_factory = tokenizer_factory or _default_tokenizer_factory
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._files: dict[str, Path] = {}
        self._input_names: tuple[str, ...] = ()
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            files = resolve_model_files(
                self.model_name,
                self._required,
                self._optional,
                model_dir=self._model_dir,
            )
            tokenizer = self._tokenizer_factory(files["tokenizer.json"])
            tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
            tokenizer.enable_padding()
            session = self._session_factory(files[ONNX_MODEL_FILE])
            names = tuple(
                port.name for port in session.get_inputs() if port.name in _FEED_NAMES
            )
            if "input_ids" not in names:
                raise ModelUnavailableError(
                    f"ONNX model for {self.model_name!r} has no input_ids input"
                )
            self._files = files
            self._input_names = names
            self._tokenizer = tokenizer
            self._on_loaded(session)
            self._session = session

    def _on_loaded(self, session: Any) -> None:
        """Hook for subclasses to read model metadata before first use."""

    def _run(self, inputs: list[Any]) -> tuple[Any, Any]:
        """Tokenise one batch and run it; return ``(first_output, attention_mask)``."""
        import numpy as np

        encodings = self._tokenizer.encode_batch(inputs)
        columns = {
            "input_ids": [enc.ids for enc in encodings],
            "attention_mask": [enc.attention_mask for enc in encodings],
            "token_type_ids": [enc.type_ids for enc in encodings],
        }
        feed = {name: np.asarray(columns[name], dtype=np.int64) for name in self._input_names}
        mask = feed.get("attention_mask")
        if mask is None:
            mask = np.asarray(columns["attention_mask"], dtype=np.int64)
        with self._run_lock:
            outputs = self._session.run(None, feed)
        return np.asarray(outputs[0], dtype=np.float32), mask


class OnnxEmbedder(_OnnxModel):
    """Lazy, local-only ONNX Runtime sentence embedder.

    Produces float32 rows L2-normalised like the sentence-transformers backend,
    pooling the first output with the model's configured pooling mode (CLS for
    the bge models, attention-mask-weighted mean otherwise).
    """

    _required = ONNX_EMBED_FILES
    _optional = ONNX_EMBED_OPTIONAL_FILES

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self._dim: int | None = None
        self._pooling: PoolingMode = "mean"

    def _on_loaded(self, session: Any) -> None:
        self._pooling = read_pooling_mode(self._files.get("1_Pooling/config.json"))
        shape = list(getattr(session.get_outputs()[0], "shape", None) or [])
        last = shape[-1] if shape else None
        if isinstance(last, int) and not isinstance(last, bool) and last > 0:
            self._dim = last

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        if self._dim is None:
            probe = self._embed_batch(["dimension probe"])
            self._dim = int(probe.shape[1])
        return self._dim

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import numpy as np

        hidden, mask = self._run(texts)
        if hidden.ndim == 2:
            pooled = hidden
        elif self._pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            weights = mask.astype(np.float32)[:, :, None]
            counts = np.clip(weights.sum(axis=1), 1e-9, None)
            pooled = (hidden * weights).sum(axis=1) / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        safe = np.where(norms > 0, norms, np.float32(1.0))
        return np.asarray(pooled / safe, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        import numpy as np

        batch = list(texts)
        self._ensure_loaded()
        if not batch:
            return np.empty((0, self.dim), dtype=np.float32)
        parts = [
            self._embed_batch(batch[start : start + BATCH_SIZE])
            for start in range(0, len(batch), BATCH_SIZE)
        ]
        matrix = np.concatenate(parts, axis=0)
        if self._dim is None:
            self._dim = int(matrix.shape[1])
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([QUERY_PREFIX + text])[0]


class OnnxReranker(_OnnxModel):
    """Lazy, local-only ONNX Runtime cross-encoder reranker.

    Scores are the model's raw logits. The torch backend may apply an
    activation, so ``--explain`` rerank scores can differ in scale between
    backends; the ranking order is what the two backends agree on.
    """

    _required = ONNX_RERANK_FILES

    def score(self, query: str, passages: list[str]) -> list[float]:
        import numpy as np

        batch = list(passages)
        if not batch:
            return []
        self._ensure_loaded()
        scores: list[float] = []
        for start in range(0, len(batch), BATCH_SIZE):
            pairs = [(query, passage) for passage in batch[start : start + BATCH_SIZE]]
            logits, _mask = self._run(pairs)
            values = np.asarray(logits, dtype=np.float64)
            if values.ndim == 2:
                values = values[:, 0]
            if values.shape != (len(pairs),):
                raise ValueError(
                    f"cross-encoder returned shape {logits.shape} for {len(pairs)} passages"
                )
            for value in values.tolist():
                if not math.isfinite(value):
                    raise ValueError(f"cross-encoder returned a non-finite score: {value!r}")
                scores.append(float(value))
        return scores


__all__ = [
    "MAX_SEQ_LEN",
    "ONNX_EMBED_FILES",
    "ONNX_MODEL_FILE",
    "ONNX_RERANK_FILES",
    "OnnxEmbedder",
    "OnnxReranker",
    "read_pooling_mode",
    "resolve_model_files",
]
