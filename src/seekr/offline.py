"""Offline model loading shared by embedding and reranking backends.

Every model load must go through :func:`load_local_model`.  It forces the
Hugging Face stack offline *before* that stack is imported, accepts either an
already provisioned model directory or a model ID that is present in the local
Hugging Face cache, and never downloads anything implicitly.

``rerank.CrossEncoderReranker`` should call ``load_local_model(..., kind=
"cross_encoder")`` so both backends share the same guarantees.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

ModelKind = Literal["embedding", "cross_encoder"]

#: Forced before importing sentence-transformers / huggingface_hub.
OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "DO_NOT_TRACK": "1",
}

_MODEL_ATTRIBUTES: dict[ModelKind, str] = {
    "embedding": "SentenceTransformer",
    "cross_encoder": "CrossEncoder",
}


class ModelUnavailableError(RuntimeError):
    """Raised when a model cannot be loaded from local files only."""


def configure_offline_environment(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Force Hugging Face offline mode and disable telemetry.

    Values are overwritten on purpose: a cached index or query must not make an
    outbound call even when the caller's shell has set an online value.
    """
    target = os.environ if environ is None else environ
    for key, value in OFFLINE_ENV.items():
        target[key] = value


def _import_sentence_transformers() -> ModuleType:
    """Import sentence-transformers after offline mode has been configured."""
    configure_offline_environment()
    import sentence_transformers

    return sentence_transformers


def _select_device() -> str:
    """Return the best available torch device, falling back to CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _looks_like_explicit_path(raw: str) -> bool:
    path = Path(raw)
    return path.is_absolute() or raw.startswith((".", "~"))


def _resolve_model_source(raw: str) -> str:
    """Resolve a local model directory, or pass a cached model ID through.

    Model IDs are deliberately left untouched: ``local_files_only=True`` and
    the offline environment make the Hugging Face loader consult only its
    cache.  This also keeps the adapter friendly to test doubles and to future
    Hub cache layouts.
    """
    path = Path(raw).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if path.exists():
        raise ModelUnavailableError(
            f"model path {raw!r} exists but is not a directory; "
            "provide a pre-provisioned model directory or a cached model ID. "
            "seekr never downloads models implicitly."
        )
    if _looks_like_explicit_path(raw):
        raise ModelUnavailableError(
            f"local model directory {raw!r} does not exist; "
            "provide a pre-provisioned model directory or a cached model ID. "
            "seekr never downloads models implicitly."
        )
    return raw


def _validate_model_name(model_name: str | Path) -> str:
    try:
        raw = os.fspath(model_name)
    except TypeError as exc:
        raise ModelUnavailableError("model_name must be a string or a path") from exc
    if not isinstance(raw, str) or not raw.strip():
        raise ModelUnavailableError("model_name must not be empty")
    return raw


def load_local_model(
    model_name: str | Path,
    *,
    kind: ModelKind = "embedding",
    device: str | None = None,
) -> Any:
    """Load a sentence-transformers model without any network access.

    Args:
        model_name: An existing local model directory, or a Hugging Face model
            ID that already exists in the local cache.
        kind: ``"embedding"`` for ``SentenceTransformer`` or
            ``"cross_encoder"`` for ``CrossEncoder``.  The reranker backend
            shares this adapter.
        device: Optional torch device.  When omitted, CUDA is used if
            available, otherwise CPU.

    Raises:
        ModelUnavailableError: The local directory is missing, or the model is
            not cached / cannot be loaded.  No download is attempted.
    """
    if kind not in _MODEL_ATTRIBUTES:
        raise ValueError(f"unknown model kind {kind!r}; expected embedding or cross_encoder")

    raw = _validate_model_name(model_name)
    configure_offline_environment()
    source = _resolve_model_source(raw)

    try:
        sentence_transformers = _import_sentence_transformers()
    except ImportError as exc:
        raise ModelUnavailableError(
            f"could not import sentence-transformers for model {raw!r}: {exc}. "
            'The PyTorch backend (embed_backend = "torch") needs the torch extra: '
            "`pipx install 'seekr-search[torch]'`, or for an existing pipx install "
            "`pipx inject seekr-search sentence-transformers`. Alternatively set "
            'embed_backend = "onnx". seekr never downloads models implicitly.'
        ) from exc

    model_class = getattr(sentence_transformers, _MODEL_ATTRIBUTES[kind])
    selected_device = device if device is not None else _select_device()

    try:
        return model_class(
            source,
            device=selected_device,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        label = "embedding" if kind == "embedding" else "cross-encoder"
        raise ModelUnavailableError(
            f"could not load {label} model {raw!r} from local files only: {exc}. "
            "seekr never downloads models implicitly; provide an existing local model "
            "directory or pre-download this model into the Hugging Face cache."
        ) from exc


__all__ = [
    "OFFLINE_ENV",
    "ModelKind",
    "ModelUnavailableError",
    "configure_offline_environment",
    "load_local_model",
]
