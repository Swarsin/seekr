"""Explicit downloading and offline inspection of models and code grammars."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import chunking
from .config import Config
from .extract import _LANGUAGE_BY_SUFFIX

TORCH_PATTERNS: tuple[str, ...] = ("*.json", "*.txt", "model.safetensors", "1_Pooling/*")
ONNX_PATTERNS: tuple[str, ...] = (
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "1_Pooling/config.json",
)
# Files whose presence means a model is usable offline by each backend.
_CACHE_MARKERS: dict[str, tuple[str, ...]] = {
    "onnx": ("onnx/model.onnx",),
    "torch": ("config.json", "model.safetensors"),
}

__all__ = [
    "ONNX_PATTERNS",
    "TORCH_PATTERNS",
    "ModelAsset",
    "ProvisionError",
    "assets_for",
    "download",
    "download_grammars",
    "estimate_bytes",
    "grammar_languages",
    "is_cached",
    "missing_grammars",
]


class ProvisionError(RuntimeError):
    """Raised when a model or code grammar cannot be downloaded or provisioned."""


@dataclass(frozen=True)
class ModelAsset:
    """Description of one model repository or directory required by seekr."""

    repo_id: str
    allow_patterns: tuple[str, ...]
    purpose: Literal["embedding", "rerank"]
    backend: Literal["onnx", "torch"] = "torch"


def assets_for(cfg: Config, *, rerank: bool = False) -> list[ModelAsset]:
    """Return the model assets configured for *cfg* and its ``embed_backend``."""
    backend: Literal["onnx", "torch"] = "onnx" if cfg.embed_backend == "onnx" else "torch"
    patterns = ONNX_PATTERNS if backend == "onnx" else TORCH_PATTERNS
    assets = [
        ModelAsset(
            repo_id=cfg.embed_model,
            allow_patterns=patterns,
            purpose="embedding",
            backend=backend,
        )
    ]
    if rerank and cfg.rerank_model and cfg.rerank_model.strip():
        assets.append(
            ModelAsset(
                repo_id=cfg.rerank_model,
                allow_patterns=patterns,
                purpose="rerank",
                backend=backend,
            )
        )
    return assets


def _local_model_dir(repo_id: str) -> Path | None:
    raw = repo_id.strip() if isinstance(repo_id, str) else ""
    if not raw:
        return None
    try:
        local_dir = Path(raw).expanduser()
        if (local_dir / "config.json").is_file():
            return local_dir
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def _hub_error_types(hf_errors: object) -> tuple[type[BaseException], ...]:
    names = (
        "HTTPError",
        "EntryNotFoundError",
        "CacheNotFound",
        "CachedRepoTreeNotFoundError",
        "CorruptedCacheException",
        "RevisionResolutionError",
        "XetDownloadError",
    )
    types: list[type[BaseException]] = []
    for name in names:
        cls = getattr(hf_errors, name, None)
        if isinstance(cls, type) and issubclass(cls, Exception):
            types.append(cls)
    return tuple(types)


def is_cached(asset: ModelAsset) -> bool:
    """Return ``True`` when *asset* is present locally without accessing the network.

    A local model directory needs ``config.json`` (and ``onnx/model.onnx`` for
    the ONNX backend); a cached Hub model needs the backend's marker files.
    """
    local_dir = _local_model_dir(asset.repo_id)
    if local_dir is not None:
        if asset.backend == "onnx":
            return (local_dir / "onnx" / "model.onnx").is_file()
        return True
    if not isinstance(asset.repo_id, str) or not asset.repo_id.strip():
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    for marker in _CACHE_MARKERS.get(asset.backend, _CACHE_MARKERS["torch"]):
        try:
            cached = try_to_load_from_cache(asset.repo_id, marker)
        except (OSError, ValueError, RuntimeError):
            return False
        if not isinstance(cached, str):
            return False
    return True


def estimate_bytes(asset: ModelAsset) -> int | None:
    """Estimate download size in bytes from Hub metadata, or ``None`` if unavailable."""
    try:
        from huggingface_hub import HfApi, constants
        from huggingface_hub import errors as hf_errors
    except ImportError:
        return None
    if constants.HF_HUB_OFFLINE:
        return None
    if not isinstance(asset.repo_id, str) or not asset.repo_id.strip():
        return None

    try:
        info = HfApi().model_info(asset.repo_id, files_metadata=True)
    except (OSError, ValueError, RuntimeError, *_hub_error_types(hf_errors)):
        return None

    siblings = getattr(info, "siblings", None)
    if not siblings:
        return None

    total = 0
    found_size = False
    for sibling in siblings:
        filename = getattr(sibling, "rfilename", None)
        if not isinstance(filename, str):
            continue
        if asset.allow_patterns and not any(
            fnmatch.fnmatchcase(filename, pattern) for pattern in asset.allow_patterns
        ):
            continue
        size = getattr(sibling, "size", None)
        if size is None:
            lfs = getattr(sibling, "lfs", None)
            size = getattr(lfs, "size", None) if lfs is not None else None
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            total += size
            found_size = True
    return total if found_size else None


def download(asset: ModelAsset) -> Path:
    """Download *asset* to the local Hugging Face cache, unless running offline."""
    try:
        from huggingface_hub import constants, snapshot_download
        from huggingface_hub import errors as hf_errors
    except ImportError as exc:
        raise ProvisionError(f"huggingface_hub is required to download models: {exc}") from exc

    if constants.HF_HUB_OFFLINE:
        raise ProvisionError(
            "downloads are disabled in this process (offline mode); "
            "run `seekr models pull` in a new shell"
        )

    local_dir = _local_model_dir(asset.repo_id)
    if local_dir is not None:
        return local_dir.resolve()

    try:
        downloaded = snapshot_download(
            asset.repo_id,
            allow_patterns=list(asset.allow_patterns),
        )
    except (OSError, ValueError, RuntimeError, *_hub_error_types(hf_errors)) as exc:
        raise ProvisionError(f"failed to download model {asset.repo_id!r}: {exc}") from exc
    return Path(str(downloaded))


def grammar_languages() -> list[str]:
    """Return sorted unique tree-sitter language names configured for code extraction."""
    return sorted(set(_LANGUAGE_BY_SUFFIX.values()))


def missing_grammars() -> list[str]:
    """Return configured tree-sitter languages not currently available offline."""
    return [lang for lang in grammar_languages() if not chunking.ast_support(lang)[0]]


def download_grammars(languages: Sequence[str]) -> None:
    """Download the requested tree-sitter grammars into the local cache."""
    names = [str(lang) for lang in languages if str(lang).strip()]
    if not names:
        return
    try:
        import tree_sitter_language_pack as tslp
    except ImportError as exc:
        raise ProvisionError(
            f"tree-sitter-language-pack is required to download code grammars: {exc}"
        ) from exc

    download_fn = getattr(tslp, "download", None)
    if not callable(download_fn):
        raise ProvisionError("installed tree-sitter-language-pack does not expose download()")

    tslp_error = getattr(tslp, "Error", None)
    error_types: tuple[type[BaseException], ...] = (
        (OSError, ValueError, RuntimeError, tslp_error)
        if isinstance(tslp_error, type) and issubclass(tslp_error, Exception)
        else (OSError, ValueError, RuntimeError)
    )
    try:
        download_fn(names)
    except error_types as exc:
        raise ProvisionError(f"failed to download code grammars: {exc}") from exc
