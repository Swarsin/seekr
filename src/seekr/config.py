"""Configuration loading and validation for seekr."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from seekr.course import DEFAULT_WEEK_PATTERN, compile_week_pattern

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_VECTOR_QUANT = "int8"
VALID_VECTOR_QUANTS = frozenset({"float32", "int8"})
DEFAULT_EMBED_BACKEND = "torch"
VALID_EMBED_BACKENDS = frozenset({"onnx", "torch"})
DEFAULT_MAX_FILE_BYTES = 20_000_000
DEFAULT_CHUNK_TARGET_WORDS = 300
DEFAULT_CHUNK_MAX_WORDS = 450
DEFAULT_CHUNK_OVERLAP_WORDS = 50
DEFAULT_POLL_INTERVAL_SECONDS = 15
DEFAULT_IGNORE_GLOBS: tuple[str, ...] = (
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
)


class ConfigError(ValueError):
    """Raised when a config file is malformed or contains invalid values."""


def _xdg_path(env_var: str, fallback: Path) -> Path:
    value = os.environ.get(env_var)
    if value:
        return Path(value).expanduser()
    return Path.home() / fallback


def default_config_path() -> Path:
    """Return $SEEKR_CONFIG or $XDG_CONFIG_HOME/seekr/config.toml (fallback ~/.config)."""
    raw = os.environ.get("SEEKR_CONFIG")
    if raw is not None and raw.strip():
        return Path(raw.strip()).expanduser()
    return _xdg_path("XDG_CONFIG_HOME", Path(".config")) / "seekr" / "config.toml"


def default_data_dir() -> Path:
    """Return $XDG_DATA_HOME/seekr, falling back to ~/.local/share/seekr."""
    return _xdg_path("XDG_DATA_HOME", Path(".local") / "share") / "seekr"


def default_db_path() -> Path:
    """Return $XDG_DATA_HOME/seekr/index.db, falling back to ~/.local/share."""
    return default_data_dir() / "index.db"


@dataclass(frozen=True)
class Config:
    """Validated seekr configuration."""

    roots: tuple[Path, ...] = ()
    db_path: Path = field(default_factory=default_db_path)
    ignore_globs: tuple[str, ...] = DEFAULT_IGNORE_GLOBS
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    embed_model: str = DEFAULT_EMBED_MODEL
    rerank_model: str = DEFAULT_RERANK_MODEL
    vector_quant: str = DEFAULT_VECTOR_QUANT
    embed_backend: str = DEFAULT_EMBED_BACKEND
    chunk_target_words: int = DEFAULT_CHUNK_TARGET_WORDS
    chunk_max_words: int = DEFAULT_CHUNK_MAX_WORDS
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS
    week_pattern: str = DEFAULT_WEEK_PATTERN
    course_structure: bool = True
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.roots, tuple) or not all(
            isinstance(root, Path) for root in self.roots
        ):
            raise ConfigError("roots must be a tuple of pathlib.Path values")
        if any(not str(root).strip() for root in self.roots):
            raise ConfigError("roots must not contain empty paths")
        if not isinstance(self.db_path, Path):
            raise ConfigError("db_path must be a pathlib.Path")
        if not str(self.db_path).strip():
            raise ConfigError("db_path must not be empty")
        if not isinstance(self.ignore_globs, tuple) or not all(
            isinstance(glob, str) for glob in self.ignore_globs
        ):
            raise ConfigError("ignore_globs must be a tuple of strings")
        if any(not glob.strip() for glob in self.ignore_globs):
            raise ConfigError("ignore_globs must not contain empty patterns")
        if not isinstance(self.embed_model, str) or not self.embed_model.strip():
            raise ConfigError("embed_model must be a non-empty string")
        if not isinstance(self.rerank_model, str) or not self.rerank_model.strip():
            raise ConfigError("rerank_model must be a non-empty string")
        if (
            not isinstance(self.vector_quant, str)
            or self.vector_quant not in VALID_VECTOR_QUANTS
        ):
            choices = ", ".join(sorted(VALID_VECTOR_QUANTS))
            raise ConfigError(
                f"vector_quant must be one of {choices}; update vector_quant in config.toml"
            )
        if (
            not isinstance(self.embed_backend, str)
            or self.embed_backend not in VALID_EMBED_BACKENDS
        ):
            choices = ", ".join(sorted(VALID_EMBED_BACKENDS))
            raise ConfigError(
                f"embed_backend must be one of {choices}; update embed_backend in config.toml"
            )
        if not isinstance(self.max_file_bytes, int) or isinstance(self.max_file_bytes, bool):
            raise ConfigError("max_file_bytes must be an integer")
        if self.max_file_bytes <= 0:
            raise ConfigError("max_file_bytes must be greater than 0")
        for name in ("chunk_target_words", "chunk_max_words", "chunk_overlap_words"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigError(f"{name} must be an integer")
        if self.chunk_target_words <= 0:
            raise ConfigError("chunk_target_words must be greater than 0")
        if self.chunk_max_words < self.chunk_target_words:
            raise ConfigError("chunk_max_words must be >= chunk_target_words")
        if self.chunk_overlap_words < 0:
            raise ConfigError("chunk_overlap_words must be >= 0")
        if self.chunk_overlap_words >= self.chunk_target_words:
            raise ConfigError("chunk_overlap_words must be less than chunk_target_words")
        if not isinstance(self.week_pattern, str):
            raise ConfigError("week_pattern must be a string")
        try:
            compile_week_pattern(self.week_pattern)
        except ValueError as exc:
            raise ConfigError(f"week_pattern is invalid: {exc}") from exc
        if not isinstance(self.course_structure, bool):
            raise ConfigError("course_structure must be a boolean")
        if (
            not isinstance(self.poll_interval_seconds, int)
            or isinstance(self.poll_interval_seconds, bool)
        ):
            raise ConfigError("poll_interval_seconds must be an integer")
        if self.poll_interval_seconds < 1:
            raise ConfigError("poll_interval_seconds must be >= 1")


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}; fix the syntax in {path} and retry") from exc
    return data


def _path_list(value: Any, key: str) -> tuple[Path, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be an array of strings")
    paths: list[Path] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{key} must contain only non-empty strings")
        paths.append(Path(item).expanduser())
    return tuple(paths)


def _glob_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ConfigError("ignore_globs must be an array of non-empty strings")
    return tuple(value)


def _db_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("db_path must be a non-empty string")
    return Path(value).expanduser()


def load_config(path: Path | None = None) -> Config:
    """Load configuration from a TOML file, using defaults when it is missing."""
    config_path = default_config_path() if path is None else Path(path).expanduser()
    data = _read_toml(config_path)

    known = {item.name for item in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        valid = ", ".join(sorted(known))
        raise ConfigError(
            f"unknown config key(s) in {config_path}: {', '.join(unknown)}; "
            f"valid keys are: {valid}"
        )

    values: dict[str, Any] = {}
    if "roots" in data:
        values["roots"] = _path_list(data["roots"], "roots")
    if "db_path" in data:
        values["db_path"] = _db_path(data["db_path"])
    if "ignore_globs" in data:
        values["ignore_globs"] = _glob_list(data["ignore_globs"])
    if "max_file_bytes" in data:
        values["max_file_bytes"] = data["max_file_bytes"]
    if "embed_model" in data:
        values["embed_model"] = data["embed_model"]
    if "rerank_model" in data:
        values["rerank_model"] = data["rerank_model"]
    if "vector_quant" in data:
        values["vector_quant"] = data["vector_quant"]
    if "embed_backend" in data:
        values["embed_backend"] = data["embed_backend"]
    if "chunk_target_words" in data:
        values["chunk_target_words"] = data["chunk_target_words"]
    if "chunk_max_words" in data:
        values["chunk_max_words"] = data["chunk_max_words"]
    if "chunk_overlap_words" in data:
        values["chunk_overlap_words"] = data["chunk_overlap_words"]
    if "week_pattern" in data:
        values["week_pattern"] = data["week_pattern"]
    if "course_structure" in data:
        values["course_structure"] = data["course_structure"]
    if "poll_interval_seconds" in data:
        values["poll_interval_seconds"] = data["poll_interval_seconds"]

    return Config(**values)
