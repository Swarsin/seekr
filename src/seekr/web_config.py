"""Read-only config inspection and safe additive roots persistence.

The local web interface deliberately supports only two TOML shapes for roots:

* no top-level ``roots`` key in a flat file (the key is appended at EOF), or
* one top-level single-line ``roots = [...]`` value (only that array span is
  replaced, preserving every other byte).

Both shapes are validated with :mod:`tomllib` and, for writes, with
``load_config`` on the staged temp file before the atomic replace.  Errors are
raised as the frozen :class:`seekr.web_contract.ApiError` values; callers do
not need a second mapping layer.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import threading
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import Config, ConfigError, load_config
from .web_contract import ApiError

__all__ = [
    "ROOTS_LIMIT",
    "ROOT_PATH_BYTES",
    "ConfigInspection",
    "RootsConfig",
    "add_root",
    "config_revision",
    "inspect_config",
    "read_roots",
    "remove_root",
    "set_roots",
    "toml_string",
]

ROOTS_LIMIT = 50
ROOT_PATH_BYTES = 4096
_MISSING_REVISION = hashlib.sha256(b"").hexdigest()
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class RootsConfig:
    """The persisted roots, their config path, and the raw-bytes revision."""

    path: Path
    roots: tuple[str, ...]
    revision: str


@dataclass(frozen=True)
class ConfigInspection:
    """A model-free, non-mutating status view of the config file."""

    state: str
    path: Path
    revision: str | None
    roots: int | None
    message: str | None


def _absolute(path: Path | str) -> Path:
    configured = Path(path).expanduser()
    if not configured.is_absolute():
        configured = Path.cwd() / configured
    return Path(os.path.abspath(os.fspath(configured)))


def _revision(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_message(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _config_invalid(path: Path, message: str) -> ApiError:
    return ApiError(
        409,
        "config_invalid",
        f"cannot safely edit {path}: {message}; fix the file manually and retry",
    )


def _invalid_root(message: str) -> ApiError:
    return ApiError(400, "invalid_root", message, {"path": message})


def config_revision(config_path: Path) -> str:
    """Return the SHA-256 hex digest of the raw config bytes.

    A missing file is the empty byte string.  An unreadable path raises
    :class:`ApiError` ``config_invalid`` rather than pretending it is missing.
    """
    path = _absolute(config_path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return _MISSING_REVISION
    except OSError as exc:
        raise _config_invalid(path, f"cannot read config file: {_safe_message(exc)}") from exc
    return _revision(raw)


def _read_raw(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""
    except OSError as exc:
        raise _config_invalid(path, f"cannot read config file: {_safe_message(exc)}") from exc


def _parse_toml(raw: bytes, path: Path) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _config_invalid(path, f"config is not valid UTF-8: {_safe_message(exc)}") from exc
    try:
        parsed = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise _config_invalid(path, f"invalid TOML: {_safe_message(exc)}") from exc
    return parsed


def _stored_roots(parsed: dict[str, object], path: Path) -> tuple[str, ...]:
    value = parsed.get("roots", [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _config_invalid(path, "roots must be an array of strings")
    return tuple(value)


def _load_validated(path: Path) -> Config:
    try:
        return load_config(path)
    except ConfigError as exc:
        raise _config_invalid(path, _safe_message(exc)) from exc


def read_roots(config_path: Path) -> RootsConfig:
    """Return the stored roots in config order and the revision.

    The file is validated with ``load_config`` so malformed or unsupported
    configuration never reaches an editor.  Missing files are the empty config.
    """
    path = _absolute(config_path)
    raw = _read_raw(path)
    parsed = _parse_toml(raw, path)
    _load_validated(path)
    return RootsConfig(path=path, roots=_stored_roots(parsed, path), revision=_revision(raw))


def inspect_config(config_path: Path) -> ConfigInspection:
    """Return a safe status snapshot without raising for malformed config."""
    path = _absolute(config_path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return ConfigInspection(
            state="ok",
            path=path,
            revision=_MISSING_REVISION,
            roots=0,
            message=None,
        )
    except OSError as exc:
        return ConfigInspection(
            state="invalid",
            path=path,
            revision=None,
            roots=None,
            message=f"cannot read config file: {_safe_message(exc)}",
        )

    revision = _revision(raw)
    try:
        parsed = _parse_toml(raw, path)
    except ApiError as exc:
        return ConfigInspection(
            state="invalid",
            path=path,
            revision=revision,
            roots=None,
            message=exc.message,
        )

    raw_roots = parsed.get("roots")
    roots_count = len(raw_roots) if isinstance(raw_roots, list) else None
    try:
        _load_validated(path)
    except ApiError as exc:
        return ConfigInspection(
            state="invalid",
            path=path,
            revision=revision,
            roots=roots_count,
            message=exc.message,
        )
    return ConfigInspection(
        state="ok",
        path=path,
        revision=revision,
        roots=0 if roots_count is None and "roots" not in parsed else roots_count,
        message=None,
    )


def _validate_root_string(path_value: object) -> str:
    if not isinstance(path_value, str):
        raise _invalid_root("root path must be a string")
    if "\x00" in path_value:
        raise _invalid_root("root path must not contain NUL")
    try:
        encoded = path_value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _invalid_root("root path must be valid UTF-8") from exc
    if len(encoded) > ROOT_PATH_BYTES:
        raise _invalid_root(
            f"root path must be at most {ROOT_PATH_BYTES} UTF-8 bytes"
        )
    if not path_value.strip():
        raise _invalid_root("root path must not be empty")
    return path_value


def _validate_submitted_root(path_value: object) -> Path:
    validated = _validate_root_string(path_value)

    try:
        expanded = Path(validated).expanduser()
        resolved = expanded.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _invalid_root(f"root path does not exist or cannot be resolved: {_safe_message(exc)}") from exc
    try:
        if not resolved.is_dir():
            raise _invalid_root("root path must be an existing directory")
    except OSError as exc:
        raise _invalid_root(f"cannot inspect root path: {_safe_message(exc)}") from exc
    if not os.access(resolved, os.R_OK | os.X_OK):
        raise _invalid_root("root directory must be readable")
    try:
        with os.scandir(resolved) as entries:
            next(entries, None)
    except OSError as exc:
        raise _invalid_root(f"root directory is not readable: {_safe_message(exc)}") from exc
    return resolved


def _canonical_existing(value: str) -> Path | None:
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def toml_string(value: str) -> str:
    """Encode a TOML basic string, escaping every required control character."""
    if "\x00" in value:
        raise ValueError("NUL is not allowed")
    out = ['"']
    for character in value:
        code = ord(character)
        if character == '"':
            out.append('\\"')
        elif character == "\\":
            out.append("\\\\")
        elif character == "\b":
            out.append("\\b")
        elif character == "\t":
            out.append("\\t")
        elif character == "\n":
            out.append("\\n")
        elif character == "\f":
            out.append("\\f")
        elif character == "\r":
            out.append("\\r")
        elif 0xD800 <= code <= 0xDFFF:
            raise ValueError("invalid Unicode surrogate")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04X}")
        else:
            out.append(character)
    out.append('"')
    return "".join(out)


def _with_new_root(value_text: str, new_value: str) -> str:
    """Insert ``new_value`` into an existing single-line TOML array literal."""
    inner = value_text[1:-1]
    if not inner.strip():
        return f"[{inner}{new_value}]"
    body = inner.rstrip(" \t")
    tail = inner[len(body) :]
    separator = " " if body.endswith(",") else ", "
    return f"[{body}{separator}{new_value}{tail}]"


def _preferred_newline(text: str) -> str:
    if "\r\n" in text and text.count("\r\n") == text.count("\n"):
        return "\r\n"
    return "\n"


def _physical_lines(text: str) -> list[str]:
    """Split only on TOML newlines, not on Unicode line separators."""
    pieces = text.split("\n")
    lines = [piece + "\n" for piece in pieces[:-1]]
    if pieces[-1]:
        lines.append(pieces[-1])
    return lines


def _line_content(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith("\n"):
        return line[:-1]
    return line


def _find_triple_end(line: str, start: int, delimiter: str) -> int | None:
    position = start
    while True:
        found = line.find(delimiter, position)
        if found < 0:
            return None
        if delimiter == '"""':
            backslashes = 0
            cursor = found - 1
            while cursor >= start and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                return found
            position = found + 3
        else:
            return found


def _visible_indices(line: str, multiline: str | None) -> tuple[list[int], str | None]:
    """Return indices outside strings/comments and the updated multiline state."""
    visible: list[int] = []
    index = 0
    length = len(line)
    while index < length:
        if multiline is not None:
            end = _find_triple_end(line, index, multiline)
            if end is None:
                return visible, multiline
            index = end + 3
            multiline = None
            continue

        character = line[index]
        if character == "#":
            break
        if character == '"':
            if line.startswith('"""', index):
                multiline = '"""'
                index += 3
                continue
            index += 1
            while index < length:
                if line[index] == "\\":
                    index += 2
                elif line[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            continue
        if character == "'":
            if line.startswith("'''", index):
                multiline = "'''"
                index += 3
                continue
            index += 1
            while index < length:
                if line[index] == "'":
                    index += 1
                    break
                index += 1
            continue
        visible.append(index)
        index += 1
    return visible, multiline


def _normalise_key(raw_key: str) -> tuple[str, ...]:
    """Split a possibly quoted/dotted key into its stripped components."""
    components: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for character in raw_key.strip():
        if quote is not None:
            current.append(character)
            if escaped:
                escaped = False
            elif character == "\\" and quote == '"':
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
            current.append(character)
        elif character == ".":
            components.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    components.append("".join(current).strip())

    normalised: list[str] = []
    for component in components:
        if len(component) >= 2 and component[0] == component[-1] and component[0] in {'"', "'"}:
            component = component[1:-1]
        normalised.append(component)
    return tuple(normalised)


def _is_roots_key(raw_key: str) -> bool:
    return "roots" in _normalise_key(raw_key)


def _find_roots_span(
    text: str,
) -> tuple[tuple[int, int] | None, bool, str | None]:
    """Find the top-level single-line roots value span.

    Returns ``(span, has_table, error)``.  ``error`` is a human-readable reason
    for an unsupported or ambiguous expression.
    """
    span: tuple[int, int] | None = None
    has_table = False
    multiline: str | None = None
    offset = 0

    for raw_line in _physical_lines(text):
        line = _line_content(raw_line)
        visible, multiline = _visible_indices(line, multiline)
        content = [(index, line[index]) for index in visible if line[index] not in " \t"]
        if not content:
            offset += len(raw_line)
            continue

        first_index = content[0][0]
        if line[first_index] == "[":
            has_table = True
            offset += len(raw_line)
            continue

        equals = next((index for index in visible if line[index] == "="), None)
        if equals is None:
            offset += len(raw_line)
            continue

        raw_key = line[:equals].strip()
        if not _is_roots_key(raw_key):
            offset += len(raw_line)
            continue
        if _normalise_key(raw_key) != ("roots",) or raw_key != "roots":
            return span, has_table, "quoted or dotted roots keys are not supported"
        if span is not None:
            return span, has_table, "duplicate roots keys are not supported"

        value_start = equals + 1
        while value_start < len(line) and line[value_start] in " \t":
            value_start += 1
        if value_start >= len(line) or line[value_start] != "[":
            return span, has_table, "roots must be a single-line TOML array"
        closing = next(
            (index for index in visible if index > value_start and line[index] == "]"),
            None,
        )
        if closing is None:
            return span, has_table, "multiline or unterminated roots arrays are not supported"
        if any(line[index] not in " \t" for index in visible if index > closing):
            return span, has_table, "unexpected content after the roots array"
        span = (offset + value_start, offset + closing + 1)
        offset += len(raw_line)

    return span, has_table, None


def _ensure_parent(parent: Path) -> None:
    if parent.exists():
        if not parent.is_dir():
            raise OSError(f"config parent is not a directory: {parent}")
        return
    missing: list[Path] = []
    cursor = parent
    while not cursor.exists() and cursor != cursor.parent:
        missing.append(cursor)
        cursor = cursor.parent
    parent.mkdir(parents=True, mode=0o700)
    for created in reversed(missing):
        os.chmod(created, 0o700)


def _fsync_directory(parent: Path) -> None:
    try:
        descriptor = os.open(str(parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _current_raw(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""
    except OSError as exc:
        raise _config_invalid(path, f"cannot re-read config file: {_safe_message(exc)}") from exc


def _format_roots_array(roots: Sequence[str]) -> str:
    return "[" + ", ".join(toml_string(item) for item in roots) + "]"


def add_root(config_path: Path, path: str, revision: str) -> RootsConfig:
    """Append one canonical directory to the config, preserving other bytes.

    Writes are serialized process-wide so two web requests cannot both stage a
    candidate from the same revision; the revision is rechecked again directly
    before the atomic replace.
    """
    with _WRITE_LOCK:
        return _add_root_locked(config_path, path, revision)


def remove_root(config_path: Path, path: str, revision: str) -> RootsConfig:
    """Remove one configured root canonically, preserving comments and other keys."""
    with _WRITE_LOCK:
        return _remove_root_locked(config_path, path, revision)


def set_roots(
    config_path: Path,
    roots: Sequence[Path],
    revision: str | None = None,
) -> RootsConfig:
    """Replace the configured ``roots`` array, preserving comments and other keys."""
    with _WRITE_LOCK:
        return _set_roots_locked(config_path, roots, revision)


def _add_root_locked(config_path: Path, path: str, revision: str) -> RootsConfig:
    resolved = _validate_submitted_root(path)
    checked_revision = revision if isinstance(revision, str) else ""

    def compute(existing_roots: tuple[str, ...]) -> tuple[str, ...]:
        for existing in existing_roots:
            candidate = _canonical_existing(existing)
            if candidate == resolved:
                raise ApiError(
                    409,
                    "duplicate_root",
                    f"root is already configured: {resolved}",
                    {"path": str(resolved)},
                )
        if len(existing_roots) >= ROOTS_LIMIT:
            raise ApiError(
                400,
                "roots_limit",
                f"at most {ROOTS_LIMIT} roots can be configured",
            )
        return (*existing_roots, str(resolved))

    return _rewrite_roots(
        config_path,
        compute,
        checked_revision,
        preserve_array_padding=True,
    )


def _remove_root_locked(config_path: Path, path: str, revision: str) -> RootsConfig:
    validated = _validate_root_string(path)
    target = _canonical_existing(validated)
    checked_revision = revision if isinstance(revision, str) else ""

    def compute(existing_roots: tuple[str, ...]) -> tuple[str, ...]:
        remaining = tuple(
            existing
            for existing in existing_roots
            if not (
                existing == validated
                or (target is not None and _canonical_existing(existing) == target)
            )
        )
        if len(remaining) == len(existing_roots):
            raise _invalid_root(f"root is not configured: {validated}")
        return remaining

    return _rewrite_roots(config_path, compute, checked_revision)


def _set_roots_locked(
    config_path: Path,
    roots: Sequence[Path],
    revision: str | None,
) -> RootsConfig:
    if len(roots) > ROOTS_LIMIT:
        raise ApiError(
            400,
            "roots_limit",
            f"at most {ROOTS_LIMIT} roots can be configured",
        )
    validated: list[str] = []
    for item in roots:
        try:
            raw_value = os.fspath(item)
        except TypeError as exc:
            raise _invalid_root("root path must be a path or string") from exc
        validated.append(_validate_root_string(raw_value))
    new_roots = tuple(validated)
    return _rewrite_roots(config_path, lambda _existing: new_roots, revision)


def _rewrite_roots(
    config_path: Path,
    compute_roots: Callable[[tuple[str, ...]], tuple[str, ...]],
    revision: str | None,
    *,
    preserve_array_padding: bool = False,
) -> RootsConfig:
    config_path = _absolute(config_path)

    if config_path.is_symlink():
        raise _config_invalid(config_path, "refusing to replace a symlink config file")

    raw = _read_raw(config_path)
    parsed = _parse_toml(raw, config_path)
    _load_validated(config_path)
    original_revision = _revision(raw)
    if revision is not None and (not isinstance(revision, str) or revision != original_revision):
        raise ApiError(
            409,
            "stale_config",
            f"config changed on disk: {config_path}; reload the folder list and retry",
        )

    existing_roots = _stored_roots(parsed, config_path)
    new_roots = compute_roots(existing_roots)

    text = raw.decode("utf-8")
    span, has_table, span_error = _find_roots_span(text)
    if span_error is not None:
        raise _config_invalid(config_path, span_error)

    if span is not None:
        start, end = span
        if preserve_array_padding and len(new_roots) == len(existing_roots) + 1:
            replacement = _with_new_root(text[start:end], toml_string(new_roots[-1]))
        else:
            replacement = _format_roots_array(new_roots)
        candidate_text = text[:start] + replacement + text[end:]
    else:
        if has_table:
            raise _config_invalid(
                config_path,
                "cannot append a top-level roots key to a file containing table headers",
            )
        newline = _preferred_newline(text)
        prefix = text
        if prefix and not prefix.endswith(("\n", "\r")):
            prefix += newline
        candidate_text = f"{prefix}roots = {_format_roots_array(new_roots)}{newline}"

    candidate = candidate_text.encode("utf-8")
    try:
        original_parsed = tomllib.loads(text)
        candidate_parsed = tomllib.loads(candidate_text)
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise _config_invalid(config_path, f"candidate TOML is invalid: {_safe_message(exc)}") from exc
    if _stored_roots(original_parsed, config_path) != existing_roots:
        raise _config_invalid(config_path, "original roots changed while editing")
    expected_roots = tuple(Path(value).expanduser() for value in new_roots)
    if tuple(Path(value).expanduser() for value in _stored_roots(candidate_parsed, config_path)) != expected_roots:
        raise _config_invalid(config_path, "candidate roots do not match the requested change")

    try:
        existing_mode = stat.S_IMODE(config_path.stat().st_mode)
    except FileNotFoundError:
        existing_mode = 0o600
    except OSError as exc:
        raise _config_invalid(config_path, f"cannot inspect config file: {_safe_message(exc)}") from exc

    parent = config_path.parent
    descriptor: int | None = None
    temp_path: Path | None = None
    try:
        _ensure_parent(parent)
        descriptor, temp_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        os.fchmod(descriptor, 0o600 if existing_mode == 0o600 else existing_mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(candidate)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            loaded = load_config(temp_path)
        except ConfigError as exc:
            raise ApiError(
                500,
                "config_write_failed",
                f"validated config could not be written to {config_path}: {_safe_message(exc)}",
            ) from exc
        if loaded.roots != expected_roots:
            raise ApiError(
                500,
                "config_write_failed",
                f"validated config roots did not match the requested change for {config_path}",
            )

        current = _current_raw(config_path)
        if _revision(current) != original_revision:
            raise ApiError(
                409,
                "stale_config",
                f"config changed on disk: {config_path}; reload the folder list and retry",
            )
        if config_path.is_symlink():
            raise _config_invalid(config_path, "refusing to replace a symlink config file")

        os.replace(temp_path, config_path)
        temp_path = None
        _fsync_directory(parent)
    except ApiError:
        raise
    except OSError as exc:
        raise ApiError(
            500,
            "config_write_failed",
            f"could not write config file {config_path}: {_safe_message(exc)}",
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass

    return RootsConfig(path=config_path, roots=new_roots, revision=_revision(candidate))
