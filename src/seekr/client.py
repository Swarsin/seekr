"""Client for the warm seekr query daemon.

``try_daemon_search`` returns ``None`` only when there is no listener at the
socket path (including a stale socket file or regular file left behind).  Once
a connection is established, timeouts, protocol problems and server errors
raise :class:`DaemonError`; the CLI must not silently fall back to a cold local
search after a partial daemon interaction.
"""

from __future__ import annotations

import errno
import json
import math
import os
import socket
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .daemon import (
    MAX_RESPONSE_BYTES,
    legacy_daemon_supported,
    socket_path,
    socket_path_problem,
)
from .models import Hit, HitExplain

__all__ = [
    "DaemonError",
    "DaemonMismatchError",
    "try_daemon_search",
    "try_daemon_status",
]

_NO_LISTENER_ERRNOS = frozenset(
    {
        errno.ENOENT,
        errno.ECONNREFUSED,
        errno.ENOTSOCK,
    }
)


class DaemonError(RuntimeError):
    """The daemon was reachable but the query could not be completed."""


class DaemonMismatchError(DaemonError):
    """The daemon's database or model no longer matches the local config."""


def _connect(path: Path, timeout: float) -> socket.socket | None:
    try:
        timeout_value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise DaemonError("daemon timeout must be a number") from exc
    if not math.isfinite(timeout_value) or timeout_value < 0:
        raise DaemonError("daemon timeout must be a non-negative finite number")

    if socket_path_problem(path) is not None:
        return None

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout_value)
    try:
        connection.connect(str(path))
    except (FileNotFoundError, ConnectionRefusedError, NotADirectoryError):
        connection.close()
        return None
    except TimeoutError as exc:
        connection.close()
        raise DaemonError(
            f"timed out connecting to seekr daemon at {path}; check or restart `seekr daemon`"
        ) from exc
    except OSError as exc:
        connection.close()
        if exc.errno in _NO_LISTENER_ERRNOS:
            return None
        raise DaemonError(f"cannot connect to seekr daemon at {path}: {exc}") from exc
    return connection


def _peer_credentials(connection: socket.socket) -> tuple[int, int, int]:
    """Return ``(pid, uid, gid)`` for the process on the other socket end."""
    if not hasattr(socket, "SO_PEERCRED"):
        raise DaemonError(
            "cannot verify seekr daemon peer credentials on this platform"
        )
    try:
        raw = connection.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        return struct.unpack("3i", raw)
    except (OSError, struct.error) as exc:
        raise DaemonError(
            f"cannot verify seekr daemon peer credentials: {exc}"
        ) from exc


def _verify_peer(connection: socket.socket) -> None:
    """Refuse to talk to a daemon owned by a different user or group."""
    _pid, uid, gid = _peer_credentials(connection)
    expected_uid = os.getuid()
    expected_gid = os.getgid()
    if uid != expected_uid or gid != expected_gid:
        raise DaemonError(
            "refusing to send a search to a seekr daemon with mismatched peer "
            f"credentials (uid={uid}, gid={gid}; expected uid={expected_uid}, "
            f"gid={expected_gid})"
        )


def _expected_payload(expected: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if expected is None:
        return None
    normalized: dict[str, Any] = {}
    for key, value in expected.items():
        if value is None:
            continue
        if key == "dim":
            try:
                normalized[key] = int(value)
            except (TypeError, ValueError) as exc:
                raise DaemonError("expected dim must be an integer") from exc
        elif key == "db_path":
            normalized[key] = str(Path(str(value)).expanduser().resolve(strict=False))
        else:
            normalized[key] = str(value)
    return normalized


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DaemonError(f"daemon hit explain field {name!r} must be an integer or null")
    return value


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DaemonError(f"daemon hit explain field {name!r} must be a number or null")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DaemonError(
            f"daemon hit explain field {name!r} must be a number or null"
        ) from exc


def _explain_from_dict(payload: object) -> HitExplain | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise DaemonError("daemon hit explain must be a JSON object")
    return HitExplain(
        keyword_rank=_optional_int(payload.get("keyword_rank"), "keyword_rank"),
        keyword_bm25=_optional_float(payload.get("keyword_bm25"), "keyword_bm25"),
        vector_rank=_optional_int(payload.get("vector_rank"), "vector_rank"),
        vector_distance=_optional_float(payload.get("vector_distance"), "vector_distance"),
        rrf_score=_optional_float(payload.get("rrf_score"), "rrf_score"),
        rerank_rank=_optional_int(payload.get("rerank_rank"), "rerank_rank"),
        rerank_score=_optional_float(payload.get("rerank_score"), "rerank_score"),
    )


def _highlights_from_wire(raw: object, snippet_len: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, (list, tuple)):
        raise DaemonError("daemon hit field 'highlights' must be a list of [start, end] pairs")
    spans: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise DaemonError("daemon hit highlight must be a [start, end] pair")
        start, end = item
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise DaemonError("daemon hit highlight bounds must be integers")
        if start < 0 or end <= start or end > snippet_len:
            raise DaemonError("daemon hit highlight bounds are out of range")
        spans.append((start, end))
    return tuple(spans)


def _hit_from_dict(payload: object) -> Hit:
    if not isinstance(payload, Mapping):
        raise DaemonError("daemon returned a hit that is not a JSON object")
    try:
        chunk_id = payload["chunk_id"]
        path = payload["path"]
        locator = payload["locator"]
        loc_start = payload["loc_start"]
        snippet = payload["snippet"]
        score = payload["score"]
        kind = payload["kind"]
    except KeyError as exc:
        raise DaemonError(f"daemon hit is missing field {exc.args[0]!r}") from exc

    if (
        isinstance(chunk_id, bool)
        or not isinstance(chunk_id, int)
        or not isinstance(path, str)
        or not isinstance(locator, str)
        or not isinstance(snippet, str)
        or not isinstance(kind, str)
    ):
        raise DaemonError("daemon hit has invalid field types")
    if isinstance(loc_start, bool) or isinstance(score, bool):
        raise DaemonError("daemon hit has invalid numeric field types")
    try:
        loc_start_value = float(loc_start)
        score_value = float(score)
    except (TypeError, ValueError) as exc:
        raise DaemonError("daemon hit has invalid numeric fields") from exc
    explain = _explain_from_dict(payload.get("explain"))
    module = payload.get("module")
    if module is not None and not isinstance(module, str):
        raise DaemonError("daemon hit field 'module' must be a string or null")
    week = payload.get("week")
    if week is not None and (isinstance(week, bool) or not isinstance(week, int)):
        raise DaemonError("daemon hit field 'week' must be an integer or null")
    highlights = _highlights_from_wire(payload.get("highlights", ()), len(snippet))
    return Hit(
        chunk_id=chunk_id,
        path=path,
        locator=locator,
        loc_start=loc_start_value,
        snippet=snippet,
        score=score_value,
        kind=kind,
        explain=explain,
        module=module,
        week=week,
        highlights=highlights,
    )


def _send_request(
    payload: Mapping[str, Any],
    sock_path: Path | None,
    timeout: float,
) -> dict[str, Any] | None:
    if not legacy_daemon_supported():
        return None
    path = socket_path() if sock_path is None else Path(sock_path)
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DaemonError(f"daemon request is not JSON serialisable: {exc}") from exc

    connection = _connect(path, timeout)
    if connection is None:
        return None

    with connection:
        _verify_peer(connection)
        try:
            connection.sendall(encoded)
        except OSError as exc:
            raise DaemonError(
                f"seekr daemon connection failed while sending: {exc}"
            ) from exc

        reader = None
        try:
            reader = connection.makefile("rb")
            line = reader.readline(MAX_RESPONSE_BYTES + 1)
        except TimeoutError as exc:
            raise DaemonError(
                f"seekr daemon at {path} timed out; check daemon health or restart `seekr daemon`"
            ) from exc
        except OSError as exc:
            raise DaemonError(
                f"cannot read from seekr daemon at {path}: {exc}"
            ) from exc
        finally:
            if reader is not None:
                try:
                    reader.close()
                except OSError:
                    pass

    if not line:
        raise DaemonError(
            "seekr daemon closed the connection without a response; restart `seekr daemon`"
        )
    if len(line) > MAX_RESPONSE_BYTES:
        raise DaemonError("seekr daemon response is too large")
    if not line.endswith(b"\n"):
        raise DaemonError("seekr daemon returned an incomplete response")

    try:
        response = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise DaemonError(f"seekr daemon returned malformed JSON: {exc}") from exc
    if not isinstance(response, dict):
        raise DaemonError("seekr daemon response is not a JSON object")

    if response.get("ok") is not True:
        error = response.get("error")
        message = str(error).strip() if error else "seekr daemon returned an error"
        if message.startswith("config mismatch"):
            raise DaemonMismatchError(message)
        raise DaemonError(message)

    return response


def try_daemon_status(
    sock_path: Path | None = None,
    timeout: float = 0.5,
) -> dict[str, Any] | None:
    """Query a running daemon for its status payload, or return ``None``."""
    return _send_request({"op": "status"}, sock_path, timeout)


def try_daemon_search(
    params: dict[str, Any],
    sock_path: Path | None = None,
    timeout: float = 2.0,
    *,
    expected: Mapping[str, Any] | None = None,
) -> list[Hit] | None:
    """Ask a running daemon for results, or return ``None`` when none listens."""
    try:
        payload = dict(params)
    except (TypeError, ValueError) as exc:
        raise DaemonError("daemon request must be a mapping") from exc
    payload.setdefault("op", "search")
    expectation = _expected_payload(expected)
    if expectation is not None:
        payload["expect"] = expectation

    response = _send_request(payload, sock_path, timeout)
    if response is None:
        return None

    hits = response.get("hits")
    if not isinstance(hits, list):
        raise DaemonError("seekr daemon response has no hits list")
    return [_hit_from_dict(hit) for hit in hits]
