"""Warm Unix-socket query daemon with a newline-delimited JSON protocol.

The daemon owns one :class:`~seekr.search.Searcher` and answers ``ping``,
``status`` and ``search`` requests.  Search requests use exactly the fields
documented in the project plan; an optional ``expect`` mapping lets a client
prove that the daemon's database and embedding model still match its own
configuration, so a stale daemon can never return silently wrong results.

The socket lives at ``$XDG_RUNTIME_DIR/seekr.sock`` with the documented
``/tmp/seekr-<uid>.sock`` fallback and is created mode ``0600``.  A regular file
or another live daemon at that path is never deleted or replaced.
"""

from __future__ import annotations

import dataclasses
import errno
import json
import os
import socket
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .models import Hit
from .search import SEARCH_MODES

__all__ = [
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "DaemonError",
    "serve",
    "socket_path",
]

_SOCKET_MODE = 0o600
_ACCEPT_TIMEOUT = 0.2
_PROBE_TIMEOUT = 0.2
_MAX_CONNECTIONS = 32
_MAX_THREAD_JOIN = 2.0
MAX_REQUEST_BYTES = 65_536
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_EXPECT_KEYS = frozenset({"db_path", "embed_model", "dim", "rerank_model"})


class DaemonError(RuntimeError):
    """A daemon socket could not be created or safely served."""


def socket_path() -> Path:
    """Return the daemon socket path, honouring ``XDG_RUNTIME_DIR``."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return Path(runtime_dir).expanduser() / "seekr.sock"
    return Path("/tmp") / f"seekr-{os.getuid()}.sock"


def _error_text(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _resolved(value: object) -> str:
    try:
        return str(Path(str(value)).expanduser().resolve(strict=False))
    except (OSError, RuntimeError):
        return str(value)


def _searcher_metadata(searcher: object) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw = getattr(searcher, "daemon_metadata", None)
    if isinstance(raw, Mapping):
        for key in _EXPECT_KEYS:
            if key not in raw:
                continue
            value = raw[key]
            if value is None:
                continue
            if key == "dim":
                try:
                    metadata[key] = int(value)
                except (TypeError, ValueError):
                    continue
            elif key == "db_path":
                metadata[key] = _resolved(value)
            else:
                metadata[key] = str(value)

    store = getattr(searcher, "store", None)
    if "db_path" not in metadata:
        db_path = getattr(store, "_db_path", None)
        if db_path is None:
            db_path = getattr(store, "db_path", None)
        if db_path is not None:
            metadata["db_path"] = _resolved(db_path)
    if "embed_model" not in metadata:
        embed_model = getattr(store, "_embed_model", None)
        if embed_model is None:
            embed_model = getattr(store, "embed_model", None)
        if embed_model is not None:
            metadata["embed_model"] = str(embed_model)
    if "dim" not in metadata:
        dim = getattr(store, "_dim", None)
        if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
            metadata["dim"] = dim
    return metadata


def _expectation_error(request: Mapping[str, Any], metadata: Mapping[str, Any]) -> str | None:
    expected = request.get("expect")
    if expected is None:
        return None
    if not isinstance(expected, Mapping):
        return "config mismatch: invalid expect payload"

    for key, wanted in expected.items():
        if key not in _EXPECT_KEYS:
            continue
        actual = metadata.get(key)
        if actual is None:
            return f"config mismatch: daemon does not report {key}"
        if key == "db_path":
            if _resolved(actual) != _resolved(wanted):
                return (
                    "config mismatch: daemon database is "
                    f"{actual}, expected {wanted}; restart the daemon"
                )
        elif key == "dim":
            try:
                if int(actual) != int(wanted):
                    return (
                        "config mismatch: daemon embedding dimension is "
                        f"{actual}, expected {wanted}; restart the daemon"
                    )
            except (TypeError, ValueError):
                return "config mismatch: invalid expected embedding dimension"
        elif str(actual) != str(wanted):
            return (
                f"config mismatch: daemon {key} is {actual!r}, "
                f"expected {wanted!r}; restart the daemon"
            )
    return None


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _search_arguments(request: Mapping[str, Any]) -> dict[str, Any]:
    query = request.get("query")
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    mode = request.get("mode", "hybrid")
    if mode not in SEARCH_MODES:
        raise ValueError(
            f"unknown search mode {mode!r}; expected 'keyword', 'vector' or 'hybrid'"
        )
    rerank = request.get("rerank", False)
    if not isinstance(rerank, bool):
        raise TypeError("rerank must be a boolean")
    path_prefix = request.get("path_prefix")
    if path_prefix is not None and not isinstance(path_prefix, str):
        raise TypeError("path_prefix must be a string or null")
    return {
        "query": query,
        "k": _positive_int(request.get("k", 10), "k"),
        "mode": mode,
        "per_file": _positive_int(request.get("per_file", 2), "per_file"),
        "rerank": rerank,
        "path_prefix": path_prefix,
    }


def _ensure_reranker(
    searcher: object,
    requested: bool,
    factory: Callable[[], object] | None,
    lock: threading.Lock,
) -> str | None:
    if not requested:
        return None
    if getattr(searcher, "reranker", None) is not None:
        return None
    if factory is None:
        return "rerank requested but this daemon has no reranker configured"
    with lock:
        if getattr(searcher, "reranker", None) is not None:
            return None
        try:
            reranker = factory()
        except Exception as exc:  # noqa: BLE001 - optional user-supplied backend
            return f"cannot build reranker: {_error_text(exc)}"
        if reranker is None:
            return "reranker factory returned no reranker"
        try:
            searcher.reranker = reranker  # type: ignore[attr-defined]
        except (AttributeError, TypeError) as exc:
            return f"cannot attach reranker: {_error_text(exc)}"
    return None


def _hit_to_dict(hit: object) -> dict[str, Any]:
    if isinstance(hit, Hit) or dataclasses.is_dataclass(hit):
        return dataclasses.asdict(hit)  # type: ignore[arg-type]
    if isinstance(hit, Mapping):
        return dict(hit)
    raise TypeError(f"search returned an unsupported hit type {type(hit).__name__}")


def _handle_request(
    searcher: object,
    request: Mapping[str, Any],
    metadata: Mapping[str, Any],
    reranker_factory: Callable[[], object] | None,
    reranker_lock: threading.Lock,
) -> dict[str, Any]:
    op = request.get("op")
    if op == "ping":
        return {"ok": True}
    if op == "status":
        store = getattr(searcher, "store", None)
        stats_fn = getattr(store, "stats", None)
        if not callable(stats_fn):
            return {"ok": False, "error": "status is unavailable: search store has no stats"}
        try:
            stats = dict(stats_fn())
        except Exception as exc:  # noqa: BLE001 - store is an injected backend
            return {"ok": False, "error": f"cannot read status: {_error_text(exc)}"}
        return {"ok": True, "stats": stats}
    if op != "search":
        return {"ok": False, "error": f"unknown op: {op!r}"}

    mismatch = _expectation_error(request, metadata)
    if mismatch is not None:
        return {"ok": False, "error": mismatch}

    try:
        arguments = _search_arguments(request)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}

    reranker_error = _ensure_reranker(
        searcher,
        bool(arguments["rerank"]),
        reranker_factory,
        reranker_lock,
    )
    if reranker_error is not None:
        return {"ok": False, "error": reranker_error}

    search = getattr(searcher, "search", None)
    if not callable(search):
        return {"ok": False, "error": "search is unavailable on this daemon"}
    try:
        hits = list(search(**arguments))
    except Exception as exc:  # noqa: BLE001 - isolate every query failure
        return {"ok": False, "error": f"search failed: {_error_text(exc)}"}
    try:
        return {"ok": True, "hits": [_hit_to_dict(hit) for hit in hits]}
    except Exception as exc:  # noqa: BLE001 - a bad hit must not kill the server
        return {"ok": False, "error": f"cannot serialise search results: {_error_text(exc)}"}


def _send_json(connection: socket.socket, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    connection.sendall(data)


def _safe_send(connection: socket.socket, payload: Mapping[str, Any]) -> bool:
    try:
        _send_json(connection, payload)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _serve_connection(
    connection: socket.socket,
    stop_event: threading.Event,
    handle_request: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> None:
    try:
        reader = connection.makefile("rb")
    except OSError:
        return
    try:
        while not stop_event.is_set():
            try:
                raw = reader.readline(MAX_REQUEST_BYTES + 1)
            except (OSError, ValueError):
                break
            if not raw:
                break
            if len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                _safe_send(
                    connection,
                    {
                        "ok": False,
                        "error": (
                            "request is too large or is not one complete newline-"
                            "terminated JSON line"
                        ),
                    },
                )
                break
            try:
                request = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                if not _safe_send(
                    connection,
                    {"ok": False, "error": f"malformed JSON: {_error_text(exc)}"},
                ):
                    break
                continue
            if not isinstance(request, Mapping):
                if not _safe_send(
                    connection,
                    {"ok": False, "error": "request must be a JSON object"},
                ):
                    break
                continue
            if not _safe_send(connection, handle_request(request)):
                break
    finally:
        try:
            reader.close()
        except OSError:
            pass


def _socket_is_active(path: Path) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(_PROBE_TIMEOUT)
    try:
        probe.connect(str(path))
    except (FileNotFoundError, ConnectionRefusedError):
        return False
    except OSError as exc:
        return exc.errno not in (errno.ENOENT, errno.ECONNREFUSED, errno.ENOTSOCK)
    else:
        return True
    finally:
        probe.close()


def _remove_owned_socket(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        current = os.lstat(path)
    except OSError:
        return
    if not stat.S_ISSOCK(current.st_mode):
        return
    if (current.st_dev, current.st_ino) != identity:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _bind_listener(path: Path) -> tuple[socket.socket, tuple[int, int]]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DaemonError(f"cannot create socket directory {path.parent}: {exc}") from exc

    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise DaemonError(f"cannot inspect socket path {path}: {exc}") from exc
    else:
        if not stat.S_ISSOCK(existing.st_mode):
            raise DaemonError(
                f"refusing to replace non-socket file at {path}; remove it manually "
                "or pass a different --socket path"
            )
        if _socket_is_active(path):
            raise DaemonError(
                f"another seekr daemon is already listening on {path}; stop it or pass "
                "a different --socket path"
            )
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise DaemonError(f"cannot remove stale socket {path}: {exc}") from exc

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bound_at: tuple[int, int] | None = None
    tmp_path = path.parent / f".seekr-{uuid.uuid4().hex[:8]}.tmp"
    tmp_created = False
    # Publish via a pre-listened sibling temp socket when it fits AF_UNIX sun_path,
    # so `path` is never visible before chmod(0600) and listen() finish.
    use_atomic_link = len(os.fsencode(tmp_path)) <= 100
    try:
        old_umask = os.umask(0o177)
        try:
            if use_atomic_link:
                listener.bind(str(tmp_path))
                tmp_created = True
            else:
                listener.bind(str(path))
        finally:
            os.umask(old_umask)
        if use_atomic_link:
            os.chmod(tmp_path, _SOCKET_MODE)
            listener.listen(_MAX_CONNECTIONS)
            listener.settimeout(_ACCEPT_TIMEOUT)
            try:
                os.link(tmp_path, path)
            except FileExistsError as exc:
                raise DaemonError(
                    f"another seekr daemon is already listening on {path}; stop it or "
                    "pass a different --socket path"
                ) from exc
            created = os.lstat(path)
            bound_at = (created.st_dev, created.st_ino)
        else:
            created = os.lstat(path)
            bound_at = (created.st_dev, created.st_ino)
            os.chmod(path, _SOCKET_MODE)
            listener.listen(_MAX_CONNECTIONS)
            listener.settimeout(_ACCEPT_TIMEOUT)
    except DaemonError:
        listener.close()
        _remove_owned_socket(path, bound_at)
        raise
    except OSError as exc:
        listener.close()
        _remove_owned_socket(path, bound_at)
        raise DaemonError(f"cannot create daemon socket {path}: {exc}") from exc
    finally:
        if tmp_created:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return listener, bound_at


def serve(
    searcher: object,
    sock_path: Path,
    stop_event: threading.Event,
    *,
    reranker_factory: Callable[[], object] | None = None,
) -> None:
    """Serve newline-delimited JSON requests until *stop_event* is set.

    Requests may be pipelined and a malformed line or unknown operation gets an
    error response without closing an otherwise usable connection.  The socket
    file is removed on clean shutdown and is never removed if it no longer
    belongs to this server.
    """
    if stop_event is None:  # pragma: no branch - defensive for direct callers
        stop_event = threading.Event()
    path = Path(sock_path)
    listener, identity = _bind_listener(path)
    metadata = _searcher_metadata(searcher)
    reranker_lock = threading.Lock()
    slots = threading.BoundedSemaphore(_MAX_CONNECTIONS)
    state_lock = threading.Lock()
    active: set[socket.socket] = set()
    threads: set[threading.Thread] = set()
    accept_error: OSError | None = None

    def handle_request(request: Mapping[str, Any]) -> dict[str, Any]:
        return _handle_request(
            searcher, request, metadata, reranker_factory, reranker_lock
        )

    def run_connection(connection: socket.socket) -> None:
        try:
            _serve_connection(connection, stop_event, handle_request)
        except Exception:  # noqa: BLE001 - one client must not kill the server
            return
        finally:
            with state_lock:
                active.discard(connection)
                threads.discard(threading.current_thread())
            try:
                connection.close()
            except OSError:
                pass
            slots.release()

    try:
        while not stop_event.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError as exc:
                if stop_event.is_set():
                    break
                accept_error = exc
                break
            if stop_event.is_set():
                connection.close()
                break
            if not slots.acquire(blocking=False):
                _safe_send(
                    connection,
                    {"ok": False, "error": "server busy: too many connections"},
                )
                connection.close()
                continue
            with state_lock:
                active.add(connection)
            thread = threading.Thread(
                target=run_connection,
                args=(connection,),
                name="seekr-daemon-client",
                daemon=True,
            )
            with state_lock:
                threads.add(thread)
            try:
                thread.start()
            except Exception:
                with state_lock:
                    active.discard(connection)
                    threads.discard(thread)
                connection.close()
                slots.release()
                raise
    finally:
        try:
            listener.close()
        except OSError:
            pass
        with state_lock:
            connections = list(active)
            pending_threads = list(threads)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
        deadline = time.monotonic() + _MAX_THREAD_JOIN
        for thread in pending_threads:
            remaining = deadline - time.monotonic()
            thread.join(timeout=max(0.0, remaining))
        for thread in pending_threads:
            # The closing connections above release clients blocked on reads,
            # but a search already in flight must finish before the caller
            # closes the Store underneath it.
            if thread.is_alive():
                thread.join()
        _remove_owned_socket(path, identity)

    if accept_error is not None:
        raise DaemonError(f"daemon accept failed: {accept_error}")
