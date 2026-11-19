"""Standalone stdlib-only offline network audit hook for seekr tests.

Used by ``tests/test_cached_pipeline.py`` to prove that seekr's indexing and
search pipelines make zero outbound attempts.

While used as a context manager, the singleton hook blocks and records every
audited outbound socket/DNS attempt (``socket.connect``,
``socket.create_connection``, ``socket.getaddrinfo``) whose target is not
local, and merely records local traffic (AF_UNIX paths, loopback hosts).
CPython raises these audit events *before* performing the socket call, so a
raise inside the hook prevents any kernel-level connection or DNS lookup from
ever happening - no real network packets are sent.

The hook is installed once per process and becomes inert when no scope is
active, so it cannot disturb unrelated tests sharing the interpreter.
"""

from __future__ import annotations

import contextlib
import ipaddress
import sys
import threading
from collections.abc import Iterator
from typing import Any

_AUDITED_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.create_connection",
        "socket.getaddrinfo",
    }
)

_LOCAL_HOST_NAMES = frozenset({"localhost"})


class OfflineNetworkAttempt(RuntimeError):
    """Raised (from inside the audit hook) when an outbound attempt is blocked."""


def _extract_target(event: str, args: tuple[Any, ...]) -> Any:
    """Return the outbound target carried by one audited event.

    ``socket.connect`` -> ``(socket, address)``; use ``address``.
    ``socket.create_connection`` -> ``(address, timeout, ...)``; use ``address``.
    ``socket.getaddrinfo`` -> ``(host, port, ...)``; use ``host``.
    """
    if event == "socket.connect":
        return args[1] if len(args) >= 2 else None
    if event == "socket.create_connection":
        return args[0] if args else None
    return args[0] if args else None  # socket.getaddrinfo


def _host_is_local(host: Any) -> bool:
    """Return True for the host part of a resolution/connect tuple.

    Only loopback literals and the ``localhost`` name count as local; every
    other host name or IP (even private ones) is outbound.
    ``socket.getaddrinfo`` hands host names here, so an unknown name is never
    local - resolving it would be an outbound DNS attempt.
    """
    if isinstance(host, (bytes, bytearray, memoryview)):
        return True  # opaque/abstract byte address: never leaves the machine
    if not isinstance(host, str):
        return False
    if host in _LOCAL_HOST_NAMES or host.startswith("127."):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]").strip()).is_loopback
    except ValueError:
        return False


def is_local_target(target: Any) -> bool:
    """Return True for connect-style targets that cannot leave the machine.

    AF_UNIX sockets (filesystem paths and abstract byte addresses) and loopback
    hosts are local; everything else is outbound.  Used for
    ``socket.connect`` addresses; resolution-stage hosts go through
    :func:`_host_is_local` instead, because there a bare ``str`` is a host
    name, not an AF_UNIX path.
    """
    if isinstance(target, (str, bytes, bytearray, memoryview)):
        # AF_UNIX sockets: filesystem paths and abstract byte addresses.
        return True
    if isinstance(target, tuple) and target:
        return _host_is_local(target[0])
    return False


def _attempt_is_local(event: str, target: Any) -> bool:
    if event == "socket.connect":
        # A bare str/bytes address here is an AF_UNIX socket, never a host.
        return is_local_target(target)
    if isinstance(target, tuple) and target:  # create_connection (host, port)
        return _host_is_local(target[0])
    return _host_is_local(target)  # getaddrinfo host


class NetworkAuditRecord:
    """Everything one offline scope blocked or allowed."""

    def __init__(self) -> None:
        self.blocked: list[tuple[str, Any]] = []
        self.allowed: list[tuple[str, Any]] = []

    @property
    def attempts(self) -> list[tuple[str, Any]]:
        """All recorded attempts, blocked ones first."""
        return self.blocked + self.allowed

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"NetworkAuditRecord(blocked={self.blocked!r}, allowed={self.allowed!r})"


_ACTIVE: list[NetworkAuditRecord] = []
_LOCK = threading.Lock()
_HOOK_INSTALLED = False
_DISPATCHING = threading.local()


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event not in _AUDITED_EVENTS:
        return
    if getattr(_DISPATCHING, "inside", False):
        # Never recurse on our own bookkeeping (defensive; list appends and
        # string formatting raise no audited events).
        return
    with _LOCK:
        active = list(_ACTIVE)
    if not active:
        return
    target = _extract_target(event, args)
    local = _attempt_is_local(event, target)
    _DISPATCHING.inside = True
    try:
        for record in active:
            if local:
                record.allowed.append((event, target))
            else:
                record.blocked.append((event, target))
    finally:
        _DISPATCHING.inside = False
    if not local:
        raise OfflineNetworkAttempt(
            f"blocked outbound network attempt: {event} target={target!r}"
        )


def _install_hook() -> None:
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _HOOK_INSTALLED = True


@contextlib.contextmanager
def offline_network_audit() -> Iterator[NetworkAuditRecord]:
    """Record and block outbound socket/DNS attempts within the block.

    Yields a :class:`NetworkAuditRecord`.  Any outbound attempt raises
    :class:`OfflineNetworkAttempt`; AF_UNIX/loopback traffic is allowed and
    still recorded under ``record.allowed``.
    """
    record = NetworkAuditRecord()
    with _LOCK:
        _install_hook()
        _ACTIVE.append(record)
    try:
        yield record
    finally:
        with _LOCK:
            try:
                _ACTIVE.remove(record)
            except ValueError:  # pragma: no cover - defensive
                pass


__all__ = [
    "NetworkAuditRecord",
    "OfflineNetworkAttempt",
    "is_local_target",
    "offline_network_audit",
]
