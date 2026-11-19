"""Stdio Model Context Protocol (MCP) JSON-RPC 2.0 server for seekr."""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from . import __version__
from .logging_config import log_event

if TYPE_CHECKING:
    from .models import Hit

__all__ = ["McpServer"]

_LOGGER = logging.getLogger(__name__)
_DEFAULT_PROTOCOL_VERSION = "2024-11-05"

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": (
            "Search locally indexed files (code, Markdown notes, PDFs, text and "
            "transcripts) using hybrid keyword and vector retrieval."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text."},
                "k": {"type": "integer", "minimum": 1, "default": 10},
                "mode": {
                    "type": "string",
                    "enum": ["keyword", "vector", "hybrid"],
                    "default": "hybrid",
                },
                "per_file": {"type": "integer", "minimum": 1, "default": 2},
                "rerank": {"type": "boolean", "default": False},
                "path": {"type": "string", "description": "Optional path prefix filter."},
                "kind": {
                    "type": "string",
                    "enum": ["code", "markdown", "pdf", "text", "transcript"],
                    "description": "Optional document kind filter.",
                },
                "since": {
                    "type": "string",
                    "description": "Optional recency filter (e.g. '7d', '12h', '2026-01-15').",
                },
                "explain": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include per-retriever rank and score breakdown.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "similar",
        "description": (
            "Find indexed files semantically related to an indexed or on-disk file."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file to match against."},
                "k": {"type": "integer", "minimum": 1, "default": 10},
                "per_file": {"type": "integer", "minimum": 1, "default": 1},
                "path_prefix": {
                    "type": "string",
                    "description": "Optional path prefix filter.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["code", "markdown", "pdf", "text", "transcript"],
                },
                "since": {
                    "type": "string",
                    "description": "Optional recency filter (e.g. '7d', '12h', '2026-01-15').",
                },
            },
            "required": ["path"],
        },
    },
]


def _default_search(*args: Any, **kwargs: Any) -> list[Hit]:
    from .cli_search import get_hits

    return get_hits(*args, **kwargs)


def _default_similar(*args: Any, **kwargs: Any) -> list[Hit]:
    from .cli_search import get_similar_hits

    return get_similar_hits(*args, **kwargs)


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_str(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or null")
    return value


def _optional_bool(value: object, name: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _optional_positive_int(value: object, name: str, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _hit_to_dict(hit: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(hit):
        data = dataclasses.asdict(hit)
    elif isinstance(hit, Mapping):
        data = dict(hit)
    else:
        raise TypeError(f"unsupported hit type {type(hit).__name__}")
    if data.get("explain") is None:
        data.pop("explain", None)
    return data


def _jsonrpc_result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


class McpServer:
    """Newline-delimited JSON-RPC 2.0 MCP server reading stdin and writing stdout."""

    def __init__(
        self,
        *,
        search_fn: Callable[..., list[Hit]] | None = None,
        similar_fn: Callable[..., list[Hit]] | None = None,
        version: str = __version__,
    ) -> None:
        self._search_fn = search_fn if search_fn is not None else _default_search
        self._similar_fn = similar_fn if similar_fn is not None else _default_similar
        self._version = version

    def handle_message(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Process one decoded JSON-RPC message and return the response object or None."""
        has_id = "id" in request
        req_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str) or not method:
            if has_id:
                return _jsonrpc_error(req_id, -32600, "invalid request: missing method")
            return None

        params_raw = request.get("params")
        if params_raw is None:
            params: Mapping[str, Any] = {}
        elif isinstance(params_raw, Mapping):
            params = params_raw
        else:
            if has_id:
                return _jsonrpc_error(req_id, -32602, "params must be a JSON object")
            return None

        if not has_id:
            log_event(_LOGGER, logging.DEBUG, "mcp.notification", method=method)
            return None

        if method == "initialize":
            protocol_version = str(
                params.get("protocolVersion") or _DEFAULT_PROTOCOL_VERSION
            )
            log_event(
                _LOGGER,
                logging.INFO,
                "mcp.initialize",
                protocol_version=protocol_version,
            )
            return _jsonrpc_result(
                req_id,
                {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "seekr", "version": self._version},
                },
            )

        if method == "ping":
            return _jsonrpc_result(req_id, {})

        if method == "tools/list":
            return _jsonrpc_result(req_id, {"tools": _TOOLS})

        if method == "tools/call":
            return _jsonrpc_result(req_id, self._handle_tool_call(params))

        return _jsonrpc_error(req_id, -32601, f"method not found: {method!r}")

    def _handle_tool_call(self, params: Mapping[str, Any]) -> dict[str, Any]:
        tool_name = params.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return {
                "content": [{"type": "text", "text": "tool name must be a non-empty string"}],
                "isError": True,
            }
        raw_args = params.get("arguments")
        if raw_args is None:
            arguments: Mapping[str, Any] = {}
        elif isinstance(raw_args, Mapping):
            arguments = raw_args
        else:
            return {
                "content": [{"type": "text", "text": "tool arguments must be a JSON object"}],
                "isError": True,
            }

        try:
            hits = self._dispatch_tool(tool_name, arguments)
            payload = [_hit_to_dict(hit) for hit in hits]
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except (
            ValueError,
            TypeError,
            LookupError,
            OSError,
            RuntimeError,
            ImportError,
            sqlite3.Error,
        ) as exc:
            log_event(
                _LOGGER,
                logging.INFO,
                "mcp.tool_error",
                tool=tool_name,
                error=str(exc),
            )
            return {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            }

        log_event(
            _LOGGER,
            logging.DEBUG,
            "mcp.tool_ok",
            tool=tool_name,
            hits=len(payload),
        )
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        }

    def _dispatch_tool(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> list[Hit]:
        if tool_name == "search":
            if "query" not in arguments:
                raise ValueError("search requires a 'query' string")
            query = _require_str(arguments.get("query"), "query")
            k = _optional_positive_int(arguments.get("k"), "k", default=10)
            mode = _optional_str(arguments.get("mode"), "mode") or "hybrid"
            per_file = _optional_positive_int(
                arguments.get("per_file"), "per_file", default=2
            )
            rerank = _optional_bool(arguments.get("rerank"), "rerank", default=False)
            path_prefix = _optional_str(arguments.get("path"), "path")
            kind = _optional_str(arguments.get("kind"), "kind")
            since = _optional_str(arguments.get("since"), "since")
            explain = _optional_bool(arguments.get("explain"), "explain", default=False)

            extra: dict[str, Any] = {}
            if explain:
                extra["explain"] = True
            if kind is not None:
                extra["kind"] = kind
            if since is not None:
                extra["since"] = since
            return self._search_fn(
                query,
                k=k,
                mode=mode,
                per_file=per_file,
                rerank=rerank,
                path_prefix=path_prefix,
                **extra,
            )

        if tool_name == "similar":
            if "path" not in arguments:
                raise ValueError("similar requires a 'path' string")
            target = Path(_require_str(arguments.get("path"), "path"))
            k = _optional_positive_int(arguments.get("k"), "k", default=10)
            per_file = _optional_positive_int(
                arguments.get("per_file"), "per_file", default=1
            )
            path_prefix = _optional_str(arguments.get("path_prefix"), "path_prefix")
            kind = _optional_str(arguments.get("kind"), "kind")
            since = _optional_str(arguments.get("since"), "since")
            return self._similar_fn(
                target,
                k=k,
                per_file=per_file,
                path_prefix=path_prefix,
                kind=kind,
                since=since,
            )

        raise LookupError(f"unknown tool: {tool_name!r}; expected 'search' or 'similar'")

    def serve_stdio(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        """Read newline-delimited JSON-RPC messages from *stdin* and reply on *stdout*."""
        reader = stdin if stdin is not None else sys.stdin
        writer = stdout if stdout is not None else sys.stdout

        log_event(_LOGGER, logging.INFO, "mcp.serve.start")
        try:
            for raw_line in reader:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError as exc:
                    response: dict[str, Any] | None = _jsonrpc_error(
                        None, -32700, f"parse error: {exc}"
                    )
                else:
                    if not isinstance(decoded, Mapping):
                        response = _jsonrpc_error(
                            None, -32600, "invalid request: message must be a JSON object"
                        )
                    else:
                        response = self.handle_message(decoded)
                if response is not None:
                    encoded = json.dumps(
                        response, ensure_ascii=False, separators=(",", ":")
                    )
                    writer.write(encoded + "\n")
                    writer.flush()
        finally:
            log_event(_LOGGER, logging.INFO, "mcp.serve.stop")
