"""Tests for the stdio MCP JSON-RPC 2.0 server (seekr mcp)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

from _platform import short_tempdir
from seekr.config import Config
from seekr.embed import FakeEmbedder
from seekr.indexer import Indexer
from seekr.mcp_server import McpServer
from seekr.models import Hit, HitExplain
from seekr.store import Store

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SQLITE_VEC = importlib.util.find_spec("sqlite_vec") is not None
_HAS_TYPER = importlib.util.find_spec("typer") is not None


def _has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return "ENABLE_FTS5" in options


_NEEDS_REAL_SUBPROCESS = _HAS_NUMPY and _HAS_SQLITE_VEC and _HAS_TYPER and _has_fts5()


class McpProtocolTests(unittest.TestCase):
    def test_stdio_protocol_tools_and_error_isolation(self) -> None:
        search_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        similar_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def fake_search(*args: Any, **kwargs: Any) -> list[Hit]:
            search_calls.append((args, kwargs))
            if args[0] == "boom":
                raise RuntimeError("index temporarily unavailable")
            return [
                Hit(
                    chunk_id=3,
                    path="/repo/notes.md",
                    locator="§Intro",
                    loc_start=1.0,
                    snippet="hybrid semantic match",
                    score=0.0328,
                    kind="markdown",
                    explain=HitExplain(keyword_rank=1, keyword_bm25=-3.2, rrf_score=0.0328),
                )
            ]

        def fake_similar(*args: Any, **kwargs: Any) -> list[Hit]:
            similar_calls.append((args, kwargs))
            return [
                Hit(
                    chunk_id=7,
                    path="/repo/related.py",
                    locator="L1-10",
                    loc_start=1.0,
                    snippet="def helper(): ...",
                    score=0.88,
                    kind="code",
                )
            ]

        messages = [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            ),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "search",
                        "arguments": {
                            "query": "semantic",
                            "k": 5,
                            "mode": "hybrid",
                            "per_file": 1,
                            "path": "/repo",
                            "kind": "markdown",
                            "since": "7d",
                            "explain": True,
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {
                        "name": "similar",
                        "arguments": {"path": "/repo/notes.md", "k": 3, "kind": "code"},
                    },
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {"name": "search", "arguments": {"query": "boom"}},
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {"name": "bogus", "arguments": {}},
                }
            ),
            "{not json",
            "[1, 2, 3]",
            json.dumps({"jsonrpc": "2.0", "id": 8, "method": "unknown/method"}),
        ]

        stdin = io.StringIO("\n".join(messages) + "\n")
        stdout = io.StringIO()
        server = McpServer(search_fn=fake_search, similar_fn=fake_similar, version="0.1.0")
        server.serve_stdio(stdin=stdin, stdout=stdout)

        responses = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        # 11 input lines minus 1 notification = 10 JSON-RPC responses.
        self.assertEqual(len(responses), 10)

        init = responses[0]
        self.assertEqual(init["id"], 1)
        self.assertEqual(init["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(init["result"]["serverInfo"], {"name": "seekr", "version": "0.1.0"})

        self.assertEqual(responses[1], {"jsonrpc": "2.0", "id": 2, "result": {}})

        tools_res = responses[2]["result"]["tools"]
        self.assertEqual([t["name"] for t in tools_res], ["search", "similar"])
        self.assertEqual(tools_res[0]["inputSchema"]["required"], ["query"])

        search_res = responses[3]["result"]
        self.assertFalse(search_res["isError"])
        search_hits = json.loads(search_res["content"][0]["text"])
        self.assertEqual(search_hits[0]["path"], "/repo/notes.md")
        self.assertEqual(search_hits[0]["explain"]["keyword_rank"], 1)
        self.assertEqual(
            search_calls[0],
            (
                ("semantic",),
                {
                    "k": 5,
                    "mode": "hybrid",
                    "per_file": 1,
                    "rerank": False,
                    "path_prefix": "/repo",
                    "explain": True,
                    "kind": "markdown",
                    "since": "7d",
                },
            ),
        )

        sim_res = responses[4]["result"]
        self.assertFalse(sim_res["isError"])
        sim_hits = json.loads(sim_res["content"][0]["text"])
        self.assertEqual(sim_hits[0]["path"], "/repo/related.py")
        self.assertEqual(
            similar_calls[0],
            (
                (Path("/repo/notes.md"),),
                {"k": 3, "per_file": 1, "path_prefix": None, "kind": "code", "since": None},
            ),
        )

        boom_res = responses[5]["result"]
        self.assertTrue(boom_res["isError"])
        self.assertIn("temporarily unavailable", boom_res["content"][0]["text"])

        unknown_tool_res = responses[6]["result"]
        self.assertTrue(unknown_tool_res["isError"])
        self.assertIn("unknown tool", unknown_tool_res["content"][0]["text"])

        self.assertEqual(responses[7]["error"]["code"], -32700)
        self.assertEqual(responses[8]["error"]["code"], -32600)
        self.assertEqual(responses[9]["error"]["code"], -32601)


@unittest.skipUnless(
    _NEEDS_REAL_SUBPROCESS,
    "numpy, sqlite-vec, typer and FTS5 are required",
)
class SubprocessMcpRoundtripTests(unittest.TestCase):
    def test_seekr_mcp_subprocess_stdio_search_roundtrip(self) -> None:
        src = Path(__file__).resolve().parents[1] / "src"
        with short_tempdir("seekr-mcp-") as tmp:
            base = Path(tmp)
            root = base / "repo"
            root.mkdir()
            note_md = root / "guide.md"
            note_md.write_text("# Search\n\nzirconium retrieval guide.\n", encoding="utf-8")
            code_py = root / "solver.py"
            code_py.write_text("def solve():\n    return 'zirconium'\n", encoding="utf-8")

            db_path = base / "index.db"
            cfg = Config(roots=(root,), db_path=db_path, embed_model="fake")
            store = Store(db_path, "fake", 8, vector_quant="int8")
            try:
                Indexer(store, FakeEmbedder(dim=8), cfg).sync([root])
            finally:
                store.close()

            cfg_dir = base / "config" / "seekr"
            cfg_dir.mkdir(parents=True)
            (cfg_dir / "config.toml").write_text(
                "\n".join(
                    [
                        f'db_path = "{db_path.as_posix()}"',
                        f'roots = ["{root.as_posix()}"]',
                        'embed_model = "fake"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["XDG_CONFIG_HOME"] = str(base / "config")
            env["XDG_DATA_HOME"] = str(base / "data")
            env["XDG_RUNTIME_DIR"] = str(base / "runtime")
            # Windows and macOS do not read XDG_CONFIG_HOME.
            env["SEEKR_CONFIG"] = str(cfg_dir / "config.toml")
            env["PYTHONPATH"] = os.pathsep.join(
                [str(src), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)

            payload = "\n".join(
                [
                    json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
                    json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                    json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 3,
                            "method": "tools/call",
                            "params": {
                                "name": "search",
                                "arguments": {
                                    "query": "zirconium",
                                    "mode": "keyword",
                                    "kind": "markdown",
                                    "explain": True,
                                },
                            },
                        }
                    ),
                ]
            ) + "\n"

            proc = subprocess.run(
                [sys.executable, "-m", "seekr.cli", "mcp"],
                input=payload,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=15.0,
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr.strip(), "")
        responses = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        self.assertEqual([r["id"] for r in responses], [1, 2, 3])
        tool_result = responses[2]["result"]
        self.assertFalse(tool_result["isError"], tool_result)
        hits = json.loads(tool_result["content"][0]["text"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["path"], str(note_md.resolve()))
        self.assertEqual(hits[0]["explain"]["keyword_rank"], 1)


class McpModuleWeekTests(unittest.TestCase):
    def test_tools_list_has_module_and_week(self) -> None:
        server = McpServer()
        response = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert response is not None
        tools = {tool["name"]: tool for tool in response["result"]["tools"]}
        for tool_name in ("search", "similar"):
            props = tools[tool_name]["inputSchema"]["properties"]
            self.assertIn("module", props)
            self.assertEqual(props["module"]["type"], "string")
            self.assertIn("week", props)
            self.assertEqual(props["week"]["type"], "string")

    def test_search_tool_passes_module_week(self) -> None:
        search_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        similar_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def fake_search(*args: Any, **kwargs: Any) -> list[Hit]:
            search_calls.append((args, kwargs))
            return [
                Hit(
                    chunk_id=1,
                    path="/course/ml/week4/trees.md",
                    locator="§Intro",
                    loc_start=1.0,
                    snippet="trees",
                    score=0.9,
                    kind="markdown",
                    module="ml",
                    week=4,
                )
            ]

        def fake_similar(*args: Any, **kwargs: Any) -> list[Hit]:
            similar_calls.append((args, kwargs))
            return []

        server = McpServer(search_fn=fake_search, similar_fn=fake_similar)
        search_resp = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "trees", "module": "ml", "week": "3-5"},
                },
            }
        )
        assert search_resp is not None
        self.assertFalse(search_resp["result"]["isError"])
        self.assertEqual(search_calls[0][1]["module"], "ml")
        self.assertEqual(search_calls[0][1]["week"], "3-5")
        payload = json.loads(search_resp["result"]["content"][0]["text"])
        self.assertEqual(payload[0]["module"], "ml")
        self.assertEqual(payload[0]["week"], 4)

        sim_resp = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "similar",
                    "arguments": {"path": "/course/a.md", "module": "ml", "week": "4"},
                },
            }
        )
        assert sim_resp is not None
        self.assertFalse(sim_resp["result"]["isError"])
        self.assertEqual(similar_calls[0][1]["module"], "ml")
        self.assertEqual(similar_calls[0][1]["week"], "4")

    def test_mcp_result_includes_highlights(self) -> None:
        def fake_search(*args: Any, **kwargs: Any) -> list[Hit]:
            return [
                Hit(
                    chunk_id=1,
                    path="/course/ml/week4/trees.md",
                    locator="§Intro",
                    loc_start=1.0,
                    snippet="decision trees",
                    score=0.9,
                    kind="markdown",
                    highlights=((9, 14),),
                )
            ]

        server = McpServer(search_fn=fake_search, similar_fn=lambda *a, **kw: [])
        resp = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"query": "trees"}},
            }
        )
        assert resp is not None
        self.assertFalse(resp["result"]["isError"])
        payload = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(payload[0]["highlights"], [[9, 14]])

    def test_tools_list_kind_enums_include_office_and_notebooks(self) -> None:
        server = McpServer()
        response = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert response is not None
        tools = {tool["name"]: tool for tool in response["result"]["tools"]}
        expected = {"code", "markdown", "notebook", "pdf", "slides", "text", "transcript", "word"}
        for tool_name in ("search", "similar"):
            enum_values = set(tools[tool_name]["inputSchema"]["properties"]["kind"]["enum"])
            self.assertEqual(enum_values, expected)

    def test_search_tool_spread(self) -> None:
        list_server = McpServer()
        list_resp = list_server.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        assert list_resp is not None
        tools = {tool["name"]: tool for tool in list_resp["result"]["tools"]}
        search_props = tools["search"]["inputSchema"]["properties"]
        self.assertIn("spread", search_props)
        self.assertEqual(search_props["spread"]["type"], "boolean")
        self.assertEqual(search_props["spread"]["default"], False)

        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        main_hit = Hit(
            chunk_id=10,
            path="/course/machine-learning/week4/trees.md",
            locator="§Trees",
            loc_start=1.0,
            snippet="decision trees",
            score=0.9,
            kind="markdown",
            module="machine-learning",
            week=4,
        )
        spread_hits = [
            main_hit,
            Hit(
                chunk_id=20,
                path="/course/machine-learning/week2/intro.md",
                locator="§Intro",
                loc_start=1.0,
                snippet="trees intro",
                score=0.7,
                kind="markdown",
                module="machine-learning",
                week=2,
            ),
        ]

        def fake_search(*args: Any, **kwargs: Any) -> list[Hit]:
            calls.append((args, kwargs))
            if args[0] == "missing":
                return []
            if len(calls) == 1:
                return [main_hit]
            return spread_hits

        server = McpServer(search_fn=fake_search, similar_fn=lambda *a, **kw: [])
        resp = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "query": "trees",
                        "k": 5,
                        "mode": "keyword",
                        "per_file": 1,
                        "rerank": True,
                        "path": "/course",
                        "kind": "markdown",
                        "since": "7d",
                        "module": "ml",
                        "week": "1-6",
                        "explain": True,
                        "spread": True,
                    },
                },
            }
        )
        assert resp is not None
        self.assertFalse(resp["result"]["isError"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0], ("trees",))
        self.assertEqual(
            calls[1][1],
            {
                "k": 30,
                "mode": "keyword",
                "per_file": 2,
                "rerank": False,
                "path_prefix": "/course",
                "kind": "markdown",
                "since": "7d",
                "module": "ml",
                "week": "1-6",
            },
        )
        payload = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(set(payload), {"hits", "spread"})
        self.assertEqual([item["chunk_id"] for item in payload["hits"]], [10])
        self.assertEqual(
            payload["spread"],
            [
                {"module": "machine-learning", "week": 2, "first": True, "chunk_ids": [20]},
                {"module": "machine-learning", "week": 4, "first": False, "chunk_ids": [10]},
            ],
        )

        calls.clear()
        empty_resp = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "missing", "spread": True},
                },
            }
        )
        assert empty_resp is not None
        self.assertFalse(empty_resp["result"]["isError"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(empty_resp["result"]["content"][0]["text"]),
            {"hits": [], "spread": []},
        )

        bad_resp = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "trees", "spread": "yes"},
                },
            }
        )
        assert bad_resp is not None
        self.assertTrue(bad_resp["result"]["isError"])
