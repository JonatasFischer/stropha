"""Smoke test: launch the MCP server as a subprocess and exchange JSON-RPC.

This tests the protocol handshake + tool invocation end-to-end. Slow-ish
(spawns a Python process) but high-fidelity.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _send(proc: subprocess.Popen, payload: dict) -> None:
    assert proc.stdin is not None
    line = (json.dumps(payload) + "\n").encode("utf-8")
    proc.stdin.write(line)
    proc.stdin.flush()


def _recv(proc: subprocess.Popen, timeout_s: float = 10.0) -> dict:
    assert proc.stdout is not None
    import selectors
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    while True:
        events = sel.select(timeout_s)
        if not events:
            raise TimeoutError("MCP server did not respond in time")
        line = proc.stdout.readline()
        if not line:
            continue
        text = line.decode("utf-8").strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        # Skip notifications without id (e.g. progress).
        if "id" in obj or "result" in obj or "error" in obj:
            return obj


@pytest.mark.slow
def test_mcp_handshake_and_tool_call(tmp_path: Path) -> None:
    """Spawn the server, initialize, list tools, call search_code."""
    # Build a tiny indexed DB so the server has something to search.
    from stropha.config import Config
    from stropha.embeddings.local import LocalEmbedder
    from stropha.ingest.pipeline import IndexPipeline
    from stropha.storage import Storage

    # Build a fixture repo with one Python file.
    repo = tmp_path / "fixture"
    repo.mkdir()
    (repo / "calc.py").write_text(
        '"""Calculator module."""\n\n'
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n"
        "    \"\"\"Multiply two numbers.\"\"\"\n"
        "    return a * b\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
    )

    db_path = tmp_path / "idx.db"
    embedder = LocalEmbedder()
    with Storage(db_path, embedding_dim=embedder.dim) as storage:
        pipeline = IndexPipeline(repo=repo, storage=storage, embedder=embedder)
        pipeline.run(rebuild=True)

    # Launch MCP server pointed at this DB and repo.
    env = os.environ.copy()
    env["STROPHA_TARGET_REPO"] = str(repo)
    env["STROPHA_INDEX_PATH"] = str(db_path)
    env["STROPHA_LOG_LEVEL"] = "ERROR"
    env.pop("VOYAGE_API_KEY", None)  # force local fallback for determinism

    proc = subprocess.Popen(
        [sys.executable, "-m", "stropha.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(tmp_path),
    )

    try:
        # 1. initialize
        _send(proc, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        })
        init_resp = _recv(proc)
        assert init_resp.get("id") == 1
        assert "result" in init_resp
        assert "serverInfo" in init_resp["result"]

        # 2. notifications/initialized (no response expected)
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        # 3. tools/list
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        list_resp = _recv(proc)
        tool_names = {t["name"] for t in list_resp["result"]["tools"]}
        assert {"search_code", "get_symbol", "get_file_outline"} <= tool_names

        # 4. tools/call search_code
        _send(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "search_code",
                "arguments": {"query": "add two numbers", "top_k": 3},
            },
        })
        call_resp = _recv(proc, timeout_s=30.0)
        assert call_resp.get("id") == 3
        assert "result" in call_resp
        # FastMCP wraps tool output; expect at least one content item.
        content = call_resp["result"].get("content", [])
        assert content, f"Empty content: {call_resp}"

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
