"""Unit tests for MCP server tools — calls Python functions directly."""
from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path


# ---------------------------------------------------------------------------
# Load mcp/server.py without conflicting with the installed `mcp` package.
# We give it a unique module name so it doesn't shadow mcp.server.fastmcp.
# ---------------------------------------------------------------------------

_SERVER_PATH = Path(__file__).parent.parent / "mcp" / "server.py"

if "tokenai_mcp_server" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("tokenai_mcp_server", _SERVER_PATH)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["tokenai_mcp_server"] = _module
    _spec.loader.exec_module(_module)
else:
    _module = sys.modules["tokenai_mcp_server"]

# Direct references — FastMCP's @mcp.tool() returns the original function unchanged.
cache_get_tool = _module.cache_get
cache_store_tool = _module.cache_store
cache_feedback_tool = _module.cache_feedback
cache_stats_tool = _module.cache_stats
compress_messages_tool = _module.compress_messages


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# 1. cache_get miss
# ---------------------------------------------------------------------------

def test_mcp_cache_get_miss():
    """cache_get returns hit=False for a brand-new unseen query."""
    result = cache_get_tool(query=f"unseen query {_uid()}", customer_id=_uid())
    assert result.get("hit") is False
    assert result.get("response") is None


# ---------------------------------------------------------------------------
# 2. cache_store then cache_get → hit
# ---------------------------------------------------------------------------

def test_mcp_cache_store_and_get():
    """Storing a query then getting it with identical text yields hit=True."""
    cid = _uid()
    query = "What is Python?"
    cache_store_tool(query=query, response="A high-level programming language.", customer_id=cid)
    result = cache_get_tool(query=query, customer_id=cid)
    assert result.get("hit") is True
    assert "response" in result


# ---------------------------------------------------------------------------
# 3. cache_feedback returns new threshold
# ---------------------------------------------------------------------------

def test_mcp_cache_feedback():
    """cache_feedback returns a float new_threshold after recording correctness."""
    cid = _uid()
    cache_store_tool(query="Define entropy.", response="Entropy measures disorder.", customer_id=cid)
    result = cache_feedback_tool(query="Define entropy.", customer_id=cid, was_correct=True)
    assert "new_threshold" in result
    assert isinstance(result["new_threshold"], float)


# ---------------------------------------------------------------------------
# 4. cache_stats returns valid threshold
# ---------------------------------------------------------------------------

def test_mcp_cache_stats():
    """cache_stats returns a threshold within the valid [0.70, 0.98] range."""
    cid = _uid()
    result = cache_stats_tool(customer_id=cid)
    assert "threshold" in result
    assert 0.70 <= result["threshold"] <= 0.98


# ---------------------------------------------------------------------------
# 5. compress_messages tool
# ---------------------------------------------------------------------------

def test_mcp_compress_tool():
    """compress_messages returns expected fields with saved_tokens >= 0."""
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am doing well, thank you!"},
    ]
    result = compress_messages_tool(messages=messages, max_tokens=10000)
    assert "saved_tokens" in result
    assert result["saved_tokens"] >= 0
    assert "messages" in result


# ---------------------------------------------------------------------------
# 6. Error handling — missing customer_id returns {"error": ...}
# ---------------------------------------------------------------------------

def test_mcp_error_handling():
    """Calling cache_get without customer_id returns an error dict, not an exception."""
    result = cache_get_tool(query="some query")  # customer_id defaults to None
    assert "error" in result
    assert isinstance(result["error"], str)
    assert "customer_id" in result["error"]
