"""MCP server exposing tokenai compression, token counting, and semantic cache as tools."""
from __future__ import annotations

import time

from mcp.server.fastmcp import FastMCP

from tokenai import compress, TokenCounter
from tokenai.cache import (
    cache_get as _cache_get,
    cache_store as _cache_store,
    cache_feedback as _cache_feedback,
)
from tokenai.cache.threshold import get_stats as _get_stats
from tokenai.api.config import settings  # shared config — version, key validation

mcp = FastMCP("tokenai")


# ---------------------------------------------------------------------------
# Existing tools (unchanged)
# ---------------------------------------------------------------------------

@mcp.tool()
def compress_messages(
    messages: list[dict],
    max_tokens: int,
    model: str = "claude-sonnet-4-6",
    aggressiveness: str = "medium",
) -> dict:
    """Compress a conversation history to fit within a token budget.

    Args:
        messages: Chat history in Anthropic/OpenAI message format
                  (list of {"role": ..., "content": ...} dicts).
        max_tokens: Target token budget for the compressed output.
        model: Claude model the messages will be sent to — used to price savings.
               One of: claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-8.
        aggressiveness: How aggressively to summarize.
                        "light" pins last 5 pairs, "medium" pins 3, "aggressive" pins 1.

    Returns:
        Dict with compressed messages and full savings metadata.
    """
    try:
        result = compress(messages, max_tokens, model=model, aggressiveness=aggressiveness)
        return {
            "messages": result.messages,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "saved_tokens": result.saved_tokens,
            "ratio": round(result.ratio, 4),
            "estimated_savings_usd": round(result.estimated_savings_usd, 6),
            "aggressiveness": result.aggressiveness,
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def count_tokens(
    model: str = "claude-sonnet-4-6",
    text: str = "",
    messages: list[dict] | None = None,
) -> dict:
    """Count tokens in a text string or a full message list for a given model.

    Pass either `text` (a plain string) or `messages` (a chat history list).
    If both are provided, `messages` takes priority.

    Args:
        model: The target model (e.g. gpt-4o, claude-sonnet-4-6).
        text: A plain string to count tokens for.
        messages: A list of {"role": ..., "content": ...} dicts to count together.

    Returns:
        Dict with token_count, model, and the mode used ("text" or "messages").
    """
    try:
        counter = TokenCounter(model)
        if messages is not None:
            count = counter.count_messages(messages)
            return {"token_count": count, "model": model, "mode": "messages"}
        count = counter.count(text)
        return {"token_count": count, "model": model, "mode": "text"}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_savings_report(
    messages: list[dict],
    max_tokens: int,
    model: str = "claude-sonnet-4-6",
    aggressiveness: str = "medium",
) -> dict:
    """Return token savings metadata without including the compressed messages.

    Useful for previewing how much a compression run would save before
    committing to it.

    Args:
        messages: Chat history in Anthropic/OpenAI message format.
        max_tokens: Target token budget.
        model: Claude model used for dollar savings estimation.
        aggressiveness: "light", "medium", or "aggressive".

    Returns:
        Dict with original_tokens, compressed_tokens, saved_tokens, ratio,
        estimated_savings_usd, aggressiveness, and model.
    """
    try:
        result = compress(messages, max_tokens, model=model, aggressiveness=aggressiveness)
        return {
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "saved_tokens": result.saved_tokens,
            "ratio": round(result.ratio, 4),
            "estimated_savings_usd": round(result.estimated_savings_usd, 6),
            "aggressiveness": result.aggressiveness,
            "model": model,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Cache tools (new)
# ---------------------------------------------------------------------------

@mcp.tool()
def cache_get(
    query: str,
    customer_id: str | None = None,
) -> dict:
    """Check if a semantically similar query has been answered before.

    Returns the cached response instantly on a hit — skipping the LLM
    entirely and saving the full token cost. Use this before every LLM
    call to avoid redundant requests.

    Args:
        query: The incoming user question or message text.
        customer_id: Identifier for the end-customer whose cache to search.
                     Required — each customer has an independent cache and
                     adaptive similarity threshold.

    Returns:
        Dict with hit (bool), response (str or null), score (float or null),
        and latency_ms (float).
    """
    try:
        if not customer_id:
            raise ValueError("customer_id is required")
        t0 = time.perf_counter()
        result = _cache_get(query, customer_id)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        if result:
            return {**result, "latency_ms": latency_ms}
        return {"hit": False, "response": None, "score": None, "latency_ms": latency_ms}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def cache_store(
    query: str,
    response: str,
    customer_id: str | None = None,
) -> dict:
    """Store a query and its LLM response in the semantic cache.

    Call this after every cache miss once you have the LLM's response,
    so future similar queries are answered from cache without an LLM call.

    Args:
        query: The user question that was answered.
        response: The LLM response to cache.
        customer_id: Identifier for the end-customer. Required.

    Returns:
        Dict with stored (bool) and customer_id (str).
    """
    try:
        if not customer_id:
            raise ValueError("customer_id is required")
        _cache_store(query, response, customer_id)
        return {"stored": True, "customer_id": customer_id}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def cache_feedback(
    query: str,
    customer_id: str | None = None,
    was_correct: bool = True,
) -> dict:
    """Tell the cache whether a cached answer was correct.

    This trains the per-customer adaptive similarity threshold:
    correct answers loosen it slightly (allow more cache hits),
    wrong answers tighten it aggressively (require higher similarity).
    Call this whenever the user confirms or rejects a cached response.

    Args:
        query: The query whose cached response is being rated.
        customer_id: Identifier for the end-customer. Required.
        was_correct: True if the cached response was useful, False if not.

    Returns:
        Dict with updated (bool), new_threshold (float), and customer_id.
    """
    try:
        if not customer_id:
            raise ValueError("customer_id is required")
        new_threshold = _cache_feedback(query, customer_id, was_correct)
        return {
            "updated": True,
            "new_threshold": new_threshold,
            "customer_id": customer_id,
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def cache_stats(
    customer_id: str | None = None,
) -> dict:
    """Get the adaptive threshold and hit-rate statistics for a customer.

    Useful for understanding how well the cache is performing and whether
    the threshold has converged to a stable value after enough feedback.

    Args:
        customer_id: Identifier for the end-customer. Required.

    Returns:
        Dict with threshold (float), total_hits (int), correct_hits (int),
        accuracy (float), and customer_id (str).
    """
    try:
        if not customer_id:
            raise ValueError("customer_id is required")
        stats = _get_stats(customer_id)
        return {**stats, "customer_id": customer_id}
    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    mcp.run()
