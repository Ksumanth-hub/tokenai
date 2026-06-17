"""Public compress() function — wraps RollingSummarizer with savings metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from tokenai._utils import extract_text
from tokenai.compressor import RollingSummarizer
from tokenai.counter import TokenCounter

_AGGRESSIVENESS_CONFIG: dict[str, int] = {
    "light": 5,
    "medium": 3,
    "aggressive": 1,
}

# USD per token ($/1M tokens divided down to per-token cost)
_MODEL_INPUT_COST_PER_TOKEN: dict[str, float] = {
    "claude-haiku-4-5": 1.00 / 1_000_000,
    "claude-haiku-4-5-20251001": 1.00 / 1_000_000,
    "claude-sonnet-4-6": 3.00 / 1_000_000,
    "claude-opus-4-8": 5.00 / 1_000_000,
}

_DEFAULT_COST_PER_TOKEN = 3.00 / 1_000_000  # fall back to Sonnet pricing


@dataclass
class CompressionResult:
    """Compression output with token counts and estimated dollar savings.

    ratio: compressed_tokens / original_tokens (lower = more compression).
    cache_hit: True when the result was served from the semantic cache.
    """
    messages: list[dict]
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    ratio: float
    estimated_savings_usd: float
    aggressiveness: str
    cache_hit: bool = field(default=False)


def compress(
    messages: list[dict],
    max_tokens: int,
    model: str = "claude-sonnet-4-6",
    aggressiveness: str = "medium",
    use_cache: bool = False,
    customer_id: str | None = None,
) -> CompressionResult:
    """Compress *messages* to fit within *max_tokens* and return savings metadata.

    Args:
        messages: Conversation history in Anthropic/OpenAI message format.
        max_tokens: Target token budget for the compressed result.
        model: The Claude model the compressed messages will be sent to.
               Used only to calculate estimated dollar savings.
        aggressiveness: One of ``"light"``, ``"medium"``, or ``"aggressive"``.
                        Controls how many recent pairs are pinned (never summarized):
                        light=5 pairs, medium=3 pairs, aggressive=1 pair.
        use_cache: When True, check the semantic cache before compressing.
                   A cache hit skips compression entirely (zero tokens consumed).
        customer_id: Required when *use_cache* is True. Each customer maintains
                     an independent cache and adaptive similarity threshold.

    Returns:
        CompressionResult with compressed messages and savings metadata.
        ``cache_hit=True`` when the response was served from cache.

    Raises:
        ValueError: If *aggressiveness* is not a recognised level, or if
                    *use_cache* is True but *customer_id* is not provided.
    """
    counter = TokenCounter("claude-3-haiku")
    cost_per_token = _MODEL_INPUT_COST_PER_TOKEN.get(model, _DEFAULT_COST_PER_TOKEN)

    # --- Semantic cache check (before any compression work) ---
    if use_cache:
        if customer_id is None:
            raise ValueError("customer_id is required when use_cache=True")

        from tokenai.cache import cache_get, cache_store  # lazy: avoid loading torch at import time

        query = _extract_last_user_message(messages)

        if query:
            hit = cache_get(query, customer_id)
            if hit:
                original_tokens = counter.count_messages(messages)
                return CompressionResult(
                    messages=[{"role": "assistant", "content": hit["response"]}],
                    original_tokens=original_tokens,
                    compressed_tokens=0,
                    saved_tokens=original_tokens,
                    ratio=1.0,
                    estimated_savings_usd=original_tokens * cost_per_token,
                    aggressiveness="cache_hit",
                    cache_hit=True,
                )

    # --- Existing compression logic ---
    if aggressiveness not in _AGGRESSIVENESS_CONFIG:
        raise ValueError(
            f"aggressiveness must be one of {list(_AGGRESSIVENESS_CONFIG)}; "
            f"got {aggressiveness!r}"
        )

    pin_last_pairs = _AGGRESSIVENESS_CONFIG[aggressiveness]
    original_tokens = counter.count_messages(messages)

    summarizer = RollingSummarizer(token_budget=max_tokens, pin_last_pairs=pin_last_pairs)
    compressed_messages = summarizer.compress(messages)
    compressed_tokens = counter.count_messages(compressed_messages)

    saved_tokens = max(0, original_tokens - compressed_tokens)
    estimated_savings_usd = saved_tokens * cost_per_token
    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

    result = CompressionResult(
        messages=compressed_messages,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        saved_tokens=saved_tokens,
        ratio=ratio,
        estimated_savings_usd=estimated_savings_usd,
        aggressiveness=aggressiveness,
        cache_hit=False,
    )

    # --- Store in cache for future hits ---
    if use_cache and customer_id is not None:
        query = _extract_last_user_message(messages)
        if query:
            from tokenai.cache import cache_store  # already imported above, but safe to repeat
            llm_response = _extract_last_assistant_message(compressed_messages)
            if not llm_response:
                llm_response = json.dumps(compressed_messages)
            cache_store(query, llm_response, customer_id)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_last_user_message(messages: list[dict]) -> str:
    """Return the text of the last user message, or empty string."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return extract_text(msg.get("content", ""))
    return ""


def _extract_last_assistant_message(messages: list[dict]) -> str:
    """Return the text of the last assistant message, or empty string."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content if isinstance(content, str) else extract_text(content)
    return ""
