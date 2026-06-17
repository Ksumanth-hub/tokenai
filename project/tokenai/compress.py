"""Public compress() function — wraps RollingSummarizer with savings metadata."""

from dataclasses import dataclass

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
    """
    messages: list[dict]
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    ratio: float
    estimated_savings_usd: float
    aggressiveness: str


def compress(
    messages: list[dict],
    max_tokens: int,
    model: str = "claude-sonnet-4-6",
    aggressiveness: str = "medium",
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

    Returns:
        CompressionResult with compressed messages and savings metadata.

    Raises:
        ValueError: If *aggressiveness* is not a recognised level.
    """
    if aggressiveness not in _AGGRESSIVENESS_CONFIG:
        raise ValueError(
            f"aggressiveness must be one of {list(_AGGRESSIVENESS_CONFIG)}; "
            f"got {aggressiveness!r}"
        )

    pin_last_pairs = _AGGRESSIVENESS_CONFIG[aggressiveness]
    counter = TokenCounter("claude-3-haiku")
    original_tokens = counter.count_messages(messages)

    summarizer = RollingSummarizer(token_budget=max_tokens, pin_last_pairs=pin_last_pairs)
    compressed_messages = summarizer.compress(messages)
    compressed_tokens = counter.count_messages(compressed_messages)

    saved_tokens = max(0, original_tokens - compressed_tokens)
    cost_per_token = _MODEL_INPUT_COST_PER_TOKEN.get(model, _DEFAULT_COST_PER_TOKEN)
    estimated_savings_usd = saved_tokens * cost_per_token
    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

    return CompressionResult(
        messages=compressed_messages,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        saved_tokens=saved_tokens,
        ratio=ratio,
        estimated_savings_usd=estimated_savings_usd,
        aggressiveness=aggressiveness,
    )
