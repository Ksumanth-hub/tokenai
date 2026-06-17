"""Token counter supporting OpenAI and Claude models via tiktoken."""
from __future__ import annotations

import tiktoken

from ctxmgr._utils import extract_text

# Map model families to tiktoken encodings.
# OpenAI models use their native encodings.
# Claude models use cl100k_base as a close approximation (~5% variance).
_MODEL_ENCODING_MAP: dict[str, str] = {
    # OpenAI
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-pro": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o4-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    # Claude (approximation via cl100k_base)
    "claude-opus-4": "cl100k_base",
    "claude-sonnet-4": "cl100k_base",
    "claude-3.5-sonnet": "cl100k_base",
    "claude-3.5-haiku": "cl100k_base",
    "claude-3-opus": "cl100k_base",
    "claude-3-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
}

_DEFAULT_ENCODING = "cl100k_base"


def _resolve_encoding(model: str) -> str:
    """Resolve a model name to a tiktoken encoding name.

    Tries exact match first, then prefix matching for versioned model names
    like 'gpt-4o-2024-08-06'.
    """
    if model in _MODEL_ENCODING_MAP:
        return _MODEL_ENCODING_MAP[model]
    for prefix, enc in _MODEL_ENCODING_MAP.items():
        if model.startswith(prefix):
            return enc
    return _DEFAULT_ENCODING


class TokenCounter:
    """Count tokens for OpenAI and Claude models using tiktoken.

    For Claude models the counts are approximate (~5% variance) since
    tiktoken doesn't ship Claude's exact vocabulary.

    Usage:
        counter = TokenCounter("gpt-4o")
        n = counter.count("Hello, world!")
        tokens = counter.encode("Hello, world!")
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self._encoding_name = _resolve_encoding(model)
        self._enc = tiktoken.get_encoding(self._encoding_name)

    @property
    def encoding_name(self) -> str:
        return self._encoding_name

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        return self._enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids back into text."""
        return self._enc.decode(tokens)

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self._enc.encode(text))

    def count_messages(self, messages: list[dict]) -> int:
        """Estimate token count for a chat-style message list.

        Accepts both plain-string content and list-of-blocks format (OpenAI
        tool calls, Anthropic multi-modal). Images and tool-use blocks are
        counted as short placeholder strings.
        """
        # https://platform.openai.com/docs/guides/chat/managing-tokens
        tokens_per_message = 4  # <|im_start|> role \n content <|im_end|>
        total = 0
        for msg in messages:
            total += tokens_per_message
            total += self.count(msg.get("role", ""))
            total += self.count(extract_text(msg.get("content")))
        total += 2  # priming tokens
        return total

    def __repr__(self) -> str:
        return f"TokenCounter(model={self.model!r}, encoding={self._encoding_name!r})"
