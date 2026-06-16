"""Unit tests for TokenCounter.count_messages with varying message arrays."""

import pytest

from ctxmgr.counter import TokenCounter

counter = TokenCounter("gpt-4o")


def test_short_messages():
    """Single-exchange message array."""
    msgs = [
        {"role": "user", "content": "Hi"},
    ]
    result = counter.count_messages(msgs)
    assert isinstance(result, int)
    # 1 message * 4 overhead + role tokens + content tokens + 2 priming
    assert result > 2  # more than just priming
    assert result == 4 + counter.count("user") + counter.count("Hi") + 2


def test_medium_messages():
    """Multi-turn conversation."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    result = counter.count_messages(msgs)
    expected = 2  # priming
    for msg in msgs:
        expected += 4  # per-message overhead
        expected += counter.count(msg["role"])
        expected += counter.count(msg["content"])
    assert result == expected


def test_long_messages():
    """Conversation with a large content block."""
    long_content = "word " * 500  # ~500 tokens
    msgs = [
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": long_content},
        {"role": "assistant", "content": "Looks good to me."},
        {"role": "user", "content": "Are you sure?"},
        {"role": "assistant", "content": "Yes, the code follows best practices."},
    ]
    result = counter.count_messages(msgs)
    # Should be dominated by the long content block
    assert result > counter.count(long_content)
    # Sanity: overhead is 5 messages * 4 + 2 priming = 22
    assert result >= counter.count(long_content) + 22


def test_empty_messages():
    """Empty message array returns only priming tokens."""
    result = counter.count_messages([])
    assert result == 2  # only priming tokens


def test_unicode_messages():
    """Messages with CJK, emoji, and mixed-script unicode."""
    msgs = [
        {"role": "user", "content": "こんにちは世界 🌍"},
        {"role": "assistant", "content": "Привет! 你好！مرحبا 🎉🚀"},
    ]
    result = counter.count_messages(msgs)
    assert isinstance(result, int)
    assert result > 2
    # Verify round-trip: encoding the content should not raise
    for msg in msgs:
        tokens = counter.encode(msg["content"])
        assert len(tokens) > 0
        assert counter.decode(tokens) == msg["content"]
