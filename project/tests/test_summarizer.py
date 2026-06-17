"""Unit tests for RollingSummarizer.

All tests mock the Anthropic client so no API key is required.
"""

from unittest.mock import MagicMock, patch

import pytest

from tokenai.compressor import RollingSummarizer
from tokenai.counter import TokenCounter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summarizer(budget: int = 4000, pin: int = 3) -> RollingSummarizer:
    s = RollingSummarizer(token_budget=budget, pin_last_pairs=pin)
    # Replace the real Anthropic client with a mock.
    s._client = MagicMock()
    _stub_haiku(s, "Mocked summary.")
    return s


def _stub_haiku(summarizer: RollingSummarizer, text: str) -> None:
    """Make _client.messages.create return *text* as the summary."""
    mock_content = MagicMock()
    mock_content.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    summarizer._client.messages.create.return_value = mock_response


def _build_conversation(n_turns: int, system: bool = True) -> list[dict]:
    """Build a conversation with *n_turns* user/assistant pairs."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": "You are a helpful assistant."})
    for i in range(1, n_turns + 1):
        msgs.append({"role": "user", "content": f"User message {i}. " + "word " * 20})
        msgs.append({"role": "assistant", "content": f"Assistant reply {i}. " + "word " * 20})
    return msgs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoOpWhenUnderBudget:
    def test_short_conversation_unchanged(self):
        s = _make_summarizer(budget=4000)
        msgs = _build_conversation(2)
        result = s.compress(msgs)
        assert result == msgs
        s._client.messages.create.assert_not_called()


class TestPriorityPinning:
    def test_system_prompt_always_first(self):
        s = _make_summarizer(budget=10)  # tiny budget forces compression
        msgs = _build_conversation(10)
        result = s.compress(msgs)
        assert result[0]["role"] == "system"

    def test_last_three_pairs_preserved(self):
        s = _make_summarizer(budget=10, pin=3)
        msgs = _build_conversation(10)
        result = s.compress(msgs)
        # Last 6 chat messages (3 pairs) must appear at the tail.
        chat_only = [m for m in msgs if m["role"] != "system"]
        expected_tail = chat_only[-6:]
        assert result[-6:] == expected_tail

    def test_no_system_prompt(self):
        s = _make_summarizer(budget=10, pin=3)
        msgs = _build_conversation(10, system=False)
        result = s.compress(msgs)
        # First message should be the summary (not a system message).
        assert result[0]["role"] == "assistant"
        assert "[Conversation summary]" in result[0]["content"]


class TestSummaryInjection:
    def test_summary_message_inserted(self):
        s = _make_summarizer(budget=10)
        _stub_haiku(s, "Key facts discussed.")
        msgs = _build_conversation(10)
        result = s.compress(msgs)
        summary_msgs = [m for m in result if "[Conversation summary]" in m.get("content", "")]
        assert len(summary_msgs) == 1

    def test_summary_contains_haiku_text(self):
        s = _make_summarizer(budget=10)
        _stub_haiku(s, "Important decision made.")
        msgs = _build_conversation(10)
        result = s.compress(msgs)
        summary = next(m for m in result if "[Conversation summary]" in m["content"])
        assert "Important decision made." in summary["content"]

    def test_haiku_called_once(self):
        s = _make_summarizer(budget=10)
        msgs = _build_conversation(10)
        s.compress(msgs)
        s._client.messages.create.assert_called_once()


class TestTooFewMessagesToSummarize:
    def test_not_enough_history_returns_unchanged(self):
        # 3 pairs total, pin=3 â†’ nothing to summarize.
        s = _make_summarizer(budget=10, pin=3)
        msgs = _build_conversation(3)
        result = s.compress(msgs)
        assert result == msgs
        s._client.messages.create.assert_not_called()


class TestTokenCount:
    def test_compressed_under_budget(self):
        """After compression the result must fit within the token budget."""
        budget = 4000
        s = _make_summarizer(budget=budget)
        # Return a very short summary so the compressed list is small.
        _stub_haiku(s, "Short summary.")
        msgs = _build_conversation(20)
        result = s.compress(msgs)
        counter = TokenCounter("claude-3-haiku")
        assert counter.count_messages(result) < budget

    def test_token_count_method(self):
        s = _make_summarizer()
        msgs = _build_conversation(5)
        assert s.token_count(msgs) == TokenCounter("claude-3-haiku").count_messages(msgs)
