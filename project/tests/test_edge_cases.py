"""Edge-case and OpenAI-format tests.

Covers:
  - Empty message list
  - Single-turn conversation (no assistant reply yet)
  - Single message that exceeds the token budget (uncompressible)
  - Content=None (missing content key)
  - OpenAI list-of-blocks format (text blocks, image blocks, tool calls)
  - Anthropic list-of-blocks format (tool_use, tool_result)
"""

from unittest.mock import patch

import pytest

from tokenai._utils import extract_text
from tokenai.compress import compress
from tokenai.counter import TokenCounter

_SUMMARIZE_PATH = "tokenai.compressor.RollingSummarizer._summarize"

counter = TokenCounter("claude-3-haiku")


# ---------------------------------------------------------------------------
# extract_text helper
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_plain_string_passthrough(self):
        assert extract_text("hello") == "hello"

    def test_none_returns_empty(self):
        assert extract_text(None) == ""

    def test_empty_list_returns_empty(self):
        assert extract_text([]) == ""

    def test_text_block(self):
        assert extract_text([{"type": "text", "text": "hi there"}]) == "hi there"

    def test_multiple_text_blocks(self):
        blocks = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]
        assert extract_text(blocks) == "Hello world"

    def test_image_block_becomes_placeholder(self):
        assert extract_text([{"type": "image", "source": {"type": "base64"}}]) == "[image]"

    def test_image_url_block_becomes_placeholder(self):
        assert extract_text([{"type": "image_url", "image_url": {"url": "https://..."}}]) == "[image]"

    def test_tool_use_block(self):
        block = [{"type": "tool_use", "id": "abc", "name": "calculator", "input": {}}]
        assert extract_text(block) == "[tool:calculator]"

    def test_tool_result_with_string_content(self):
        block = [{"type": "tool_result", "tool_use_id": "abc", "content": "42"}]
        assert extract_text(block) == "42"

    def test_tool_result_with_nested_blocks(self):
        block = [{"type": "tool_result", "content": [{"type": "text", "text": "result"}]}]
        assert extract_text(block) == "result"

    def test_mixed_blocks(self):
        blocks = [
            {"type": "text", "text": "Here is an image:"},
            {"type": "image", "source": {}},
            {"type": "text", "text": "and a tool call"},
            {"type": "tool_use", "name": "search", "input": {}},
        ]
        result = extract_text(blocks)
        assert "Here is an image:" in result
        assert "[image]" in result
        assert "and a tool call" in result
        assert "[tool:search]" in result


# ---------------------------------------------------------------------------
# TokenCounter â€” content format robustness
# ---------------------------------------------------------------------------

class TestCountMessagesFormats:
    def test_none_content_does_not_crash(self):
        msgs = [{"role": "user", "content": None}]
        total = counter.count_messages(msgs)
        assert isinstance(total, int)
        assert total >= 2

    def test_missing_content_key(self):
        msgs = [{"role": "user"}]
        total = counter.count_messages(msgs)
        assert isinstance(total, int)

    def test_openai_list_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]
        plain = [{"role": "user", "content": "What is 2+2?"}]
        # List format and string format should yield the same count.
        assert counter.count_messages(msgs) == counter.count_messages(plain)

    def test_openai_image_message(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]}
        ]
        total = counter.count_messages(msgs)
        assert total > counter.count("user") + 2  # has actual content tokens


# ---------------------------------------------------------------------------
# compress() â€” edge cases
# ---------------------------------------------------------------------------

class TestEmptyHistory:
    def test_empty_list_returns_empty(self):
        with patch(_SUMMARIZE_PATH, return_value="x") as mock_fn:
            result = compress([], max_tokens=4000)
            mock_fn.assert_not_called()
        assert result.messages == []
        assert result.saved_tokens == 0
        assert result.ratio == 1.0

    def test_system_only_returns_unchanged(self):
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        with patch(_SUMMARIZE_PATH, return_value="x") as mock_fn:
            result = compress(msgs, max_tokens=10)
            mock_fn.assert_not_called()
        assert result.messages == msgs
        assert result.saved_tokens == 0


class TestSingleTurn:
    def test_single_user_message_under_budget(self):
        msgs = [{"role": "user", "content": "Hello, can you help me?"}]
        with patch(_SUMMARIZE_PATH, return_value="x") as mock_fn:
            result = compress(msgs, max_tokens=4000)
            mock_fn.assert_not_called()
        assert result.messages == msgs

    def test_single_user_message_over_budget_returns_unchanged(self):
        # Can't summarize a single message â€” nothing to pin vs. compress.
        msgs = [{"role": "user", "content": "word " * 2000}]
        with patch(_SUMMARIZE_PATH, return_value="x") as mock_fn:
            result = compress(msgs, max_tokens=100)
            mock_fn.assert_not_called()
        assert result.messages == msgs
        assert result.saved_tokens == 0


class TestVeryLongSingleMessage:
    def test_over_budget_single_message_is_uncompressible(self):
        long_content = "The quick brown fox jumps over the lazy dog. " * 500
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": long_content},
        ]
        original_tokens = counter.count_messages(msgs)
        assert original_tokens > 4000  # verify fixture is actually over budget

        with patch(_SUMMARIZE_PATH, return_value="x") as mock_fn:
            result = compress(msgs, max_tokens=4000)
            mock_fn.assert_not_called()
        assert result.messages == msgs
        assert result.saved_tokens == 0
        assert result.compressed_tokens == result.original_tokens


# ---------------------------------------------------------------------------
# compress() â€” OpenAI list-format messages
# ---------------------------------------------------------------------------

class TestOpenAIFormat:
    def test_list_content_compresses(self):
        """compress() works end-to-end with OpenAI-style list content."""
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        for i in range(1, 8):
            msgs.append({"role": "user", "content": [{"type": "text", "text": f"Question {i}? " + "word " * 30}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"Answer {i}. " + "word " * 30}]})

        with patch(_SUMMARIZE_PATH, return_value="Summary of the conversation."):
            result = compress(msgs, max_tokens=200)

        assert result.original_tokens > 0
        assert isinstance(result.estimated_savings_usd, float)

    def test_mixed_string_and_list_content(self):
        """Messages with a mix of string and list content are counted correctly."""
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Plain string message."},
            {"role": "assistant", "content": [{"type": "text", "text": "List content reply."}]},
        ]
        total = counter.count_messages(msgs)
        assert total > 2
