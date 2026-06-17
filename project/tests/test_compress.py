"""Unit tests for the public compress() function.

Tests three realistic conversation types:
  - support_chat: customer-service billing dialogue
  - coding_assistant: developer debugging session with code snippets
  - rag_qa: retrieval-augmented Q&A

All tests mock RollingSummarizer._summarize so no API key is required.
"""

from unittest.mock import patch

import pytest

from tokenai.compress import CompressionResult, compress

# ---------------------------------------------------------------------------
# Conversation fixtures
# ---------------------------------------------------------------------------

SUPPORT_CHAT: list[dict] = [
    {"role": "system", "content": "You are a helpful billing support agent for Acme Corp."},
    {"role": "user", "content": "Hi, I was charged twice for my subscription last month. Can you help?"},
    {"role": "assistant", "content": "I'm sorry to hear that. Could you provide your account email or order number?"},
    {"role": "user", "content": "Sure, my email is john.doe@example.com and the orders are #A1234 and #A1235."},
    {"role": "assistant", "content": "I can see both charges. #A1234 is your regular subscription at $29.99 and #A1235 is a duplicate. I will initiate a refund of $29.99 right away."},
    {"role": "user", "content": "How long will the refund take?"},
    {"role": "assistant", "content": "Refunds typically take 5-7 business days depending on your bank. You will receive a confirmation email within the hour."},
    {"role": "user", "content": "Can you make sure this does not happen again? I have had this issue twice now."},
    {"role": "assistant", "content": "Absolutely. I have flagged your account to prevent duplicate billing and added a note to escalate any future anomalies to our senior team before processing."},
    {"role": "user", "content": "Thank you. Also, can I upgrade to the Pro plan while I have you?"},
    {"role": "assistant", "content": "Of course! The Pro plan is $59.99/month with unlimited projects, priority support, and API access. I will apply a 10% loyalty discount, bringing it to $53.99/month. Shall I proceed?"},
    {"role": "user", "content": "Yes please. And can I get that discount applied to my current month too?"},
    {"role": "assistant", "content": "Done. I have upgraded you to Pro at $53.99/month and applied a prorated credit of $8.50 to your current billing cycle."},
    {"role": "user", "content": "Perfect. Can I get a receipt for the refund for my expense report?"},
    {"role": "assistant", "content": "I have emailed a detailed receipt to john.doe@example.com covering the $29.99 refund and your new Pro plan details. Is there anything else I can help you with?"},
]

CODING_ASSISTANT: list[dict] = [
    {"role": "system", "content": "You are an expert Python developer helping debug and improve code."},
    {"role": "user", "content": "My function throws a KeyError but I cannot figure out why.\n\n```python\ndef get_user_score(data, user_id):\n    return data['users'][user_id]['score']\n```"},
    {"role": "assistant", "content": "The KeyError occurs when 'users', user_id, or 'score' is missing. Use .get() for safe access:\n\n```python\ndef get_user_score(data, user_id):\n    return data.get('users', {}).get(user_id, {}).get('score')\n```"},
    {"role": "user", "content": "That works, but now it returns None sometimes. I need it to raise a meaningful error instead."},
    {"role": "assistant", "content": "Raise a custom exception with context:\n\n```python\ndef get_user_score(data, user_id):\n    users = data.get('users')\n    if users is None:\n        raise ValueError('data missing users key')\n    user = users.get(user_id)\n    if user is None:\n        raise KeyError(f'user {user_id!r} not found')\n    score = user.get('score')\n    if score is None:\n        raise KeyError(f'user {user_id!r} has no score')\n    return score\n```"},
    {"role": "user", "content": "Now I need to handle a list of user_ids and collect errors separately instead of stopping on the first failure."},
    {"role": "assistant", "content": "Use a results/errors pattern:\n\n```python\ndef get_scores_bulk(data, user_ids):\n    results, errors = {}, {}\n    for uid in user_ids:\n        try:\n            results[uid] = get_user_score(data, uid)\n        except (KeyError, ValueError) as e:\n            errors[uid] = str(e)\n    return results, errors\n```"},
    {"role": "user", "content": "Can I make this async? We are hitting a remote API for each user."},
    {"role": "assistant", "content": "Yes. Use asyncio.gather with a semaphore to cap concurrency:\n\n```python\nimport asyncio, aiohttp\n\nasync def fetch_user_score(session, user_id):\n    async with session.get(f'/api/users/{user_id}/score') as resp:\n        resp.raise_for_status()\n        return (await resp.json())['score']\n\nasync def get_scores_bulk(user_ids, max_concurrent=10):\n    sem = asyncio.Semaphore(max_concurrent)\n    results, errors = {}, {}\n    async with aiohttp.ClientSession() as session:\n        async def fetch(uid):\n            async with sem:\n                try:\n                    results[uid] = await fetch_user_score(session, uid)\n                except Exception as e:\n                    errors[uid] = str(e)\n        await asyncio.gather(*[fetch(uid) for uid in user_ids])\n    return results, errors\n```"},
    {"role": "user", "content": "How do I add retry logic for transient failures?"},
    {"role": "assistant", "content": "Use tenacity for exponential backoff:\n\n```python\nfrom tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type\n\n@retry(\n    stop=stop_after_attempt(3),\n    wait=wait_exponential(multiplier=1, min=1, max=10),\n    retry=retry_if_exception_type(aiohttp.ClientError),\n)\nasync def fetch_user_score(session, user_id):\n    async with session.get(f'/api/users/{user_id}/score') as resp:\n        resp.raise_for_status()\n        return (await resp.json())['score']\n```\nThis retries up to 3 times with 1-10s backoff on any aiohttp.ClientError."},
]

RAG_QA: list[dict] = [
    {"role": "system", "content": "You are a Q&A assistant. Answer only from the provided context. Say 'I don't know' if the context lacks the answer."},
    {"role": "user", "content": "Context: TokenAI is a Python library for managing LLM context windows. It provides token counting via tiktoken and rolling summarization via Claude Haiku.\n\nQuestion: What does TokenAI use for token counting?"},
    {"role": "assistant", "content": "TokenAI uses tiktoken for token counting."},
    {"role": "user", "content": "Context: Rolling summarization replaces the oldest N messages with a summary produced by Claude Haiku. The system prompt and last 3 user/assistant pairs are always preserved.\n\nQuestion: What messages are preserved during rolling summarization?"},
    {"role": "assistant", "content": "The system prompt and the last 3 user/assistant pairs (6 messages total) are always preserved."},
    {"role": "user", "content": "Context: Claude Haiku costs $1.00 per million input tokens. Claude Sonnet costs $3.00 per million. Claude Opus costs $5.00 per million.\n\nQuestion: How much does processing 500,000 tokens with Claude Sonnet cost?"},
    {"role": "assistant", "content": "Processing 500,000 tokens with Claude Sonnet costs $1.50 (500,000 / 1,000,000 * $3.00)."},
    {"role": "user", "content": "Context: compress() accepts aggressiveness levels: light (pin 5 pairs), medium (pin 3 pairs), aggressive (pin 1 pair). Higher aggressiveness compresses more.\n\nQuestion: Which level pins the fewest recent messages?"},
    {"role": "assistant", "content": "The 'aggressive' level pins only 1 recent pair, the fewest of the three levels."},
    {"role": "user", "content": "Context: CompressionResult contains: messages, original_tokens, compressed_tokens, saved_tokens, ratio, estimated_savings_usd, aggressiveness.\n\nQuestion: What fields does CompressionResult contain?"},
    {"role": "assistant", "content": "CompressionResult contains: messages, original_tokens, compressed_tokens, saved_tokens, ratio, estimated_savings_usd, and aggressiveness."},
    {"role": "user", "content": "Context: Token counting overhead: each message adds 4 tokens, plus 2 priming tokens for the whole conversation.\n\nQuestion: How many overhead tokens does a 5-message conversation add?"},
    {"role": "assistant", "content": "A 5-message conversation adds 22 overhead tokens: 5 * 4 = 20 per-message tokens plus 2 priming tokens."},
    {"role": "user", "content": "Question: What is the boiling point of water on Mars?"},
    {"role": "assistant", "content": "I don't know -- the provided context does not contain information about the boiling point of water on Mars."},
]

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SUMMARIZE_PATH = "tokenai.compressor.RollingSummarizer._summarize"
_MOCK_SUMMARY = "Mocked summary text."
_TIGHT_BUDGET = 300  # forces compression on all three fixtures


# ---------------------------------------------------------------------------
# CompressionResult structure
# ---------------------------------------------------------------------------

class TestCompressionResultFields:
    def test_returns_dataclass(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            result = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET)
        assert isinstance(result, CompressionResult)

    def test_saved_tokens_equals_difference(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            result = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET)
        assert result.saved_tokens == result.original_tokens - result.compressed_tokens

    def test_ratio_is_between_zero_and_one(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            result = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET)
        assert 0.0 < result.ratio <= 1.0

    def test_savings_usd_is_nonnegative(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            result = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET)
        assert result.estimated_savings_usd >= 0.0

    def test_no_compression_when_under_budget(self):
        short = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY) as mock_fn:
            result = compress(short, max_tokens=10_000)
            mock_fn.assert_not_called()
        assert result.messages == short
        assert result.saved_tokens == 0
        assert result.ratio == 1.0
        assert result.estimated_savings_usd == 0.0


# ---------------------------------------------------------------------------
# Aggressiveness config
# ---------------------------------------------------------------------------

class TestAggressiveness:
    def test_default_is_medium(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            result = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET)
        assert result.aggressiveness == "medium"

    def test_light_accepted(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            result = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, aggressiveness="light")
        assert result.aggressiveness == "light"

    def test_aggressive_accepted(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            result = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, aggressiveness="aggressive")
        assert result.aggressiveness == "aggressive"

    def test_invalid_level_raises_value_error(self):
        with pytest.raises(ValueError, match="aggressiveness"):
            compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, aggressiveness="extreme")

    def test_aggressive_pins_fewer_messages_than_light(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            light = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, aggressiveness="light")
            aggressive = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, aggressiveness="aggressive")
        assert len(aggressive.messages) <= len(light.messages)


# ---------------------------------------------------------------------------
# Dollar savings by model
# ---------------------------------------------------------------------------

class TestDollarSavings:
    def test_haiku_cheaper_than_sonnet(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            haiku = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, model="claude-haiku-4-5")
            sonnet = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, model="claude-sonnet-4-6")
        assert haiku.estimated_savings_usd < sonnet.estimated_savings_usd

    def test_sonnet_cheaper_than_opus(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            sonnet = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, model="claude-sonnet-4-6")
            opus = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, model="claude-opus-4-8")
        assert sonnet.estimated_savings_usd < opus.estimated_savings_usd

    def test_savings_formula_sonnet(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            result = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, model="claude-sonnet-4-6")
        expected = result.saved_tokens * (3.00 / 1_000_000)
        assert abs(result.estimated_savings_usd - expected) < 1e-12

    def test_savings_formula_opus(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            result = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, model="claude-opus-4-8")
        expected = result.saved_tokens * (5.00 / 1_000_000)
        assert abs(result.estimated_savings_usd - expected) < 1e-12

    def test_unknown_model_defaults_to_sonnet_pricing(self):
        with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
            unknown = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, model="claude-unknown-9")
            sonnet = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, model="claude-sonnet-4-6")
        assert abs(unknown.estimated_savings_usd - sonnet.estimated_savings_usd) < 1e-12


# ---------------------------------------------------------------------------
# Three conversation types â€” savings report
# ---------------------------------------------------------------------------

def _print_savings_report(label: str, result: CompressionResult) -> None:
    sep = "=" * 52
    reduction_pct = (1 - result.ratio) * 100
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Aggressiveness  : {result.aggressiveness}")
    print(f"  Original tokens : {result.original_tokens}")
    print(f"  Compressed      : {result.compressed_tokens}")
    print(f"  Saved           : {result.saved_tokens} ({reduction_pct:.1f}% reduction)")
    print(f"  Est. savings    : ${result.estimated_savings_usd:.6f} per call (Sonnet 4.6)")
    print(sep)


class TestConversationTypes:
    def test_support_chat_compresses(self, capsys):
        with patch(_SUMMARIZE_PATH, return_value="Customer billed twice; refund issued; plan upgraded to Pro at $53.99."):
            result = compress(SUPPORT_CHAT, max_tokens=_TIGHT_BUDGET, model="claude-sonnet-4-6")
        _print_savings_report("Support Chat", result)
        assert result.compressed_tokens < result.original_tokens
        assert result.saved_tokens > 0
        assert result.estimated_savings_usd > 0.0

    def test_coding_assistant_compresses(self, capsys):
        with patch(_SUMMARIZE_PATH, return_value="User debugged a KeyError; added bulk async fetch with tenacity retry."):
            result = compress(CODING_ASSISTANT, max_tokens=_TIGHT_BUDGET, model="claude-sonnet-4-6")
        _print_savings_report("Coding Assistant", result)
        assert result.compressed_tokens < result.original_tokens
        assert result.saved_tokens > 0
        assert result.estimated_savings_usd > 0.0

    def test_rag_qa_compresses(self, capsys):
        with patch(_SUMMARIZE_PATH, return_value="User asked several TokenAI Q&A questions about pricing and compression."):
            result = compress(RAG_QA, max_tokens=_TIGHT_BUDGET, model="claude-sonnet-4-6")
        _print_savings_report("RAG Q&A", result)
        assert result.compressed_tokens < result.original_tokens
        assert result.saved_tokens > 0
        assert result.estimated_savings_usd > 0.0

    def test_system_prompt_preserved_across_types(self):
        fixtures = [SUPPORT_CHAT, CODING_ASSISTANT, RAG_QA]
        for msgs in fixtures:
            with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
                result = compress(msgs, max_tokens=_TIGHT_BUDGET)
            assert result.messages[0]["role"] == "system"

    def test_summary_message_present_when_compressed(self):
        for msgs in [SUPPORT_CHAT, CODING_ASSISTANT, RAG_QA]:
            with patch(_SUMMARIZE_PATH, return_value=_MOCK_SUMMARY):
                result = compress(msgs, max_tokens=_TIGHT_BUDGET)
            summary_msgs = [
                m for m in result.messages
                if "[Conversation summary]" in m.get("content", "")
            ]
            assert len(summary_msgs) == 1
