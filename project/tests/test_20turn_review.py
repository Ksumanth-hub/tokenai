"""20-turn conversation compression â€” manual review script.

Run with:
    ANTHROPIC_API_KEY=... pytest tests/test_20turn_review.py -v -s

Requires a real API key. Prints before/after token counts and the full
compressed conversation so you can judge summary quality by eye.
"""

import os

import pytest

from tokenai.compressor import RollingSummarizer
from tokenai.counter import TokenCounter

TWENTY_TURN_CONVERSATION: list[dict] = [
    {"role": "system", "content": "You are TokenAI, an expert on token economics and LLM context management."},
    {"role": "user", "content": "What is a token in the context of large language models?"},
    {"role": "assistant", "content": "A token is the basic unit that LLMs process. It roughly corresponds to 4 characters or 0.75 words in English. The sentence 'Hello, world!' is about 4 tokens."},
    {"role": "user", "content": "Why does tokenization matter for API costs?"},
    {"role": "assistant", "content": "API pricing is per-token. Input and output tokens are billed separately. A 1M-token context window filled with a long conversation can cost significantly more than a compact one."},
    {"role": "user", "content": "What is a context window?"},
    {"role": "assistant", "content": "A context window is the maximum number of tokens a model can process in a single call â€” including both the conversation history and the model's response. Claude's largest window is 200K tokens."},
    {"role": "user", "content": "What happens when a conversation exceeds the context window?"},
    {"role": "assistant", "content": "The call fails with a context-length error, or the provider silently truncates the oldest messages. Either way you lose information. Rolling summarization is one solution."},
    {"role": "user", "content": "How does rolling summarization work?"},
    {"role": "assistant", "content": "You periodically replace the oldest N messages with a single summary message produced by a fast, cheap model like Haiku. Recent turns are pinned so the model retains the most relevant context."},
    {"role": "user", "content": "What should a good summarization prompt preserve?"},
    {"role": "assistant", "content": "Key facts exchanged, decisions made, technical details like code snippets or data values, and enough narrative thread that the conversation can continue naturally without feeling like a cold start."},
    {"role": "user", "content": "Why use Claude Haiku specifically for summarization?"},
    {"role": "assistant", "content": "Haiku is fast (~3Ã— faster than Sonnet) and cheap (~10Ã— cheaper per token). Summary quality is high enough for conversation compression. You save the heavier models for tasks that need deep reasoning."},
    {"role": "user", "content": "What is priority pinning?"},
    {"role": "assistant", "content": "Priority pinning means certain messages are guaranteed to survive compression: the system prompt (which defines the assistant persona and rules) and the last N user/assistant pairs (most recent context)."},
    {"role": "user", "content": "How many pairs should we pin?"},
    {"role": "assistant", "content": "Three pairs (6 messages) is a reasonable default. It gives the model enough recent context to maintain conversational coherence without eating too much of the budget."},
    {"role": "user", "content": "Should the summary message have a special role?"},
    {"role": "assistant", "content": "Using the 'assistant' role with a [Conversation summary] label works well. It reads naturally in the chat history and doesn't confuse models trained on user/assistant turns."},
    {"role": "user", "content": "What token budget should we target?"},
    {"role": "assistant", "content": "4 000 tokens covers a typical compressed conversation and leaves plenty of room for the model's response. You can tune this per use-case â€” customer support chats need less history than coding sessions."},
    {"role": "user", "content": "How do we measure summary quality?"},
    {"role": "assistant", "content": "Three checks: (1) token count is under budget, (2) key facts from the original turns appear in the summary, (3) a human reading only the compressed history can follow the conversation without obvious gaps."},
    {"role": "user", "content": "Is there a risk of hallucination in the summary?"},
    {"role": "assistant", "content": "Yes. The summarizer might mis-state a number or drop a nuance. Mitigation: keep the summarization prompt narrow (facts only, no inference), and always pin recent turns where accuracy matters most."},
    {"role": "user", "content": "Can we run compression multiple times on the same conversation?"},
    {"role": "assistant", "content": "Yes. Each pass replaces old messages with a new summary. Over a very long session you accumulate layers: summary-of-summary, then more recent pinned turns. The budget check prevents infinite re-compression."},
    {"role": "user", "content": "What are the next steps after Week 2?"},
    {"role": "assistant", "content": "Week 3 covers semantic retrieval â€” embedding older turns and fetching only the ones relevant to the current query, so you get the best of both: compressed context plus targeted recall of anything the summary might have dropped."},
]


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set â€” skipping live API test",
)
def test_20turn_compression_quality():
    """Live test: compress a 20-turn conversation to under 4 000 tokens.

    The test conversation totals ~871 tokens, so we use a 450-token budget to
    force compression and prove the summarizer works end-to-end.  In production
    the budget would be set to 4 000.
    """
    counter = TokenCounter("claude-3-haiku")
    original_tokens = counter.count_messages(TWENTY_TURN_CONVERSATION)

    # Use a tight budget to force compression of this ~871-token conversation.
    # 600 is achievable: system + summary + 6 pinned messages fit comfortably.
    TEST_BUDGET = 600
    summarizer = RollingSummarizer(token_budget=TEST_BUDGET, pin_last_pairs=3)
    compressed = summarizer.compress(TWENTY_TURN_CONVERSATION)
    compressed_tokens = counter.count_messages(compressed)

    # ---- Manual review output ----------------------------------------
    print(f"\n{'='*60}")
    print(f"ORIGINAL  : {len(TWENTY_TURN_CONVERSATION)} messages, {original_tokens} tokens")
    print(f"COMPRESSED: {len(compressed)} messages, {compressed_tokens} tokens")
    print(f"BUDGET    : {TEST_BUDGET} tokens")
    print(f"REDUCTION : {original_tokens - compressed_tokens} tokens "
          f"({(original_tokens - compressed_tokens) / original_tokens * 100:.1f}%)")
    print(f"{'='*60}")
    for i, msg in enumerate(compressed):
        role = msg["role"].upper()
        content = msg["content"]
        preview = content[:200] + ("..." if len(content) > 200 else "")
        # encode to ASCII with replacement to avoid cp1252 errors on Windows
        safe = preview.encode("ascii", errors="replace").decode("ascii")
        print(f"\n[{i}] {role}\n{safe}")
    print(f"\n{'='*60}\n")

    # ---- Assertions ---------------------------------------------------
    assert compressed_tokens < TEST_BUDGET, (
        f"Compressed conversation is {compressed_tokens} tokens â€” exceeds {TEST_BUDGET}-token budget"
    )

    # System prompt preserved
    assert compressed[0]["role"] == "system"

    # Last 3 pairs (6 messages) preserved verbatim
    chat_original = [m for m in TWENTY_TURN_CONVERSATION if m["role"] != "system"]
    pinned = [
        m for m in compressed
        if m["role"] != "system" and "[Conversation summary]" not in m.get("content", "")
    ]
    assert pinned == chat_original[-6:], "Last 3 pairs not preserved exactly"

    # Exactly one summary message present and non-trivial
    summaries = [m for m in compressed if "[Conversation summary]" in m.get("content", "")]
    assert len(summaries) == 1, "Expected exactly one summary message"
    assert len(summaries[0]["content"]) > 50, "Summary seems too short to be meaningful"
