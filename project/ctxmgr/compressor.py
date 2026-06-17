"""Rolling summarizer that compresses old conversation turns via Claude Haiku."""
from __future__ import annotations

import anthropic

from ctxmgr._utils import extract_text
from ctxmgr.counter import TokenCounter

# The prompt sent to Haiku to produce a summary.
SUMMARIZE_PROMPT = """\
You are a conversation summarizer. Given the messages below, write a concise \
summary that preserves:
- Key facts, data, and technical details exchanged
- Decisions made or conclusions reached
- Any code snippets or structured data referenced (quote briefly)
- Context needed to continue the conversation naturally

Rules:
- Output ONLY the summary text — no headings, labels, or preamble
- Keep it under 300 words
- Write in past tense, third person ("The user asked...", "The assistant explained...")

Messages to summarize:
{messages}"""


class RollingSummarizer:
    """Compress a conversation to stay under a token budget.

    Priority pinning guarantees that the system prompt (if present) and the
    last ``pin_last_pairs`` user/assistant exchanges are never summarized.
    Everything older than that is collapsed into one summary message via
    Claude Haiku.

    Usage::

        summarizer = RollingSummarizer(token_budget=4000)
        messages = summarizer.compress(messages)
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        token_budget: int = 4000,
        pin_last_pairs: int = 3,
        counter_model: str = "claude-3-haiku",
    ) -> None:
        self._client = anthropic.Anthropic()
        self._model = model
        self._token_budget = token_budget
        self._pin_last_pairs = pin_last_pairs
        self._counter = TokenCounter(counter_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, messages: list[dict]) -> list[dict]:
        """Return a compressed copy of *messages* within the token budget.

        If the conversation is already within budget it is returned unchanged.
        Otherwise the oldest turns (excluding pinned messages) are replaced by
        a single summary message produced by Claude Haiku.
        """
        if self._counter.count_messages(messages) <= self._token_budget:
            return messages

        system_msg, chat_msgs = self._split_system(messages)
        pin_count = self._pin_last_pairs * 2  # each pair = user + assistant

        if len(chat_msgs) <= pin_count:
            # Nothing old enough to summarize; return as-is.
            return messages

        to_summarize = chat_msgs[:-pin_count]
        pinned_tail = chat_msgs[-pin_count:]

        summary_text = self._summarize(to_summarize)
        summary_msg = {
            "role": "assistant",
            "content": f"[Conversation summary]\n{summary_text}",
        }

        result: list[dict] = []
        if system_msg is not None:
            result.append(system_msg)
        result.append(summary_msg)
        result.extend(pinned_tail)
        return result

    def token_count(self, messages: list[dict]) -> int:
        """Return the estimated token count for *messages*."""
        return self._counter.count_messages(messages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_system(self, messages: list[dict]) -> tuple[dict | None, list[dict]]:
        """Return (system_message_or_None, remaining_chat_messages)."""
        if messages and messages[0]["role"] == "system":
            return messages[0], list(messages[1:])
        return None, list(messages)

    def _summarize(self, messages: list[dict]) -> str:
        """Call Claude Haiku to summarize *messages* and return the text."""
        formatted = "\n".join(
            f"{m['role'].upper()}: {extract_text(m.get('content'))}" for m in messages
        )
        prompt = SUMMARIZE_PROMPT.format(messages=formatted)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
