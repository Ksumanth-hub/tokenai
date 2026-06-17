"""Internal helpers shared across ctxmgr modules."""
from __future__ import annotations


def extract_text(content: str | list | None) -> str:
    """Return plain text from a message content field.

    Handles both plain strings and the list-of-typed-blocks format used by
    OpenAI (tool calls, images) and Anthropic (multi-modal, tool use).
    Non-text blocks become short placeholders so token counts stay meaningful.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type in ("image", "image_url"):
            parts.append("[image]")
        elif block_type == "tool_use":
            parts.append(f"[tool:{block.get('name', '')}]")
        elif block_type == "tool_result":
            # Content of a tool result can itself be a string or list.
            parts.append(extract_text(block.get("content", "")))
        else:
            # Unknown block — try common text fields before giving up.
            parts.append(block.get("text", block.get("value", "")))
    return " ".join(p for p in parts if p)
