# tokenai

LLM context management for Python — token counting, rolling summarization, and dollar savings reports.

LLM APIs charge per token. A 20-turn support conversation can hit 800+ tokens before the user asks their real question. **tokenai** compresses the history with Claude Haiku (cheap and fast) so you send fewer tokens to your expensive model — and returns the exact dollar amount saved per call.

```
pip install tokenai
```

Requires Python 3.10+ and an `ANTHROPIC_API_KEY` (only needed when compression runs; token counting is fully local).

---

## Quick start

```python
import os
import anthropic
from tokenai import compress

# Your conversation history (Anthropic or OpenAI format)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a token?"},
    {"role": "assistant", "content": "A token is the basic unit LLMs process — roughly 4 characters or 0.75 words."},
    # ... many more turns ...
]

# Compress to fit within 4,000 tokens, report savings vs Sonnet pricing
result = compress(messages, max_tokens=4000, model="claude-sonnet-4-6")

print(f"Before : {result.original_tokens} tokens")
print(f"After  : {result.compressed_tokens} tokens")
print(f"Saved  : {result.saved_tokens} tokens (${result.estimated_savings_usd:.4f} per call)")

# Pass result.messages to your LLM call as usual
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=result.messages,
)
```

---

## Real before / after

Live test — a 20-turn conversation about token economics, compressed end-to-end via Claude Haiku:

```
ORIGINAL  : 31 messages, 879 tokens
COMPRESSED:  8 messages, 573 tokens
BUDGET    : 600 tokens
REDUCTION : 306 tokens (34.8%)
Time      : ~6 seconds (includes Haiku API call)
```

The compressed result contains:
- **[0] SYSTEM** — persona pinned verbatim, always
- **[1] ASSISTANT** — Haiku's summary of the 14 oldest turns (under 300 words)
- **[2]–[7]** — last 3 user/assistant pairs kept word-for-word

Estimated savings per call on the 306 saved tokens:

| Model | $/call saved | At 10k calls/day |
|---|---|---|
| claude-haiku-4-5 | $0.000306 | $3.06/day |
| claude-sonnet-4-6 | $0.000918 | $9.18/day |
| claude-opus-4-8 | $0.001530 | $15.30/day |

---

## Benchmarks — three conversation types

Measured with `aggressiveness="medium"` (3 pinned pairs):

| Type | Original | Compressed | Reduction | $/call (Sonnet) |
|---|---|---|---|---|
| Support chat | 426 tok | 237 tok | 44.4% | $0.000567 |
| Coding assistant | 726 tok | 515 tok | 29.1% | $0.000633 |
| RAG Q&A | 541 tok | 251 tok | 53.6% | $0.000870 |

---

## API reference

### `compress(messages, max_tokens, model, aggressiveness)`

The main function. Returns a `CompressionResult` with the compressed messages and savings metadata.

```python
from tokenai import compress, CompressionResult

result: CompressionResult = compress(
    messages,                      # list of {"role": ..., "content": ...}
    max_tokens=4000,               # token budget for the compressed result
    model="claude-sonnet-4-6",     # your target model (used for $ savings calc)
    aggressiveness="medium",       # "light" | "medium" | "aggressive"
)
```

**`CompressionResult` fields:**

| Field | Type | Description |
|---|---|---|
| `messages` | `list[dict]` | Compressed conversation, ready to send to your LLM |
| `original_tokens` | `int` | Token count before compression |
| `compressed_tokens` | `int` | Token count after compression |
| `saved_tokens` | `int` | `original_tokens - compressed_tokens` |
| `ratio` | `float` | `compressed / original` — lower means more compression |
| `estimated_savings_usd` | `float` | `saved_tokens × model input price per token` |
| `aggressiveness` | `str` | The level that was used |

**Aggressiveness levels** — controls how many recent pairs are pinned (never summarized):

| Level | Pinned pairs | Best for |
|---|---|---|
| `"light"` | 5 pairs | Long coding sessions, high coherence needed |
| `"medium"` | 3 pairs | General purpose (default) |
| `"aggressive"` | 1 pair | Support chats, RAG lookups, cost-critical apps |

---

### `TokenCounter`

Count tokens locally without any API call. Works for both Claude and OpenAI models.

```python
from tokenai import TokenCounter

counter = TokenCounter("claude-sonnet-4-6")

# Count a string
print(counter.count("Hello, world!"))       # 4

# Count a full conversation
print(counter.count_messages(messages))     # e.g. 879

# Encode / decode
tokens = counter.encode("Hello")
text   = counter.decode(tokens)
```

Supported: all `claude-*` and `gpt-*` model names. Unknown models fall back to `cl100k_base`.

---

### `RollingSummarizer`

Lower-level class if you want direct control over the summarizer.

```python
from tokenai import RollingSummarizer

summarizer = RollingSummarizer(
    model="claude-haiku-4-5-20251001",  # model used to write summaries
    token_budget=4000,                  # compress until under this limit
    pin_last_pairs=3,                   # never summarize the last N pairs
)

compressed_messages = summarizer.compress(messages)
```

---

## Message format support

Both **Anthropic** and **OpenAI** message formats work out of the box:

```python
# Plain string — works with both APIs
{"role": "user", "content": "What is a token?"}

# OpenAI list-of-blocks (vision, tool calls)
{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
]}

# Anthropic list-of-blocks (tool use)
{"role": "assistant", "content": [
    {"type": "text", "text": "I will look that up."},
    {"type": "tool_use", "id": "tu_01", "name": "search", "input": {"q": "tokens"}},
]}
```

Images become `[image]` and tool-use blocks become `[tool:name]` in token counts and summaries.

---

## Edge cases

| Scenario | What happens |
|---|---|
| Empty history `[]` | Returns `[]`, `saved_tokens=0`, no API call |
| Single-turn (no assistant reply yet) | Returns unchanged, no API call |
| One message already over budget | Returns unchanged — cannot split a single message |
| Already under budget | Returns unchanged, no API call |
| `content=None` | Treated as empty string — no crash |

---

## How it works

```
messages (879 tokens)
       |
       v
  TokenCounter          <- counts tokens locally via tiktoken
       |
  over budget?
   yes |
       v
  Split: system + oldest turns + last-3 pairs
       |
       v
  Claude Haiku          <- summarizes the oldest turns cheaply
       |
       v
  [system] + [summary] + [last-3 pairs]  (573 tokens)
       |
       v
  CompressionResult     <- tokens saved, ratio, $ saved
```

---

## Requirements

- Python 3.10+
- `anthropic >= 0.100.0`
- `tiktoken >= 0.7.0`
- `ANTHROPIC_API_KEY` environment variable

Set your key:
```bash
# Linux / macOS
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## License

MIT — see [LICENSE](LICENSE)
