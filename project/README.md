# tokenai

LLM context management for Python — token counting, rolling summarization, adaptive semantic caching, and dollar savings reports.

LLM APIs charge per token. **tokenai** gives you three layers of cost reduction:

1. **Token counting** — count tokens locally, no API call needed
2. **Rolling summarization** — compress long conversations with Claude Haiku before sending to expensive models
3. **Adaptive semantic cache** — skip the LLM entirely for similar queries using vector similarity search with a per-customer self-tuning threshold

```bash
pip install tokenai
```

Requires Python 3.10+ and an `ANTHROPIC_API_KEY` (only needed when compression runs).

---

## Quick start

```python
from tokenai import compress

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a token?"},
    {"role": "assistant", "content": "A token is the basic unit LLMs process — roughly 4 characters."},
    # ... many more turns ...
]

result = compress(messages, max_tokens=4000, model="claude-sonnet-4-6")

print(f"Before : {result.original_tokens} tokens")
print(f"After  : {result.compressed_tokens} tokens")
print(f"Saved  : {result.saved_tokens} tokens (${result.estimated_savings_usd:.4f})")

import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=result.messages,
)
```

---

## Semantic Cache (Layer 3)

The cache stores query-response pairs as vector embeddings. When a new query arrives, it checks for a semantically similar past query — if the similarity exceeds the customer's threshold, the cached response is returned in milliseconds instead of making an LLM call.

```python
from tokenai.cache import cache_store, cache_get, cache_feedback

# Store a response after an LLM call
cache_store(
    query="What is machine learning?",
    response="Machine learning is a branch of AI where systems learn from data...",
    customer_id="acme-corp",
)

# On the next similar query — no LLM needed
result = cache_get("Explain ML to me", customer_id="acme-corp")
if result:
    print(result["response"])  # cached answer
    print(result["score"])     # similarity: e.g. 0.91
```

### Use the cache inside `compress()`

```python
result = compress(
    messages,
    max_tokens=4000,
    model="claude-sonnet-4-6",
    use_cache=True,
    customer_id="acme-corp",   # required when use_cache=True
)

if result.cache_hit:
    print("Served from cache — no LLM call made")
```

### Adaptive threshold

Each customer starts with a similarity threshold of `0.85`. The threshold self-tunes based on feedback:

- Correct hit → threshold `− 0.005` (slightly more permissive)
- Wrong hit → threshold `+ 0.020` (tightened 4× harder)
- Clamped to `[0.70, 0.98]`

```python
from tokenai.cache import cache_feedback
from tokenai.cache.threshold import get_stats

# After the user says the cached answer was right
cache_feedback("Explain ML to me", customer_id="acme-corp", was_correct=True)

stats = get_stats("acme-corp")
print(stats)  # {"threshold": 0.845, "total_hits": 12, "correct_hits": 11, "accuracy": 0.917}
```

Cache data is stored locally in `.tokenai_cache/` (auto-created, gitignored):
- `chroma/` — ChromaDB vector store (one collection per customer)
- `thresholds.db` — SQLite with per-customer thresholds and hit-rate statistics

---

## REST API

Start the API server:

```bash
python -m tokenai.api.server
# Listening on http://localhost:8000
# Interactive demo at http://localhost:8000/demo
# API docs at http://localhost:8000/docs
```

All endpoints require an `x-api-key` header. The default dev key is `dev-key-local`. For production, set `TOKENAI_API_KEYS` in `.env`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/compress` | Compress a conversation (optionally with cache) |
| `POST` | `/cache/get` | Check for a semantic cache hit |
| `POST` | `/cache/store` | Store a query-response pair |
| `POST` | `/cache/feedback` | Record whether a hit was correct |
| `GET` | `/cache/stats` | Per-customer threshold and accuracy stats |
| `GET` | `/health` | Liveness check |
| `GET` | `/demo` | Interactive HTML demo |
| `GET` | `/docs` | Swagger UI |

### Example — compress

```bash
curl -X POST http://localhost:8000/compress \
  -H "x-api-key: dev-key-local" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is overfitting?"},
      {"role": "assistant", "content": "Overfitting is when a model memorizes training data..."},
      {"role": "user", "content": "How do I prevent it?"}
    ],
    "max_tokens": 2000,
    "model": "claude-sonnet-4-6",
    "use_cache": true,
    "customer_id": "acme-corp"
  }'
```

```json
{
  "messages": [...],
  "saved_tokens": 48,
  "saved_usd": 0.000144,
  "compression_ratio": 0.82,
  "cache_hit": false,
  "latency_ms": 312.4
}
```

### Example — cache

```bash
# Store
curl -X POST http://localhost:8000/cache/store \
  -H "x-api-key: dev-key-local" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is gradient descent?", "response": "An optimization algorithm...", "customer_id": "acme-corp"}'

# Check (semantically similar query)
curl -X POST http://localhost:8000/cache/get \
  -H "x-api-key: dev-key-local" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain gradient descent", "customer_id": "acme-corp"}'
```

```json
{"hit": true, "response": "An optimization algorithm...", "score": 0.94, "latency_ms": 18.2}
```

### Configuration (`.env`)

```env
TOKENAI_API_KEYS=key-customer-acme,key-customer-beta
TOKENAI_DEV_KEY=dev-key-local
PORT=8000
```

Copy `.env.example` to `.env` to get started.

---

## Interactive Demo

Visit `http://localhost:8000/demo` after starting the server for a live dashboard with three tabs:

- **Cache Playground** — type any query to see a HIT or MISS with similarity score; seed a knowledge base of 5 AI/ML Q&A pairs; give feedback and watch the threshold respond
- **Compress** — build a conversation, set a token budget, compress with or without cache; see before/after token counts and USD saved
- **Adaptive Learning** — live threshold gauge, feedback buttons (Correct / Wrong), scenario runners that animate the threshold converging in real time

---

## MCP Integration

Connect tokenai to Claude Desktop so Claude can call your compression and cache tools live.

**Install the MCP SDK:**

```bash
pip install mcp
```

**Add to Claude Desktop config** (`~/AppData/Roaming/Claude/claude_desktop_config.json` on Windows, `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "tokenai": {
      "command": "python",
      "args": ["C:/path/to/your/project/mcp/server.py"]
    }
  }
}
```

Restart Claude Desktop — it will show seven new tools:

| Tool | What it does |
|---|---|
| `compress_messages` | Compresses a conversation and returns result + savings metadata |
| `count_tokens` | Counts tokens in a string or message list for any model |
| `get_savings_report` | Returns savings metadata only (no compressed messages) |
| `cache_get` | Checks for a semantic cache hit for a given query |
| `cache_store` | Stores a query-response pair in the semantic cache |
| `cache_feedback` | Records whether a cached answer was correct, updating the threshold |
| `cache_stats` | Returns threshold and hit-rate stats for a customer |

**Example prompts:**

> "Count the tokens in this text for claude-sonnet-4-6: …"
> "Compress this 20-turn conversation to 4000 tokens with aggressive mode."
> "Check the cache for 'What is gradient descent?' for customer acme-corp."
> "Store this Q&A pair in the cache for customer acme-corp."

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

### `compress(messages, max_tokens, model, aggressiveness, use_cache, customer_id)`

```python
from tokenai import compress, CompressionResult

result: CompressionResult = compress(
    messages,                      # list of {"role": ..., "content": ...}
    max_tokens=4000,               # token budget
    model="claude-sonnet-4-6",     # target model (used for $ savings calc)
    aggressiveness="medium",       # "light" | "medium" | "aggressive"
    use_cache=False,               # check semantic cache before compressing
    customer_id=None,              # required when use_cache=True
)
```

**`CompressionResult` fields:**

| Field | Type | Description |
|---|---|---|
| `messages` | `list[dict]` | Compressed conversation |
| `original_tokens` | `int` | Token count before compression |
| `compressed_tokens` | `int` | Token count after compression |
| `saved_tokens` | `int` | `original_tokens − compressed_tokens` |
| `ratio` | `float` | `compressed / original` |
| `estimated_savings_usd` | `float` | `saved_tokens × model input price` |
| `aggressiveness` | `str` | Level used (or `"cache_hit"` on a hit) |
| `cache_hit` | `bool` | `True` when served from semantic cache |

**Aggressiveness levels:**

| Level | Pinned pairs | Best for |
|---|---|---|
| `"light"` | 5 pairs | Long coding sessions, high coherence needed |
| `"medium"` | 3 pairs | General purpose (default) |
| `"aggressive"` | 1 pair | Support chats, RAG lookups, cost-critical apps |

---

### `cache_get / cache_store / cache_feedback`

```python
from tokenai.cache import cache_get, cache_store, cache_feedback
from tokenai.cache.threshold import get_stats

# Store
cache_store(query, response, customer_id)

# Retrieve
result = cache_get(query, customer_id)
# Returns {"response": str, "score": float, "hit": True} or None

# Feedback
new_threshold = cache_feedback(query, customer_id, was_correct=True)

# Stats
stats = get_stats(customer_id)
# {"threshold": float, "total_hits": int, "correct_hits": int, "accuracy": float}
```

---

### `TokenCounter`

```python
from tokenai import TokenCounter

counter = TokenCounter("claude-sonnet-4-6")
print(counter.count("Hello, world!"))       # 4
print(counter.count_messages(messages))     # e.g. 879
```

Supported: all `claude-*` and `gpt-*` model names. Unknown models fall back to `cl100k_base`.

---

### `RollingSummarizer`

```python
from tokenai import RollingSummarizer

summarizer = RollingSummarizer(
    model="claude-haiku-4-5-20251001",
    token_budget=4000,
    pin_last_pairs=3,
)
compressed_messages = summarizer.compress(messages)
```

---

## Message format support

Both **Anthropic** and **OpenAI** message formats work out of the box:

```python
# Plain string
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
| Cache miss | Falls through to normal compression |
| New customer (no threshold set) | Starts at `0.85` |

---

## How it works

```
query arrives
      |
      v
  cache_get()              <- embed query, cosine search in ChromaDB
      |
  score >= threshold?
   yes |                    no |
       v                       v
  cached response         compress()
  (milliseconds)               |
                          TokenCounter   <- local token count via tiktoken
                               |
                          over budget?
                           yes |
                               v
                          Split: system + oldest turns + pinned pairs
                               |
                               v
                          Claude Haiku   <- cheap summarization
                               |
                               v
                          [system] + [summary] + [pinned pairs]
                               |
                               v
                          cache_store()  <- save for next time
                               |
                               v
                          CompressionResult
```

---

## Requirements

- Python 3.10+
- `anthropic >= 0.100.0`
- `tiktoken >= 0.7.0`
- `sentence-transformers >= 2.2.0`
- `chromadb >= 0.4.0`
- `fastapi >= 0.110.0`
- `uvicorn >= 0.29.0`
- `python-dotenv >= 1.0.0`
- `pydantic >= 2.0.0`
- `ANTHROPIC_API_KEY` environment variable

```bash
export ANTHROPIC_API_KEY="sk-ant-..."          # Linux/macOS
$env:ANTHROPIC_API_KEY = "sk-ant-..."          # Windows PowerShell
```

---

## Project structure

```
tokenai/
  compress.py          # compress() — main entry point
  counter.py           # TokenCounter
  compressor.py        # RollingSummarizer
  cache/
    __init__.py        # cache_get, cache_store, cache_feedback
    embedder.py        # sentence-transformers (all-MiniLM-L6-v2, CPU)
    store.py           # ChromaDB vector store
    threshold.py       # SQLite adaptive threshold
  api/
    server.py          # FastAPI app + /demo route
    config.py          # Settings (loaded from .env)
    auth.py            # API key verification
    models.py          # Pydantic request/response models
    routes/
      cache.py         # /cache/* endpoints
      compress.py      # /compress endpoint
mcp/
  server.py            # FastMCP server (7 tools)
tests/
  test_compress.py
  test_counter.py
  test_layer3.py
  test_api.py
  test_mcp.py
demo.html              # Interactive frontend (served at /demo)
```

---

## License

MIT — see [LICENSE](LICENSE)
