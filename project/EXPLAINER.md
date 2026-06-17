# How tokenai Works — A Plain-English Guide

This document explains the entire tokenai project from scratch. No coding experience required.

---

## The Problem We Are Solving

When you chat with an AI assistant like Claude or ChatGPT, the AI does not have memory between messages on its own. Every single time you send a message, your app has to re-send the **entire conversation from the beginning** — every message you wrote, every reply the AI gave — so the AI can understand the context.

Imagine calling customer support and every time you ask a new question, the agent forgets who you are and you have to re-explain your entire situation from scratch. That is what happens under the hood with AI apps.

Now here is the expensive part: **AI companies charge by the word** (more precisely, by something called a "token"). The more text you send, the more you pay. In a long conversation with 20 back-and-forth messages, you might be re-sending 800+ words every single time — even though most of those old messages are no longer very important.

**tokenai solves this by shrinking the conversation before you send it**, so you send fewer tokens, pay less money, and get faster responses.

---

## What Is a Token?

Before anything else, you need to understand what a "token" is, because everything in this project revolves around the concept.

A token is roughly **4 characters of text**, or about **¾ of a word**.

Examples:
- `"Hello"` = 1 token
- `"Hello, world!"` = 4 tokens
- `"What is machine learning?"` = 5 tokens
- A full paragraph = roughly 75–100 tokens

AI companies like Anthropic (makers of Claude) charge something like **$3 per million tokens** sent to the AI. That sounds tiny, but a busy app sending thousands of conversations per day adds up fast.

The goal of tokenai is simple: **send fewer tokens, save real money.**

---

## The Three Layers of tokenai

tokenai attacks the problem in three different ways, each building on the previous one.

```
Layer 1: Count  →  Layer 2: Compress  →  Layer 3: Skip the AI entirely
```

### Layer 1 — Token Counter

**What it does:** Counts exactly how many tokens are in any message or conversation, without making any API call or spending any money.

**Why it matters:** Before you can shrink a conversation, you need to know how big it actually is. This layer gives you that number instantly and for free.

**Analogy:** Like a word count tool in Microsoft Word — it just reads what you have and tells you the number.

---

### Layer 2 — Rolling Summarizer (Compression)

**What it does:** When a conversation is too long, it summarizes the old parts and keeps the recent parts word-for-word.

**Why it matters:** The old messages in a conversation are usually less important than the recent ones. If you and an AI spent 15 messages discussing background context and then 5 messages on the actual task, you probably only need a summary of those first 15, not every word.

**How it works, step by step:**

1. Count the tokens in the full conversation
2. If it fits within the budget → do nothing, send as-is
3. If it is too long → split the conversation into three parts:
   - **System message** (the AI's instructions) → always kept word-for-word
   - **Old messages** (everything except the most recent few) → sent to a cheap AI (Claude Haiku) to be summarized into a short paragraph
   - **Recent messages** (the last 3–5 back-and-forth pairs) → kept word-for-word, because they contain what the user is actually asking about right now
4. Combine: [system] + [summary paragraph] + [recent messages]
5. The result is much shorter, but still makes sense to the AI

**Analogy:** Imagine a meeting where someone catches a latecomer up: "Earlier we discussed A, B, and C — here's the short version. Now we're talking about D." The latecomer understands the context without sitting through the whole meeting.

**Real example:**
```
Before: 31 messages, 879 tokens
After:   8 messages, 573 tokens
Saved:  306 tokens (34.8% reduction)
Cost:   About $0.0009 saved per conversation on Claude Sonnet
```

---

### Layer 3 — Adaptive Semantic Cache

This is the most powerful layer. Instead of compressing the conversation and still sending it to the AI, **it skips the AI entirely** if it has seen a similar question before.

**The Core Idea:**

Imagine a library with millions of answered questions. When a new question comes in, instead of asking an AI to answer it fresh, you first check: "Has someone asked something very similar before?" If yes, return that old answer. The AI never gets called. No tokens spent. No API cost.

But "similar" is the tricky part. A simple word-match would miss obvious pairs like:
- *"What is ML?"* and *"What is machine learning?"* — different words, same meaning
- *"How does backprop work?"* and *"Explain backpropagation"* — totally different words, identical question

tokenai solves this with **vector embeddings**.

---

#### What Is a Vector Embedding?

Every piece of text can be converted into a list of numbers (a "vector") that represents its *meaning* — not its exact words. Texts that mean similar things produce vectors that are mathematically close to each other.

**Analogy:** Think of it like GPS coordinates. "The Eiffel Tower" and "the iron tower in Paris" are very different strings of text, but their GPS coordinates are nearly identical. If you search for "things near 48.8584° N, 2.2945° E," you find both.

tokenai converts every stored question into one of these number-vectors and stores them in a special database (ChromaDB). When a new question arrives:

1. Convert the new question to its vector
2. Search the database for the closest stored vector
3. If the closest match is similar enough (above a "threshold") → return the cached answer
4. If nothing is close enough → fall through to Layer 2, compress, and call the AI

---

#### The Adaptive Threshold

The "threshold" is the minimum similarity score required to count as a match. If set too low, you get false positives (wrong answers returned). Too high, and you miss obvious matches.

tokenai makes the threshold **self-adjusting per customer**:

- **Starting value:** 0.85 (85% similarity required)
- **After a correct cache hit:** threshold drops by 0.005 → becomes slightly more permissive (reward good matches)
- **After a wrong cache hit:** threshold rises by 0.020 → becomes stricter (penalize bad matches — 4× harder than the reward)
- **Limits:** Never goes below 0.70 (too permissive) or above 0.98 (too strict)

**Analogy:** It works like a human learning from experience. If you keep getting right answers when you take a shortcut, you use that shortcut more. If you get burned once, you become much more careful.

Each customer (company using the API) gets their own independent threshold that learns from their specific use patterns.

---

## The REST API

All three layers are available over the internet as a web service. This means any app — mobile, web, or desktop — can use tokenai without installing any Python code.

**What a REST API is:** A set of web addresses (called "endpoints") that your app can send requests to, like filling out a form and getting a result back.

**How it works:**

1. Your app sends a request to `http://localhost:8000/compress` with the conversation
2. The API checks the cache (Layer 3), compresses if needed (Layer 2), counts tokens (Layer 1)
3. The API sends back the result — smaller conversation, token count, money saved

**Authentication:** Every request must include a secret key (`x-api-key: dev-key-local` for development). This prevents random people from using your API for free.

**Endpoints (web addresses):**

| Address | What it does |
|---|---|
| `/compress` | Submit a conversation, get back a compressed version with cost savings |
| `/cache/get` | Check if a question is already in the cache |
| `/cache/store` | Add a new question-answer pair to the cache |
| `/cache/feedback` | Tell the system whether a cached answer was correct or wrong |
| `/cache/stats` | See the current threshold and accuracy for a customer |
| `/health` | Check if the server is running |
| `/demo` | Open the interactive demo in your browser |
| `/docs` | Auto-generated API documentation |

---

## The Interactive Demo

Visit `http://localhost:8000/demo` in your browser after starting the server.

It has three tabs you can click through:

**Tab 1 — Cache Playground**
- Type any question
- Click "Check Cache"
- See whether it is a HIT (found in cache, answered instantly) or MISS (not found)
- Hits show a similarity score (e.g., 91%) and response time (e.g., 18ms vs 2,000ms for an AI call)
- Click "Seed Demo" to pre-load 5 AI/ML questions so you can immediately test hits
- After a hit, rate it as ✓ Correct or ✗ Wrong and watch the threshold adjust

**Tab 2 — Compress**
- Build a fake conversation by adding user and assistant messages
- Or click "Load Sample" to get a realistic 10-message ML conversation
- Click "Compress" to see the before and after token counts
- Toggle "Use Cache" to route through the cache first

**Tab 3 — Adaptive Learning**
- See the current similarity threshold as a large number (e.g., 0.850)
- A visual bar shows where you are between "permissive" (0.70) and "strict" (0.98)
- Click "5× Correct Feedback" or "3× Wrong Feedback" to watch the threshold move in real time
- See the hit-rate accuracy (how often cached answers were correct)

The top of every page shows live counters: total queries sent, cache hits, tokens saved, and dollars saved.

---

## The MCP Integration

MCP stands for "Model Context Protocol." It is a standard way to give AI assistants like Claude Desktop access to external tools — similar to how apps on your phone can use your camera or GPS.

When tokenai's MCP server is connected to Claude Desktop, Claude can:
- Count tokens in text you paste directly into the chat
- Compress a conversation you describe
- Check and update the semantic cache
- Report how much money would be saved

You do not need to write any code. You just ask Claude in plain English: *"Compress this conversation to 2000 tokens"* and Claude calls tokenai behind the scenes and shows you the result.

---

## How All the Pieces Fit Together

Here is the full picture of what happens when a user sends a message in an app built on tokenai:

```
User sends a message
        |
        v
  [Layer 3] Check semantic cache
        |
   Similar question found?
   Yes  |                     No
        v                      v
  Return cached answer    [Layer 2] Is conversation too long?
  (milliseconds, $0)       No  |              Yes  |
                               v                   v
                          Send as-is         Summarize old turns
                          to the AI          with Claude Haiku
                                                   |
                                                   v
                                             Send shorter version
                                             to the main AI
                                                   |
                                                   v
                                             AI responds
                                                   |
                                                   v
                                             Store in cache
                                             for next time
```

---

## The Technology Stack (What Software Was Used)

You do not need to know these to use tokenai, but here is what powers each piece:

| Component | Technology | What it is |
|---|---|---|
| Token counting | `tiktoken` | A library made by OpenAI that counts tokens fast, locally |
| Summarization | `anthropic` | The official Python library to call Claude AI |
| Text-to-vector conversion | `sentence-transformers` | Converts text into meaning-vectors using a small AI model |
| Cache database | `ChromaDB` | A database designed specifically for storing and searching vectors |
| Threshold storage | `SQLite` | A simple, lightweight database built into Python |
| REST API | `FastAPI` | A modern Python web framework for building APIs |
| Web server | `uvicorn` | The program that runs the API and serves web requests |
| AI desktop tools | `mcp` (FastMCP) | The Model Context Protocol SDK for Claude Desktop integration |
| Frontend | HTML + Tailwind CSS + vanilla JavaScript | The interactive demo page |

---

## The Files in the Project

```
project/
│
├── tokenai/               ← The Python package (all the core logic)
│   ├── counter.py         ← Layer 1: counts tokens
│   ├── compressor.py      ← Layer 2: rolls and summarizes conversations
│   ├── compress.py        ← The main function that ties layers 1 and 2 together
│   └── cache/             ← Layer 3: semantic cache
│       ├── embedder.py    ← Converts text into number-vectors
│       ├── store.py       ← Saves and searches vectors in ChromaDB
│       └── threshold.py   ← Manages per-customer thresholds in SQLite
│   └── api/               ← The REST API
│       ├── server.py      ← The web server
│       ├── auth.py        ← Checks API keys
│       ├── models.py      ← Defines the shape of requests and responses
│       └── routes/        ← The individual endpoints (/compress, /cache/*)
│
├── mcp/
│   └── server.py          ← The Claude Desktop integration (7 tools)
│
├── tests/                 ← Automated tests that verify everything works
│   ├── test_compress.py
│   ├── test_counter.py
│   ├── test_layer3.py
│   ├── test_api.py
│   └── test_mcp.py
│
├── demo.html              ← The interactive browser demo (served at /demo)
├── demo_layer3.py         ← A command-line demo script for the cache
├── .tokenai_cache/        ← Auto-created folder where cache data is stored
├── .env                   ← Your private config (API keys, port) — not committed to git
├── .env.example           ← A safe template showing what goes in .env
├── pyproject.toml         ← Package definition and dependencies
├── README.md              ← Technical reference documentation
└── EXPLAINER.md           ← This document
```

---

## Frequently Asked Questions

**Q: Does my conversation get stored somewhere I don't control?**
All data (vectors, thresholds, cached responses) is stored locally in the `.tokenai_cache/` folder on your own machine or server. Nothing is sent to a third-party service except when calling the Anthropic API for compression.

**Q: What happens if two customers' questions are similar — do they share cache?**
No. Every customer has a separate cache collection (identified by their `customer_id`). Customer A's cached answers are never returned to Customer B.

**Q: How accurate is the semantic cache?**
It depends on your use case. Highly repetitive domains (customer support, FAQ bots, coding assistants) see hit rates of 40–80%. The adaptive threshold means accuracy improves over time as the system learns what similarity level works for your specific queries.

**Q: Do I need a GPU to run this?**
No. The embedding model (`all-MiniLM-L6-v2`) runs on CPU. It takes about 50–200ms to embed a query on a typical laptop, which is still much faster than waiting for an LLM response.

**Q: How much money can this actually save?**
It depends on your traffic and conversation length. In internal tests, a busy support chatbot sending 10,000 conversations/day through Claude Sonnet saves roughly $9–15/day from compression alone, before accounting for cache hits. With a 50% cache hit rate, that doubles.

**Q: Can I use this with OpenAI models (GPT-4, etc.) instead of Claude?**
The token counting works with any model. The compression step currently calls Claude Haiku to write summaries, but the output (compressed messages) can be sent to any LLM. The semantic cache is model-agnostic — it stores and retrieves based on meaning, not which AI you use.

---

## How to Run It

1. **Install dependencies:**
   ```bash
   pip install tokenai sentence-transformers chromadb fastapi uvicorn python-dotenv
   ```

2. **Set your API key:**
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

3. **Start the API server:**
   ```bash
   python -m tokenai.api.server
   ```

4. **Open the demo in your browser:**
   ```
   http://localhost:8000/demo
   ```

5. **Click "Seed Demo"** to load sample questions, then try similar queries to see cache hits in action.

---

*tokenai is an open project. If something is unclear or you find a bug, open an issue on GitHub.*
