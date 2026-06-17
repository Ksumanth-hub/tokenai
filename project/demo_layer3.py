"""Live demo of tokenai Layer 3: Adaptive Semantic Cache."""
import sys
import time

sys.path.insert(0, ".")

from tokenai.cache.embedder import embed, get_dimension
from tokenai.cache.store import save, search
from tokenai.cache.threshold import get as get_threshold, update as update_threshold, get_stats
from tokenai.cache import cache_get, cache_store, cache_feedback
from tokenai import compress

SEP = "-" * 60
CUSTOMER = "demo_customer_001"


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# -- 1. Embedder ----------------------------------------------------------------
section("1. EMBEDDER")

t0 = time.perf_counter()
vec = embed("What is a transformer model?")
elapsed = (time.perf_counter() - t0) * 1000

print(f"  Model dimension : {get_dimension()}")
print(f"  Vector length   : {len(vec)}")
print(f"  First 5 values  : {[round(v, 4) for v in vec[:5]]}")
print(f"  Encode time     : {elapsed:.1f} ms  (model already warm)")

zero = embed("")
print(f"  Empty string    : zero vector  all({all(v == 0.0 for v in zero)})")


# -- 2. Similarity Search ------------------------------------------------------─
section("2. VECTOR STORE — save & search")

PAIRS = [
    ("What is machine learning?",         "ML is a subset of AI that learns from data."),
    ("How do neural networks work?",       "Neural nets use layers of weighted connections."),
    ("What is gradient descent?",         "It minimizes loss by following the steepest downhill gradient."),
    ("Explain the attention mechanism.",   "Attention weighs token relevance to each other in a sequence."),
]

for q, r in PAIRS:
    save(q, r, CUSTOMER)
    print(f"  stored: \"{q[:45]}\"")

print()

QUERIES = [
    ("What is ML?",                          "should match: 'machine learning'"),
    ("How does backpropagation work?",       "should match: 'gradient descent'"),
    ("Tell me about self-attention.",        "should match: 'attention mechanism'"),
    ("What is the capital of France?",      "low score — unrelated"),
]

for q, note in QUERIES:
    results = search(q, CUSTOMER, top_k=1)
    if results:
        r = results[0]
        print(f"  query : \"{q}\"")
        print(f"  match : \"{r['query']}\"  score={r['score']:.3f}  ({note})")
    else:
        print(f"  query : \"{q}\"  -> no results")
    print()


# -- 3. Adaptive Threshold ----------------------------------------------------─
section("3. ADAPTIVE THRESHOLD")

CUST_T = "thresh_demo_001"
print(f"  Initial threshold : {get_threshold(CUST_T):.3f}  (default for new customer)")

print("\n  Simulating 3 correct cache hits:")
for i in range(3):
    t = update_threshold(CUST_T, was_correct=True)
    print(f"    correct hit #{i+1}  -> threshold = {t:.3f}")

print("\n  Simulating 2 wrong cache hits:")
for i in range(2):
    t = update_threshold(CUST_T, was_correct=False)
    print(f"    wrong hit #{i+1}   -> threshold = {t:.3f}")

stats = get_stats(CUST_T)
print(f"\n  Final stats:")
print(f"    threshold    = {stats['threshold']:.3f}")
print(f"    total_hits   = {stats['total_hits']}")
print(f"    correct_hits = {stats['correct_hits']}")
print(f"    accuracy     = {stats['accuracy']:.1%}")


# -- 4. High-Level Cache API ----------------------------------------------------
section("4. CACHE API — cache_get / cache_store / cache_feedback")

CUST_API = "api_demo_001"

print("  [MISS] First lookup — nothing in cache yet")
result = cache_get("What is tokenization in NLP?", CUST_API)
print(f"  cache_get -> {result}\n")

print("  Storing query + response …")
cache_store(
    "What is tokenization in NLP?",
    "Tokenization splits text into subword pieces a model can process.",
    CUST_API,
)

print("  [HIT?] Exact same query:")
result = cache_get("What is tokenization in NLP?", CUST_API)
print(f"  cache_get -> hit={result['hit']}  score={result['score']:.3f}")
print(f"  response  -> \"{result['response'][:60]}…\"\n")

print("  [HIT?] Semantically similar (different wording):")
result2 = cache_get("How does tokenization work?", CUST_API)
if result2:
    print(f"  cache_get -> hit={result2['hit']}  score={result2['score']:.3f}")
    print(f"  response  -> \"{result2['response'][:60]}…\"")
else:
    print("  cache_get -> None  (score below threshold)")

print(f"\n  Recording feedback on the hit …")
new_t = cache_feedback("What is tokenization in NLP?", CUST_API, was_correct=True)
print(f"  new threshold = {new_t:.3f}")


# -- 5. compress() with use_cache ----------------------------------------------─
section("5. compress() WITH use_cache=True")

CUST_COMPRESS = "compress_demo_001"

MESSAGES = [
    {"role": "system",    "content": "You are a helpful AI assistant."},
    {"role": "user",      "content": "What is a large language model?"},
    {"role": "assistant", "content": "An LLM is a neural network trained on massive text corpora to predict and generate language."},
    {"role": "user",      "content": "How many parameters does GPT-4 have?"},
    {"role": "assistant", "content": "GPT-4's exact parameter count is undisclosed, but estimates put it around 1.8 trillion."},
    {"role": "user",      "content": "What is context length?"},
]

print("  CALL 1 — cache cold (expected MISS)")
t0 = time.perf_counter()
r1 = compress(MESSAGES, max_tokens=10_000, use_cache=True, customer_id=CUST_COMPRESS)
t1 = (time.perf_counter() - t0) * 1000
print(f"  cache_hit        = {r1.cache_hit}")
print(f"  original_tokens  = {r1.original_tokens}")
print(f"  compressed_tokens= {r1.compressed_tokens}")
print(f"  saved_tokens     = {r1.saved_tokens}")
print(f"  time             = {t1:.1f} ms\n")

print("  CALL 2 — same messages (expected HIT)")
t0 = time.perf_counter()
r2 = compress(MESSAGES, max_tokens=10_000, use_cache=True, customer_id=CUST_COMPRESS)
t1 = (time.perf_counter() - t0) * 1000
print(f"  cache_hit        = {r2.cache_hit}")
print(f"  original_tokens  = {r2.original_tokens}")
print(f"  saved_tokens     = {r2.saved_tokens}")
print(f"  estimated_savings= ${r2.estimated_savings_usd:.6f}")
print(f"  time             = {t1:.1f} ms  ← no LLM call needed")

print(f"\n  Returned message content:")
for m in r2.messages:
    print(f"    [{m['role']}]: {str(m['content'])[:80]}")


# -- 6. ValueError guard --------------------------------------------------------
section("6. ERROR GUARD — use_cache=True without customer_id")

try:
    compress(MESSAGES, max_tokens=10_000, use_cache=True)
    print("  ERROR: should have raised ValueError")
except ValueError as e:
    print(f"  Correctly raised ValueError: {e}")


print(f"\n{SEP}")
print("  ALL DONE — Layer 3 working correctly.")
print(SEP)
