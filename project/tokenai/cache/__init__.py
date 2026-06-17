"""Public cache API for tokenai Layer 3: Adaptive Semantic Cache."""
from __future__ import annotations

from tokenai.cache import embedder, store, threshold


def cache_get(query: str, customer_id: str) -> dict | None:
    """Return a cached response if one exists above the customer's threshold.

    Steps:
    1. Embed *query* via embedder.
    2. Search the customer's store for the nearest neighbour.
    3. Fetch the customer's current similarity threshold.
    4. Return ``{"response": str, "score": float, "hit": True}`` if the top
       score meets the threshold, else ``None``.
    """
    results = store.search(query, customer_id, top_k=1)
    if not results:
        return None
    top = results[0]
    cutoff = threshold.get(customer_id)
    if top["score"] >= cutoff:
        return {"response": top["response"], "score": top["score"], "hit": True}
    return None


def cache_store(query: str, response: str, customer_id: str) -> None:
    """Persist a query-response pair after a cache miss.

    Call this immediately after receiving the LLM response so that future
    similar queries can be served from the cache.
    """
    store.save(query, response, customer_id)


def cache_feedback(query: str, customer_id: str, was_correct: bool) -> float:
    """Record whether a cache hit was correct and return the updated threshold.

    Correct hits lower the threshold slightly (allow more hits).
    Wrong hits raise it significantly (require higher similarity before trusting).
    """
    return threshold.update(customer_id, was_correct)
