"""Cache endpoints — /cache/get, /cache/store, /cache/feedback, /cache/stats."""
from __future__ import annotations

import time
from time import perf_counter

from fastapi import APIRouter, Depends, Query

from tokenai.api.auth import verify_api_key
from tokenai.api.models import (
    CacheFeedbackRequest,
    CacheFeedbackResponse,
    CacheGetRequest,
    CacheGetResponse,
    CacheStoreRequest,
    CacheStoreResponse,
    CacheStatsResponse,
)
from tokenai.cache import cache_get, cache_store, cache_feedback
from tokenai.cache.threshold import get_stats

router = APIRouter(prefix="/cache", tags=["cache"])


def _resolve(body_id: str | None, key_id: str) -> str:
    """Return body customer_id when given, else fall back to the API key."""
    return body_id if body_id else key_id


@router.post("/get", response_model=CacheGetResponse)
def cache_get_endpoint(
    body: CacheGetRequest,
    key_customer_id: str = Depends(verify_api_key),
) -> CacheGetResponse:
    """Check whether a semantically similar query is already cached.

    Returns the cached response and similarity score on a hit, or
    ``hit=False`` when nothing similar is found above the threshold.
    """
    cid = _resolve(body.customer_id, key_customer_id)
    t0 = perf_counter()
    result = cache_get(body.query, cid)
    latency_ms = max(0.001, (perf_counter() - t0) * 1000)

    if result:
        return CacheGetResponse(
            hit=True,
            response=result["response"],
            score=result["score"],
            customer_id=cid,
            latency_ms=latency_ms,
        )
    return CacheGetResponse(
        hit=False,
        response=None,
        score=None,
        customer_id=cid,
        latency_ms=latency_ms,
    )


@router.post("/store", response_model=CacheStoreResponse)
def cache_store_endpoint(
    body: CacheStoreRequest,
    key_customer_id: str = Depends(verify_api_key),
) -> CacheStoreResponse:
    """Persist a query-response pair in the semantic cache.

    Call this after every LLM response so future similar queries can be
    served from cache without an LLM call.
    """
    cid = _resolve(body.customer_id, key_customer_id)
    cache_store(body.query, body.response, cid)
    return CacheStoreResponse(stored=True, customer_id=cid)


@router.post("/feedback", response_model=CacheFeedbackResponse)
def cache_feedback_endpoint(
    body: CacheFeedbackRequest,
    key_customer_id: str = Depends(verify_api_key),
) -> CacheFeedbackResponse:
    """Record whether a cached answer was correct.

    Trains the per-customer adaptive threshold: correct hits lower it
    slightly (allow more cache hits), wrong hits raise it aggressively.
    """
    cid = _resolve(body.customer_id, key_customer_id)
    new_threshold = cache_feedback(body.query, cid, body.was_correct)
    return CacheFeedbackResponse(
        updated=True,
        new_threshold=new_threshold,
        customer_id=cid,
    )


@router.get("/stats", response_model=CacheStatsResponse)
def cache_stats_endpoint(
    key_customer_id: str = Depends(verify_api_key),
    customer_id: str | None = Query(default=None),
) -> CacheStatsResponse:
    """Return adaptive threshold and hit-rate statistics for a customer.

    Pass ``?customer_id=xxx`` to override the API-key-derived customer.
    """
    cid = _resolve(customer_id, key_customer_id)
    stats = get_stats(cid)
    return CacheStatsResponse(customer_id=cid, **stats)
