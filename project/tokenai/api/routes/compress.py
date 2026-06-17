"""Compress endpoint — POST /compress."""
from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException

from tokenai import compress
from tokenai.api.auth import verify_api_key
from tokenai.api.models import CompressRequest, CompressResponse

router = APIRouter(prefix="/compress", tags=["compress"])


@router.post("", response_model=CompressResponse)
def compress_endpoint(
    body: CompressRequest,
    key_customer_id: str = Depends(verify_api_key),
) -> CompressResponse:
    """Compress a conversation history to fit within a token budget.

    When ``use_cache=True``, checks the semantic cache before compressing
    — returning in milliseconds on a hit and storing the result on a miss.
    ``customer_id`` must be explicitly provided in the request body when
    ``use_cache=True`` so one API key can serve multiple end-customers.
    """
    if body.use_cache and not body.customer_id:
        raise HTTPException(
            status_code=400,
            detail="customer_id is required in the request body when use_cache=True.",
        )

    customer_id = body.customer_id if body.customer_id else key_customer_id

    t0 = perf_counter()
    try:
        result = compress(
            messages=body.messages,
            max_tokens=body.max_tokens,
            model=body.model,
            use_cache=body.use_cache,
            customer_id=customer_id if body.use_cache else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = max(0.001, (perf_counter() - t0) * 1000)

    return CompressResponse(
        messages=result.messages,
        saved_tokens=result.saved_tokens,
        saved_usd=result.estimated_savings_usd,
        compression_ratio=result.ratio,
        cache_hit=result.cache_hit,
        latency_ms=latency_ms,
    )
