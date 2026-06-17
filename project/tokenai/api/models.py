"""Pydantic request and response models for the tokenai REST API."""
from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Cache models
# ---------------------------------------------------------------------------

class CacheGetRequest(BaseModel):
    query: str
    customer_id: str | None = None


class CacheGetResponse(BaseModel):
    hit: bool
    response: str | None = None
    score: float | None = None
    customer_id: str
    latency_ms: float


class CacheStoreRequest(BaseModel):
    query: str
    response: str
    customer_id: str | None = None


class CacheStoreResponse(BaseModel):
    stored: bool
    customer_id: str


class CacheFeedbackRequest(BaseModel):
    query: str
    was_correct: bool
    customer_id: str | None = None


class CacheFeedbackResponse(BaseModel):
    updated: bool
    new_threshold: float
    customer_id: str


class CacheStatsResponse(BaseModel):
    customer_id: str
    threshold: float
    total_hits: int
    correct_hits: int
    accuracy: float


# ---------------------------------------------------------------------------
# Compress model
# ---------------------------------------------------------------------------

class CompressRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 4000
    model: str = "claude-haiku"
    use_cache: bool = False
    customer_id: str | None = None


class CompressResponse(BaseModel):
    messages: list[dict]
    saved_tokens: int
    saved_usd: float
    compression_ratio: float
    cache_hit: bool = False
    latency_ms: float


# ---------------------------------------------------------------------------
# Health model
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    cache_ready: bool
    mcp_available: bool
