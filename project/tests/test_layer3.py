"""Tests for tokenai Layer 3: Adaptive Semantic Cache."""
from __future__ import annotations

import uuid

import pytest

from tokenai.cache.embedder import embed, get_dimension
from tokenai.cache.store import save, search
from tokenai.cache.threshold import get as threshold_get, update as threshold_update
from tokenai.cache import cache_get, cache_store, cache_feedback
from tokenai import compress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_id(prefix: str = "test") -> str:
    """Generate a unique customer_id so tests never share cache state."""
    return f"{prefix}_{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# embedder.py
# ---------------------------------------------------------------------------

def test_embed_returns_vector():
    """embed() returns a list of 384 floats."""
    result = embed("hello world")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)


def test_embed_empty_string():
    """embed() on empty string returns a zero vector without raising."""
    result = embed("")
    assert isinstance(result, list)
    assert len(result) == 384


# ---------------------------------------------------------------------------
# store.py
# ---------------------------------------------------------------------------

def test_store_save_and_search():
    """Saving a query then searching with the same text yields score > 0.8."""
    cid = _unique_id("store_hit")
    save("What is machine learning?", "It is a branch of AI.", cid)
    results = search("What is machine learning?", cid, top_k=1)
    assert len(results) == 1
    assert results[0]["score"] > 0.8
    assert results[0]["response"] == "It is a branch of AI."


def test_store_miss():
    """Searching a brand-new customer collection returns an empty list."""
    cid = _unique_id("store_miss")
    results = search("completely unrelated query", cid)
    assert results == []


# ---------------------------------------------------------------------------
# threshold.py
# ---------------------------------------------------------------------------

def test_threshold_default():
    """A brand-new customer gets the default threshold of 0.85."""
    cid = _unique_id("thresh_default")
    assert threshold_get(cid) == pytest.approx(0.85)


def test_threshold_correct_feedback():
    """One correct hit lowers the threshold by 0.005 (0.85 → 0.845)."""
    cid = _unique_id("thresh_correct")
    new = threshold_update(cid, was_correct=True)
    assert new == pytest.approx(0.845)


def test_threshold_wrong_feedback():
    """One wrong hit raises the threshold by 0.020 (0.85 → 0.870)."""
    cid = _unique_id("thresh_wrong")
    new = threshold_update(cid, was_correct=False)
    assert new == pytest.approx(0.870)


def test_threshold_clamp_lower():
    """Threshold never falls below 0.70 no matter how many correct feedbacks."""
    cid = _unique_id("thresh_clamp_low")
    for _ in range(100):
        threshold_update(cid, was_correct=True)
    assert threshold_get(cid) >= 0.70


def test_threshold_clamp_upper():
    """Threshold never exceeds 0.98 no matter how many wrong feedbacks."""
    cid = _unique_id("thresh_clamp_high")
    for _ in range(100):
        threshold_update(cid, was_correct=False)
    assert threshold_get(cid) <= 0.98


# ---------------------------------------------------------------------------
# cache/__init__.py — public API
# ---------------------------------------------------------------------------

def test_cache_get_hit():
    """cache_get returns a hit dict when a matching entry is in the store."""
    cid = _unique_id("cache_hit")
    cache_store("What is a neural network?", "It is a model inspired by brains.", cid)
    result = cache_get("What is a neural network?", cid)
    assert result is not None
    assert result["hit"] is True
    assert "response" in result
    assert "score" in result


def test_cache_get_miss():
    """cache_get returns None for a query with no cached similar entry."""
    cid = _unique_id("cache_miss")
    result = cache_get("a completely novel unseen query xyz", cid)
    assert result is None


# ---------------------------------------------------------------------------
# compress() integration
# ---------------------------------------------------------------------------

def test_compress_cache_hit_on_second_call():
    """Second compress() call with the same query returns cache_hit=True."""
    cid = _unique_id("compress_cache")
    messages = [
        {"role": "user", "content": "Explain transformer architecture briefly."},
        {"role": "assistant", "content": "Transformers use self-attention to process sequences in parallel."},
        {"role": "user", "content": "What is tokenization?"},
    ]

    result1 = compress(messages, max_tokens=10000, use_cache=True, customer_id=cid)
    assert result1.cache_hit is False

    result2 = compress(messages, max_tokens=10000, use_cache=True, customer_id=cid)
    assert result2.cache_hit is True
    assert result2.saved_tokens > 0


def test_compress_no_cache_unchanged():
    """compress() without use_cache works exactly as before; cache_hit defaults to False."""
    messages = [
        {"role": "user", "content": "What is a token?"},
        {"role": "assistant", "content": "A token is roughly 4 characters."},
    ]
    result = compress(messages, max_tokens=10000)
    assert result.cache_hit is False
    assert isinstance(result.messages, list)
    assert isinstance(result.saved_tokens, int)
    assert isinstance(result.estimated_savings_usd, float)
