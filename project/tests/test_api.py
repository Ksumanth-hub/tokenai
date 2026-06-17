"""REST API tests using FastAPI TestClient."""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from tokenai.api.server import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dev_headers():
    return {"x-api-key": "dev-key-local"}


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# 1. Root endpoint
# ---------------------------------------------------------------------------

def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "tokenai"


# ---------------------------------------------------------------------------
# 2. Health endpoint
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["mcp_available"] is True


# ---------------------------------------------------------------------------
# 3. Missing API key → 422
# ---------------------------------------------------------------------------

def test_missing_api_key(client):
    r = client.post("/cache/get", json={"query": "test"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 4. Invalid API key → 401
# ---------------------------------------------------------------------------

def test_invalid_api_key(client):
    r = client.post(
        "/cache/get",
        headers={"x-api-key": "bad-key"},
        json={"query": "test"},
    )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# 5. Dev key accepted → 200
# ---------------------------------------------------------------------------

def test_dev_key_accepted(client, dev_headers):
    r = client.post("/cache/get", headers=dev_headers, json={"query": "hello"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# 6. Store then get → hit
# ---------------------------------------------------------------------------

def test_cache_store_then_get_hit(client, dev_headers):
    cid = _uid()
    query = f"unique test query {cid}"
    response_text = "This is the answer to that unique query."

    store_r = client.post(
        "/cache/store",
        headers=dev_headers,
        json={"query": query, "response": response_text, "customer_id": cid},
    )
    assert store_r.status_code == 200
    assert store_r.json()["stored"] is True

    get_r = client.post(
        "/cache/get",
        headers=dev_headers,
        json={"query": query, "customer_id": cid},
    )
    assert get_r.status_code == 200
    data = get_r.json()
    assert data["hit"] is True
    assert data["score"] > 0.8


# ---------------------------------------------------------------------------
# 7. Cache get miss
# ---------------------------------------------------------------------------

def test_cache_get_miss(client, dev_headers):
    cid = _uid()
    r = client.post(
        "/cache/get",
        headers=dev_headers,
        json={"query": f"completely unseen query {cid}", "customer_id": cid},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["hit"] is False
    assert data["response"] is None


# ---------------------------------------------------------------------------
# 8. Feedback correct → threshold decreases
# ---------------------------------------------------------------------------

def test_cache_feedback_correct(client, dev_headers):
    cid = _uid()
    query = "What is the meaning of life?"

    client.post(
        "/cache/store",
        headers=dev_headers,
        json={"query": query, "response": "42.", "customer_id": cid},
    )

    r = client.post(
        "/cache/feedback",
        headers=dev_headers,
        json={"query": query, "was_correct": True, "customer_id": cid},
    )
    assert r.status_code == 200
    assert r.json()["new_threshold"] < 0.85  # 0.85 - 0.005 = 0.845


# ---------------------------------------------------------------------------
# 9. Feedback wrong → threshold increases
# ---------------------------------------------------------------------------

def test_cache_feedback_wrong(client, dev_headers):
    cid = _uid()
    query = "What is the speed of light?"

    client.post(
        "/cache/store",
        headers=dev_headers,
        json={"query": query, "response": "~3×10⁸ m/s.", "customer_id": cid},
    )

    r = client.post(
        "/cache/feedback",
        headers=dev_headers,
        json={"query": query, "was_correct": False, "customer_id": cid},
    )
    assert r.status_code == 200
    assert r.json()["new_threshold"] > 0.85  # 0.85 + 0.020 = 0.870


# ---------------------------------------------------------------------------
# 10. Stats endpoint
# ---------------------------------------------------------------------------

def test_cache_stats(client, dev_headers):
    r = client.get("/cache/stats", headers=dev_headers)
    assert r.status_code == 200
    data = r.json()
    assert "threshold" in data
    assert "total_hits" in data
    assert "accuracy" in data
    assert 0.70 <= data["threshold"] <= 0.98


# ---------------------------------------------------------------------------
# 11. Compress endpoint
# ---------------------------------------------------------------------------

def test_compress_endpoint(client, dev_headers):
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "ML learns patterns from data to make predictions."},
        {"role": "user", "content": "Can you give an example?"},
    ]
    r = client.post(
        "/compress",
        headers=dev_headers,
        json={"messages": messages, "max_tokens": 10000},
    )
    assert r.status_code == 200
    data = r.json()
    assert "messages" in data
    assert "saved_tokens" in data
    assert "saved_usd" in data
    assert "compression_ratio" in data
    assert data["saved_tokens"] >= 0


# ---------------------------------------------------------------------------
# 12. Compress: cache hit on second call
# ---------------------------------------------------------------------------

def test_compress_cache_hit_second_call(client, dev_headers):
    cid = _uid()
    messages = [
        {"role": "user", "content": "What is deep learning?"},
        {"role": "assistant", "content": "Deep learning uses multi-layer neural networks."},
        {"role": "user", "content": "How is it different from ML?"},
    ]
    payload = {
        "messages": messages,
        "max_tokens": 10000,
        "use_cache": True,
        "customer_id": cid,
    }

    r1 = client.post("/compress", headers=dev_headers, json=payload)
    assert r1.status_code == 200
    assert r1.json()["cache_hit"] is False

    r2 = client.post("/compress", headers=dev_headers, json=payload)
    assert r2.status_code == 200
    assert r2.json()["cache_hit"] is True
    assert r2.json()["saved_tokens"] > 0


# ---------------------------------------------------------------------------
# 13. Compress: use_cache=True without customer_id → 400
# ---------------------------------------------------------------------------

def test_compress_missing_customer_id(client, dev_headers):
    r = client.post(
        "/compress",
        headers=dev_headers,
        json={
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10000,
            "use_cache": True,
            # customer_id intentionally omitted
        },
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 14. latency_ms present and positive on all timed responses
# ---------------------------------------------------------------------------

def test_latency_ms_present(client, dev_headers):
    cid = _uid()

    endpoints = [
        ("/cache/get", "POST", {"query": "latency test", "customer_id": cid}),
        ("/cache/store", "POST", {"query": "latency test", "response": "answer", "customer_id": cid}),
        ("/cache/feedback", "POST", {"query": "latency test", "was_correct": True, "customer_id": cid}),
        ("/compress", "POST", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10000}),
    ]

    for path, method, payload in endpoints:
        if method == "POST":
            r = client.post(path, headers=dev_headers, json=payload)
        else:
            r = client.get(path, headers=dev_headers)
        assert r.status_code == 200, f"{path} failed: {r.text}"
        data = r.json()
        if "latency_ms" in data:
            assert data["latency_ms"] > 0, f"{path} latency_ms should be > 0"
