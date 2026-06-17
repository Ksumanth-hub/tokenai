"""Persist and search query-response pairs by vector similarity using ChromaDB."""
from __future__ import annotations

import os
import uuid

import chromadb

from tokenai.cache import embedder

_CACHE_DIR = ".tokenai_cache/chroma"

_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    """Return a singleton ChromaDB persistent client, creating the directory if needed."""
    global _client
    if _client is None:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=_CACHE_DIR)
    return _client


def _collection_name(customer_id: str) -> str:
    return f"tokenai_{customer_id}"


def save(query: str, response: str, customer_id: str) -> None:
    """Store a query-response pair for *customer_id*.

    Auto-creates the collection on first call for a new customer.
    """
    client = _get_client()
    collection = client.get_or_create_collection(
        name=_collection_name(customer_id),
        metadata={"hnsw:space": "cosine"},
    )
    vector = embedder.embed(query)
    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[vector],
        documents=[query],
        metadatas=[{"response": response}],
    )


def search(query: str, customer_id: str, top_k: int = 1) -> list[dict]:
    """Return the top-k most similar cached entries for *query*.

    Each result is ``{"query": str, "response": str, "score": float}`` where
    score is cosine similarity (0.0 – 1.0, higher = more similar).
    Returns ``[]`` if the collection does not exist or is empty.
    """
    client = _get_client()
    name = _collection_name(customer_id)
    try:
        collection = client.get_collection(name)
    except Exception:
        return []

    count = collection.count()
    if count == 0:
        return []

    vector = embedder.embed(query)
    results = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, count),
    )

    output: list[dict] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        score = max(0.0, 1.0 - dist)  # cosine similarity = 1 - cosine distance
        output.append({"query": doc, "response": meta["response"], "score": score})

    return output
