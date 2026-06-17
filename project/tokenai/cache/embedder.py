"""Convert text to dense vectors using sentence-transformers."""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIMENSION = 384

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load model once on CPU, cache in module-level variable."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _model


def embed(text: str) -> list[float]:
    """Return a 384-dim float vector for *text*.

    Returns a zero vector for empty input without raising.
    """
    if not text or not text.strip():
        return [0.0] * _DIMENSION
    return _get_model().encode(text).tolist()


def get_dimension() -> int:
    """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
    return _DIMENSION
