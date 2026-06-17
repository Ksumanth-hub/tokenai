"""FastAPI dependency for x-api-key header authentication."""
from __future__ import annotations

from fastapi import Header, HTTPException

from tokenai.api.config import settings


def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Validate the x-api-key request header.

    Accepts any key listed in TOKENAI_API_KEYS plus the dev key.
    Returns the key itself as the caller's customer_id.
    Raises HTTP 401 for unrecognised keys.
    """
    if not settings.is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return x_api_key
