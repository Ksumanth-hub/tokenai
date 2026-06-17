"""Central configuration — single source of truth for both API and MCP server."""
from __future__ import annotations

import os

from dotenv import load_dotenv


class Settings:
    """Application settings populated from environment variables and .env file."""

    def __init__(self) -> None:
        self.dev_key: str = "dev-key-local"
        self.api_keys: list[str] = []
        self.cache_dir: str = ".tokenai_cache"
        self.version: str = "0.2.0"
        self.port: int = 8000

    def load(self) -> "Settings":
        """Read .env then override from environment. Safe to call multiple times."""
        load_dotenv()
        raw = os.getenv("TOKENAI_API_KEYS", "")
        self.api_keys = [k.strip() for k in raw.split(",") if k.strip()]
        self.dev_key = os.getenv("TOKENAI_DEV_KEY", "dev-key-local")
        try:
            self.port = int(os.getenv("PORT", "8000"))
        except ValueError:
            self.port = 8000

        if not self.api_keys:
            print(
                "WARNING: TOKENAI_API_KEYS is not set — "
                "only the dev key will be accepted. "
                "Set TOKENAI_API_KEYS in your .env for production."
            )
        return self

    def is_valid_key(self, key: str) -> bool:
        """Return True if *key* is the dev key or a registered customer key."""
        return key == self.dev_key or key in self.api_keys


settings = Settings().load()
