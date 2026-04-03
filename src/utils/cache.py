"""TTL-based in-memory cache for meme images."""

import time
from typing import Any, Optional


class TTLCache:
    """Simple in-memory cache with per-entry TTL expiry.

    Args:
        ttl_seconds: Time-to-live in seconds for each cached entry.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if present and not expired, else None.

        Args:
            key: Cache key.

        Returns:
            Cached value or None.
        """
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value under key with current timestamp.

        Args:
            key: Cache key.
            value: Value to store.
        """
        self._store[key] = (value, time.monotonic())

    def invalidate(self, key: str) -> None:
        """Remove a key from the cache regardless of TTL.

        Args:
            key: Cache key to remove.
        """
        self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()
