"""Reddit public JSON API meme fetcher backend."""

import random
import time
from typing import Optional

import cv2
import numpy as np
import requests

from src.meme.fetcher import MemeFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)

REDDIT_BASE = "https://www.reddit.com/r/catmemes"
HEADERS = {"User-Agent": "catpose-meme-machine/1.0"}


class RedditMemeFetcher(MemeFetcher):
    """Fetches cat memes from r/catmemes public JSON endpoint.

    No API key required. Rate-limited — enforces 2-second minimum delay.

    Args:
        config: Full config dict from config.yaml.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._last_fetch: float = 0.0
        self._posts: list[str] = []

    def fetch(self, pose_label: str) -> Optional[np.ndarray]:
        """Fetch a cat meme from Reddit.

        Args:
            pose_label: One of the 7 pose class strings (unused — Reddit has no tag search).

        Returns:
            BGR numpy array or None on failure.
        """
        elapsed = time.monotonic() - self._last_fetch
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)

        if not self._posts:
            self._posts = self._load_posts()

        if not self._posts:
            return None

        url = random.choice(self._posts)
        self._last_fetch = time.monotonic()

        try:
            resp = requests.get(url, timeout=self._timeout, headers=HEADERS)
            resp.raise_for_status()
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            logger.debug("Fetched Reddit meme from %s", url)
            return img
        except Exception as exc:
            logger.warning("Reddit image fetch failed: %s", exc)
            return None

    def _load_posts(self) -> list[str]:
        """Load image URLs from the r/catmemes hot feed."""
        try:
            resp = requests.get(
                f"{REDDIT_BASE}/hot.json?limit=25",
                timeout=self._timeout,
                headers=HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()
            urls = []
            for post in data.get("data", {}).get("children", []):
                pd = post.get("data", {})
                url = pd.get("url", "")
                if url.lower().endswith((".jpg", ".jpeg", ".png")):
                    urls.append(url)
            logger.info("Loaded %d Reddit image posts", len(urls))
            return urls
        except Exception as exc:
            logger.error("Failed to load Reddit posts: %s", exc)
            return []
