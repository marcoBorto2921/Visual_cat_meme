"""Giphy API meme fetcher backend."""

import os
import random
from typing import Optional
from urllib.parse import urlencode

import cv2
import numpy as np
import requests

from src.meme.fetcher import MemeFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)

GIPHY_API_BASE = "https://api.giphy.com/v1/gifs/search"


class GiphyMemeFetcher(MemeFetcher):
    """Fetches cat GIFs from the Giphy search API.

    Requires a free GIPHY_API_KEY env var or config key.

    Args:
        config: Full config dict from config.yaml.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._api_key = os.environ.get(
            "GIPHY_API_KEY",
            config.get("meme", {}).get("giphy_api_key", ""),
        )
        if not self._api_key:
            logger.warning("GIPHY_API_KEY not set — Giphy backend will fail")

    def fetch(self, pose_label: str) -> Optional[np.ndarray]:
        """Fetch a cat GIF from Giphy matching the pose label.

        Args:
            pose_label: One of the 7 pose class strings.

        Returns:
            BGR numpy array (first frame of GIF) or None on failure.
        """
        tags = self._tags_for_pose(pose_label)
        query = random.choice(tags)
        params = urlencode({
            "api_key": self._api_key,
            "q": query,
            "limit": 10,
            "rating": "g",
            "lang": "en",
        })
        try:
            resp = requests.get(f"{GIPHY_API_BASE}?{params}", timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            gifs = data.get("data", [])
            if not gifs:
                logger.warning("No Giphy results for query '%s'", query)
                return None
            gif = random.choice(gifs)
            gif_url = gif["images"]["fixed_height"]["url"]
            img_resp = requests.get(gif_url, timeout=self._timeout)
            img_resp.raise_for_status()
            img_array = np.frombuffer(img_resp.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode GIF frame")
            logger.debug("Fetched Giphy GIF for query '%s'", query)
            return img
        except Exception as exc:
            logger.warning("Giphy fetch failed: %s", exc)
            return None
