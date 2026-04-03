"""CATAAS (Cat as a Service) meme fetcher backend."""

import random
from typing import Optional
from urllib.parse import quote

import cv2
import numpy as np
import requests

from src.meme.fetcher import MemeFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)

CATAAS_BASE = "https://cataas.com"


class CataasMemeFetcher(MemeFetcher):
    """Fetches cat images from cataas.com using pose-based tags.

    No API key required. Falls back to a plain cat image on tag failure.

    Args:
        config: Full config dict from config.yaml.
    """

    def fetch(self, pose_label: str) -> Optional[np.ndarray]:
        """Fetch a cat meme from CATAAS matching the pose label.

        Args:
            pose_label: One of the 7 pose class strings.

        Returns:
            BGR numpy array or None on failure.
        """
        tags = self._tags_for_pose(pose_label)
        tag = random.choice(tags)
        # CATAAS accepts text overlay on the image
        encoded_tag = quote(tag)
        url = f"{CATAAS_BASE}/cat/says/{encoded_tag}?width=400&height=400&fontSize=30&fontColor=white"
        try:
            resp = requests.get(url, timeout=self._timeout)
            resp.raise_for_status()
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            logger.debug("Fetched CATAAS image for tag '%s'", tag)
            return img
        except Exception as exc:
            logger.warning("CATAAS fetch failed for tag '%s': %s", tag, exc)
            # Try plain cat fallback
            return self._fetch_plain_cat()

    def _fetch_plain_cat(self) -> Optional[np.ndarray]:
        """Fetch a plain cat image as fallback."""
        try:
            resp = requests.get(f"{CATAAS_BASE}/cat?width=400&height=400", timeout=self._timeout)
            resp.raise_for_status()
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as exc:
            logger.error("CATAAS fallback also failed: %s", exc)
            return None
