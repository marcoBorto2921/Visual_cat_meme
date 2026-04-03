"""Abstract base class for meme fetcher backends."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class MemeFetcher(ABC):
    """Abstract meme fetcher. Each backend implements fetch().

    Args:
        config: Full config dict from config.yaml.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._timeout = config["meme"]["request_timeout_seconds"]

    @abstractmethod
    def fetch(self, pose_label: str) -> Optional[np.ndarray]:
        """Fetch a meme image for the given pose label.

        Args:
            pose_label: One of the 7 pose class strings.

        Returns:
            BGR numpy array of the meme image, or None on failure.
        """

    def _tags_for_pose(self, pose_label: str) -> list[str]:
        """Return list of search tags for the given pose label.

        Args:
            pose_label: Pose class string.

        Returns:
            List of tag strings from config.
        """
        return self._config.get("pose_to_tags", {}).get(pose_label, ["cat"])
