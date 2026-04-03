"""CatPose Meme Machine — entry point.

Wires PoseDetector, RuleBasedClassifier, MemeFetcher, TTLCache, and Renderer
into a real-time webcam loop.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from src.display.renderer import Renderer
from src.meme.cataas import CataasMemeFetcher
from src.meme.fetcher import MemeFetcher
from src.meme.giphy import GiphyMemeFetcher
from src.meme.reddit import RedditMemeFetcher
from src.pose.classifier import RuleBasedClassifier
from src.pose.detector import PoseDetector
from src.utils.cache import TTLCache
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load and return the YAML config dict.

    Args:
        path: Path to config.yaml relative to project root.

    Returns:
        Config dict.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_fetcher(config: dict) -> MemeFetcher:
    """Instantiate the configured meme fetcher backend.

    Args:
        config: Full config dict.

    Returns:
        MemeFetcher instance.
    """
    backend = config["meme"]["backend"]
    if backend == "reddit":
        return RedditMemeFetcher(config)
    elif backend == "giphy":
        return GiphyMemeFetcher(config)
    else:
        return CataasMemeFetcher(config)


def save_screenshot(frame: np.ndarray) -> None:
    """Save the webcam frame to screenshots/ with a timestamp filename."""
    Path("screenshots").mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"screenshots/catpose_{ts}.png"
    cv2.imwrite(path, frame)
    logger.info("Screenshot saved: %s", path)


def main() -> None:
    """Run the CatPose Meme Machine main loop."""
    config = load_config()

    cam_cfg = config["camera"]
    cap = cv2.VideoCapture(cam_cfg["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

    if not cap.isOpened():
        logger.error("Cannot open camera index %d", cam_cfg["index"])
        sys.exit(1)

    pose_cfg = config["pose"]
    detector = PoseDetector(
        visibility_threshold=pose_cfg["visibility_threshold"],
        model_complexity=1,
    )
    classifier = RuleBasedClassifier(
        visibility_threshold=pose_cfg["visibility_threshold"],
        smoothing_window=pose_cfg["smoothing_window"],
    )
    fetcher: MemeFetcher = build_fetcher(config)
    cache = TTLCache(ttl_seconds=config["meme"]["cache_ttl_seconds"])
    renderer = Renderer(config)

    current_meme: Optional[np.ndarray] = None
    last_label: str = ""
    fps: float = 0.0
    tick_freq = cv2.getTickFrequency()
    last_tick = cv2.getTickCount()

    # Pre-built fetcher instances for runtime backend switching
    fetchers: dict[str, tuple[str, MemeFetcher]] = {
        "1": ("cataas", CataasMemeFetcher(config)),
        "2": ("reddit", RedditMemeFetcher(config)),
        "3": ("giphy", GiphyMemeFetcher(config)),
    }

    logger.info("CatPose Meme Machine started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Empty frame — skipping")
            continue

        annotated, landmarks = detector.process(frame)
        label = classifier.classify(landmarks)

        # Fetch meme when label changes or cache expired
        cached = cache.get(label)
        if cached is not None:
            current_meme = cached
        elif label != last_label or current_meme is None:
            meme = fetcher.fetch(label)
            if meme is not None:
                cache.set(label, meme)
                current_meme = meme
            # If meme is None, keep previous meme displayed

        last_label = label

        # FPS calculation
        now = cv2.getTickCount()
        fps = tick_freq / max(now - last_tick, 1)
        last_tick = now

        renderer.render(annotated, label, current_meme, fps, classifier.active_rule)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("Quit key pressed")
            break
        elif key == ord("d"):
            renderer.toggle_debug()
        elif key == ord("r"):
            cache.invalidate(label)
            logger.info("Cache invalidated for '%s'", label)
        elif key == ord("s"):
            save_screenshot(annotated)
        elif chr(key) in fetchers:
            name, new_fetcher = fetchers[chr(key)]
            fetcher = new_fetcher
            cache.clear()
            logger.info("Switched to %s backend", name)

    cap.release()
    detector.close()
    renderer.close()


if __name__ == "__main__":
    main()
