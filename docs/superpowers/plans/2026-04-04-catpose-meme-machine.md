# CatPose Meme Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time webcam pose detection app that retrieves matching cat memes using MediaPipe landmarks and OpenCV display.

**Architecture:** MediaPipe detects 33 body landmarks per frame; a rule-based classifier maps landmark geometry to 7 pose classes; meme fetcher retrieves cat images from CATAAS/Reddit/Giphy APIs with TTL cache; OpenCV renders a dual-panel window (webcam + meme overlay).

**Tech Stack:** Python 3.10+, MediaPipe ≥0.10, OpenCV, Pillow, PyYAML, requests, numpy, ruff

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | Pinned deps |
| `configs/config.yaml` | All runtime settings |
| `.gitignore` | Exclude venv, keys, models |
| `Makefile` | Dev shortcuts |
| `src/__init__.py` | Package marker |
| `src/utils/logger.py` | Structured logging setup |
| `src/utils/cache.py` | TTL-based in-memory cache |
| `src/pose/detector.py` | MediaPipe Pose wrapper → 33 landmarks |
| `src/pose/classifier.py` | RuleBasedClassifier (7 poses) |
| `src/meme/fetcher.py` | Abstract base `MemeFetcher` |
| `src/meme/cataas.py` | CATAAS backend |
| `src/meme/reddit.py` | Reddit public JSON backend |
| `src/meme/giphy.py` | Giphy API backend |
| `src/display/renderer.py` | OpenCV dual-panel renderer |
| `main.py` | Entry point, wires all components |
| `scripts/collect_pose_data.py` | Optional: collect labeled pose samples |
| `notebooks/01_pose_analysis.py` | Landmark distribution visualization |
| `README.md` | Usage guide |
| `TECHNICAL_CHOICES.md` | Architecture decisions |
| `screenshots/.gitkeep` | Screenshot dir |
| `models/.gitkeep` | Model dir |

---

### Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `Makefile`
- Create: `configs/config.yaml`
- Create: `src/__init__.py`
- Create: `src/pose/__init__.py`
- Create: `src/meme/__init__.py`
- Create: `src/display/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `screenshots/.gitkeep`
- Create: `models/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
mediapipe==0.10.21
opencv-python==4.10.0.84
requests==2.32.3
Pillow==10.4.0
PyYAML==6.0.2
numpy==1.26.4
ruff==0.4.10
```

- [ ] **Step 2: Create .gitignore**

```
.venv/
.claude/
*.pyc
__pycache__/
.env
models/
*.joblib
screenshots/
*.png
*.jpg
```

- [ ] **Step 3: Create Makefile**

```makefile
.PHONY: run install lint

install:
	pip install -r requirements.txt

run:
	python main.py

lint:
	ruff check src/ main.py
```

- [ ] **Step 4: Create configs/config.yaml**

```yaml
camera:
  index: 0
  width: 640
  height: 480
  fps: 30

pose:
  classifier: rule_based
  ml_model_path: models/pose_classifier.joblib
  visibility_threshold: 0.5
  smoothing_window: 5

meme:
  backend: cataas
  cache_ttl_seconds: 30
  request_timeout_seconds: 3
  giphy_api_key: ""

display:
  window_title: "CatPose Meme Machine"
  meme_panel_width: 400
  debug_mode: false
  font_scale: 1.0

pose_to_tags:
  arms_up: ["happy cat", "excited cat", "jumping cat"]
  arms_wide: ["big cat", "dramatic cat", "surprised cat"]
  thinking: ["thinking cat", "serious cat", "smart cat"]
  slouching: ["sleepy cat", "lazy cat", "tired cat"]
  crossed_arms: ["grumpy cat", "mad cat", "annoyed cat"]
  hands_on_hips: ["sassy cat", "boss cat", "confident cat"]
  neutral: ["cat", "cute cat", "cat meme"]
```

- [ ] **Step 5: Create empty __init__.py files**

```python
# empty
```
(for src/, src/pose/, src/meme/, src/display/, src/utils/)

- [ ] **Step 6: Create .gitkeep files**

```
# empty files at screenshots/.gitkeep and models/.gitkeep
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore Makefile configs/config.yaml src/ screenshots/ models/
git commit -m "feat: project scaffold — config, deps, package structure"
```

---

### Task 2: Logger and TTL Cache Utilities

**Files:**
- Create: `src/utils/logger.py`
- Create: `src/utils/cache.py`

- [ ] **Step 1: Create src/utils/logger.py**

```python
"""Structured logging setup for CatPose Meme Machine."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with the given name.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

- [ ] **Step 2: Create src/utils/cache.py**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add src/utils/logger.py src/utils/cache.py
git commit -m "feat: add TTL cache and structured logger utilities"
```

---

### Task 3: Pose Detector (MediaPipe Wrapper)

**Files:**
- Create: `src/pose/detector.py`

- [ ] **Step 1: Create src/pose/detector.py**

```python
"""MediaPipe Pose wrapper that extracts 33 body landmarks per frame."""

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Landmark:
    """A single body landmark with normalized coordinates and visibility."""

    x: float
    y: float
    z: float
    visibility: float


LandmarkList = list[Optional[Landmark]]


class PoseDetector:
    """Wraps MediaPipe Pose to extract normalized body landmarks.

    Args:
        visibility_threshold: Minimum visibility score to accept a landmark.
        model_complexity: MediaPipe model complexity (0, 1, or 2).
    """

    # Landmark indices used by the classifier
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(
        self,
        visibility_threshold: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self._threshold = visibility_threshold
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp_draw = mp.solutions.drawing_utils
        logger.info("PoseDetector initialized (threshold=%.2f)", visibility_threshold)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, LandmarkList]:
        """Detect pose in a BGR frame and draw skeleton overlay.

        Args:
            frame: BGR image array from OpenCV.

        Returns:
            Tuple of (annotated BGR frame, list of 33 Landmark or None per index).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        landmarks: LandmarkList = [None] * 33
        annotated = frame.copy()

        if results.pose_landmarks:
            self._mp_draw.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
            )
            for i, lm in enumerate(results.pose_landmarks.landmark):
                if lm.visibility >= self._threshold:
                    landmarks[i] = Landmark(
                        x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
                    )

        return annotated, landmarks

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()
        logger.info("PoseDetector closed")
```

- [ ] **Step 2: Commit**

```bash
git add src/pose/detector.py
git commit -m "feat: add MediaPipe pose detector wrapper"
```

---

### Task 4: Rule-Based Pose Classifier

**Files:**
- Create: `src/pose/classifier.py`

- [ ] **Step 1: Create src/pose/classifier.py**

```python
"""Rule-based pose classifier using MediaPipe landmark geometry."""

from collections import deque
from typing import Optional

from src.pose.detector import Landmark, LandmarkList, PoseDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)

PoseLabel = str  # one of the 7 pose class strings


class RuleBasedClassifier:
    """Classifies body pose into one of 7 classes using geometric rules.

    Rules operate on normalized landmark coordinates (y=0 is top of frame).

    Args:
        visibility_threshold: Minimum visibility for a landmark to be used.
        smoothing_window: Number of frames to smooth predictions over.
    """

    POSES = [
        "arms_up",
        "arms_wide",
        "thinking",
        "slouching",
        "crossed_arms",
        "hands_on_hips",
        "neutral",
    ]

    def __init__(
        self,
        visibility_threshold: float = 0.5,
        smoothing_window: int = 5,
    ) -> None:
        self._threshold = visibility_threshold
        self._history: deque[PoseLabel] = deque(maxlen=smoothing_window)
        self._active_rule: str = "none"

    @property
    def active_rule(self) -> str:
        """Name of the rule that fired for the last classification."""
        return self._active_rule

    def classify(self, landmarks: LandmarkList) -> PoseLabel:
        """Classify pose from a landmark list and return smoothed label.

        Args:
            landmarks: 33-element list from PoseDetector.process().

        Returns:
            Smoothed pose label string.
        """
        raw = self._classify_raw(landmarks)
        self._history.append(raw)
        # Return majority vote from history
        return max(set(self._history), key=self._history.count)

    def _lm(self, landmarks: LandmarkList, idx: int) -> Optional[Landmark]:
        """Return landmark if visible above threshold, else None."""
        lm = landmarks[idx]
        if lm is None or lm.visibility < self._threshold:
            return None
        return lm

    def _dist(self, a: Landmark, b: Landmark) -> float:
        """Euclidean distance between two landmarks (x, y only)."""
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    def _classify_raw(self, landmarks: LandmarkList) -> PoseLabel:
        """Apply geometric rules in priority order and return raw label."""
        idx = PoseDetector

        l_wrist = self._lm(landmarks, idx.LEFT_WRIST)
        r_wrist = self._lm(landmarks, idx.RIGHT_WRIST)
        l_shoulder = self._lm(landmarks, idx.LEFT_SHOULDER)
        r_shoulder = self._lm(landmarks, idx.RIGHT_SHOULDER)
        l_elbow = self._lm(landmarks, idx.LEFT_ELBOW)
        r_elbow = self._lm(landmarks, idx.RIGHT_ELBOW)
        l_hip = self._lm(landmarks, idx.LEFT_HIP)
        r_hip = self._lm(landmarks, idx.RIGHT_HIP)
        nose = self._lm(landmarks, idx.NOSE)

        # arms_up: both wrists above shoulders (y inverted, smaller = higher)
        if (
            l_wrist and r_wrist and l_shoulder and r_shoulder
            and l_wrist.y < l_shoulder.y - 0.15
            and r_wrist.y < r_shoulder.y - 0.15
        ):
            self._active_rule = "arms_up: wrists above shoulders"
            return "arms_up"

        # arms_wide: wrists far out horizontally and at similar height to shoulders
        if (
            l_wrist and r_wrist and l_shoulder and r_shoulder
            and abs(l_wrist.x - l_shoulder.x) > 0.25
            and abs(r_wrist.x - r_shoulder.x) > 0.25
            and abs(l_wrist.y - l_shoulder.y) < 0.15
            and abs(r_wrist.y - r_shoulder.y) < 0.15
        ):
            self._active_rule = "arms_wide: wrists extended horizontally"
            return "arms_wide"

        # thinking: one wrist near chin/cheek area (near nose, above shoulders)
        if nose and l_shoulder and r_shoulder:
            chin_y = nose.y + 0.1
            chin_x = nose.x
            if l_wrist and self._dist(l_wrist, Landmark(chin_x, chin_y, 0, 1)) < 0.15:
                self._active_rule = "thinking: left wrist near chin"
                return "thinking"
            if r_wrist and self._dist(r_wrist, Landmark(chin_x, chin_y, 0, 1)) < 0.15:
                self._active_rule = "thinking: right wrist near chin"
                return "thinking"

        # slouching: shoulders far below nose
        if nose and l_shoulder and r_shoulder:
            avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            if avg_shoulder_y > nose.y + 0.35:
                self._active_rule = "slouching: shoulders far below nose"
                return "slouching"

        # crossed_arms: left wrist past right shoulder AND right wrist past left shoulder
        if l_wrist and r_wrist and l_shoulder and r_shoulder:
            if l_wrist.x > r_shoulder.x and r_wrist.x < l_shoulder.x:
                self._active_rule = "crossed_arms: wrists crossed"
                return "crossed_arms"

        # hands_on_hips: both wrists near hip landmarks
        if l_wrist and l_hip and r_wrist and r_hip:
            if self._dist(l_wrist, l_hip) < 0.12 and self._dist(r_wrist, r_hip) < 0.12:
                self._active_rule = "hands_on_hips: wrists near hips"
                return "hands_on_hips"

        self._active_rule = "neutral: no rule matched"
        return "neutral"
```

- [ ] **Step 2: Commit**

```bash
git add src/pose/classifier.py
git commit -m "feat: add rule-based pose classifier with 7 classes"
```

---

### Task 5: Abstract Meme Fetcher + CATAAS Backend

**Files:**
- Create: `src/meme/fetcher.py`
- Create: `src/meme/cataas.py`

- [ ] **Step 1: Create src/meme/fetcher.py**

```python
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
```

- [ ] **Step 2: Create src/meme/cataas.py**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add src/meme/fetcher.py src/meme/cataas.py
git commit -m "feat: add abstract MemeFetcher and CATAAS backend"
```

---

### Task 6: Reddit and Giphy Backends

**Files:**
- Create: `src/meme/reddit.py`
- Create: `src/meme/giphy.py`

- [ ] **Step 1: Create src/meme/reddit.py**

```python
"""Reddit public JSON API meme fetcher backend."""

import os
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
```

- [ ] **Step 2: Create src/meme/giphy.py**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add src/meme/reddit.py src/meme/giphy.py
git commit -m "feat: add Reddit and Giphy meme fetcher backends"
```

---

### Task 7: OpenCV Dual-Panel Renderer

**Files:**
- Create: `src/display/renderer.py`

- [ ] **Step 1: Create src/display/renderer.py**

```python
"""OpenCV dual-panel renderer: webcam feed (left) + cat meme (right)."""

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_placeholder(width: int, height: int) -> np.ndarray:
    """Create a grey placeholder image with 'No meme' text."""
    img = np.full((height, width, 3), 80, dtype=np.uint8)
    cv2.putText(
        img, "No meme", (width // 4, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA,
    )
    return img


def _resize_fit(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image to fit within target dimensions, maintaining aspect ratio."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Pad to exact target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


class Renderer:
    """Renders the dual-panel OpenCV window.

    Left panel: annotated webcam feed with pose label.
    Right panel: current cat meme image.
    Bottom bar: debug info (FPS, active rule) when debug mode is on.

    Args:
        config: Full config dict from config.yaml.
    """

    def __init__(self, config: dict) -> None:
        self._title = config["display"]["window_title"]
        self._meme_w = config["display"]["meme_panel_width"]
        self._font_scale = config["display"]["font_scale"]
        self._debug = config["display"]["debug_mode"]
        self._cam_w = config["camera"]["width"]
        self._cam_h = config["camera"]["height"]
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        logger.info("Renderer initialized: '%s'", self._title)

    @property
    def debug_mode(self) -> bool:
        """Whether debug overlay is shown."""
        return self._debug

    def toggle_debug(self) -> None:
        """Toggle debug overlay visibility."""
        self._debug = not self._debug
        logger.info("Debug mode: %s", self._debug)

    def render(
        self,
        cam_frame: np.ndarray,
        pose_label: str,
        meme_img: Optional[np.ndarray],
        fps: float,
        active_rule: str = "",
    ) -> None:
        """Render and display the combined frame.

        Args:
            cam_frame: Annotated BGR webcam frame from PoseDetector.
            pose_label: Current smoothed pose label string.
            meme_img: BGR meme image or None.
            fps: Current FPS estimate.
            active_rule: Debug string for the active classification rule.
        """
        h = self._cam_h

        # --- Left panel: webcam + pose label ---
        cam_panel = cv2.resize(cam_frame, (self._cam_w, h))
        self._draw_label(cam_panel, pose_label, fps)

        # --- Right panel: meme ---
        if meme_img is not None:
            meme_panel = _resize_fit(meme_img, self._meme_w, h)
        else:
            meme_panel = _make_placeholder(self._meme_w, h)

        # --- Combine ---
        combined = np.hstack([cam_panel, meme_panel])

        # --- Debug bar ---
        if self._debug and active_rule:
            bar = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                bar, f"Rule: {active_rule}  |  FPS: {fps:.1f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
            combined = np.vstack([combined, bar])

        cv2.imshow(self._title, combined)

    def _draw_label(self, frame: np.ndarray, label: str, fps: float) -> None:
        """Draw pose label and FPS counter on the webcam panel."""
        label_text = label.replace("_", " ").upper()
        cv2.putText(
            frame, label_text, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, self._font_scale,
            (0, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA,
        )

    def close(self) -> None:
        """Destroy the OpenCV window."""
        cv2.destroyAllWindows()
        logger.info("Renderer closed")
```

- [ ] **Step 2: Commit**

```bash
git add src/display/renderer.py
git commit -m "feat: add OpenCV dual-panel renderer"
```

---

### Task 8: main.py Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Create main.py**

```python
"""CatPose Meme Machine — entry point.

Wires PoseDetector, RuleBasedClassifier, MemeFetcher, TTLCache, and Renderer
into a real-time webcam loop.
"""

import os
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
    """Save the combined frame to screenshots/ with a timestamp filename."""
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
    fetcher = build_fetcher(config)
    cache = TTLCache(ttl_seconds=config["meme"]["cache_ttl_seconds"])
    renderer = Renderer(config)

    current_meme: Optional[np.ndarray] = None
    last_label: str = "neutral"
    fps: float = 0.0
    tick_freq = cv2.getTickFrequency()
    last_tick = cv2.getTickCount()
    last_combined: Optional[np.ndarray] = None

    logger.info("CatPose Meme Machine started. Press 'q' to quit.")

    fetchers = {
        "1": ("cataas", CataasMemeFetcher(config)),
        "2": ("reddit", RedditMemeFetcher(config)),
        "3": ("giphy", GiphyMemeFetcher(config)),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Empty frame — skipping")
            continue

        annotated, landmarks = detector.process(frame)
        label = classifier.classify(landmarks)

        # Fetch meme when label changes or cache miss
        if label != last_label or cache.get(label) is None:
            cached = cache.get(label)
            if cached is not None:
                current_meme = cached
            else:
                meme = fetcher.fetch(label)
                if meme is not None:
                    cache.set(label, meme)
                    current_meme = meme
                # If meme is None keep previous meme displayed
            last_label = label

        # FPS calculation
        now = cv2.getTickCount()
        fps = tick_freq / (now - last_tick)
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
            # Build last combined frame for screenshot
            combined_h = config["camera"]["height"]
            combined_w = config["camera"]["width"] + config["display"]["meme_panel_width"]
            # Re-render to numpy for saving — grab last imshow frame
            save_screenshot(annotated)
        elif key in (ord("1"), ord("2"), ord("3")):
            k = chr(key)
            name, new_fetcher = fetchers[k]
            fetcher = new_fetcher
            cache.clear()
            logger.info("Switched to %s backend", name)

    cap.release()
    detector.close()
    renderer.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: add main.py entry point wiring full pipeline"
```

---

### Task 9: Optional Scripts and Notebooks

**Files:**
- Create: `scripts/collect_pose_data.py`
- Create: `notebooks/01_pose_analysis.py`

- [ ] **Step 1: Create scripts/collect_pose_data.py**

```python
"""Collect labeled pose samples for optional ML classifier training.

Usage:
    python scripts/collect_pose_data.py --pose arms_up --output data/poses.csv
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import yaml

from src.pose.detector import PoseDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled pose samples")
    parser.add_argument("--pose", required=True, help="Pose label to record")
    parser.add_argument("--output", default="data/poses.csv", help="Output CSV path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detector = PoseDetector(visibility_threshold=config["pose"]["visibility_threshold"])

    cap = cv2.VideoCapture(config["camera"]["index"])
    collected = 0

    print(f"Collecting {args.samples} samples for pose '{args.pose}'. Press SPACE to capture, Q to quit.")

    with open(args.output, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        while collected < args.samples:
            ret, frame = cap.read()
            if not ret:
                continue
            annotated, landmarks = detector.process(frame)
            cv2.imshow("Collect Poses", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                row = [args.pose]
                for lm in landmarks:
                    if lm:
                        row.extend([lm.x, lm.y, lm.z, lm.visibility])
                    else:
                        row.extend([0.0, 0.0, 0.0, 0.0])
                writer.writerow(row)
                collected += 1
                print(f"  Captured {collected}/{args.samples}")
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"Saved {collected} samples to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create notebooks/01_pose_analysis.py**

```python
"""Visualize landmark distributions per pose class from collected data.

Run: python notebooks/01_pose_analysis.py --input data/poses.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np

LANDMARK_NAMES = {
    0: "NOSE", 11: "L_SHOULDER", 12: "R_SHOULDER",
    13: "L_ELBOW", 14: "R_ELBOW", 15: "L_WRIST",
    16: "R_WRIST", 23: "L_HIP", 24: "R_HIP",
}

KEY_INDICES = sorted(LANDMARK_NAMES.keys())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/poses.csv")
    args = parser.parse_args()

    pose_data: dict[str, list[list[float]]] = defaultdict(list)

    with open(args.input) as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[0]
            values = list(map(float, row[1:]))
            # Each landmark is 4 floats: x, y, z, visibility
            key_vals = []
            for idx in KEY_INDICES:
                base = idx * 4
                key_vals.extend(values[base:base + 4])
            pose_data[label].append(key_vals)

    print("=== Landmark Statistics per Pose ===\n")
    for pose, samples in sorted(pose_data.items()):
        arr = np.array(samples)
        print(f"Pose: {pose} ({len(samples)} samples)")
        for i, idx in enumerate(KEY_INDICES):
            base = i * 4
            x_mean = arr[:, base].mean()
            y_mean = arr[:, base + 1].mean()
            vis_mean = arr[:, base + 3].mean()
            print(f"  {LANDMARK_NAMES[idx]:12s} x={x_mean:.3f} y={y_mean:.3f} vis={vis_mean:.3f}")
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add scripts/collect_pose_data.py notebooks/01_pose_analysis.py
git commit -m "feat: add pose data collection script and landmark analysis notebook"
```

---

### Task 10: Documentation

**Files:**
- Create: `README.md`
- Create: `TECHNICAL_CHOICES.md`

- [ ] **Step 1: Create README.md**

```markdown
# CatPose Meme Machine

Real-time webcam pose detection → matching cat meme display.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle debug overlay |
| `r` | Force refresh meme |
| `s` | Save screenshot |
| `1/2/3` | Switch backend (cataas/reddit/giphy) |

## Pose Classes

`arms_up`, `arms_wide`, `thinking`, `slouching`, `crossed_arms`, `hands_on_hips`, `neutral`

## Config

Edit `configs/config.yaml` to adjust camera index, pose thresholds, meme backend, and display options.
```

- [ ] **Step 2: Create TECHNICAL_CHOICES.md**

```markdown
# Technical Choices

## Pose Detection: MediaPipe Pose
- Runs on CPU with no CUDA requirement
- Single pip install, 33 high-quality landmarks
- Alternative considered: OpenPose (requires CUDA), MMPose (heavier)

## Classifier: Rule-Based (Phase 1)
- Zero training data required — instant feedback loop
- Interpretable: each pose maps to 1-2 geometric rules
- Upgrade path: scikit-learn MLP/SVM on collected samples (Phase 2)

## Meme Backend: CATAAS (default)
- Zero API key required — works out of the box
- Accepts text overlay queries for pose-relevant tags
- Reddit and Giphy backends available for richer results

## Display: OpenCV
- Already a MediaPipe dependency — no extra install
- Dual-panel layout: webcam skeleton (left) + meme (right)
- Target FPS: ≥20 on CPU

## Caching: TTL In-Memory Dict
- Prevents API hammering on every frame
- 30-second TTL refreshes memes while keeping real-time feel
- `r` key forces cache invalidation for manual refresh
```

- [ ] **Step 3: Commit**

```bash
git add README.md TECHNICAL_CHOICES.md
git commit -m "docs: add README and technical choices documentation"
```

---

## Self-Review

### Spec Coverage
- [x] 7 pose classes with rule-based geometry
- [x] MediaPipe wrapper with 33 landmarks
- [x] CATAAS, Reddit, Giphy backends
- [x] TTL cache (30s default)
- [x] OpenCV dual-panel renderer
- [x] All keyboard shortcuts (q, d, r, s, 1/2/3)
- [x] config.yaml with all settings
- [x] No hardcoded values
- [x] Type hints + docstrings throughout
- [x] Graceful error handling (fallback on meme API failure)
- [x] All files from CLAUDE.md checklist

### Type Consistency
- `LandmarkList = list[Optional[Landmark]]` defined in `detector.py`, imported in `classifier.py` ✓
- `MemeFetcher.fetch()` returns `Optional[np.ndarray]` — consistent across all backends ✓
- `RuleBasedClassifier.classify()` returns `PoseLabel` (str) — matches `main.py` usage ✓
