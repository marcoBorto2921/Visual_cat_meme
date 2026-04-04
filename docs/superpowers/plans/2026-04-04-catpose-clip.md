# CatPose CLIP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real-time webcam pose detection → CLIP-based zero-shot retrieval of the most similar cat photo from a local folder.

**Architecture:** MediaPipe detects 33 body landmarks → geometric rules classify pose → CLIP text embedding of pose description → cosine similarity against pre-indexed CLIP image embeddings of cat photos → best match displayed in dual-panel OpenCV window.

**Tech Stack:** Python 3.10+, MediaPipe 0.10.x, OpenCV, HuggingFace Transformers (CLIP ViT-B/32), NumPy, PyYAML, Pillow

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | Pinned dependencies |
| `configs/config.yaml` | All runtime parameters |
| `src/utils/logger.py` | Structured logging setup |
| `src/pose/detector.py` | MediaPipe Pose wrapper → 33 landmarks dict |
| `src/pose/describer.py` | Landmark geometry → pose label + text description |
| `src/clip_index/indexer.py` | Load cat images → CLIP embeddings → save .npy + .json |
| `src/clip_index/retriever.py` | Load index, cosine similarity, return top-k paths |
| `src/display/renderer.py` | OpenCV dual-panel renderer with overlays |
| `scripts/build_index.py` | Standalone CLI to build/rebuild the CLIP index |
| `main.py` | Entry point: wire all components, main loop |
| `.gitignore`, `Makefile`, `README.md`, `TECHNICAL_CHOICES.md` | Infra/docs |

---

### Task 1: Project skeleton and config

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `configs/config.yaml`
- Create: `assets/cats/.gitkeep`, `clip_index/.gitkeep`, `screenshots/.gitkeep`
- Create: all `__init__.py` stubs

- [ ] **Step 1: Create .gitignore**

```
.venv/
.claude/
__pycache__/
*.pyc
.env
clip_index/embeddings.npy
clip_index/filenames.json
assets/cats/*
!assets/cats/.gitkeep
screenshots/
```

- [ ] **Step 2: Create requirements.txt**

```
mediapipe==0.10.33
opencv-python==4.11.0.86
transformers==4.51.3
torch==2.6.0
Pillow==11.2.1
PyYAML==6.0.2
numpy==2.2.4
ruff==0.11.4
```

- [ ] **Step 3: Create configs/config.yaml**

```yaml
camera:
  index: 0
  width: 640
  height: 480
  fps: 30

pose:
  visibility_threshold: 0.5
  smoothing_window: 5

clip:
  model_name: "openai/clip-vit-base-patch32"
  index_dir: "clip_index"
  cats_dir: "assets/cats"
  top_k: 3

display:
  window_title: "CatPose CLIP"
  cat_panel_width: 480
  debug_mode: false
  font_scale: 1.0
  similarity_threshold: 0.15

pose_descriptions:
  arms_up:       "a person with both arms raised above the head, excited"
  arms_wide:     "a person with arms stretched wide open to the sides"
  thinking:      "a person touching their face with one hand, thinking"
  slouching:     "a person slouching forward with drooping shoulders, tired"
  crossed_arms:  "a person with arms crossed on the chest, grumpy"
  hands_on_hips: "a person with hands on hips, confident and sassy"
  neutral:       "a person standing normally, calm and relaxed"
```

- [ ] **Step 4: Create placeholder directories and __init__.py files**

```bash
mkdir -p assets/cats clip_index screenshots src/pose src/clip_index src/display src/utils scripts configs
touch assets/cats/.gitkeep clip_index/.gitkeep screenshots/.gitkeep
touch src/__init__.py src/pose/__init__.py src/clip_index/__init__.py src/display/__init__.py src/utils/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: project skeleton, config and gitignore"
```

---

### Task 2: Logger

**Files:**
- Create: `src/utils/logger.py`

- [ ] **Step 1: Write logger.py**

```python
"""Structured logging setup for CatPose CLIP."""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and return a configured logger.

    Args:
        name: Logger name (typically __name__).
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
```

- [ ] **Step 2: Commit**

```bash
git add src/utils/logger.py
git commit -m "feat: add structured logger"
```

---

### Task 3: Pose detector (MediaPipe wrapper)

**Files:**
- Create: `src/pose/detector.py`

- [ ] **Step 1: Write detector.py**

```python
"""MediaPipe Pose wrapper returning normalized landmarks."""

from __future__ import annotations

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Key landmark indices
LANDMARK_INDICES = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
}


class PoseDetector:
    """Wraps MediaPipe Pose Tasks API for real-time landmark detection.

    Args:
        visibility_threshold: Minimum landmark visibility to consider valid.
    """

    def __init__(self, visibility_threshold: float = 0.5) -> None:
        self.visibility_threshold = visibility_threshold
        base_options = mp_python.BaseOptions(
            delegate=mp_python.BaseOptions.Delegate.CPU
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        logger.info("PoseDetector initialized (Tasks API, CPU)")

    def detect(self, frame_bgr: np.ndarray) -> dict[str, NormalizedLandmark] | None:
        """Detect pose landmarks in a BGR frame.

        Args:
            frame_bgr: BGR image from OpenCV.

        Returns:
            Dict mapping landmark name to NormalizedLandmark, or None if no pose found.
        """
        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_bgr[:, :, ::-1]),
        )
        result = self._landmarker.detect(rgb)
        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        visible: dict[str, NormalizedLandmark] = {}
        for name, idx in LANDMARK_INDICES.items():
            lm = landmarks[idx]
            if lm.visibility >= self.visibility_threshold:
                visible[name] = lm
        return visible if visible else None

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("PoseDetector closed")
```

- [ ] **Step 2: Commit**

```bash
git add src/pose/detector.py
git commit -m "feat: MediaPipe Pose detector (Tasks API)"
```

---

### Task 4: Pose describer (geometry → text)

**Files:**
- Create: `src/pose/describer.py`

- [ ] **Step 1: Write describer.py**

```python
"""Convert MediaPipe landmarks to pose label and CLIP text description."""

from __future__ import annotations

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from src.utils.logger import get_logger

logger = get_logger(__name__)


def classify_pose(
    landmarks: dict[str, NormalizedLandmark],
    descriptions: dict[str, str],
) -> tuple[str, str]:
    """Classify body pose from landmarks using geometric rules.

    Args:
        landmarks: Dict of visible landmark name → NormalizedLandmark.
        descriptions: Pose label → text description mapping (from config).

    Returns:
        Tuple of (pose_label, text_description).
    """
    def get(name: str) -> NormalizedLandmark | None:
        return landmarks.get(name)

    nose = get("NOSE")
    ls = get("LEFT_SHOULDER")
    rs = get("RIGHT_SHOULDER")
    lw = get("LEFT_WRIST")
    rw = get("RIGHT_WRIST")
    lh = get("LEFT_HIP")
    rh = get("RIGHT_HIP")

    # arms_up: both wrists above nose
    if lw and rw and nose:
        if lw.y < nose.y and rw.y < nose.y:
            label = "arms_up"
            return label, descriptions.get(label, label)

    # arms_wide: wrists far apart horizontally relative to shoulder width
    if lw and rw and ls and rs:
        wrist_span = abs(lw.x - rw.x)
        shoulder_span = abs(ls.x - rs.x)
        if wrist_span > shoulder_span * 1.8:
            label = "arms_wide"
            return label, descriptions.get(label, label)

    # thinking: one wrist near nose
    if nose:
        for wrist in [lw, rw]:
            if wrist:
                dist = ((wrist.x - nose.x) ** 2 + (wrist.y - nose.y) ** 2) ** 0.5
                if dist < 0.15:
                    label = "thinking"
                    return label, descriptions.get(label, label)

    # crossed_arms: wrists crossed (left wrist right of right wrist) near torso
    if lw and rw and ls and rs:
        mid_shoulder_y = (ls.y + rs.y) / 2
        mid_hip_y = ((lh.y + rh.y) / 2) if lh and rh else mid_shoulder_y + 0.2
        torso_region = lw.y > mid_shoulder_y and rw.y > mid_shoulder_y
        if lw.x > rw.x and torso_region:
            label = "crossed_arms"
            return label, descriptions.get(label, label)

    # hands_on_hips: wrists near hip y-level
    if lw and rw and lh and rh:
        lw_near_hip = abs(lw.y - lh.y) < 0.12
        rw_near_hip = abs(rw.y - rh.y) < 0.12
        if lw_near_hip and rw_near_hip:
            label = "hands_on_hips"
            return label, descriptions.get(label, label)

    # slouching: shoulders below nose by less than normal + head forward
    if ls and rs and nose:
        shoulder_y = (ls.y + rs.y) / 2
        if shoulder_y - nose.y < 0.12:
            label = "slouching"
            return label, descriptions.get(label, label)

    label = "neutral"
    return label, descriptions.get(label, label)
```

- [ ] **Step 2: Commit**

```bash
git add src/pose/describer.py
git commit -m "feat: pose describer with geometric rules"
```

---

### Task 5: CLIP indexer

**Files:**
- Create: `src/clip_index/indexer.py`

- [ ] **Step 1: Write indexer.py**

```python
"""Build and save CLIP image embedding index from assets/cats/."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class CLIPIndexer:
    """Builds a CLIP image embedding index from a folder of cat images.

    Args:
        model_name: HuggingFace model identifier.
        cats_dir: Path to folder containing cat images.
        index_dir: Path to folder where index files will be saved.
    """

    def __init__(self, model_name: str, cats_dir: str, index_dir: str) -> None:
        self.cats_dir = Path(cats_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading CLIP model: %s", model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        logger.info("CLIP model loaded")

    def build(self) -> tuple[np.ndarray, list[str]]:
        """Build embedding index from all images in cats_dir.

        Returns:
            Tuple of (embeddings array [N, 512], list of file paths).
        """
        image_paths = [
            p for p in self.cats_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS
        ]
        if not image_paths:
            logger.warning(
                "No images found in '%s'. Index will be empty.", self.cats_dir
            )
            return np.zeros((0, 512), dtype=np.float32), []

        logger.info("Indexing %d images...", len(image_paths))
        embeddings: list[np.ndarray] = []
        valid_paths: list[str] = []

        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                import torch
                with torch.no_grad():
                    feats = self.model.get_image_features(**inputs)
                feats = feats.numpy().astype(np.float32)
                feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
                embeddings.append(feats[0])
                valid_paths.append(str(path))
                logger.debug("Indexed: %s", path.name)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path.name, exc)

        emb_array = np.stack(embeddings, axis=0)
        return emb_array, valid_paths

    def save(self, embeddings: np.ndarray, filenames: list[str]) -> None:
        """Save index to disk.

        Args:
            embeddings: Array of shape [N, 512].
            filenames: List of image file paths.
        """
        np.save(self.index_dir / "embeddings.npy", embeddings)
        with open(self.index_dir / "filenames.json", "w") as f:
            json.dump(filenames, f)
        logger.info("Index saved: %d images", len(filenames))

    def build_and_save(self) -> tuple[np.ndarray, list[str]]:
        """Build the index and save it. Returns (embeddings, filenames)."""
        emb, paths = self.build()
        self.save(emb, paths)
        return emb, paths
```

- [ ] **Step 2: Commit**

```bash
git add src/clip_index/indexer.py
git commit -m "feat: CLIP image indexer"
```

---

### Task 6: CLIP retriever

**Files:**
- Create: `src/clip_index/retriever.py`

- [ ] **Step 1: Write retriever.py**

```python
"""Load CLIP index and retrieve top-k cat images by text query."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CLIPRetriever:
    """Retrieves cat images by cosine similarity to a text query.

    Args:
        model_name: HuggingFace model identifier.
        index_dir: Path to folder containing embeddings.npy and filenames.json.
        top_k: Number of top results to return.
    """

    def __init__(self, model_name: str, index_dir: str, top_k: int = 3) -> None:
        self.index_dir = Path(index_dir)
        self.top_k = top_k
        self._embeddings: np.ndarray | None = None
        self._filenames: list[str] = []
        self._last_text: str = ""
        self._last_embedding: np.ndarray | None = None

        logger.info("Loading CLIP model for retrieval: %s", model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        logger.info("CLIP retriever model loaded")

    def load_index(self) -> bool:
        """Load pre-built index from disk.

        Returns:
            True if index loaded successfully with at least one image, else False.
        """
        emb_path = self.index_dir / "embeddings.npy"
        fn_path = self.index_dir / "filenames.json"
        if not emb_path.exists() or not fn_path.exists():
            logger.warning("Index not found at '%s'. Run build_index.py first.", self.index_dir)
            return False
        self._embeddings = np.load(emb_path)
        with open(fn_path) as f:
            self._filenames = json.load(f)
        if len(self._filenames) == 0:
            logger.warning("Index is empty — no cat images indexed yet.")
            return False
        logger.info("Index loaded: %d images", len(self._filenames))
        return True

    def load_from_arrays(self, embeddings: np.ndarray, filenames: list[str]) -> None:
        """Load index directly from arrays (used after build_and_save at runtime).

        Args:
            embeddings: Array of shape [N, 512].
            filenames: Corresponding file paths.
        """
        self._embeddings = embeddings
        self._filenames = filenames
        self._last_text = ""
        self._last_embedding = None

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Compute L2-normalized CLIP text embedding, with caching.

        Args:
            text: Input text description.

        Returns:
            Normalized embedding of shape [512].
        """
        if text == self._last_text and self._last_embedding is not None:
            return self._last_embedding
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        emb = feats.numpy().astype(np.float32)[0]
        emb /= np.linalg.norm(emb) + 1e-8
        self._last_text = text
        self._last_embedding = emb
        return emb

    def retrieve(self, text: str) -> list[tuple[str, float]]:
        """Retrieve top-k cat images most similar to the text.

        Args:
            text: Pose description string.

        Returns:
            List of (file_path, similarity_score) sorted descending.
            Empty list if index is empty.
        """
        if self._embeddings is None or len(self._filenames) == 0:
            return []
        text_emb = self._get_text_embedding(text)
        sims = self._embeddings @ text_emb  # cosine sim (both L2-normalized)
        k = min(self.top_k, len(self._filenames))
        top_indices = np.argsort(sims)[::-1][:k]
        return [(self._filenames[i], float(sims[i])) for i in top_indices]

    def best_match(self, text: str) -> tuple[str, float] | None:
        """Return best matching (path, score) or None if index empty.

        Args:
            text: Pose description string.

        Returns:
            Tuple of (file_path, score) or None.
        """
        results = self.retrieve(text)
        return results[0] if results else None
```

- [ ] **Step 2: Commit**

```bash
git add src/clip_index/retriever.py
git commit -m "feat: CLIP retriever with text caching"
```

---

### Task 7: OpenCV renderer

**Files:**
- Create: `src/display/renderer.py`

- [ ] **Step 1: Write renderer.py**

```python
"""OpenCV dual-panel renderer: webcam left, cat image right."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Renderer:
    """Manages the dual-panel OpenCV display window.

    Args:
        window_title: OpenCV window name.
        cat_panel_width: Width in pixels of the right panel.
        font_scale: Scale factor for all text overlays.
        debug_mode: Whether to show debug overlay on startup.
        similarity_threshold: Min score to show a cat image.
        screenshots_dir: Where to save screenshots.
    """

    def __init__(
        self,
        window_title: str,
        cat_panel_width: int,
        font_scale: float,
        debug_mode: bool,
        similarity_threshold: float,
        screenshots_dir: str = "screenshots",
    ) -> None:
        self.window_title = window_title
        self.cat_panel_width = cat_panel_width
        self.font_scale = font_scale
        self.debug_mode = debug_mode
        self.similarity_threshold = similarity_threshold
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True)
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        logger.info("Renderer initialized: window='%s'", window_title)

    def _make_cat_panel(
        self,
        cat_path: str | None,
        score: float,
        height: int,
        top_k_results: list[tuple[str, float]] | None,
    ) -> np.ndarray:
        """Render the right panel with the cat image and optional debug overlay.

        Args:
            cat_path: Path to cat image file, or None.
            score: Similarity score of the best match.
            height: Panel height to match webcam frame.
            top_k_results: List of (path, score) for debug overlay.

        Returns:
            BGR panel of shape [height, cat_panel_width, 3].
        """
        panel = np.zeros((height, self.cat_panel_width, 3), dtype=np.uint8)

        if cat_path and Path(cat_path).exists() and score >= self.similarity_threshold:
            try:
                pil_img = Image.open(cat_path).convert("RGB")
                orig_w, orig_h = pil_img.size
                scale = min(self.cat_panel_width / orig_w, height / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                x_off = (self.cat_panel_width - new_w) // 2
                y_off = (height - new_h) // 2
                panel[y_off:y_off + new_h, x_off:x_off + new_w] = img_bgr
            except Exception as exc:
                logger.warning("Cannot load cat image '%s': %s", cat_path, exc)
        else:
            msg = "No match" if score < self.similarity_threshold else "No index"
            cv2.putText(
                panel, msg,
                (self.cat_panel_width // 4, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (200, 200, 200), 2,
            )

        # Debug overlay
        if self.debug_mode and top_k_results:
            for i, (path, sim) in enumerate(top_k_results):
                name = Path(path).name[:30]
                text = f"#{i+1} {name} {sim:.3f}"
                cv2.putText(
                    panel, text,
                    (8, height - 20 - i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * self.font_scale,
                    (0, 255, 255), 1,
                )
        return panel

    def render(
        self,
        frame_bgr: np.ndarray,
        pose_label: str,
        score: float,
        cat_path: str | None,
        fps: float,
        top_k_results: list[tuple[str, float]] | None = None,
    ) -> np.ndarray:
        """Compose the full dual-panel frame.

        Args:
            frame_bgr: Webcam frame (BGR).
            pose_label: Current pose classification label.
            score: Best match similarity score.
            cat_path: Path to best matching cat image, or None.
            fps: Current frames per second.
            top_k_results: Top-k results for debug overlay.

        Returns:
            Composed BGR image for display.
        """
        h, w = frame_bgr.shape[:2]

        # Left panel overlays
        left = frame_bgr.copy()
        cv2.putText(
            left, f"Pose: {pose_label}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), 2,
        )
        cv2.putText(
            left, f"Sim: {score:.3f}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, (255, 255, 0), 2,
        )
        cv2.putText(
            left, f"FPS: {fps:.1f}",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.font_scale, (200, 200, 200), 1,
        )
        if self.debug_mode:
            cv2.putText(
                left, "[DEBUG]",
                (w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2,
            )

        right = self._make_cat_panel(cat_path, score, h, top_k_results)
        composed = np.hstack([left, right])
        cv2.imshow(self.window_title, composed)
        return composed

    def save_screenshot(self, frame: np.ndarray) -> str:
        """Save current frame to screenshots directory.

        Args:
            frame: BGR image to save.

        Returns:
            Path to saved file.
        """
        import time
        filename = self.screenshots_dir / f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info("Screenshot saved: %s", filename)
        return str(filename)

    def destroy(self) -> None:
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()
        logger.info("Renderer destroyed")
```

- [ ] **Step 2: Commit**

```bash
git add src/display/renderer.py
git commit -m "feat: dual-panel OpenCV renderer"
```

---

### Task 8: Standalone build_index script

**Files:**
- Create: `scripts/build_index.py`

- [ ] **Step 1: Write build_index.py**

```python
"""Standalone script to build or rebuild the CLIP image index."""

import sys
from pathlib import Path

import yaml

# Allow running as: python scripts/build_index.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clip_index.indexer import CLIPIndexer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Load config and build CLIP index from assets/cats/."""
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    clip_cfg = config["clip"]
    indexer = CLIPIndexer(
        model_name=clip_cfg["model_name"],
        cats_dir=clip_cfg["cats_dir"],
        index_dir=clip_cfg["index_dir"],
    )
    emb, paths = indexer.build_and_save()
    if len(paths) == 0:
        logger.warning(
            "Index built but empty. Add JPG/PNG images to '%s' and re-run.",
            clip_cfg["cats_dir"],
        )
    else:
        logger.info("Done. Indexed %d cat images.", len(paths))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/build_index.py
git commit -m "feat: standalone build_index script"
```

---

### Task 9: main.py entry point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write main.py**

```python
"""CatPose CLIP — entry point.

Keyboard shortcuts:
  q — quit
  d — toggle debug overlay
  r — force re-retrieval (clear cache)
  i — re-index assets/cats/ at runtime
  s — save screenshot
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import yaml

from src.clip_index.indexer import CLIPIndexer
from src.clip_index.retriever import CLIPRetriever
from src.display.renderer import Renderer
from src.pose.describer import classify_pose
from src.pose.detector import PoseDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML config file.

    Args:
        path: Path to config file.

    Returns:
        Config dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run the CatPose CLIP pipeline."""
    config = load_config()
    cam_cfg = config["camera"]
    pose_cfg = config["pose"]
    clip_cfg = config["clip"]
    disp_cfg = config["display"]
    pose_descs: dict[str, str] = config["pose_descriptions"]

    # --- Init components ---
    detector = PoseDetector(visibility_threshold=pose_cfg["visibility_threshold"])

    indexer = CLIPIndexer(
        model_name=clip_cfg["model_name"],
        cats_dir=clip_cfg["cats_dir"],
        index_dir=clip_cfg["index_dir"],
    )
    retriever = CLIPRetriever(
        model_name=clip_cfg["model_name"],
        index_dir=clip_cfg["index_dir"],
        top_k=clip_cfg["top_k"],
    )

    # Try loading pre-built index; build if absent
    if not retriever.load_index():
        logger.info("No pre-built index found — building now...")
        emb, paths = indexer.build_and_save()
        retriever.load_from_arrays(emb, paths)

    renderer = Renderer(
        window_title=disp_cfg["window_title"],
        cat_panel_width=disp_cfg["cat_panel_width"],
        font_scale=disp_cfg["font_scale"],
        debug_mode=disp_cfg["debug_mode"],
        similarity_threshold=disp_cfg["similarity_threshold"],
    )

    cap = cv2.VideoCapture(cam_cfg["index"])
    if not cap.isOpened():
        logger.error("Cannot open camera index %d", cam_cfg["index"])
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

    # State
    smoothing: deque[str] = deque(maxlen=pose_cfg["smoothing_window"])
    current_cat: str | None = None
    current_score: float = 0.0
    top_k_results: list[tuple[str, float]] = []
    prev_label: str = ""
    force_retrieval: bool = False

    # FPS tracking
    fps_counter: deque[float] = deque(maxlen=30)
    prev_time = time.time()

    logger.info("Starting main loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            continue

        frame = cv2.flip(frame, 1)  # mirror

        # FPS
        now = time.time()
        fps_counter.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = sum(fps_counter) / len(fps_counter)

        # Pose detection
        landmarks = detector.detect(frame)
        if landmarks:
            label, description = classify_pose(landmarks, pose_descs)
        else:
            label, description = "neutral", pose_descs.get("neutral", "neutral")

        # Smoothing
        smoothing.append(label)
        stable_label = max(set(smoothing), key=list(smoothing).count)

        # Retrieval (only when pose changes or forced)
        if stable_label != prev_label or force_retrieval:
            top_k_results = retriever.retrieve(description)
            if top_k_results:
                current_cat, current_score = top_k_results[0]
            else:
                current_cat, current_score = None, 0.0
            prev_label = stable_label
            force_retrieval = False

        # Render
        composed = renderer.render(
            frame_bgr=frame,
            pose_label=stable_label,
            score=current_score,
            cat_path=current_cat,
            fps=fps,
            top_k_results=top_k_results,
        )

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            renderer.debug_mode = not renderer.debug_mode
            logger.info("Debug mode: %s", renderer.debug_mode)
        elif key == ord("r"):
            force_retrieval = True
            logger.info("Forced re-retrieval")
        elif key == ord("i"):
            logger.info("Re-indexing assets/cats/ ...")
            emb, paths = indexer.build_and_save()
            retriever.load_from_arrays(emb, paths)
            force_retrieval = True
            logger.info("Re-index complete: %d images", len(paths))
        elif key == ord("s"):
            renderer.save_screenshot(composed)

    cap.release()
    detector.close()
    renderer.destroy()
    logger.info("Bye!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: main entry point with full pipeline wiring"
```

---

### Task 10: Docs and Makefile

**Files:**
- Create: `README.md`, `TECHNICAL_CHOICES.md`, `Makefile`

- [ ] **Step 1: Write README.md**

```markdown
# CatPose CLIP

Real-time webcam pose detection → CLIP-based zero-shot cat photo retrieval.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Add JPG/PNG cat photos to `assets/cats/`
2. Build the CLIP index: `python scripts/build_index.py`
3. Run: `python main.py`

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle debug overlay |
| `r` | Force re-retrieval |
| `i` | Re-index assets/cats/ |
| `s` | Save screenshot |
```

- [ ] **Step 2: Write TECHNICAL_CHOICES.md**

```markdown
# Technical Choices

## Pose Detection — MediaPipe Tasks API
MediaPipe 0.10+ uses the Tasks API (PoseLandmarker) which works on Windows without legacy Python bindings.

## Image Retrieval — CLIP ViT-B/32 via HuggingFace
Zero-shot: no labelled data, no training. Text embedding of a geometric pose description is matched against pre-indexed image embeddings via cosine similarity.

## No GPU required
CLIP inference on CPU is fast enough for the use case (~5-10ms per text embedding).

## Caching
Text embeddings are cached in the retriever — recomputed only on pose label change.
```

- [ ] **Step 3: Write Makefile**

```makefile
.PHONY: index run lint

index:
	python scripts/build_index.py

run:
	python main.py

lint:
	ruff check src/ scripts/ main.py
```

- [ ] **Step 4: Commit**

```bash
git add README.md TECHNICAL_CHOICES.md Makefile
git commit -m "docs: README, technical choices, Makefile"
```

---

### Task 11: Final push

- [ ] **Step 1: Verify everything**

```bash
python scripts/build_index.py  # should warn "empty" gracefully
python -c "from src.pose.detector import PoseDetector; print('OK')"
python -c "from src.clip_index.retriever import CLIPRetriever; print('OK')"
```

- [ ] **Step 2: Push**

```bash
git push
```
