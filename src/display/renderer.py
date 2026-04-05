"""OpenCV dual-panel renderer: webcam left, cat image right."""

from __future__ import annotations

import time
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
        confidence_threshold: Min confidence to show a cat image.
        cats_dir: Directory containing cat images (label.jpg).
        screenshots_dir: Where to save screenshots.
    """

    def __init__(
        self,
        window_title: str,
        cat_panel_width: int,
        font_scale: float,
        debug_mode: bool,
        confidence_threshold: float,
        cats_dir: str,
        screenshots_dir: str = "screenshots",
    ) -> None:
        self.window_title = window_title
        self.cat_panel_width = cat_panel_width
        self.font_scale = font_scale
        self.debug_mode = debug_mode
        self.confidence_threshold = confidence_threshold
        self.cats_dir = Path(cats_dir)
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True)
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        logger.info("Renderer initialized: window='%s'", window_title)

    def _find_cat_image(self, label: str) -> Path | None:
        """Find the image file for a given label.

        Tries common extensions in order. Returns None if not found.

        Args:
            label: Pose class name (filename without extension).

        Returns:
            Path to the image file, or None.
        """
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = self.cats_dir / f"{label}{ext}"
            if p.exists():
                return p
        return None

    def _make_cat_panel(
        self,
        label: str | None,
        confidence: float,
        height: int,
        top3: list[tuple[str, float]] | None,
    ) -> np.ndarray:
        """Render the right panel with the cat image and optional debug overlay.

        Args:
            label: Predicted pose label (maps to assets/cats/{label}.jpg).
            confidence: Confidence score of the prediction.
            height: Panel height to match webcam frame.
            top3: Top-3 (label, confidence) pairs for debug overlay.

        Returns:
            BGR panel of shape [height, cat_panel_width, 3].
        """
        panel = np.zeros((height, self.cat_panel_width, 3), dtype=np.uint8)

        if label and confidence >= self.confidence_threshold:
            cat_path = self._find_cat_image(label)
            if cat_path:
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
                    panel[y_off : y_off + new_h, x_off : x_off + new_w] = img_bgr
                except Exception as exc:
                    logger.warning("Cannot load cat image '%s': %s", cat_path, exc)
            else:
                msg = f"No image for '{label}'"
                cv2.putText(
                    panel, msg,
                    (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.font_scale,
                    (200, 200, 200), 1,
                )
        else:
            msg = "?" if label else "No model"
            cv2.putText(
                panel, msg,
                (self.cat_panel_width // 2 - 20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0 * self.font_scale,
                (100, 100, 100), 3,
            )

        # Debug overlay: top-3 predictions
        if self.debug_mode and top3:
            for i, (lbl, conf) in enumerate(top3):
                text = f"#{i + 1} {lbl}  {conf:.2f}"
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
        pose_label: str | None,
        confidence: float,
        fps: float,
        top3: list[tuple[str, float]] | None = None,
    ) -> np.ndarray:
        """Compose the full dual-panel frame and display it.

        Args:
            frame_bgr: Webcam frame (BGR).
            pose_label: Predicted pose class name, or None.
            confidence: Confidence score of the prediction.
            fps: Current frames per second.
            top3: Top-3 (label, confidence) pairs for debug overlay.

        Returns:
            Composed BGR image (left + right panels).
        """
        h, w = frame_bgr.shape[:2]

        left = frame_bgr.copy()
        label_text = pose_label if pose_label else "?"
        cv2.putText(
            left, f"Pose: {label_text}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), 2,
        )
        cv2.putText(
            left, f"Conf: {confidence:.2f}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8,
            (255, 255, 0), 2,
        )
        cv2.putText(
            left, f"FPS: {fps:.1f}",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.font_scale,
            (200, 200, 200), 1,
        )
        if self.debug_mode:
            cv2.putText(
                left, "[DEBUG]",
                (w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2,
            )

        right = self._make_cat_panel(pose_label, confidence, h, top3)
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
        filename = self.screenshots_dir / f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info("Screenshot saved: %s", filename)
        return str(filename)

    def destroy(self) -> None:
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()
        logger.info("Renderer destroyed")
