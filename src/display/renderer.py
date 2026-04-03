"""OpenCV dual-panel renderer: webcam feed (left) + cat meme (right)."""

from typing import Optional

import cv2
import numpy as np

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
