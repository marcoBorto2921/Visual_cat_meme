"""OpenCV renderer: full-screen webcam feed with cartoon speech bubble overlay."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Speech bubble constants
_BUBBLE_TARGET_SIZE = 220   # target side length (px) at full scale
_BUBBLE_MARGIN = 16         # gap from frame edge (px)
_BUBBLE_PADDING = 8         # inner padding between bubble wall and cat image (px)
_BUBBLE_RADIUS = 18         # corner radius (px)
_BORDER_THICKNESS = 3       # border stroke (px)
_TAIL_W = 30                # tail base width (px)
_TAIL_LEN = 28              # tail tip offset below/left of bubble (px)
_SCALE_STEP = 0.15          # bubble scale increment per frame (0→1)


def _draw_rounded_rect_filled(
    img: np.ndarray,
    x: int, y: int, w: int, h: int,
    r: int,
    color: tuple[int, int, int],
) -> None:
    """Fill a rounded rectangle on *img* in-place.

    Args:
        img: BGR image.
        x, y: Top-left corner.
        w, h: Width and height.
        r: Corner radius.
        color: BGR fill colour.
    """
    r = min(r, w // 2, h // 2)
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in [
        (x + r,     y + r),
        (x + w - r, y + r),
        (x + r,     y + h - r),
        (x + w - r, y + h - r),
    ]:
        cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 360, color, -1)


def _draw_rounded_rect_border(
    img: np.ndarray,
    x: int, y: int, w: int, h: int,
    r: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw the border of a rounded rectangle on *img* in-place.

    Args:
        img: BGR image.
        x, y: Top-left corner.
        w, h: Width and height.
        r: Corner radius.
        color: BGR border colour.
        thickness: Stroke width.
    """
    r = min(r, w // 2, h // 2)
    cv2.line(img, (x + r, y),     (x + w - r, y),     color, thickness)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness)
    cv2.line(img, (x,     y + r), (x,     y + h - r), color, thickness)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness)
    cv2.ellipse(img, (x + r,     y + r),     (r, r), 180,  0,  90, color, thickness)
    cv2.ellipse(img, (x + w - r, y + r),     (r, r), 270,  0,  90, color, thickness)
    cv2.ellipse(img, (x + r,     y + h - r), (r, r),  90,  0,  90, color, thickness)
    cv2.ellipse(img, (x + w - r, y + h - r), (r, r),   0,  0,  90, color, thickness)


class Renderer:
    """Full-screen webcam renderer with a cartoon speech-bubble cat overlay.

    The right side-panel is gone.  Instead, a speech bubble (top-right corner)
    shows the predicted cat image.  When the label changes the bubble plays a
    quick pop-in animation (scale 0 → 1 over ~8 frames).

    Args:
        window_title: OpenCV window name.
        cat_panel_width: Kept for API compatibility; no longer used for layout.
        font_scale: Scale factor for all text overlays.
        debug_mode: Whether to show debug overlay (top-3) on startup.
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
        self.cat_panel_width = cat_panel_width  # kept for compat, unused
        self.font_scale = font_scale
        self.debug_mode = debug_mode
        self.confidence_threshold = confidence_threshold
        self.cats_dir = Path(cats_dir)
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True)

        # Pop-in animation state
        self._bubble_scale: float = 0.0
        self._current_label: str | None = None

        # Image cache: label → BGR ndarray (full resolution, loaded once)
        self._cat_cache: dict[str, np.ndarray | None] = {}

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        logger.info("Renderer initialised: window='%s'", window_title)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_cat_image(self, label: str) -> Path | None:
        """Return the image path for *label*, or None if not found.

        Args:
            label: Pose class name (filename without extension).
        """
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = self.cats_dir / f"{label}{ext}"
            if p.exists():
                return p
        return None

    def _load_cat_bgr(self, label: str) -> np.ndarray | None:
        """Load (and cache) the cat image for *label*.

        Args:
            label: Pose class name.

        Returns:
            BGR ndarray or None if not found / unreadable.
        """
        if label not in self._cat_cache:
            path = self._find_cat_image(label)
            if path is None:
                self._cat_cache[label] = None
            else:
                try:
                    pil = Image.open(path).convert("RGB")
                    self._cat_cache[label] = cv2.cvtColor(
                        np.array(pil), cv2.COLOR_RGB2BGR
                    )
                except Exception as exc:
                    logger.warning("Cannot load cat image '%s': %s", path, exc)
                    self._cat_cache[label] = None
        return self._cat_cache[label]

    def _advance_bubble_animation(self, new_label: str | None) -> None:
        """Update pop-in scale; reset to 0 when the label changes.

        Args:
            new_label: The label that will be displayed this frame.
        """
        if new_label != self._current_label:
            self._current_label = new_label
            self._bubble_scale = 0.0
        if self._bubble_scale < 1.0:
            self._bubble_scale = min(1.0, self._bubble_scale + _SCALE_STEP)

    def _draw_speech_bubble(
        self,
        frame: np.ndarray,
        label: str | None,
        confidence: float,
        top3: list[tuple[str, float]] | None,
    ) -> None:
        """Draw the speech bubble (and optional debug overlay) on *frame* in-place.

        Args:
            frame: Full webcam BGR frame (will be mutated).
            label: Predicted pose label, or None.
            confidence: Prediction confidence.
            top3: Top-3 (label, confidence) pairs; used when debug_mode is True.
        """
        fh, fw = frame.shape[:2]

        # Scale target size by animation factor
        sz = int(_BUBBLE_TARGET_SIZE * self._bubble_scale)
        if sz < 4:
            return  # too small to draw yet

        r = max(2, int(_BUBBLE_RADIUS * self._bubble_scale))
        pad = max(1, int(_BUBBLE_PADDING * self._bubble_scale))
        border = max(1, int(_BORDER_THICKNESS * self._bubble_scale))

        # Bubble position: top-right corner of the frame
        bx = fw - _BUBBLE_MARGIN - sz   # left edge of bubble
        by = _BUBBLE_MARGIN              # top edge of bubble

        # --- Tail (triangle pointing bottom-left, toward user's head) ----------
        tail_base_left  = (bx + 6,             by + sz)
        tail_base_right = (bx + 6 + _TAIL_W,   by + sz)
        tail_tip        = (bx - _TAIL_LEN,      by + sz + _TAIL_LEN)
        tail_pts = np.array(
            [tail_base_left, tail_base_right, tail_tip], dtype=np.int32
        )
        cv2.fillPoly(frame, [tail_pts], (255, 255, 255))
        cv2.polylines(
            frame, [tail_pts], isClosed=True,
            color=(0, 0, 0), thickness=border,
        )

        # --- Bubble background ------------------------------------------------
        _draw_rounded_rect_filled(frame, bx, by, sz, sz, r, (255, 255, 255))
        _draw_rounded_rect_border(frame, bx, by, sz, sz, r, (0, 0, 0), border)

        # --- Content inside bubble -------------------------------------------
        inner_x = bx + pad
        inner_y = by + pad
        inner_w = sz - 2 * pad
        inner_h = sz - 2 * pad

        show_cat = (
            label is not None
            and confidence >= self.confidence_threshold
            and self._bubble_scale >= 1.0  # only show image after animation ends
        )

        if show_cat:
            cat_bgr = self._load_cat_bgr(label)  # type: ignore[arg-type]
            if cat_bgr is not None:
                oh, ow = cat_bgr.shape[:2]
                scale = min(inner_w / ow, inner_h / oh)
                nw = int(ow * scale)
                nh = int(oh * scale)
                resized = cv2.resize(cat_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
                ox = inner_x + (inner_w - nw) // 2
                oy = inner_y + (inner_h - nh) // 2
                frame[oy : oy + nh, ox : ox + nw] = resized
            else:
                # Image file missing — show label text
                cv2.putText(
                    frame, label or "?",
                    (inner_x + 4, inner_y + inner_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.font_scale,
                    (80, 80, 80), 1, cv2.LINE_AA,
                )
        else:
            # No recognised pose or still animating — draw "?"
            q_font = cv2.FONT_HERSHEY_SIMPLEX
            q_scale = 2.5 * self.font_scale * self._bubble_scale
            q_thick = max(1, int(3 * self._bubble_scale))
            (tw, th), _ = cv2.getTextSize("?", q_font, q_scale, q_thick)
            tx = inner_x + (inner_w - tw) // 2
            ty = inner_y + (inner_h + th) // 2
            cv2.putText(
                frame, "?", (tx, ty), q_font, q_scale,
                (100, 100, 100), q_thick, cv2.LINE_AA,
            )

        # --- Debug overlay: top-3 just below the bubble ----------------------
        if self.debug_mode and top3:
            for i, (lbl, conf) in enumerate(top3):
                text = f"#{i + 1} {lbl}  {conf:.2f}"
                ty_debug = by + sz + _TAIL_LEN + 20 + i * 22
                cv2.putText(
                    frame, text,
                    (fw - _BUBBLE_MARGIN - 220, ty_debug),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.font_scale,
                    (0, 255, 255), 1, cv2.LINE_AA,
                )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def render(
        self,
        frame_bgr: np.ndarray,
        pose_label: str | None,
        confidence: float,
        fps: float,
        top3: list[tuple[str, float]] | None = None,
    ) -> np.ndarray:
        """Compose and display the annotated full-screen webcam frame.

        Args:
            frame_bgr: Raw webcam frame (BGR).
            pose_label: Predicted pose class name, or None.
            confidence: Confidence score of the prediction.
            fps: Current frames per second.
            top3: Top-3 (label, confidence) pairs for debug overlay.

        Returns:
            Annotated BGR image (same resolution as *frame_bgr*).
        """
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        # Determine the effective label to display
        show_label = (
            pose_label
            if (pose_label and confidence >= self.confidence_threshold)
            else None
        )

        # Advance pop-in animation (must happen before _draw_speech_bubble)
        self._advance_bubble_animation(show_label)

        # FPS — top-left
        cv2.putText(
            out, f"FPS: {fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65 * self.font_scale,
            (200, 200, 200), 1, cv2.LINE_AA,
        )

        if self.debug_mode:
            cv2.putText(
                out, "[DEBUG]",
                (w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA,
            )

        # Speech bubble (drawn last so it sits on top)
        self._draw_speech_bubble(out, show_label, confidence, top3)

        cv2.imshow(self.window_title, out)
        return out

    def save_screenshot(self, frame: np.ndarray) -> str:
        """Save *frame* to the screenshots directory.

        Args:
            frame: BGR image to save.

        Returns:
            Absolute path of the saved file.
        """
        filename = self.screenshots_dir / f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info("Screenshot saved: %s", filename)
        return str(filename)

    def destroy(self) -> None:
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()
        logger.info("Renderer destroyed")
