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

        frame = cv2.flip(frame, 1)  # mirror view

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
            label = "neutral"
            description = pose_descs.get("neutral", "a person standing normally")

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
