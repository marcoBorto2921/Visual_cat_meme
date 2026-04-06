"""CatPose — entry point.

Real-time pose classification via webcam: imita un gatto e vedi la sua foto.

Keyboard shortcuts:
  q — quit
  d — toggle debug overlay (top-3 predictions with confidence)
  r — reset smoothing window
  s — save screenshot
"""

from __future__ import annotations

import sys
import time
from collections import deque

import cv2
import yaml

from src.classifier.features import extract_features
from src.classifier.predictor import Predictor
from src.display.renderer import Renderer
from src.face.detector import FaceDetector
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
    """Run the CatPose real-time pipeline."""
    config = load_config()
    cam_cfg = config["camera"]
    pose_cfg = config["pose"]
    face_cfg = config.get("face", {"enabled": False})
    clf_cfg = config["classifier"]
    disp_cfg = config["display"]
    paths_cfg = config["paths"]

    # --- Init detectors ---
    detector = PoseDetector(visibility_threshold=pose_cfg["visibility_threshold"])
    face_detector: FaceDetector | None = None
    if face_cfg.get("enabled", False):
        face_detector = FaceDetector(model_path=face_cfg.get("model_path"))

    # --- Init predictor (exits with instructions if model missing) ---
    try:
        predictor = Predictor(
            classifier_path=paths_cfg["classifier"],
            label_encoder_path=paths_cfg["label_encoder"],
        )
    except FileNotFoundError as exc:
        logger.error(
            "Modello non trovato.\n%s\n\n"
            "Segui questi passi:\n"
            "  1. python scripts/collect_samples.py\n"
            "  2. python scripts/train_classifier.py\n"
            "  3. python main.py",
            exc,
        )
        sys.exit(1)

    # --- Init renderer ---
    renderer = Renderer(
        window_title=disp_cfg["window_title"],
        cat_panel_width=disp_cfg["cat_panel_width"],
        font_scale=disp_cfg["font_scale"],
        debug_mode=disp_cfg["debug_mode"],
        confidence_threshold=clf_cfg["confidence_threshold"],
        cats_dir=config["data_collection"]["cats_dir"],
        screenshots_dir=paths_cfg["screenshots"],
    )

    # --- Open webcam ---
    cap = cv2.VideoCapture(cam_cfg["index"])
    if not cap.isOpened():
        logger.error("Cannot open camera index %d", cam_cfg["index"])
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

    # --- State ---
    smoothing: deque[str] = deque(maxlen=clf_cfg["smoothing_window"])
    current_label: str | None = None
    current_confidence: float = 0.0
    current_top3: list[tuple[str, float]] = []

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

        # Pose + face detection → feature extraction
        landmarks = detector.detect(frame)
        face_landmarks = face_detector.detect(frame) if face_detector else None
        if landmarks:
            features = extract_features(
                landmarks, pose_cfg["visibility_threshold"], face_landmarks
            )
            label, confidence, top3 = predictor.predict(features)
            smoothing.append(label)
            stable_label = max(set(smoothing), key=list(smoothing).count)

            if confidence >= clf_cfg["confidence_threshold"]:
                current_label = stable_label
                current_confidence = confidence
                current_top3 = top3
            else:
                current_label = None
                current_confidence = confidence
                current_top3 = top3
        else:
            # No pose detected — clear smoothing, show "?"
            smoothing.clear()
            current_label = None
            current_confidence = 0.0
            current_top3 = []

        # Render
        composed = renderer.render(
            frame_bgr=frame,
            pose_label=current_label,
            confidence=current_confidence,
            fps=fps,
            top3=current_top3,
        )

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            renderer.debug_mode = not renderer.debug_mode
            logger.info("Debug mode: %s", renderer.debug_mode)
        elif key == ord("r"):
            smoothing.clear()
            logger.info("Smoothing window reset")
        elif key == ord("s"):
            renderer.save_screenshot(composed)

    cap.release()
    detector.close()
    if face_detector:
        face_detector.close()
    renderer.destroy()
    logger.info("Bye!")


if __name__ == "__main__":
    main()
