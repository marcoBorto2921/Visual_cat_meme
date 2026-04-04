"""MediaPipe Pose wrapper using the Tasks API (mediapipe >= 0.10)."""

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = "models/pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# Skeleton connections for manual drawing
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def _download_model() -> None:
    """Download the pose landmarker model file if not already present."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs("models", exist_ok=True)
    logger.info("Downloading pose landmarker model (~29MB)...")
    resp = requests.get(MODEL_URL, stream=True, timeout=60)
    resp.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    logger.info("Model downloaded to %s", MODEL_PATH)


@dataclass
class Landmark:
    """A single body landmark with normalized coordinates and visibility."""

    x: float
    y: float
    z: float
    visibility: float


LandmarkList = list[Optional[Landmark]]


class PoseDetector:
    """Wraps MediaPipe Tasks PoseLandmarker to extract 33 body landmarks.

    Args:
        visibility_threshold: Minimum visibility score to accept a landmark.
        model_complexity: Ignored (kept for API compatibility; use lite/full via MODEL_PATH).
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
        _download_model()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_ts_ms: int = 0
        logger.info("PoseDetector initialized (Tasks API, threshold=%.2f)", visibility_threshold)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, LandmarkList]:
        """Detect pose in a BGR frame and draw skeleton overlay.

        Args:
            frame: BGR image array from OpenCV.

        Returns:
            Tuple of (annotated BGR frame, list of 33 Landmark or None per index).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Increment timestamp — must be strictly increasing for VIDEO mode
        self._frame_ts_ms += 33
        results = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        landmarks: LandmarkList = [None] * 33
        annotated = frame.copy()

        if results.pose_landmarks:
            pose_lms = results.pose_landmarks[0]  # first detected person
            h, w = frame.shape[:2]

            for i, lm in enumerate(pose_lms):
                vis = getattr(lm, "visibility", 1.0) or 0.0
                if vis >= self._threshold:
                    landmarks[i] = Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=vis)

            # Draw connections
            for start_idx, end_idx in POSE_CONNECTIONS:
                lm_a = landmarks[start_idx]
                lm_b = landmarks[end_idx]
                if lm_a and lm_b:
                    pt_a = (int(lm_a.x * w), int(lm_a.y * h))
                    pt_b = (int(lm_b.x * w), int(lm_b.y * h))
                    cv2.line(annotated, pt_a, pt_b, (0, 200, 255), 2, cv2.LINE_AA)

            # Draw joint circles
            for lm in landmarks:
                if lm:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1, cv2.LINE_AA)

        return annotated, landmarks

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("PoseDetector closed")
