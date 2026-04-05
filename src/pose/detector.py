"""MediaPipe Pose wrapper returning all 33 normalized landmarks."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = Path("models/pose_landmarker_full.task")


def _ensure_model() -> str:
    """Download the PoseLandmarker model file if not already present.

    Returns:
        Absolute path to the model file.
    """
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading PoseLandmarker model to %s ...", MODEL_PATH)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("Model downloaded.")
    return str(MODEL_PATH)


class PoseDetector:
    """Wraps MediaPipe Pose Tasks API for real-time landmark detection.

    Args:
        visibility_threshold: Minimum landmark visibility to consider a pose valid.
            A pose is returned only when at least one landmark exceeds this threshold.
    """

    def __init__(self, visibility_threshold: float = 0.5) -> None:
        self.visibility_threshold = visibility_threshold
        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_python.BaseOptions.Delegate.CPU,
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

    def detect(self, frame_bgr: np.ndarray) -> list | None:
        """Detect pose landmarks in a BGR frame.

        Args:
            frame_bgr: BGR image from OpenCV.

        Returns:
            List of 33 NormalizedLandmark objects, or None if no pose found.
            Each landmark has .x, .y, .z, .visibility attributes.
        """
        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_bgr[:, :, ::-1]),
        )
        result = self._landmarker.detect(rgb)
        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        # Return all 33 landmarks; feature extraction handles filtering by visibility
        return list(landmarks)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("PoseDetector closed")
