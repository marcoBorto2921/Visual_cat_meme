"""MediaPipe Pose wrapper returning normalized landmarks."""

from __future__ import annotations

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

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

    def detect(self, frame_bgr: np.ndarray) -> dict | None:
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
        visible: dict = {}
        for name, idx in LANDMARK_INDICES.items():
            lm = landmarks[idx]
            if lm.visibility >= self.visibility_threshold:
                visible[name] = lm
        return visible if visible else None

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("PoseDetector closed")
