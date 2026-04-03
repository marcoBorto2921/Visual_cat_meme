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
