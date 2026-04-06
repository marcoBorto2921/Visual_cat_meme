"""MediaPipe FaceLandmarker wrapper returning 478 face landmarks + blendshapes."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path("models/face_landmarker.task")


@dataclass
class FaceResult:
    """Result of a FaceDetector inference.

    Attributes:
        landmarks: List of 478 NormalizedLandmark objects (.x, .y, .z in [0,1]).
        blendshapes: List of Category objects (.category_name, .score in [0,1]).
            52 ARKit blendshape scores including tongueOut, jawOpen, etc.
        transform_matrix: Row-major 4x4 face-to-camera transformation matrix
            as a flat list of 16 floats, or None if unavailable.
    """

    landmarks: list
    blendshapes: list
    transform_matrix: list | None


def _ensure_model() -> str:
    """Download the FaceLandmarker model file if not already present.

    Returns:
        Absolute path to the model file.
    """
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading FaceLandmarker model to %s ...", MODEL_PATH)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("FaceLandmarker model downloaded.")
    return str(MODEL_PATH)


class FaceDetector:
    """Wraps MediaPipe Face Tasks API for real-time face landmark detection.

    Returns landmarks and ARKit blendshapes (including tongueOut, jawOpen)
    for accurate mouth and expression classification.

    Args:
        model_path: Path to the FaceLandmarker .task file. If None, the model
            is downloaded automatically to models/face_landmarker.task.
    """

    def __init__(self, model_path: str | None = None) -> None:
        resolved_path = model_path if model_path else _ensure_model()
        base_options = mp_python.BaseOptions(
            model_asset_path=resolved_path,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        logger.info("FaceDetector initialized (Tasks API, CPU, blendshapes=True)")

    def detect(self, frame_bgr: np.ndarray) -> FaceResult | None:
        """Detect face landmarks and blendshapes in a BGR frame.

        Args:
            frame_bgr: BGR image from OpenCV.

        Returns:
            FaceResult with landmarks and blendshapes, or None if no face found.
        """
        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_bgr[:, :, ::-1]),
        )
        result = self._landmarker.detect(rgb)
        if not result.face_landmarks:
            return None
        matrix: list | None = None
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0]).flatten().tolist()
        return FaceResult(
            landmarks=list(result.face_landmarks[0]),
            blendshapes=list(result.face_blendshapes[0]) if result.face_blendshapes else [],
            transform_matrix=matrix,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("FaceDetector closed")
