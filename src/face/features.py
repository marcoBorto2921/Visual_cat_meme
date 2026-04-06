"""Face feature extraction from FaceLandmarker results.

Produces a 5-dimensional float32 vector:
  - Features 0–1: ARKit blendshape scores (jaw open, tongue out) — direct model output.
  - Features 2–4: Head euler angles from the 4x4 facial transformation matrix,
    which gives accurate pitch/yaw/roll regardless of face position in frame.
    Falls back to landmark geometry if the matrix is unavailable.
"""

from __future__ import annotations

import numpy as np

from src.face.detector import FaceResult

FACE_FEATURE_DIM = 5

# Normalization bounds for euler angles (radians).
# Clipped to ±MAX_ANGLE before normalizing to [-1, 1].
_MAX_ANGLE = np.pi / 3  # ±60 degrees covers all realistic head poses

# Face mesh landmark indices (MediaPipe 478-landmark model) — fallback only
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454


def _euler_from_matrix(matrix_data: list) -> tuple[float, float, float]:
    """Extract pitch, yaw, roll from a row-major 4x4 facial transformation matrix.

    The matrix maps face space to camera space. We extract euler angles using
    ZYX convention (yaw → pitch → roll).

    Args:
        matrix_data: Flat list of 16 floats, row-major.

    Returns:
        Tuple of (pitch, yaw, roll) in radians. Each normalized to [-1, 1]
        by dividing by MAX_ANGLE and clipping.
    """
    m = np.array(matrix_data, dtype=np.float32).reshape(4, 4)
    R = m[:3, :3]

    # ZYX Euler decomposition
    yaw = float(np.arctan2(R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    pitch = float(np.arctan2(-R[2, 1], R[2, 2]))
    roll = float(np.arctan2(-R[1, 0], R[0, 0]))

    def norm(angle: float) -> float:
        return float(np.clip(angle / _MAX_ANGLE, -1.0, 1.0))

    return norm(pitch), norm(yaw), norm(roll)


def _euler_from_landmarks(face_result: FaceResult) -> tuple[float, float, float]:
    """Fallback head orientation from landmark geometry.

    Args:
        face_result: FaceResult with landmarks populated.

    Returns:
        Tuple of (pitch, yaw, roll), each normalized to [-1, 1].
    """
    lm = face_result.landmarks
    nose = lm[_NOSE_TIP]
    chin = lm[_CHIN]
    left_cheek = lm[_LEFT_CHEEK]
    right_cheek = lm[_RIGHT_CHEEK]

    face_center_x = (left_cheek.x + right_cheek.x) / 2.0
    face_width = abs(right_cheek.x - left_cheek.x) + 1e-6
    yaw = float(np.clip((nose.x - face_center_x) / face_width, -1.0, 1.0))

    le_outer = lm[_LEFT_EYE_OUTER]
    re_outer = lm[_RIGHT_EYE_OUTER]
    eye_y = (le_outer.y + re_outer.y) / 2.0
    face_height = abs(float(chin.y) - float(eye_y)) + 1e-6
    expected_nose_y = eye_y + face_height * 0.45
    pitch = float(np.clip((nose.y - expected_nose_y) / face_height, -1.0, 1.0))

    le_inner = lm[_LEFT_EYE_INNER]
    re_inner = lm[_RIGHT_EYE_INNER]
    left_eye_x = (float(le_outer.x) + float(le_inner.x)) / 2.0
    left_eye_y = (float(le_outer.y) + float(le_inner.y)) / 2.0
    right_eye_x = (float(re_outer.x) + float(re_inner.x)) / 2.0
    right_eye_y = (float(re_outer.y) + float(re_inner.y)) / 2.0
    roll = float(
        np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x + 1e-6) / np.pi
    )

    return pitch, yaw, roll


def extract_face_features(face_result: FaceResult) -> np.ndarray:
    """Extract 5 normalized face features from a FaceResult.

    Features:
        0  jaw_open     blendshape  jawOpen score [0,1]
        1  tongue_out   blendshape  tongueOut score [0,1]
        2  head_yaw     matrix      horizontal rotation, normalized to [-1, 1]
        3  head_pitch   matrix      vertical tilt, normalized to [-1, 1]
        4  head_roll    matrix      lateral tilt, normalized to [-1, 1]

    Args:
        face_result: FaceResult returned by FaceDetector.detect().

    Returns:
        Float32 ndarray of shape (5,).
    """
    features = np.zeros(FACE_FEATURE_DIM, dtype=np.float32)

    # Features 0–1: blendshapes
    blendshape_map = {b.category_name: float(b.score) for b in face_result.blendshapes}
    features[0] = blendshape_map.get("jawOpen", 0.0)
    features[1] = blendshape_map.get("tongueOut", 0.0)

    # Features 2–4: head orientation
    if face_result.transform_matrix is not None:
        pitch, yaw, roll = _euler_from_matrix(face_result.transform_matrix)
    else:
        pitch, yaw, roll = _euler_from_landmarks(face_result)

    features[2] = yaw
    features[3] = pitch
    features[4] = roll

    return features
