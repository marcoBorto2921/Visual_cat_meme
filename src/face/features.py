"""Face feature extraction from FaceLandmarker results.

Produces a 5-dimensional float32 vector from blendshape scores (for mouth/tongue)
and landmark geometry (for head orientation). Using blendshapes for mouth/tongue
is significantly more accurate than geometric approximations because MediaPipe's
internal face model is trained to detect these states directly.
"""

from __future__ import annotations

import numpy as np

from src.face.detector import FaceResult

FACE_FEATURE_DIM = 5

# Face mesh landmark indices (MediaPipe 478-landmark model)
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454


def extract_face_features(face_result: FaceResult) -> np.ndarray:
    """Extract 5 normalized face features from a FaceResult.

    Features (index → name → source → description):
        0  jaw_open      blendshape  jawOpen score [0,1]; 0=closed, 1=wide open
        1  tongue_out    blendshape  tongueOut score [0,1]; direct tongue detection
        2  head_yaw      landmarks   horizontal rotation; 0=frontal, ±1=sideways
        3  head_pitch    landmarks   vertical tilt; 0=level, positive=looking down
        4  head_roll     landmarks   lateral tilt; normalized angle in [-0.5, 0.5]

    Args:
        face_result: FaceResult returned by FaceDetector.detect().

    Returns:
        Float32 ndarray of shape (5,).
    """
    features = np.zeros(FACE_FEATURE_DIM, dtype=np.float32)

    # --- Features 0–1: blendshapes (direct model scores) ---
    blendshape_map = {b.category_name: float(b.score) for b in face_result.blendshapes}
    features[0] = blendshape_map.get("jawOpen", 0.0)
    features[1] = blendshape_map.get("tongueOut", 0.0)

    # --- Features 2–4: head orientation from landmarks ---
    lm = face_result.landmarks

    nose = lm[_NOSE_TIP]
    chin = lm[_CHIN]
    left_cheek = lm[_LEFT_CHEEK]
    right_cheek = lm[_RIGHT_CHEEK]

    face_center_x = (left_cheek.x + right_cheek.x) / 2.0
    face_width = abs(right_cheek.x - left_cheek.x) + 1e-6

    # head_yaw: nose offset from face center normalized by face width
    features[2] = float(nose.x - face_center_x) / face_width

    # head_pitch: nose offset from expected position in the eye–chin axis
    le_outer = lm[_LEFT_EYE_OUTER]
    re_outer = lm[_RIGHT_EYE_OUTER]
    eye_y = (le_outer.y + re_outer.y) / 2.0
    face_height = abs(float(chin.y) - float(eye_y)) + 1e-6
    expected_nose_y = eye_y + face_height * 0.45
    features[3] = float(nose.y - expected_nose_y) / face_height

    # head_roll: angle of the eye line, normalized to [-0.5, 0.5]
    le_inner = lm[_LEFT_EYE_INNER]
    re_inner = lm[_RIGHT_EYE_INNER]
    left_eye_x = (float(le_outer.x) + float(le_inner.x)) / 2.0
    left_eye_y = (float(le_outer.y) + float(le_inner.y)) / 2.0
    right_eye_x = (float(re_outer.x) + float(re_inner.x)) / 2.0
    right_eye_y = (float(re_outer.y) + float(re_inner.y)) / 2.0
    features[4] = float(
        np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x + 1e-6) / np.pi
    )

    return features
