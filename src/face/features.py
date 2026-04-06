"""Face landmark feature extraction.

Converts a list of 478 MediaPipe FaceLandmarker NormalizedLandmark objects
into a 5-dimensional float32 feature vector capturing mouth openness and
head orientation.
"""

from __future__ import annotations

import numpy as np

FACE_FEATURE_DIM = 5

# Face mesh landmark indices (MediaPipe 478-landmark model)
_UPPER_LIP = 13
_LOWER_LIP = 14
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454


def extract_face_features(landmarks: list) -> np.ndarray:
    """Extract 5 normalized face features from MediaPipe FaceLandmarker landmarks.

    Features (index → name → description):
        0  mouth_open_ratio   vertical lip gap divided by mouth width; ~0 closed, >0.5 open
        1  lower_lip_drop     lower lip descent below mouth-corner line; proxy for tongue out
        2  head_yaw           horizontal head rotation; 0 = frontal, ±1 = fully sideways
        3  head_pitch         vertical tilt; 0 = level, positive = looking down
        4  head_roll          lateral tilt; normalized angle of eye line in [-0.5, 0.5]

    Args:
        landmarks: List of 478 NormalizedLandmark objects from FaceDetector.detect().

    Returns:
        Float32 ndarray of shape (5,).
    """
    features = np.zeros(FACE_FEATURE_DIM, dtype=np.float32)

    ul = landmarks[_UPPER_LIP]
    ll = landmarks[_LOWER_LIP]
    ml = landmarks[_MOUTH_LEFT]
    mr = landmarks[_MOUTH_RIGHT]

    mouth_width = abs(mr.x - ml.x) + 1e-6
    mouth_corner_y = (ml.y + mr.y) / 2.0

    # 0: mouth_open_ratio — how open the mouth is relative to its width
    vertical_gap = max(0.0, float(ll.y - ul.y))
    features[0] = vertical_gap / mouth_width

    # 1: lower_lip_drop — how far the lower lip hangs below the corner line
    features[1] = max(0.0, float(ll.y - mouth_corner_y)) / mouth_width

    # Reference points for head orientation
    nose = landmarks[_NOSE_TIP]
    chin = landmarks[_CHIN]
    left_cheek = landmarks[_LEFT_CHEEK]
    right_cheek = landmarks[_RIGHT_CHEEK]

    face_center_x = (left_cheek.x + right_cheek.x) / 2.0
    face_width = abs(right_cheek.x - left_cheek.x) + 1e-6

    # 2: head_yaw — nose offset from face center, normalized by face width
    features[2] = float(nose.x - face_center_x) / face_width

    # 3: head_pitch — nose position relative to the eye–chin axis
    le_outer = landmarks[_LEFT_EYE_OUTER]
    re_outer = landmarks[_RIGHT_EYE_OUTER]
    eye_y = (le_outer.y + re_outer.y) / 2.0
    face_height = abs(float(chin.y) - float(eye_y)) + 1e-6
    # Nose is ~45% of the way from eyes to chin when head is level
    expected_nose_y = eye_y + face_height * 0.45
    features[3] = float(nose.y - expected_nose_y) / face_height

    # 4: head_roll — angle of the eye line, normalized to [-0.5, 0.5]
    le_inner = landmarks[_LEFT_EYE_INNER]
    re_inner = landmarks[_RIGHT_EYE_INNER]
    left_eye_x = (float(le_outer.x) + float(le_inner.x)) / 2.0
    left_eye_y = (float(le_outer.y) + float(le_inner.y)) / 2.0
    right_eye_x = (float(re_outer.x) + float(re_inner.x)) / 2.0
    right_eye_y = (float(re_outer.y) + float(re_inner.y)) / 2.0
    features[4] = float(
        np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x + 1e-6) / np.pi
    )

    return features
