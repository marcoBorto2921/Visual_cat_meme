"""Landmark feature extraction and normalization.

Converts pose landmarks (33 MediaPipe NormalizedLandmark objects) and optional
face landmarks (478 MediaPipe NormalizedLandmark objects) into a single
translation- and scale-invariant 104-dimensional float32 feature vector.

Layout: [pose_features (99)] + [face_features (5)] = 104 total.
Face features are zero-padded when not available.
"""

from __future__ import annotations

import numpy as np

from src.face.detector import FaceResult
from src.face.features import extract_face_features, FACE_FEATURE_DIM

NUM_POSE_LANDMARKS = 33
POSE_FEATURE_DIM = NUM_POSE_LANDMARKS * 3  # x, y, z per landmark
FEATURE_DIM = POSE_FEATURE_DIM + FACE_FEATURE_DIM  # 99 + 5 = 104

# Anchor indices used for pose normalization
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


def extract_features(
    landmarks: list,
    visibility_threshold: float,
    face_result: FaceResult | None = None,
) -> np.ndarray:
    """Extract a normalized feature vector from pose and optional face landmarks.

    Pose normalization steps:
    1. Center coordinates on the hip midpoint (translation invariance).
    2. Scale by shoulder-to-shoulder distance (scale invariance).
    3. Zero-pad coordinates of landmarks below the visibility threshold.

    Face features (5-dim) are appended after the pose features. If face_result
    is None, the last 5 dimensions are zero.

    Args:
        landmarks: List of 33 NormalizedLandmark objects from PoseDetector.
        visibility_threshold: Landmarks with visibility below this value are zeroed out.
        face_result: Optional FaceResult from FaceDetector.detect(). Pass None
            when face detection is disabled.

    Returns:
        Float32 ndarray of shape (104,) — 99 pose features + 5 face features.
    """
    # --- Pose features (99-dim) ---
    lh = landmarks[LEFT_HIP_IDX]
    rh = landmarks[RIGHT_HIP_IDX]
    hip_x = (lh.x + rh.x) / 2.0
    hip_y = (lh.y + rh.y) / 2.0
    hip_z = (lh.z + rh.z) / 2.0

    ls = landmarks[LEFT_SHOULDER_IDX]
    rs = landmarks[RIGHT_SHOULDER_IDX]
    shoulder_dist = float(
        np.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2 + (ls.z - rs.z) ** 2)
    )
    shoulder_dist = max(shoulder_dist, 1e-6)

    pose_coords = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        if lm.visibility >= visibility_threshold:
            pose_coords[i, 0] = (lm.x - hip_x) / shoulder_dist
            pose_coords[i, 1] = (lm.y - hip_y) / shoulder_dist
            pose_coords[i, 2] = (lm.z - hip_z) / shoulder_dist
        # invisible landmark → stays (0, 0, 0)

    pose_features = pose_coords.flatten()

    # --- Face features (5-dim) ---
    if face_result is not None:
        face_features = extract_face_features(face_result)
    else:
        face_features = np.zeros(FACE_FEATURE_DIM, dtype=np.float32)

    return np.concatenate([pose_features, face_features])
