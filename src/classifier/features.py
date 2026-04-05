"""Landmark feature extraction and normalization.

Converts a list of 33 MediaPipe NormalizedLandmark objects into a
translation- and scale-invariant 99-dimensional float32 feature vector.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 33
FEATURE_DIM = NUM_LANDMARKS * 3  # x, y, z per landmark

# Anchor indices used for normalization
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


def extract_features(landmarks: list, visibility_threshold: float) -> np.ndarray:
    """Extract a normalized feature vector from MediaPipe pose landmarks.

    Normalization steps:
    1. Center coordinates on the hip midpoint (translation invariance).
    2. Scale by shoulder-to-shoulder distance (scale invariance).
    3. Zero-pad coordinates of landmarks below the visibility threshold.

    Args:
        landmarks: List of 33 NormalizedLandmark objects from MediaPipe.
        visibility_threshold: Landmarks with visibility below this value are zeroed out.

    Returns:
        Float32 ndarray of shape (99,) — 33 landmarks × (x, y, z).
    """
    # Hip center for translation normalization
    lh = landmarks[LEFT_HIP_IDX]
    rh = landmarks[RIGHT_HIP_IDX]
    hip_x = (lh.x + rh.x) / 2.0
    hip_y = (lh.y + rh.y) / 2.0
    hip_z = (lh.z + rh.z) / 2.0

    # Shoulder distance for scale normalization
    ls = landmarks[LEFT_SHOULDER_IDX]
    rs = landmarks[RIGHT_SHOULDER_IDX]
    shoulder_dist = float(
        np.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2 + (ls.z - rs.z) ** 2)
    )
    shoulder_dist = max(shoulder_dist, 1e-6)

    coords = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        if lm.visibility >= visibility_threshold:
            coords[i, 0] = (lm.x - hip_x) / shoulder_dist
            coords[i, 1] = (lm.y - hip_y) / shoulder_dist
            coords[i, 2] = (lm.z - hip_z) / shoulder_dist
        # invisible landmark → stays (0, 0, 0)

    return coords.flatten()
