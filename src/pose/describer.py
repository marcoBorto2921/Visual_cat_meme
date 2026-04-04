"""Convert MediaPipe landmarks to pose label and CLIP text description."""

from __future__ import annotations


def classify_pose(
    landmarks: dict,
    descriptions: dict[str, str],
) -> tuple[str, str]:
    """Classify body pose from landmarks using geometric rules.

    Args:
        landmarks: Dict of visible landmark name to NormalizedLandmark.
        descriptions: Pose label to text description mapping (from config).

    Returns:
        Tuple of (pose_label, text_description).
    """
    def get(name: str):
        return landmarks.get(name)

    nose = get("NOSE")
    ls = get("LEFT_SHOULDER")
    rs = get("RIGHT_SHOULDER")
    lw = get("LEFT_WRIST")
    rw = get("RIGHT_WRIST")
    lh = get("LEFT_HIP")
    rh = get("RIGHT_HIP")

    # arms_up: both wrists above nose
    if lw and rw and nose:
        if lw.y < nose.y and rw.y < nose.y:
            label = "arms_up"
            return label, descriptions.get(label, label)

    # arms_wide: wrists far apart horizontally relative to shoulder width
    if lw and rw and ls and rs:
        wrist_span = abs(lw.x - rw.x)
        shoulder_span = abs(ls.x - rs.x)
        if wrist_span > shoulder_span * 1.8:
            label = "arms_wide"
            return label, descriptions.get(label, label)

    # thinking: one wrist near nose
    if nose:
        for wrist in [lw, rw]:
            if wrist:
                dist = ((wrist.x - nose.x) ** 2 + (wrist.y - nose.y) ** 2) ** 0.5
                if dist < 0.15:
                    label = "thinking"
                    return label, descriptions.get(label, label)

    # crossed_arms: wrists crossed (left wrist right of right wrist) near torso
    if lw and rw and ls and rs:
        mid_shoulder_y = (ls.y + rs.y) / 2
        torso_region = lw.y > mid_shoulder_y and rw.y > mid_shoulder_y
        if lw.x > rw.x and torso_region:
            label = "crossed_arms"
            return label, descriptions.get(label, label)

    # hands_on_hips: wrists near hip y-level
    if lw and rw and lh and rh:
        lw_near_hip = abs(lw.y - lh.y) < 0.12
        rw_near_hip = abs(rw.y - rh.y) < 0.12
        if lw_near_hip and rw_near_hip:
            label = "hands_on_hips"
            return label, descriptions.get(label, label)

    # slouching: shoulders close to nose vertically
    if ls and rs and nose:
        shoulder_y = (ls.y + rs.y) / 2
        if shoulder_y - nose.y < 0.12:
            label = "slouching"
            return label, descriptions.get(label, label)

    label = "neutral"
    return label, descriptions.get(label, label)
