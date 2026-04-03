"""Rule-based pose classifier using MediaPipe landmark geometry."""

from collections import deque
from typing import Optional

from src.pose.detector import Landmark, LandmarkList, PoseDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)

PoseLabel = str  # one of the 7 pose class strings


class RuleBasedClassifier:
    """Classifies body pose into one of 7 classes using geometric rules.

    Rules operate on normalized landmark coordinates (y=0 is top of frame).

    Args:
        visibility_threshold: Minimum visibility for a landmark to be used.
        smoothing_window: Number of frames to smooth predictions over.
    """

    POSES = [
        "arms_up",
        "arms_wide",
        "thinking",
        "slouching",
        "crossed_arms",
        "hands_on_hips",
        "neutral",
    ]

    def __init__(
        self,
        visibility_threshold: float = 0.5,
        smoothing_window: int = 5,
    ) -> None:
        self._threshold = visibility_threshold
        self._history: deque[PoseLabel] = deque(maxlen=smoothing_window)
        self._active_rule: str = "none"

    @property
    def active_rule(self) -> str:
        """Name of the rule that fired for the last classification."""
        return self._active_rule

    def classify(self, landmarks: LandmarkList) -> PoseLabel:
        """Classify pose from a landmark list and return smoothed label.

        Args:
            landmarks: 33-element list from PoseDetector.process().

        Returns:
            Smoothed pose label string.
        """
        raw = self._classify_raw(landmarks)
        self._history.append(raw)
        # Return majority vote from history
        return max(set(self._history), key=self._history.count)

    def _lm(self, landmarks: LandmarkList, idx: int) -> Optional[Landmark]:
        """Return landmark if visible above threshold, else None."""
        lm = landmarks[idx]
        if lm is None or lm.visibility < self._threshold:
            return None
        return lm

    def _dist(self, a: Landmark, b: Landmark) -> float:
        """Euclidean distance between two landmarks (x, y only)."""
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    def _classify_raw(self, landmarks: LandmarkList) -> PoseLabel:
        """Apply geometric rules in priority order and return raw label."""
        idx = PoseDetector

        l_wrist = self._lm(landmarks, idx.LEFT_WRIST)
        r_wrist = self._lm(landmarks, idx.RIGHT_WRIST)
        l_shoulder = self._lm(landmarks, idx.LEFT_SHOULDER)
        r_shoulder = self._lm(landmarks, idx.RIGHT_SHOULDER)
        l_elbow = self._lm(landmarks, idx.LEFT_ELBOW)
        r_elbow = self._lm(landmarks, idx.RIGHT_ELBOW)
        l_hip = self._lm(landmarks, idx.LEFT_HIP)
        r_hip = self._lm(landmarks, idx.RIGHT_HIP)
        nose = self._lm(landmarks, idx.NOSE)

        # arms_up: both wrists above shoulders (y inverted, smaller = higher)
        if (
            l_wrist and r_wrist and l_shoulder and r_shoulder
            and l_wrist.y < l_shoulder.y - 0.15
            and r_wrist.y < r_shoulder.y - 0.15
        ):
            self._active_rule = "arms_up: wrists above shoulders"
            return "arms_up"

        # arms_wide: wrists far out horizontally and at similar height to shoulders
        if (
            l_wrist and r_wrist and l_shoulder and r_shoulder
            and abs(l_wrist.x - l_shoulder.x) > 0.25
            and abs(r_wrist.x - r_shoulder.x) > 0.25
            and abs(l_wrist.y - l_shoulder.y) < 0.15
            and abs(r_wrist.y - r_shoulder.y) < 0.15
        ):
            self._active_rule = "arms_wide: wrists extended horizontally"
            return "arms_wide"

        # thinking: one wrist near chin/cheek area (near nose, above shoulders)
        if nose and l_shoulder and r_shoulder:
            chin_y = nose.y + 0.1
            chin_x = nose.x
            if l_wrist and self._dist(l_wrist, Landmark(chin_x, chin_y, 0, 1)) < 0.15:
                self._active_rule = "thinking: left wrist near chin"
                return "thinking"
            if r_wrist and self._dist(r_wrist, Landmark(chin_x, chin_y, 0, 1)) < 0.15:
                self._active_rule = "thinking: right wrist near chin"
                return "thinking"

        # slouching: shoulders far below nose
        if nose and l_shoulder and r_shoulder:
            avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            if avg_shoulder_y > nose.y + 0.35:
                self._active_rule = "slouching: shoulders far below nose"
                return "slouching"

        # crossed_arms: left wrist past right shoulder AND right wrist past left shoulder
        if l_wrist and r_wrist and l_shoulder and r_shoulder:
            if l_wrist.x > r_shoulder.x and r_wrist.x < l_shoulder.x:
                self._active_rule = "crossed_arms: wrists crossed"
                return "crossed_arms"

        # hands_on_hips: both wrists near hip landmarks
        if l_wrist and l_hip and r_wrist and r_hip:
            if self._dist(l_wrist, l_hip) < 0.12 and self._dist(r_wrist, r_hip) < 0.12:
                self._active_rule = "hands_on_hips: wrists near hips"
                return "hands_on_hips"

        self._active_rule = "neutral: no rule matched"
        return "neutral"
