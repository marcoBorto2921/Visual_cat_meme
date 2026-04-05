"""Load a trained classifier and predict pose labels with confidence scores."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Loads a trained SVM/MLP classifier and produces pose predictions.

    Args:
        classifier_path: Path to the pickled classifier.
        label_encoder_path: Path to the pickled LabelEncoder.
    """

    def __init__(self, classifier_path: str, label_encoder_path: str) -> None:
        clf_path = Path(classifier_path)
        le_path = Path(label_encoder_path)

        if not clf_path.exists() or not le_path.exists():
            raise FileNotFoundError(
                f"Model files not found: {clf_path}, {le_path}\n"
                "Run `python scripts/train_classifier.py` first."
            )

        with open(clf_path, "rb") as f:
            self._clf = pickle.load(f)
        with open(le_path, "rb") as f:
            self._le = pickle.load(f)

        logger.info("Predictor loaded: %d classes", len(self._le.classes_))

    @property
    def classes(self) -> list[str]:
        """Return the list of class names."""
        return list(self._le.classes_)

    def predict(
        self, features: np.ndarray
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Predict the pose label for a feature vector.

        Args:
            features: Float32 ndarray of shape (99,).

        Returns:
            Tuple of:
            - label: Predicted class name.
            - confidence: Probability of the predicted class (0–1).
            - top3: List of up to 3 (label, confidence) pairs, sorted descending.
        """
        x = features.reshape(1, -1)
        proba = self._clf.predict_proba(x)[0]
        top_indices = np.argsort(proba)[::-1]

        best_idx = top_indices[0]
        label = str(self._le.classes_[best_idx])
        confidence = float(proba[best_idx])

        top3 = [
            (str(self._le.classes_[i]), float(proba[i]))
            for i in top_indices[:3]
        ]

        return label, confidence, top3
