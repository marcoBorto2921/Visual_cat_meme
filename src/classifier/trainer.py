"""Train SVM or MLP classifier on collected landmark samples."""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.utils.logger import get_logger

logger = get_logger(__name__)

LOW_ACCURACY_THRESHOLD = 0.7


def train(config: dict) -> None:
    """Train a pose classifier from collected samples.

    Reads data/samples.csv, trains SVM or MLP, saves models to disk, and
    prints a classification report on the validation split.

    Args:
        config: Full application config dict (from config.yaml).

    Raises:
        SystemExit: If samples file is missing or has insufficient data.
    """
    samples_path = Path(config["data_collection"]["output_file"])
    clf_path = Path(config["paths"]["classifier"])
    le_path = Path(config["paths"]["label_encoder"])
    clf_cfg = config["classifier"]
    model_type: str = clf_cfg["model"]

    if not samples_path.exists():
        logger.error(
            "Samples file not found: %s\n"
            "Run `python scripts/collect_samples.py` first.",
            samples_path,
        )
        sys.exit(1)

    with open(samples_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.error("Samples file is empty. Collect training data first.")
        sys.exit(1)

    labels = np.array([r["label"] for r in rows])
    feature_cols = [k for k in rows[0] if k != "label"]
    features = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float32)

    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        logger.error(
            "Need at least 2 classes to train. Found: %s", list(unique_classes)
        )
        sys.exit(1)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(
        "Training %s classifier on %d samples (%d classes)...",
        model_type.upper(),
        len(X_train),
        len(unique_classes),
    )

    if model_type == "svm":
        svm_cfg = clf_cfg["svm"]
        clf = SVC(
            C=float(svm_cfg["C"]),
            kernel=str(svm_cfg["kernel"]),
            gamma=str(svm_cfg["gamma"]),
            probability=bool(svm_cfg["probability"]),
        )
    elif model_type == "mlp":
        mlp_cfg = clf_cfg["mlp"]
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(mlp_cfg["hidden_layer_sizes"]),
            max_iter=int(mlp_cfg["max_iter"]),
            random_state=int(mlp_cfg["random_state"]),
        )
    else:
        logger.error("Unknown model type '%s'. Use 'svm' or 'mlp'.", model_type)
        sys.exit(1)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)

    print("\n" + "=" * 50)
    print(f"Validation accuracy: {val_accuracy:.3f}")
    print("=" * 50)
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    if val_accuracy < LOW_ACCURACY_THRESHOLD:
        logger.warning(
            "Accuracy bassa (%.2f) — raccogli più campioni o controlla le pose.",
            val_accuracy,
        )

    clf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    logger.info("Classifier saved to %s", clf_path)
    logger.info("Label encoder saved to %s", le_path)
