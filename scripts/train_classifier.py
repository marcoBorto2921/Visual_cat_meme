"""Entry point for training the pose classifier.

Usage:
    python scripts/train_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.trainer import train
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Train the pose classifier using collected samples."""
    config = load_config()
    train(config)


if __name__ == "__main__":
    main()
