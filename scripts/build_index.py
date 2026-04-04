"""Standalone script to build or rebuild the CLIP image index."""

import sys
from pathlib import Path

import yaml

# Allow running as: python scripts/build_index.py from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clip_index.indexer import CLIPIndexer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Load config and build CLIP index from assets/cats/."""
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    clip_cfg = config["clip"]
    indexer = CLIPIndexer(
        model_name=clip_cfg["model_name"],
        cats_dir=clip_cfg["cats_dir"],
        index_dir=clip_cfg["index_dir"],
    )
    emb, paths = indexer.build_and_save()
    if len(paths) == 0:
        logger.warning(
            "Index built but empty. Add JPG/PNG images to '%s' and re-run.",
            clip_cfg["cats_dir"],
        )
    else:
        logger.info("Done. Indexed %d cat images.", len(paths))


if __name__ == "__main__":
    main()
