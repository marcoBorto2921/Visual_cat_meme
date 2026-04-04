"""Build and save CLIP image embedding index from assets/cats/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class CLIPIndexer:
    """Builds a CLIP image embedding index from a folder of cat images.

    Args:
        model_name: HuggingFace model identifier.
        cats_dir: Path to folder containing cat images.
        index_dir: Path to folder where index files will be saved.
    """

    def __init__(self, model_name: str, cats_dir: str, index_dir: str) -> None:
        self.cats_dir = Path(cats_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading CLIP model: %s", model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        logger.info("CLIP model loaded")

    def build(self) -> tuple[np.ndarray, list[str]]:
        """Build embedding index from all images in cats_dir.

        Returns:
            Tuple of (embeddings array [N, 512], list of file paths).
        """
        image_paths = [
            p for p in self.cats_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS
        ]
        if not image_paths:
            logger.warning(
                "No images found in '%s'. Index will be empty.", self.cats_dir
            )
            return np.zeros((0, 512), dtype=np.float32), []

        logger.info("Indexing %d images...", len(image_paths))
        embeddings: list[np.ndarray] = []
        valid_paths: list[str] = []

        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    feats = self.model.get_image_features(**inputs)
                feats_np = feats.numpy().astype(np.float32)
                feats_np /= np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-8
                embeddings.append(feats_np[0])
                valid_paths.append(str(path))
                logger.debug("Indexed: %s", path.name)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path.name, exc)

        emb_array = np.stack(embeddings, axis=0)
        return emb_array, valid_paths

    def save(self, embeddings: np.ndarray, filenames: list[str]) -> None:
        """Save index to disk.

        Args:
            embeddings: Array of shape [N, 512].
            filenames: List of image file paths.
        """
        np.save(self.index_dir / "embeddings.npy", embeddings)
        with open(self.index_dir / "filenames.json", "w") as f:
            json.dump(filenames, f)
        logger.info("Index saved: %d images", len(filenames))

    def build_and_save(self) -> tuple[np.ndarray, list[str]]:
        """Build the index and save it.

        Returns:
            Tuple of (embeddings, filenames).
        """
        emb, paths = self.build()
        self.save(emb, paths)
        return emb, paths
