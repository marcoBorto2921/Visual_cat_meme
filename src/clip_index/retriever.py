"""Load CLIP index and retrieve top-k cat images by text query."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CLIPRetriever:
    """Retrieves cat images by cosine similarity to a text query.

    Args:
        model_name: HuggingFace model identifier.
        index_dir: Path to folder containing embeddings.npy and filenames.json.
        top_k: Number of top results to return.
    """

    def __init__(self, model_name: str, index_dir: str, top_k: int = 3) -> None:
        self.index_dir = Path(index_dir)
        self.top_k = top_k
        self._embeddings: np.ndarray | None = None
        self._filenames: list[str] = []
        self._last_text: str = ""
        self._last_embedding: np.ndarray | None = None

        logger.info("Loading CLIP model for retrieval: %s", model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        logger.info("CLIP retriever model loaded")

    def load_index(self) -> bool:
        """Load pre-built index from disk.

        Returns:
            True if index loaded successfully with at least one image, else False.
        """
        emb_path = self.index_dir / "embeddings.npy"
        fn_path = self.index_dir / "filenames.json"
        if not emb_path.exists() or not fn_path.exists():
            logger.warning(
                "Index not found at '%s'. Run build_index.py first.", self.index_dir
            )
            return False
        self._embeddings = np.load(emb_path)
        with open(fn_path) as f:
            self._filenames = json.load(f)
        if len(self._filenames) == 0:
            logger.warning("Index is empty — no cat images indexed yet.")
            return False
        logger.info("Index loaded: %d images", len(self._filenames))
        return True

    def load_from_arrays(self, embeddings: np.ndarray, filenames: list[str]) -> None:
        """Load index directly from arrays (used after build_and_save at runtime).

        Args:
            embeddings: Array of shape [N, 512].
            filenames: Corresponding file paths.
        """
        self._embeddings = embeddings
        self._filenames = filenames
        self._last_text = ""
        self._last_embedding = None

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Compute L2-normalized CLIP text embedding, with caching.

        Args:
            text: Input text description.

        Returns:
            Normalized embedding of shape [512].
        """
        if text == self._last_text and self._last_embedding is not None:
            return self._last_embedding
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        emb = feats.numpy().astype(np.float32)[0]
        emb /= np.linalg.norm(emb) + 1e-8
        self._last_text = text
        self._last_embedding = emb
        return emb

    def retrieve(self, text: str) -> list[tuple[str, float]]:
        """Retrieve top-k cat images most similar to the text.

        Args:
            text: Pose description string.

        Returns:
            List of (file_path, similarity_score) sorted descending.
            Empty list if index is empty.
        """
        if self._embeddings is None or len(self._filenames) == 0:
            return []
        text_emb = self._get_text_embedding(text)
        sims = self._embeddings @ text_emb  # cosine sim (both L2-normalized)
        k = min(self.top_k, len(self._filenames))
        top_indices = np.argsort(sims)[::-1][:k]
        return [(self._filenames[i], float(sims[i])) for i in top_indices]

    def best_match(self, text: str) -> tuple[str, float] | None:
        """Return best matching (path, score) or None if index empty.

        Args:
            text: Pose description string.

        Returns:
            Tuple of (file_path, score) or None.
        """
        results = self.retrieve(text)
        return results[0] if results else None
