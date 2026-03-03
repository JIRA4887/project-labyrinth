"""
labyrinth.encoder
-----------------
Semantic encoder using sentence-transformers.

Provides dense vector embeddings for text blocks and cosine similarity
scoring — the foundation of Recursive Semantic Anchoring (RSA).

Model: all-MiniLM-L6-v2 (384-dim, ~80MB, fast, strong MTEB scores)
"""

from __future__ import annotations

import numpy as np
from typing import List, Union


class SemanticEncoder:
    """
    Lazy-loaded sentence-transformer encoder.

    The model is downloaded on first use (~80 MB) and cached locally.
    All subsequent calls use the cached model — no network required.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None  # lazy load

    def _load(self):
        """Load the sentence-transformer model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with:\n"
                    "  pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into dense embedding vectors.

        Args:
            text: A string or list of strings to encode.

        Returns:
            np.ndarray of shape (n_texts, 384) or (384,) for single string.
        """
        self._load()
        single = isinstance(text, str)
        texts = [text] if single else text
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings[0] if single else embeddings

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two normalized embedding vectors.

        Since encode() normalizes by default, this is just the dot product.

        Args:
            a: First embedding vector (shape: 384,)
            b: Second embedding vector (shape: 384,)

        Returns:
            float in [-1, 1], where 1.0 = identical semantics.
        """
        return float(np.dot(a, b))

    def batch_similarity(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        Cosine similarity between one query and multiple candidate vectors.

        Args:
            query:      Shape (384,)
            candidates: Shape (n, 384)

        Returns:
            np.ndarray of shape (n,) with similarity scores.
        """
        return candidates @ query
