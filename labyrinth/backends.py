"""
labyrinth.backends
------------------
L3 archive backend abstraction.

Two backends are provided:

  NumpyL3Backend (default)
    - Pure Python + NumPy — works on every Python version including 3.14.
    - In-memory cosine-similarity search over stored embeddings.
    - Optional persistence: pass persist_path="path/to/archive.npz" to
      load/save the archive across sessions.
    - Zero extra dependencies beyond numpy (already required).

  ChromaL3Backend (optional)
    - Uses ChromaDB for persistent vector storage.
    - Only available on Python <= 3.12 due to Pydantic V1 incompatibility.
    - Automatically selected if chromadb is importable AND Python < 3.14.
    - Falls back to NumpyL3Backend with a warning if unavailable.

Usage:
    # Auto-select best available backend (recommended):
    backend = auto_select_backend()

    # Or explicitly:
    backend = NumpyL3Backend(persist_path="labyrinth_archive.npz")
    backend.add("block-id", embedding, "raw text", {"hash": "abc123"})
    results = backend.query(query_embedding, n_results=3)
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Abstract Base ─────────────────────────────────────────────────────────────

class L3Backend(ABC):
    """Abstract interface for an L3 vector archive backend."""

    @abstractmethod
    def add(
        self,
        doc_id: str,
        embedding: np.ndarray,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Store a raw text block with its embedding."""

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 2,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the top-n most similar blocks for a query embedding.

        Returns: list of (text, similarity_score) tuples, highest first.
        """

    @abstractmethod
    def get_by_id(self, doc_id: str) -> Optional[str]:
        """Retrieve raw text by exact document ID."""

    @abstractmethod
    def count(self) -> int:
        """Return number of stored blocks."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored blocks while preserving configuration."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""


# ── Numpy Backend ─────────────────────────────────────────────────────────────

class NumpyL3Backend(L3Backend):
    """
    Pure-numpy L3 archive. Works on all Python versions.

    Storage layout (in-memory):
      _ids       : list[str]           — document IDs
      _texts     : list[str]           — raw text blocks
      _metadata  : list[dict]          — per-document metadata (e.g. hash)
      _matrix    : np.ndarray (N, D)   — stacked unit-norm embeddings

    Cosine similarity = dot product of unit-norm vectors.

    Persistence (optional):
      Pass persist_path to __init__. The archive is loaded on init if the
      file exists, and saved automatically on every add().
      Format: .npz for embeddings, .json sidecar for texts/metadata.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metadata: List[Dict] = []
        self._matrix: Optional[np.ndarray] = None  # shape (N, D)
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    @property
    def name(self) -> str:
        return "NumpyL3Backend"

    def add(
        self,
        doc_id: str,
        embedding: np.ndarray,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        norm = np.linalg.norm(embedding)
        unit = embedding / norm if norm > 0 else embedding

        self._ids.append(doc_id)
        self._texts.append(text)
        self._metadata.append(metadata or {})

        if self._matrix is None:
            self._matrix = unit.reshape(1, -1)
        else:
            if unit.shape[0] != self._matrix.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self._matrix.shape[1]}, "
                    f"got {unit.shape[0]}. Ensure your encoder hasn't changed."
                )
            self._matrix = np.vstack([self._matrix, unit.reshape(1, -1)])

        if self._persist_path:
            self._save()

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 2,
    ) -> List[Tuple[str, float]]:
        if self._matrix is None or len(self._ids) == 0:
            return []

        norm = np.linalg.norm(query_embedding)
        q = query_embedding / norm if norm > 0 else query_embedding
        
        if q.shape[0] != self._matrix.shape[1]:
            logger.warning(
                f"Query dimension mismatch: expected {self._matrix.shape[1]}, got {q.shape[0]}. "
                "Returning empty results to prevent crash."
            )
            return []

        scores = self._matrix @ q               # (N,) cosine similarities
        n = min(n_results, len(self._ids))
        top_idx = np.argsort(scores)[::-1][:n]  # descending

        return [(self._texts[i], float(scores[i])) for i in top_idx]

    def get_by_id(self, doc_id: str) -> Optional[str]:
        try:
            idx = self._ids.index(doc_id)
            return self._texts[idx]
        except ValueError:
            return None

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids.clear()
        self._texts.clear()
        self._metadata.clear()
        self._matrix = None
        if self._persist_path:
            # Overwrite persisted files with empty state
            self._save()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        """Save embeddings to .npz and texts/metadata to .json sidecar."""
        try:
            npz_path = self._persist_path.with_suffix(".npz")
            json_path = self._persist_path.with_suffix(".json")
            np.savez_compressed(str(npz_path), matrix=self._matrix)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"ids": self._ids, "texts": self._texts, "meta": self._metadata}, f)
        except Exception as e:
            logger.warning(f"NumpyL3Backend: failed to persist: {e}")

    def _load(self):
        """Load previously persisted archive."""
        try:
            npz_path = self._persist_path.with_suffix(".npz")
            json_path = self._persist_path.with_suffix(".json")
            if npz_path.exists() and json_path.exists():
                data = np.load(str(npz_path))
                self._matrix = data["matrix"]
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._ids = meta["ids"]
                self._texts = meta["texts"]
                self._metadata = meta.get("meta", [{} for _ in self._ids])
                logger.info(f"NumpyL3Backend: loaded {len(self._ids)} blocks from {npz_path}")
        except Exception as e:
            logger.warning(f"NumpyL3Backend: failed to load persisted archive: {e}")


# ── Chroma Backend ────────────────────────────────────────────────────────────

class ChromaL3Backend(L3Backend):
    """
    ChromaDB-backed L3 archive. Persistent, supports large corpora.
    Only works on Python <= 3.12 (ChromaDB uses Pydantic V1).
    """

    def __init__(self, collection_name: str = "labyrinth_archive"):
        import chromadb
        self._client = chromadb.Client()
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def name(self) -> str:
        return "ChromaL3Backend"

    def add(
        self,
        doc_id: str,
        embedding: np.ndarray,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        self._col.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 2,
    ) -> List[Tuple[str, float]]:
        n = min(n_results, self._col.count())
        if n == 0:
            return []
        results = self._col.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n,
        )
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        # ChromaDB cosine distance → similarity: sim = 1 - dist
        return [(doc, 1.0 - dist) for doc, dist in zip(docs, dists)]

    def get_by_id(self, doc_id: str) -> Optional[str]:
        try:
            result = self._col.get(ids=[doc_id])
            docs = result.get("documents", [])
            return docs[0] if docs else None
        except Exception:
            return None

    def count(self) -> int:
        return self._col.count()

    def clear(self) -> None:
        self._client.delete_collection(self._col.name)
        self._col = self._client.get_or_create_collection(
            name="labyrinth_archive",
            metadata={"hnsw:space": "cosine"},
        )


# ── Auto-selector ─────────────────────────────────────────────────────────────

def auto_select_backend(
    collection_name: str = "labyrinth_archive",
    persist_path: Optional[str] = None,
) -> L3Backend:
    """
    Returns the best available L3 backend for the current Python environment.

    Logic:
      1. If Python >= 3.13 → always use NumpyL3Backend (ChromaDB incompatible).
      2. Else try ChromaL3Backend; fall back to NumpyL3Backend if import fails.
    """
    py = sys.version_info
    if py >= (3, 13):
        logger.info(
            f"Python {py.major}.{py.minor} detected — using NumpyL3Backend "
            "(ChromaDB requires Python <= 3.12). Full L3 functionality available."
        )
        return NumpyL3Backend(persist_path=persist_path)

    try:
        backend = ChromaL3Backend(collection_name=collection_name)
        logger.info(f"L3 backend: ChromaL3Backend (Python {py.major}.{py.minor})")
        return backend
    except Exception as e:
        logger.warning(
            f"ChromaDB unavailable ({e}) — falling back to NumpyL3Backend. "
            "Install chromadb for persistent cross-session storage."
        )
        return NumpyL3Backend(persist_path=persist_path)
