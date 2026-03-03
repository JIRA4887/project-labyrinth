"""
labyrinth.memory
----------------
Three-layer memory system: L1 (Working), L2 (Index), L3 (Archive).

    L1 — Working Memory:  Recent raw text, always in context.
    L2 — Semantic Index:  Anchor Tokens (compressed embeddings + summaries).
    L3 — Archive:         Vector store of all raw blocks.
                          Uses NumpyL3Backend (pure numpy, all Python versions)
                          or ChromaL3Backend (Python <= 3.12, if chromadb installed).
                          Backend is auto-selected at init time.

This is the heart of Project Labyrinth's approach to unbounded context.
"""

from __future__ import annotations

import hashlib
import uuid
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np

from .encoder import SemanticEncoder
from .backends import L3Backend, auto_select_backend

logger = logging.getLogger(__name__)


@dataclass
class AnchorToken:
    """
    A compressed representation of a raw text block.

    The Anchor Token does NOT store every word — it stores the semantic
    essence as a dense vector, plus a short human-readable summary.
    The raw block lives in L3 and is always retrievable.

    content_hash (SHA-256): Fingerprint of the source_text at compression time.
    Used by the Truth-Loop to detect silent content changes (data fingerprinting).
    If the hash of the L3-stored block no longer matches content_hash, the anchor
    is considered stale and must be re-compressed.
    """
    id: str
    embedding: np.ndarray       # 384-dim semantic vector
    summary: str                # Short natural-language summary of the block
    source_text: str            # The raw text that was compressed (kept for L3 archival)
    token_count: int            # Token count of the original block
    content_hash: str = ""      # SHA-256 fingerprint of source_text (set on creation)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Auto-compute hash if not provided (backwards compatible)
        if not self.content_hash and self.source_text:
            self.content_hash = hashlib.sha256(
                self.source_text.encode("utf-8")
            ).hexdigest()

    def verify_integrity(self, raw_text: str) -> bool:
        """
        Verify that `raw_text` (e.g., retrieved from L3) still matches the
        SHA-256 fingerprint captured at compression time.

        Returns True if content is unchanged (anchor is still valid).
        Returns False if the underlying source has been silently modified.
        """
        current_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        return current_hash == self.content_hash

    def __repr__(self):
        return (f"AnchorToken(id={self.id[:8]}..., tokens={self.token_count}, "
                f"hash={self.content_hash[:8]}..., summary={self.summary[:40]!r})")


class LabyrinthMemory:
    """
    The three-layer memory manager for Project Labyrinth.

    Manages compression from L1 → L2 → L3 automatically as content grows,
    and provides semantic similarity search over all archived content.

    Args:
        l1_max_tokens:    Maximum tokens in the L1 working memory buffer.
        block_size_tokens: Size of blocks compressed into Anchor Tokens.
        collection_name:  Backend collection name (used by ChromaDB backend).
        encoder:          SemanticEncoder instance (shared or standalone).
        use_l3:           Enable L3 archive. If True, backend auto-selected.
        l3_backend:       Pass a custom L3Backend instance to override auto-selection.
        persist_path:     File path for NumpyL3Backend persistence (.npz/.json).
    """

    def __init__(
        self,
        l1_max_tokens: int = 4096,
        block_size_tokens: int = 512,
        collection_name: str = "labyrinth_archive",
        encoder: Optional[SemanticEncoder] = None,
        use_l3: bool = True,
        l3_backend: Optional[L3Backend] = None,
        persist_path: Optional[str] = None,
    ):
        self.l1_max_tokens = l1_max_tokens
        self.block_size_tokens = block_size_tokens
        self.encoder = encoder or SemanticEncoder()
        self.use_l3 = use_l3

        # L1 — raw text chunks (ring-buffer behaviour via deque)
        self._l1_chunks: deque[str] = deque()
        self._l1_token_counts: deque[int] = deque()
        self._l1_total_tokens: int = 0

        # L2 — list of AnchorTokens
        self._l2_anchors: List[AnchorToken] = []

        # L3 — pluggable backend (numpy or chroma, auto-selected)
        self._l3: Optional[L3Backend] = None
        if use_l3:
            self._l3 = l3_backend or auto_select_backend(
                collection_name=collection_name,
                persist_path=persist_path,
            )
            logger.info(f"L3 backend: {self._l3.name}")

        # Stats
        self.stats: Dict = {
            "l1_pushes": 0,
            "l2_compressions": 0,
            "l3_retrievals": 0,
            "total_tokens_seen": 0,
        }

    # ── L1 Operations ─────────────────────────────────────────────────────────

    def push_to_l1(self, text: str, token_count: int) -> Optional[AnchorToken]:
        """
        Push a new text chunk into L1 working memory.

        If L1 overflows, the oldest block is compressed and moved to L2/L3.

        Args:
            text:        The raw text chunk to add.
            token_count: Pre-computed token count for the chunk.

        Returns:
            The AnchorToken created if L1 overflow occurred, else None.
        """
        self._l1_chunks.append(text)
        self._l1_token_counts.append(token_count)
        self._l1_total_tokens += token_count
        self.stats["l1_pushes"] += 1
        self.stats["total_tokens_seen"] += token_count

        anchor = None
        # Drain from L1 front until within limit
        while self._l1_total_tokens > self.l1_max_tokens and len(self._l1_chunks) > 1:
            overflow_text = self._l1_chunks.popleft()
            overflow_tokens = self._l1_token_counts.popleft()
            self._l1_total_tokens -= overflow_tokens
            anchor = self._compress_to_l2(overflow_text, overflow_tokens)

        return anchor

    # ── L2 Compression ────────────────────────────────────────────────────────

    def _compress_to_l2(self, text: str, token_count: int) -> AnchorToken:
        """
        Encode a raw text block into an AnchorToken and archive to L3.
        """
        embedding = self.encoder.encode(text)
        summary = self._summarise(text)
        anchor = AnchorToken(
            id=str(uuid.uuid4()),
            embedding=embedding,
            summary=summary,
            source_text=text,
            token_count=token_count,
        )
        self._l2_anchors.append(anchor)
        self.stats["l2_compressions"] += 1
        logger.debug(f"Compressed block → L2: {anchor}")

        if self.use_l3:
            self._archive_to_l3(anchor)

        return anchor

    @staticmethod
    def _summarise(text: str, max_chars: int = 200) -> str:
        """
        Lightweight extractive summary: first two sentences or max_chars.
        No LLM required — pure string operations.
        """
        text = text.strip()
        sentences = text.replace("\n", " ").split(". ")
        summary = ". ".join(sentences[:2])
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
        return summary

    # ── L3 Archive ────────────────────────────────────────────────────────────

    def _archive_to_l3(self, anchor: AnchorToken):
        """Store a raw block in the L3 backend for future Truth-Loop retrieval."""
        if self._l3 is None:
            return
        try:
            self._l3.add(
                doc_id=anchor.id,
                embedding=anchor.embedding,
                text=anchor.source_text,
                metadata={
                    "summary": anchor.summary,
                    "token_count": anchor.token_count,
                    "timestamp": anchor.timestamp,
                    "content_hash": anchor.content_hash,
                },
            )
        except Exception as e:
            logger.error(f"L3 archive failed for anchor {anchor.id}: {e}")

    def verify_anchor_integrity(self, anchor_id: str) -> Optional[bool]:
        """
        Cross-check an AnchorToken's SHA-256 fingerprint against its L3 raw block.

        Returns:
            True  — content unchanged (anchor is valid).
            False — content changed (anchor is stale; should be invalidated).
            None  — anchor or L3 block not found (inconclusive).
        """
        anchor = self.get_anchor_by_id(anchor_id)
        if anchor is None:
            return None
        raw_text = self.retrieve_from_l3_by_id(anchor_id)
        if raw_text is None:
            return True  # L3 not available — L2 source_text is authoritative
        return anchor.verify_integrity(raw_text)

    def retrieve_from_l3(
        self,
        query_embedding: np.ndarray,
        n_results: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Semantic search over the L3 archive.

        Args:
            query_embedding: Query vector (384-dim).
            n_results:       Number of results to return.

        Returns:
            List of (raw_text, similarity_score) tuples, sorted by relevance.
        """
        if not self.use_l3 or self._l3 is None:
            return []
        self.stats["l3_retrievals"] += 1
        return self._l3.query(query_embedding, n_results=n_results)

    def retrieve_from_l3_by_id(self, anchor_id: str) -> Optional[str]:
        """Retrieve raw text for a specific anchor by ID."""
        if not self.use_l3 or self._l3 is None:
            anchor = self.get_anchor_by_id(anchor_id)
            return anchor.source_text if anchor else None
        return self._l3.get_by_id(anchor_id)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def l1_text(self) -> str:
        """Full concatenated L1 working memory text."""
        return "\n".join(self._l1_chunks)

    @property
    def l1_token_count(self) -> int:
        return self._l1_total_tokens

    @property
    def l2_anchors(self) -> List[AnchorToken]:
        return list(self._l2_anchors)

    @property
    def l2_anchor_count(self) -> int:
        return len(self._l2_anchors)

    @property
    def l3_count(self) -> int:
        if not self.use_l3 or self._l3 is None:
            return 0
        return self._l3.count()

    @property
    def l3_backend_name(self) -> str:
        """Name of the active L3 backend (e.g. 'NumpyL3Backend')."""
        return self._l3.name if self._l3 else "disabled"

    def get_anchor_by_id(self, anchor_id: str) -> Optional[AnchorToken]:
        for a in self._l2_anchors:
            if a.id == anchor_id:
                return a
        return None

    def get_l2_embeddings(self) -> np.ndarray:
        """All L2 anchor embeddings as a matrix (n_anchors, 384)."""
        if not self._l2_anchors:
            return np.empty((0, 384), dtype=np.float32)
        return np.stack([a.embedding for a in self._l2_anchors])

    def reset(self):
        """Clear all memory layers. Useful between sessions."""
        self._l1_chunks.clear()
        self._l1_token_counts.clear()
        self._l1_total_tokens = 0
        self._l2_anchors.clear()
        # L3 Backend: clear state via backend-specific reset
        if self.use_l3 and self._l3 is not None:
            self._l3.clear()
        self.stats = {k: 0 for k in self.stats}

    def summary(self) -> str:
        """Human-readable memory state summary."""
        total_original = self.stats["total_tokens_seen"]
        current = self.l1_token_count + self.l2_anchor_count * 50  # anchors ~50 tokens each
        ratio = (1 - current / total_original) * 100 if total_original > 0 else 0
        return (
            f"LabyrinthMemory State\n"
            f"  L1 (Working):  {self.l1_token_count:,} tokens ({len(self._l1_chunks)} chunks)\n"
            f"  L2 (Index):    {self.l2_anchor_count:,} anchor tokens\n"
            f"  L3 (Archive):  {self.l3_count:,} raw blocks\n"
            f"  Total seen:    {total_original:,} tokens\n"
            f"  Compression:   {ratio:.1f}% reduction\n"
            f"  Stats:         {self.stats}"
        )
