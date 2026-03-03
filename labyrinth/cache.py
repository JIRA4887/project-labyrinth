"""
labyrinth.cache
---------------
Semantic Answer Cache (Tier 1) — Zero-Cost Query Deduplication.

Inspired by: "Zero-Waste Agentic RAG" (Partha Sarkar, Towards Data Science)

Problem: In multi-turn or multi-user sessions, semantically identical queries
trigger the full LLM pipeline repeatedly at full token cost. This is wasteful.

Solution: Cache LLM answers against query embeddings. On a new query, compute
cosine similarity to all cached entries. If similarity > hit_threshold (default
95%), return the cached answer instantly at $0 token cost.

Two-level design:
  Tier 1 — Semantic Hit (>= hit_threshold):  Return stored answer immediately.
  Tier 2 — Context Hit  (>= topic_threshold): Return stored retrieval context
            (skips vector DB search; LLM still generates a fresh answer).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .encoder import SemanticEncoder

logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """
    A single cached query→answer pair with validation metadata.
    """
    query: str
    query_embedding: np.ndarray
    answer: str
    context_chunks: List[str]           # Raw retrieval context blocks
    context_hash: str                   # SHA-256 of answer for staleness detection
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    last_hit_at: Optional[float] = None

    def is_stale(self, max_age_seconds: Optional[float] = None) -> bool:
        """Check if this entry has exceeded its TTL."""
        if max_age_seconds is None:
            return False
        return (time.time() - self.created_at) > max_age_seconds

    def record_hit(self):
        self.hit_count += 1
        self.last_hit_at = time.time()


@dataclass
class CacheResult:
    """The result of a cache lookup."""
    tier: int               # 0 = miss, 1 = semantic hit, 2 = context hit
    similarity: float       # Best similarity score found
    answer: Optional[str]   # Tier-1 hit: cached answer; Tier-2/miss: None
    context: List[str]      # Tier-2 hit: cached context chunks; else []
    entry_age_seconds: float = 0.0
    latency_ms: float = 0.0

    @property
    def is_miss(self) -> bool:
        return self.tier == 0

    @property
    def is_semantic_hit(self) -> bool:
        return self.tier == 1

    @property
    def is_context_hit(self) -> bool:
        return self.tier == 2


# ── Temporal Intent Detection ──────────────────────────────────────────────────

_TEMPORAL_KEYWORDS = frozenset([
    "latest", "current", "recent", "today", "now", "real-time", "realtime",
    "live", "right now", "at the moment", "as of", "up-to-date", "newest",
    "last hour", "last 24", "this week", "this month", "just happened",
    "breaking", "updated", "fresh",
])

def has_temporal_intent(query: str) -> bool:
    """
    Detect if a query is asking for time-sensitive / real-time data.

    Such queries should BYPASS the cache entirely and always go to the
    live source (L3 archive or the LLM with no compression bypass).

    >>> has_temporal_intent("What are the latest results?")
    True
    >>> has_temporal_intent("Explain how attention works")
    False
    """
    q_lower = query.lower()
    return any(kw in q_lower for kw in _TEMPORAL_KEYWORDS)


# ── Main Cache Class ────────────────────────────────────────────────────────────

class SemanticAnswerCache:
    """
    Two-tier semantic cache for LLM answers and retrieval context.

    Tier 1 — Semantic Hit (similarity >= hit_threshold):
        Returns the full cached LLM answer + zero token cost.

    Tier 2 — Context Hit (similarity >= topic_threshold):
        Returns cached retrieval context chunks. LLM still runs, but
        the expensive vector DB / L3 lookup is skipped.

    Args:
        encoder:         SemanticEncoder instance (share with the rest of Labyrinth).
        hit_threshold:   Cosine similarity for a Tier-1 full-answer cache hit (0.95).
        topic_threshold: Cosine similarity for a Tier-2 context cache hit (0.70).
        max_entries:     Maximum number of entries to keep (LRU eviction).
        max_age_seconds: Optional TTL. Entries older than this are considered stale.
        bypass_temporal: If True, queries with temporal intent bypass the cache.
    """

    def __init__(
        self,
        encoder: Optional[SemanticEncoder] = None,
        hit_threshold: float = 0.95,
        topic_threshold: float = 0.70,
        max_entries: int = 512,
        max_age_seconds: Optional[float] = None,
        bypass_temporal: bool = True,
    ):
        self.encoder = encoder or SemanticEncoder()
        self.hit_threshold = hit_threshold
        self.topic_threshold = topic_threshold
        self.max_entries = max_entries
        self.max_age_seconds = max_age_seconds
        self.bypass_temporal = bypass_temporal

        self._entries: List[CacheEntry] = []

        # Stats
        self._total_queries: int = 0
        self._tier1_hits: int = 0
        self._tier2_hits: int = 0
        self._bypasses: int = 0  # temporal intent bypasses

    # ── Core Lookup ────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> CacheResult:
        """
        Look up a query in the cache.

        Returns a CacheResult with tier=0 (miss), 1 (answer hit), or 2 (context hit).

        The caller should:
          - On Tier-1 hit: return result.answer directly to user (no LLM call).
          - On Tier-2 hit: use result.context as retrieval context for LLM.
          - On miss: run full pipeline, then call store() with the answer.
        """
        t0 = time.perf_counter()
        self._total_queries += 1

        # Temporal bypass: never serve cached data for time-sensitive queries
        if self.bypass_temporal and has_temporal_intent(query):
            self._bypasses += 1
            logger.debug(f"Cache BYPASS (temporal intent): {query[:60]}")
            return CacheResult(
                tier=0, similarity=0.0, answer=None, context=[],
                latency_ms=(time.perf_counter() - t0) * 1000
            )

        if not self._entries:
            return CacheResult(
                tier=0, similarity=0.0, answer=None, context=[],
                latency_ms=(time.perf_counter() - t0) * 1000
            )

        query_emb = self.encoder.encode(query)
        best_sim, best_entry = self._find_best(query_emb)

        latency_ms = (time.perf_counter() - t0) * 1000

        if best_entry is None or best_sim < self.topic_threshold:
            logger.debug(f"Cache MISS (sim={best_sim:.3f}): {query[:60]}")
            return CacheResult(tier=0, similarity=best_sim, answer=None,
                               context=[], latency_ms=latency_ms)

        age = time.time() - best_entry.created_at

        # Check staleness
        if best_entry.is_stale(self.max_age_seconds):
            logger.debug(f"Cache STALE (age={age:.0f}s): invalidating entry.")
            self._entries.remove(best_entry)
            return CacheResult(tier=0, similarity=best_sim, answer=None,
                               context=[], latency_ms=latency_ms)

        best_entry.record_hit()

        if best_sim >= self.hit_threshold:
            # Tier-1: full semantic answer hit
            self._tier1_hits += 1
            logger.info(
                f"Cache TIER-1 HIT (sim={best_sim:.3f}, age={age:.0f}s, "
                f"hits={best_entry.hit_count}): {query[:60]}"
            )
            return CacheResult(
                tier=1, similarity=best_sim,
                answer=best_entry.answer,
                context=best_entry.context_chunks,
                entry_age_seconds=age,
                latency_ms=latency_ms,
            )

        # Tier-2: topic/context hit — skip vector DB, reuse raw context
        self._tier2_hits += 1
        logger.info(
            f"Cache TIER-2 HIT (sim={best_sim:.3f}, age={age:.0f}s): {query[:60]}"
        )
        return CacheResult(
            tier=2, similarity=best_sim,
            answer=None,
            context=best_entry.context_chunks,
            entry_age_seconds=age,
            latency_ms=latency_ms,
        )

    def _find_best(
        self, query_emb: np.ndarray
    ) -> Tuple[float, Optional[CacheEntry]]:
        """Find the most similar cache entry to the query embedding."""
        if not self._entries:
            return 0.0, None

        cache_embs = np.stack([e.query_embedding for e in self._entries])
        sims = self.encoder.batch_similarity(query_emb, cache_embs)
        best_idx = int(np.argmax(sims))
        return float(sims[best_idx]), self._entries[best_idx]

    # ── Store ──────────────────────────────────────────────────────────────────

    def store(
        self,
        query: str,
        answer: str,
        context_chunks: Optional[List[str]] = None,
    ) -> CacheEntry:
        """
        Store a new query→answer pair in the cache.

        Args:
            query:          The user query text.
            answer:         The LLM-generated answer.
            context_chunks: The raw retrieval context blocks used to generate
                            the answer (stored for Tier-2 hits on related queries).

        Returns:
            The created CacheEntry.
        """
        # Evict LRU entry if at capacity
        if len(self._entries) >= self.max_entries:
            oldest = min(self._entries, key=lambda e: e.last_hit_at or e.created_at)
            self._entries.remove(oldest)
            logger.debug("Cache evicted oldest entry (LRU).")

        query_emb = self.encoder.encode(query)
        content_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()

        entry = CacheEntry(
            query=query,
            query_embedding=query_emb,
            answer=answer,
            context_chunks=context_chunks or [],
            context_hash=content_hash,
        )
        self._entries.append(entry)
        logger.debug(f"Cache STORED: {query[:60]} (hash={content_hash[:8]})")
        return entry

    # ── Invalidation ───────────────────────────────────────────────────────────

    def invalidate(self, query: str, similarity_threshold: float = 0.95) -> int:
        """
        Invalidate all cache entries semantically similar to the given query.
        Use when the underlying data has changed (e.g., after a document update).

        Returns number of entries removed.
        """
        if not self._entries:
            return 0
        query_emb = self.encoder.encode(query)
        cache_embs = np.stack([e.query_embedding for e in self._entries])
        sims = self.encoder.batch_similarity(query_emb, cache_embs)
        to_remove = [e for e, s in zip(self._entries, sims) if s >= similarity_threshold]
        for e in to_remove:
            self._entries.remove(e)
        if to_remove:
            logger.info(f"Cache invalidated {len(to_remove)} entries for: {query[:60]}")
        return len(to_remove)

    def invalidate_by_content_hash(self, answer_text: str) -> int:
        """
        Invalidate any entry whose stored answer matches the SHA-256 of `answer_text`.
        Useful when you know exactly which cached answer changed.
        """
        target_hash = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()
        to_remove = [e for e in self._entries if e.context_hash == target_hash]
        for e in to_remove:
            self._entries.remove(e)
        return len(to_remove)

    def clear(self):
        """Remove all cache entries."""
        self._entries.clear()
        logger.info("SemanticAnswerCache cleared.")

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def tier1_hit_rate(self) -> float:
        if self._total_queries == 0:
            return 0.0
        return self._tier1_hits / self._total_queries

    @property
    def tier2_hit_rate(self) -> float:
        if self._total_queries == 0:
            return 0.0
        return self._tier2_hits / self._total_queries

    @property
    def overall_hit_rate(self) -> float:
        if self._total_queries == 0:
            return 0.0
        return (self._tier1_hits + self._tier2_hits) / self._total_queries

    @property
    def total_queries(self) -> int:
        return self._total_queries

    def report(self) -> str:
        lines = [
            "-" * 50,
            "  Semantic Answer Cache -- Report",
            "-" * 50,
            f"  Entries cached:     {self.size} / {self.max_entries}",
            f"  Total queries:      {self._total_queries}",
            f"  Tier-1 hits:        {self._tier1_hits} ({self.tier1_hit_rate * 100:.1f}%)",
            f"  Tier-2 hits:        {self._tier2_hits} ({self.tier2_hit_rate * 100:.1f}%)",
            f"  Temporal bypasses:  {self._bypasses}",
            f"  Overall hit rate:   {self.overall_hit_rate * 100:.1f}%",
            f"  Hit threshold:      {self.hit_threshold}",
            f"  Topic threshold:    {self.topic_threshold}",
            "-" * 50,
        ]
        return "\n".join(lines)
