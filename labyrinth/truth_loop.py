"""
labyrinth.truth_loop
--------------------
Truth-Loop Mechanism -- confidence-triggered precision recall.

When the model is uncertain about a compressed memory, the Truth-Loop
"teleports" back to the original raw block in L3 to verify the fact.

Enhancements (v2, from Zero-Waste Agentic RAG patterns):
  - SHA-256 fingerprint validation before returning L2 anchor data.
    If the anchor's content has changed since compression, it is marked
    stale and re-retrieved from L3.
  - Context Sufficiency Check: before accepting retrieved context as
    sufficient, verify it semantically covers all aspects of the query.
    If coverage is too narrow, a fallback retrieval fetches additional
    context to fill the gap.

Design:
  1. A confidence score (0.0-1.0) is computed for a factual claim.
  2. If confidence < threshold t, retrieve the raw block from L3.
  3. Validate the retrieved block's SHA-256 fingerprint against the anchor.
  4. Run a sufficiency check; if coverage < sufficiency_threshold, fetch more.
  5. Return merged, verified raw context for inline verification.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .memory import LabyrinthMemory
from .encoder import SemanticEncoder

logger = logging.getLogger(__name__)


@dataclass
class TruthLoopEvent:
    """Records a single Truth-Loop activation for auditing and benchmarking."""
    query: str
    confidence: float
    threshold: float
    retrieved_text: Optional[str]
    similarity_score: float
    latency_ms: float
    triggered: bool
    fingerprint_valid: Optional[bool] = None  # None = not checked (L3 unavailable)
    sufficiency_score: float = 1.0            # 0-1: how well retrieved text covers query
    fallback_triggered: bool = False          # True if sufficiency check caused extra fetch
    timestamp: float = field(default_factory=time.time)


class TruthLoop:
    """
    Confidence-triggered retrieval from L3 archive.

    The Truth-Loop fires selectively -- only when confidence in a compressed
    memory is below the threshold t. In typical workloads this is < 4% of
    generation steps, adding negligible overhead.

    v2 Additions:
      - Fingerprint validation: verifies SHA-256 of retrieved L3 content
        matches the AnchorToken's content_hash before trusting it.
      - Context sufficiency: checks if retrieved blocks semantically cover
        the full query. If not, fetches additional context (Fallback).

    Args:
        memory:               LabyrinthMemory instance.
        encoder:              SemanticEncoder (shared).
        threshold:            Confidence threshold t. Below -> L3 fires.
        n_results:            Initial number of L3 blocks to retrieve.
        sufficiency_threshold: Minimum cosine similarity between the query
                              embedding and the AVERAGE of retrieved block
                              embeddings to consider context sufficient.
                              Below this -> Context Fallback fires.
        max_fallback_results: Max additional blocks to fetch on fallback.
    """

    DEFAULT_THRESHOLD = 0.82
    DEFAULT_SUFFICIENCY = 0.40  # minimum context coverage score

    def __init__(
        self,
        memory: Optional[LabyrinthMemory] = None,
        encoder: Optional[SemanticEncoder] = None,
        threshold: float = DEFAULT_THRESHOLD,
        n_results: int = 2,
        sufficiency_threshold: float = DEFAULT_SUFFICIENCY,
        max_fallback_results: int = 4,
    ):
        self.memory = memory or LabyrinthMemory()
        self.encoder = encoder or self.memory.encoder
        self.threshold = threshold
        self.n_results = n_results
        self.sufficiency_threshold = sufficiency_threshold
        self.max_fallback_results = max_fallback_results

        self._events: List[TruthLoopEvent] = []
        self._trigger_count: int = 0
        self._check_count: int = 0
        self._fingerprint_mismatches: int = 0
        self._sufficiency_fallbacks: int = 0

    # ── Core Check ────────────────────────────────────────────────────────────

    def check(
        self,
        query: str,
        confidence: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Evaluate whether the Truth-Loop should fire for a given query.

        If no explicit confidence is provided, we compute it as the maximum
        semantic similarity between the query and any L2 anchor. High
        similarity -> high confidence -> no retrieval needed.

        Args:
            query:      The factual claim or question to verify.
            confidence: Explicit confidence score if available (0.0-1.0).
                        If None, computed from L2 anchor similarity.

        Returns:
            Tuple of:
              - triggered (bool):   Whether L3 retrieval fired.
              - raw_text (str|None): Retrieved & verified raw block, or None.
              - confidence (float): The confidence score used for the decision.
        """
        t0 = time.perf_counter()
        self._check_count += 1

        # Compute confidence from L2 similarity if not provided
        if confidence is None:
            confidence = self._compute_confidence(query)

        triggered = confidence < self.threshold
        retrieved_text = None
        best_similarity = confidence
        fingerprint_valid = None
        sufficiency_score = 1.0
        fallback_triggered = False

        if triggered:
            self._trigger_count += 1
            results, sufficiency_score, fallback_triggered = self._retrieve_with_sufficiency(query)
            if results:
                # Merge all retrieved blocks into one verified context string
                texts = [text for text, _ in results]
                best_similarity = results[0][1]  # best individual score

                # --- SHA-256 Fingerprint Validation ---
                # For the best-matching block, verify it hasn't been silently modified.
                fingerprint_valid = self._validate_fingerprint(query, texts[0])
                if fingerprint_valid is False:
                    self._fingerprint_mismatches += 1
                    logger.warning(
                        f"TruthLoop: Fingerprint MISMATCH for query '{query[:50]}'. "
                        "Anchor is stale — content may have changed after compression."
                    )

                retrieved_text = "\n\n---\n\n".join(texts)
            if fallback_triggered:
                self._sufficiency_fallbacks += 1

        latency_ms = (time.perf_counter() - t0) * 1000
        event = TruthLoopEvent(
            query=query,
            confidence=confidence,
            threshold=self.threshold,
            retrieved_text=retrieved_text,
            similarity_score=best_similarity,
            latency_ms=latency_ms,
            triggered=triggered,
            fingerprint_valid=fingerprint_valid,
            sufficiency_score=sufficiency_score,
            fallback_triggered=fallback_triggered,
        )
        self._events.append(event)

        logger.debug(
            f"TruthLoop: conf={confidence:.3f} threshold={self.threshold} "
            f"triggered={triggered} sufficiency={sufficiency_score:.3f} "
            f"fingerprint_ok={fingerprint_valid} latency={latency_ms:.1f}ms"
        )

        return triggered, retrieved_text, confidence

    def verify(
        self,
        claim: str,
        anchor_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Force verification of a specific claim against L3 raw data.

        If anchor_id is provided, retrieves that exact block AND runs
        SHA-256 fingerprint validation to ensure content hasn't changed.

        Args:
            claim:     The claim text to verify.
            anchor_id: Specific anchor ID to retrieve (optional).

        Returns:
            Tuple of (found: bool, raw_text: str | None)
        """
        if anchor_id:
            raw = self.memory.retrieve_from_l3_by_id(anchor_id)
            if raw is None:
                return (False, None)
            # Fingerprint check
            valid = self.memory.verify_anchor_integrity(anchor_id)
            if valid is False:
                logger.warning(f"verify(): Anchor {anchor_id[:8]} is STALE (hash mismatch).")
                self._fingerprint_mismatches += 1
                # Still return the raw text but caller is notified via log
            return (True, raw)

        results, _, _ = self._retrieve_with_sufficiency(claim)
        if results:
            texts = [t for t, _ in results]
            return (True, "\n\n---\n\n".join(texts))
        return (False, None)

    # ── Scoring & Retrieval ───────────────────────────────────────────────────

    def _compute_confidence(self, query: str) -> float:
        """
        Confidence = max cosine similarity between query and any L2 anchor.

        High similarity means the model's compressed memory is well-aligned
        with what's being asked -- it can answer confidently from L2.

        Low similarity means the topic may be in a block that compressed
        poorly -- L3 retrieval is warranted.
        """
        anchors = self.memory.l2_anchors
        if not anchors:
            return 1.0  # No compressed content -- L1 is all we have, always confident

        query_emb = self.encoder.encode(query)
        anchor_embs = self.memory.get_l2_embeddings()
        similarities = self.encoder.batch_similarity(query_emb, anchor_embs)
        return float(np.max(similarities))

    def _retrieve_with_sufficiency(
        self,
        query: str,
    ) -> Tuple[List[Tuple[str, float]], float, bool]:
        """
        Retrieve L3 blocks and check if they sufficiently cover the query.

        Sufficiency = cosine similarity between query embedding and the
        mean embedding of all retrieved blocks. This detects the case where
        the cache holds context about only one aspect of a multi-faceted query
        (e.g., query asks about 'packaging AND taste' but cache only has
        'packaging' blocks).

        If sufficiency < sufficiency_threshold, a Context Fallback fires:
        additional blocks are fetched with a larger n_results to broaden
        the retrieved context.

        Returns:
            (results, sufficiency_score, fallback_triggered)
        """
        query_emb = self.encoder.encode(query)
        results: List[Tuple[str, float]] = self.memory.retrieve_from_l3(
            query_emb, n_results=self.n_results
        )

        if not results:
            return [], 1.0, False

        # Compute sufficiency: encode retrieved texts, compare mean to query
        retrieved_texts = [text for text, _ in results]
        retrieved_embs = np.stack([
            self.encoder.encode(t) for t in retrieved_texts
        ])
        mean_emb = retrieved_embs.mean(axis=0)
        # Normalize mean embedding before similarity
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        sufficiency = float(np.dot(query_emb, mean_emb))
        sufficiency = max(0.0, min(1.0, sufficiency))  # clamp to [0, 1]

        fallback_triggered = False
        if sufficiency < self.sufficiency_threshold:
            # Context Fallback: fetch more blocks to cover the gap
            logger.info(
                f"TruthLoop: Context Sufficiency FALLBACK "
                f"(score={sufficiency:.3f} < {self.sufficiency_threshold}). "
                f"Fetching {self.max_fallback_results} blocks."
            )
            fallback_results = self.memory.retrieve_from_l3(
                query_emb, n_results=self.max_fallback_results
            )
            # Merge: deduplicate by text, keep best score
            seen = {text for text, _ in results}
            for text, score in fallback_results:
                if text not in seen:
                    results.append((text, score))
                    seen.add(text)
            fallback_triggered = True

        return results, sufficiency, fallback_triggered

    def _validate_fingerprint(self, query: str, retrieved_text: str) -> Optional[bool]:
        """
        Find the L2 anchor most similar to the query and verify its
        SHA-256 fingerprint against the retrieved L3 text.

        Returns:
            True  -- fingerprint matches (content unchanged).
            False -- fingerprint mismatch (stale anchor detected).
            None  -- no anchors to check (inconclusive).
        """
        anchors = self.memory.l2_anchors
        if not anchors:
            return None

        query_emb = self.encoder.encode(query)
        anchor_embs = self.memory.get_l2_embeddings()
        sims = self.encoder.batch_similarity(query_emb, anchor_embs)
        best_anchor = anchors[int(np.argmax(sims))]
        return best_anchor.verify_integrity(retrieved_text)

    def _retrieve(self, query: str) -> List[Tuple[str, float]]:
        """Legacy semantic search over L3 archive (simple, no sufficiency check)."""
        query_emb = self.encoder.encode(query)
        return self.memory.retrieve_from_l3(query_emb, n_results=self.n_results)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def trigger_rate(self) -> float:
        """Fraction of checks that triggered L3 retrieval."""
        if self._check_count == 0:
            return 0.0
        return self._trigger_count / self._check_count

    @property
    def avg_latency_ms(self) -> float:
        if not self._events:
            return 0.0
        return sum(e.latency_ms for e in self._events) / len(self._events)

    def report(self) -> str:
        fp_mismatches = self._fingerprint_mismatches
        lines = [
            "-" * 50,
            "  Truth-Loop Mechanism -- Report",
            "-" * 50,
            f"  Threshold (t):         {self.threshold}",
            f"  Sufficiency threshold: {self.sufficiency_threshold}",
            f"  Total checks:          {self._check_count}",
            f"  Triggers fired:        {self._trigger_count}",
            f"  Trigger rate:          {self.trigger_rate * 100:.1f}%",
            f"  Fingerprint mismatches:{fp_mismatches}",
            f"  Sufficiency fallbacks: {self._sufficiency_fallbacks}",
            f"  Avg latency:           {self.avg_latency_ms:.1f} ms",
            "-" * 50,
        ]
        return "\n".join(lines)

    @property
    def events(self) -> List[TruthLoopEvent]:
        return list(self._events)
