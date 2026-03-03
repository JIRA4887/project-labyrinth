"""
labyrinth.proxy
---------------
OpenAI-compatible proxy wrapper — v2.

Drop-in replacement for the OpenAI messages list.
Intercepts outgoing chat requests, compresses history via DeltaProtocol,
checks the SemanticAnswerCache before calling the LLM, and returns a new
messages list with the compressed context.

v2 Enhancements (Zero-Waste Agentic RAG patterns):
  - Tier-1 Semantic Answer Cache: Near-identical queries (>=95% sim) return
    a cached LLM answer instantly at $0 token cost.
  - Tier-2 Context Cache: Related queries (>=70% sim) reuse cached retrieval
    context to skip L3 search overhead.
  - Temporal Intent Bypass: Queries with keywords like "latest", "current",
    "today" bypass the cache and go straight to the live L3 / LLM pipeline,
    ensuring real-time queries never return stale cached data.

Usage:
    from labyrinth import LabyrinthProxy

    proxy = LabyrinthProxy()

    # Ask a question (auto-checks cache):
    cache_result, messages = proxy.ask("What is the JWT secret location?")
    if cache_result.is_semantic_hit:
        print(cache_result.answer)  # instant, $0 cost
    else:
        response = openai.chat.completions.create(model="gpt-4o", messages=messages)
        proxy.push_assistant(response.choices[0].message.content)
        # Store in cache for future deduplication:
        proxy.store_answer(query, response.choices[0].message.content)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple

from .memory import LabyrinthMemory
from .delta import DeltaProtocol
from .truth_loop import TruthLoop
from .encoder import SemanticEncoder
from .cache import SemanticAnswerCache, CacheResult, has_temporal_intent

logger = logging.getLogger(__name__)

# OpenAI message format
Message = Dict[str, str]  # {"role": "...", "content": "..."}


class LabyrinthProxy:
    """
    OpenAI-compatible proxy that transparently compresses conversation history
    and eliminates redundant LLM calls via semantic caching.

    This is the "drop-in proxy" described in the paper. Any application that
    currently builds a `messages` list and sends it to an LLM can use this
    class with minimal code changes.

    Args:
        l1_max_tokens:      Working memory size. Default: 4096 tokens.
        system_prompt:      Persistent system prompt to always include.
        truth_loop_tau:     Confidence threshold for Truth-Loop activation.
        use_l3:             Enable ChromaDB L3 archive. Default: True.
        cache_hit_threshold: Cosine similarity for Tier-1 answer cache hit (0.95).
        cache_topic_threshold: Cosine similarity for Tier-2 context hit (0.70).
        cache_max_age_seconds: Optional TTL for cache entries (None = no expiry).
        enable_cache:       Master switch to enable/disable the answer cache.
    """

    def __init__(
        self,
        l1_max_tokens: int = 4096,
        system_prompt: str = "",
        truth_loop_tau: float = 0.82,
        use_l3: bool = True,
        cache_hit_threshold: float = 0.95,
        cache_topic_threshold: float = 0.70,
        cache_max_age_seconds: Optional[float] = None,
        enable_cache: bool = True,
    ):
        self._encoder = SemanticEncoder()
        self._memory = LabyrinthMemory(
            l1_max_tokens=l1_max_tokens,
            encoder=self._encoder,
            use_l3=use_l3,
        )
        self._delta = DeltaProtocol(
            memory=self._memory,
            system_prompt=system_prompt,
        )
        self._truth_loop = TruthLoop(
            memory=self._memory,
            encoder=self._encoder,
            threshold=truth_loop_tau,
        )
        self._cache = SemanticAnswerCache(
            encoder=self._encoder,
            hit_threshold=cache_hit_threshold,
            topic_threshold=cache_topic_threshold,
            max_age_seconds=cache_max_age_seconds,
        ) if enable_cache else None

        self._message_log: List[Message] = []
        self._cache_enabled = enable_cache

    # ── Message Ingestion ─────────────────────────────────────────────────────

    def push(self, role: str, content: str):
        """
        Add a message to the Labyrinth memory system.

        Args:
            role:    "user", "assistant", or "system"
            content: Message text content.
        """
        formatted = f"[{role.upper()}]: {content}"
        self._delta.push(formatted)
        self._message_log.append({"role": role, "content": content})

    def push_user(self, content: str):
        """Convenience method for user messages."""
        self.push("user", content)

    def push_assistant(self, content: str):
        """Convenience method for assistant responses."""
        self.push("assistant", content)

    # ── Cache-Aware Ask ───────────────────────────────────────────────────────

    def ask(self, query: str) -> Tuple[CacheResult, List[Message]]:
        """
        Primary entry-point for sending a query through Labyrinth.

        Checks the Semantic Answer Cache first:
          - Tier-1 hit (>=95% sim): Returns cached answer, no LLM call needed.
          - Tier-2 hit (>=70% sim): Returns cached context + compressed messages.
          - Miss / temporal bypass: Returns full compressed message list.

        Temporal intent queries (containing "latest", "current", "now", etc.)
        ALWAYS bypass the cache to ensure fresh data.

        Args:
            query: The user's question or message.

        Returns:
            Tuple of (CacheResult, messages_list).
            - CacheResult.tier == 1: Use result.answer directly (skip LLM).
            - CacheResult.tier == 2: Inject result.context into your prompt.
            - CacheResult.tier == 0: Call LLM with the returned messages.
        """
        # Push the user query into memory
        self.push_user(query)

        # Temporal intent check — always log but let cache handle bypass
        if has_temporal_intent(query):
            logger.info(f"Temporal intent detected in query: '{query[:60]}' -> bypassing cache.")

        # Check cache
        if self._cache is not None:
            cache_result = self._cache.lookup(query)
        else:
            cache_result = CacheResult(tier=0, similarity=0.0, answer=None, context=[])

        # Build the compressed messages either way
        messages = self._build_messages(cache_result)

        return cache_result, messages

    def _build_messages(self, cache_result: CacheResult) -> List[Message]:
        """Assemble the messages list for the LLM, injecting cache context if available."""
        compressed_context = self._delta.assemble_context()

        # If Tier-2 hit, prepend the cached context blocks to the context
        if cache_result.is_context_hit and cache_result.context:
            context_section = "\n\n[CACHED RETRIEVAL CONTEXT]\n" + "\n---\n".join(cache_result.context)
            compressed_context = context_section + "\n\n" + compressed_context if compressed_context else context_section

        messages = []
        if compressed_context:
            messages.append({"role": "system", "content": compressed_context})

        return messages

    def store_answer(
        self,
        query: str,
        answer: str,
        context_chunks: Optional[List[str]] = None,
    ):
        """
        Store an LLM answer in the Semantic Answer Cache.

        Call this after receiving a response from the LLM to enable
        future Tier-1/2 cache hits for semantically similar queries.

        Args:
            query:          The user query that produced this answer.
            answer:         The LLM-generated answer text.
            context_chunks: The raw retrieval context used (for Tier-2 hits).
        """
        if self._cache is not None:
            self._cache.store(query, answer, context_chunks)

    # ── Legacy compress() interface ───────────────────────────────────────────

    def compress(self, messages: Optional[List[Message]] = None) -> List[Message]:
        """
        Compress the conversation history into a minimal messages list.

        Legacy interface (v1 compatible). Does not use the answer cache.
        For cache-aware usage, use ask() instead.

        Args:
            messages: Optional list of new messages to ingest before compressing.

        Returns:
            A new messages list for any OpenAI-compatible API.
        """
        if messages:
            for msg in messages:
                self.push(msg.get("role", "user"), msg.get("content", ""))

        compressed_context = self._delta.assemble_context()
        compressed_messages = []
        if compressed_context:
            compressed_messages.append({
                "role": "system",
                "content": compressed_context,
            })
        return compressed_messages

    def check_truth_loop(self, query: str, confidence: Optional[float] = None) -> Optional[str]:
        """
        Run a Truth-Loop check for a given query.

        If confidence is below tau, retrieves, fingerprint-validates, and
        returns the relevant raw block from L3. Returns None otherwise.

        Args:
            query:      The factual claim or question.
            confidence: Explicit confidence score (0-1). Computed if None.

        Returns:
            Raw source text if Truth-Loop fired, else None.
        """
        triggered, raw_text, conf = self._truth_loop.check(query, confidence)
        if triggered and raw_text:
            logger.info(f"Truth-Loop fired: conf={conf:.3f} -> injecting raw context.")
            return raw_text
        return None

    def invalidate_cache(self, query: str) -> int:
        """
        Invalidate cache entries similar to the given query.

        Use when the underlying knowledge has changed (e.g., document updated)
        to prevent stale Tier-1 hits.

        Returns number of entries removed.
        """
        if self._cache is None:
            return 0
        return self._cache.invalidate(query)

    # ── Stats & Reports ───────────────────────────────────────────────────────

    def report(self) -> str:
        """Full compression, cache, and truth-loop statistics report."""
        sections = [
            self._delta.report(),
            self._truth_loop.report(),
            self._memory.summary(),
        ]
        if self._cache is not None:
            sections.insert(1, self._cache.report())
        return "\n\n".join(sections)

    @property
    def delta(self) -> DeltaProtocol:
        return self._delta

    @property
    def memory(self) -> LabyrinthMemory:
        return self._memory

    @property
    def truth_loop(self) -> TruthLoop:
        return self._truth_loop

    @property
    def cache(self) -> Optional[SemanticAnswerCache]:
        return self._cache

    @property
    def message_count(self) -> int:
        return len(self._message_log)
