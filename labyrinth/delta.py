"""
labyrinth.delta
---------------
Delta Update Protocol — the efficiency engine of Project Labyrinth.

Instead of sending the full conversation history with every prompt,
the Delta Protocol transmits only newly added content and assembles
a compressed context from L2 anchor summaries + L1 raw text.

Cost comparison (paper claim):
  Standard:  100,000 tokens @ $0.015/1K = $1.50 per prompt
  Labyrinth: ~500    tokens @ $0.015/1K = $0.01 per prompt (99.5% reduction)
"""

from __future__ import annotations

import time
import logging
from typing import List, Optional, Dict, Any

from .memory import LabyrinthMemory

logger = logging.getLogger(__name__)

# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _get_tokenizer():
    """Load tiktoken tokenizer (cl100k_base = GPT-4/GPT-4o encoding)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        raise ImportError(
            "tiktoken is required for accurate token counting.\n"
            "Install with: pip install tiktoken"
        )


def count_tokens(text: str) -> int:
    """Count tokens in a string using OpenAI's cl100k_base tokenizer."""
    enc = _get_tokenizer()
    return len(enc.encode(text))


# ── Delta Protocol ────────────────────────────────────────────────────────────

class DeltaProtocol:
    """
    Manages the conversation state and assembles compressed context prompts.

    The key insight: instead of passing the full message history to the LLM
    on every turn, we maintain a compressed state and only pass the delta
    (new content) plus a compact index (anchor summaries) plus the L1 window.

    Args:
        memory:            LabyrinthMemory instance to use for storage.
        system_prompt:     Optional system prompt to always prepend.
        anchor_in_context: Whether to include anchor summaries in context.
    """

    def __init__(
        self,
        memory: Optional[LabyrinthMemory] = None,
        system_prompt: str = "",
        anchor_in_context: bool = True,
    ):
        self.memory = memory or LabyrinthMemory()
        self.system_prompt = system_prompt
        self.anchor_in_context = anchor_in_context

        self._raw_history: List[str] = []   # Full uncompressed history (for baseline comparison)
        self._turn_count: int = 0
        self._timing: Dict[str, float] = {}

    # ── Push ──────────────────────────────────────────────────────────────────

    def push(self, text: str) -> int:
        """
        Add a new message/turn to the system.

        Tokenizes the text, pushes to L1, and handles overflow compression
        via the LabyrinthMemory manager.

        Args:
            text: The new text to add (user message, assistant response, etc.)

        Returns:
            Token count of the pushed text.
        """
        tokens = count_tokens(text)
        self._raw_history.append(text)
        self._turn_count += 1
        self.memory.push_to_l1(text, tokens)
        logger.debug(f"Turn {self._turn_count}: pushed {tokens} tokens")
        return tokens

    # ── Assemble Context ──────────────────────────────────────────────────────

    def assemble_context(self) -> str:
        """
        Build the compressed context string to send to the LLM.

        Structure:
            [SYSTEM PROMPT]
            [MEMORY INDEX — L2 anchor summaries]
            [WORKING MEMORY — L1 raw text]

        Returns:
            The assembled context string.
        """
        t0 = time.perf_counter()
        parts: List[str] = []

        # System prompt
        if self.system_prompt:
            parts.append(f"[SYSTEM]\n{self.system_prompt}\n")

        # L2 Anchor Index
        if self.anchor_in_context and self.memory.l2_anchors:
            index_parts = ["[MEMORY INDEX — Compressed Historical Context]"]
            for i, anchor in enumerate(self.memory.l2_anchors):
                index_parts.append(f"  [{i+1}] {anchor.summary}")
            parts.append("\n".join(index_parts))

        # L1 Working Memory
        l1 = self.memory.l1_text
        if l1:
            parts.append(f"[WORKING MEMORY — Recent Context]\n{l1}")

        context = "\n\n".join(parts)
        self._timing["assemble_ms"] = (time.perf_counter() - t0) * 1000
        return context

    # ── Token Measurement ─────────────────────────────────────────────────────

    def raw_token_count(self) -> int:
        """Token count of the full uncompressed history (what standard LLMs would send)."""
        return count_tokens("\n".join(self._raw_history))

    def labyrinth_token_count(self) -> int:
        """Token count of the Labyrinth-compressed context."""
        return count_tokens(self.assemble_context())

    def compression_ratio(self) -> float:
        """
        Fraction of tokens saved vs. raw history.
        0.92 = 92% token reduction (Labyrinth sends 8% of what raw would send).
        """
        raw = self.raw_token_count()
        if raw == 0:
            return 0.0
        compressed = self.labyrinth_token_count()
        return 1.0 - (compressed / raw)

    def cost_comparison(self, price_per_1k: float = 0.015) -> Dict[str, Any]:
        """
        Estimated cost comparison for current context state.

        Args:
            price_per_1k: Cost per 1,000 tokens in USD. Default: $0.015 (GPT-4o class).

        Returns:
            Dict with raw_cost, labyrinth_cost, savings_pct, savings_usd.
        """
        raw = self.raw_token_count()
        compressed = self.labyrinth_token_count()
        raw_cost = (raw / 1000) * price_per_1k
        lby_cost = (compressed / 1000) * price_per_1k
        return {
            "raw_tokens": raw,
            "labyrinth_tokens": compressed,
            "raw_cost_usd": round(raw_cost, 4),
            "labyrinth_cost_usd": round(lby_cost, 4),
            "savings_pct": round(self.compression_ratio() * 100, 1),
            "savings_usd": round(raw_cost - lby_cost, 4),
            "turn_count": self._turn_count,
            "assemble_ms": round(self._timing.get("assemble_ms", 0), 2),
        }

    def report(self) -> str:
        """Print a formatted report of compression statistics."""
        c = self.cost_comparison()
        lines = [
            "─" * 55,
            "  Project Labyrinth — Delta Protocol Report",
            "─" * 55,
            f"  Turns processed:      {c['turn_count']}",
            f"  Raw token count:      {c['raw_tokens']:,}",
            f"  Labyrinth tokens:     {c['labyrinth_tokens']:,}",
            f"  Compression:          {c['savings_pct']}% reduction",
            f"  Raw cost:             ${c['raw_cost_usd']:.4f}",
            f"  Labyrinth cost:       ${c['labyrinth_cost_usd']:.4f}",
            f"  Savings:              ${c['savings_usd']:.4f}",
            f"  Assembly latency:     {c['assemble_ms']} ms",
            "─" * 55,
        ]
        return "\n".join(lines)
