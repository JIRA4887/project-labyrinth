"""
tests/test_delta.py
Unit tests for the Delta Update Protocol.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from labyrinth.memory import LabyrinthMemory
from labyrinth.delta import DeltaProtocol, count_tokens


# ── Tokenizer Tests ───────────────────────────────────────────────────────────

class TestTokenCounter:

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_tokens_hello(self):
        # "Hello, world!" = known token count
        count = count_tokens("Hello, world!")
        assert count > 0

    def test_count_tokens_scales_with_length(self):
        short = count_tokens("Hello")
        long  = count_tokens("Hello " * 100)
        assert long > short * 50  # rough proportionality


# ── DeltaProtocol Tests ───────────────────────────────────────────────────────

@pytest.fixture
def dp():
    """Fresh DeltaProtocol with small L1 window for test control."""
    mem = LabyrinthMemory(l1_max_tokens=200, use_l3=False)
    return DeltaProtocol(memory=mem, system_prompt="You are a helpful assistant.")


class TestDeltaProtocol:

    def test_push_returns_token_count(self, dp):
        tokens = dp.push("Hello, how are you?")
        assert tokens > 0

    def test_raw_token_count_grows_with_turns(self, dp):
        dp.push("First message about machine learning.")
        count1 = dp.raw_token_count()
        dp.push("Second message about neural networks.")
        count2 = dp.raw_token_count()
        assert count2 > count1

    def test_assemble_context_includes_system_prompt(self, dp):
        dp.push("Test message.")
        ctx = dp.assemble_context()
        assert "You are a helpful assistant." in ctx

    def test_assemble_context_includes_l1_content(self, dp):
        dp.push("The quick brown fox jumps.")
        ctx = dp.assemble_context()
        assert "quick brown fox" in ctx

    def test_compression_ratio_improves_with_more_turns(self, dp):
        """After many turns (forcing L2 compression), Labyrinth should save tokens."""
        # Use a fresh DeltaProtocol with larger L1 so we can push bigger blocks
        big_mem = LabyrinthMemory(l1_max_tokens=2000, use_l3=False)
        big_dp  = DeltaProtocol(memory=big_mem)

        # Each turn is ~60 tokens; 80 turns ≈ 4,800 tokens > 2,000 L1 → forces overflow
        long_turn = (
            "This is a detailed software engineering discussion turn. "
            "We are reviewing the authentication module and its implementation details. "
            "The JWT token flow, validation logic, and refresh token mechanism are all key components. "
        ) * 3  # ~60 tokens

        for i in range(80):
            big_dp.push(f"Turn {i}: " + long_turn)

        ratio = big_dp.compression_ratio()
        # We expect Labyrinth to save at least 30% once L2 compression kicks in
        assert ratio > 0.30, f"Compression ratio {ratio:.2%} is lower than expected"

    def test_labyrinth_tokens_less_than_raw(self, dp):
        """Labyrinth context should use fewer tokens than raw history after compression."""
        for i in range(30):
            dp.push("Repeated content block " * 10 + f" turn {i}")
        raw = dp.raw_token_count()
        compressed = dp.labyrinth_token_count()
        assert compressed < raw, (
            f"Expected compressed ({compressed}) < raw ({raw})"
        )

    def test_context_includes_anchor_summaries_after_overflow(self, dp):
        """After L1 overflow, assemble_context() should include L2 anchor summaries."""
        for _ in range(30):
            dp.push("Overflow content block that causes L2 compression to trigger now.")
        ctx = dp.assemble_context()
        if dp.memory.l2_anchor_count > 0:
            assert "MEMORY INDEX" in ctx

    def test_cost_comparison_structure(self, dp):
        """cost_comparison() should return all expected keys."""
        dp.push("Test message for cost comparison output check.")
        result = dp.cost_comparison()
        expected_keys = [
            "raw_tokens", "labyrinth_tokens", "raw_cost_usd",
            "labyrinth_cost_usd", "savings_pct", "savings_usd",
            "turn_count", "assemble_ms"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_turn_count_increments(self, dp):
        """Turn count should reflect number of pushes."""
        for _ in range(5):
            dp.push("A message.")
        assert dp._turn_count == 5

    def test_report_is_string(self, dp):
        dp.push("Generate a report.")
        report = dp.report()
        assert isinstance(report, str)
        assert "Compression" in report
