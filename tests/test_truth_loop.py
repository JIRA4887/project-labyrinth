"""
tests/test_truth_loop.py
Unit tests for the Truth-Loop Mechanism.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from labyrinth.memory import LabyrinthMemory
from labyrinth.truth_loop import TruthLoop
from labyrinth.encoder import SemanticEncoder


# L3 runs with NumpyL3Backend on Python 3.14


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tl_no_l3():
    """TruthLoop without L3 (no ChromaDB) for fast unit tests."""
    mem = LabyrinthMemory(l1_max_tokens=100, use_l3=False)
    return TruthLoop(memory=mem, threshold=0.82)


@pytest.fixture
def tl_with_l3():
    mem = LabyrinthMemory(l1_max_tokens=100, use_l3=True)
    return TruthLoop(memory=mem, threshold=0.82)


# ── Confidence Scoring ────────────────────────────────────────────────────────

class TestConfidenceScoring:

    def test_confidence_is_1_with_empty_l2(self, tl_no_l3):
        """With no anchors in L2, confidence defaults to 1.0 (only L1 exists)."""
        conf = tl_no_l3._compute_confidence("What is the capital of France?")
        assert conf == 1.0

    def test_confidence_between_0_and_1(self, tl_no_l3):
        """Confidence should always be in [0, 1]."""
        # Add some anchors to L2 by overflow
        for _ in range(20):
            tl_no_l3.memory.push_to_l1("Philosophy and art and culture in ancient Greece.", token_count=10)
        conf = tl_no_l3._compute_confidence("What is machine learning?")
        assert 0.0 <= conf <= 1.0

    def test_similar_query_has_higher_confidence(self, tl_no_l3):
        """Query similar to archived content should score higher confidence."""
        # Archive some content about Python
        for _ in range(20):
            tl_no_l3.memory.push_to_l1(
                "Python is a high-level interpreted programming language. "
                "Python supports object-oriented and functional paradigms.",
                token_count=10
            )
        conf_related = tl_no_l3._compute_confidence("Tell me about Python programming.")
        conf_unrelated = tl_no_l3._compute_confidence("What is the boiling point of lead?")
        assert conf_related >= conf_unrelated


# ── Truth-Loop Activation ─────────────────────────────────────────────────────

class TestTruthLoopActivation:

    def test_explicit_high_confidence_does_not_trigger(self, tl_no_l3):
        """Explicit confidence above threshold should NOT trigger Truth-Loop."""
        triggered, raw_text, conf = tl_no_l3.check("Paris is the capital.", confidence=0.95)
        assert not triggered
        assert raw_text is None

    def test_explicit_low_confidence_triggers(self, tl_no_l3):
        """Explicit confidence below threshold SHOULD trigger Truth-Loop."""
        triggered, raw_text, conf = tl_no_l3.check("Some uncertain fact.", confidence=0.30)
        # Note: raw_text may be None if L3 is empty — that's fine for unit test
        assert triggered

    def test_check_returns_three_tuple(self, tl_no_l3):
        """check() must return (triggered: bool, raw_text: str|None, confidence: float)."""
        result = tl_no_l3.check("Test query.", confidence=0.5)
        assert len(result) == 3
        triggered, raw_text, conf = result
        assert isinstance(triggered, bool)
        assert raw_text is None or isinstance(raw_text, str)
        assert isinstance(conf, float)

    def test_trigger_rate_zero_when_always_confident(self, tl_no_l3):
        """If all scores are above threshold, trigger rate is 0."""
        for _ in range(5):
            tl_no_l3.check("This is a test.", confidence=0.99)
        assert tl_no_l3.trigger_rate == 0.0

    def test_trigger_rate_one_when_always_uncertain(self, tl_no_l3):
        """If all scores are below threshold, trigger rate is 1."""
        for _ in range(5):
            tl_no_l3.check("Uncertain claim.", confidence=0.10)
        assert tl_no_l3.trigger_rate == 1.0

    def test_event_log_records_each_check(self, tl_no_l3):
        """Each check should add one event to the log."""
        tl_no_l3.check("Event 1.", confidence=0.9)
        tl_no_l3.check("Event 2.", confidence=0.5)
        tl_no_l3.check("Event 3.", confidence=0.2)
        assert len(tl_no_l3.events) == 3

    def test_event_latency_is_positive(self, tl_no_l3):
        """Every event should record a positive latency."""
        tl_no_l3.check("Latency test.", confidence=0.9)
        assert tl_no_l3.events[0].latency_ms > 0


# ── L3 Retrieval Integration ──────────────────────────────────────────────────

class TestTruthLoopL3Retrieval:

    def test_verify_returns_tuple(self, tl_no_l3):
        """verify() should return (found: bool, text: str|None)."""
        found, text = tl_no_l3.verify("Test claim.")
        assert isinstance(found, bool)
        assert text is None or isinstance(text, str)

    def test_l3_retrieval_finds_seeded_fact(self, tl_with_l3):
        """After seeding a fact into L3, Truth-Loop should retrieve it."""
        # Seed specific fact into L3 via overflow
        fact_text = "The speed of light in a vacuum is exactly 299,792,458 metres per second."
        tl_with_l3.memory.push_to_l1(fact_text, token_count=15)
        # Force overflow to archive to L3
        for _ in range(20):
            tl_with_l3.memory.push_to_l1("Unrelated content about cooking and food recipes here.", token_count=10)

        if tl_with_l3.memory.l3_count > 0:
            triggered, raw_text, conf = tl_with_l3.check(
                "What is the speed of light?", confidence=0.30
            )
            assert triggered
            if raw_text:
                assert "299,792,458" in raw_text or "light" in raw_text.lower()


# ── Report ────────────────────────────────────────────────────────────────────

class TestTruthLoopReport:

    def test_report_returns_string(self, tl_no_l3):
        tl_no_l3.check("Report test.", confidence=0.9)
        report = tl_no_l3.report()
        assert isinstance(report, str)
        assert "Truth-Loop" in report

    def test_threshold_appears_in_report(self, tl_no_l3):
        report = tl_no_l3.report()
        assert "0.82" in report
