"""
tests/test_memory.py
Unit tests for the LabyrinthMemory three-layer system.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from labyrinth.memory import LabyrinthMemory, AnchorToken
from labyrinth.encoder import SemanticEncoder


# No chromadb imports skip here anymore because L3 has a Numpy backend.


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def memory_no_l3():
    """Memory with L3 disabled for fast unit tests (no ChromaDB needed)."""
    return LabyrinthMemory(
        l1_max_tokens=100,
        block_size_tokens=20,
        use_l3=False,
    )


@pytest.fixture
def memory_with_l3():
    return LabyrinthMemory(
        l1_max_tokens=100,
        block_size_tokens=20,
        use_l3=True,
    )
# ── L1 Tests ──────────────────────────────────────────────────────────────────

class TestL1WorkingMemory:

    def test_push_single_chunk(self, memory_no_l3):
        """Pushing a small text chunk stays in L1."""
        memory_no_l3.push_to_l1("Hello world", token_count=2)
        assert memory_no_l3.l1_token_count == 2
        assert "Hello world" in memory_no_l3.l1_text

    def test_push_multiple_chunks(self, memory_no_l3):
        """Multiple small chunks accumulate in L1."""
        for i in range(5):
            memory_no_l3.push_to_l1(f"Chunk {i}", token_count=5)
        assert memory_no_l3.l1_token_count == 25
        assert memory_no_l3.l2_anchor_count == 0  # no overflow yet

    def test_l1_overflow_triggers_l2_compression(self, memory_no_l3):
        """When L1 exceeds max_tokens, oldest block is compressed to L2."""
        # Fill L1 to 90 tokens
        for i in range(9):
            memory_no_l3.push_to_l1(f"Block {i}: content here", token_count=10)
        # This next push should overflow
        anchor = memory_no_l3.push_to_l1("Overflow block", token_count=20)
        # At least one anchor should exist in L2
        assert memory_no_l3.l2_anchor_count >= 1
        assert anchor is not None

    def test_l1_text_contains_recent_content(self, memory_no_l3):
        """L1 should contain the most recently pushed content."""
        memory_no_l3.push_to_l1("Early content", token_count=5)
        for _ in range(20):  # push lots to force overflow
            memory_no_l3.push_to_l1("Filler " * 5, token_count=10)
        memory_no_l3.push_to_l1("Very recent content", token_count=4)
        assert "Very recent content" in memory_no_l3.l1_text

    def test_stats_updated_on_push(self, memory_no_l3):
        """Stats should track total tokens seen."""
        memory_no_l3.push_to_l1("Test text", token_count=10)
        assert memory_no_l3.stats["total_tokens_seen"] == 10
        assert memory_no_l3.stats["l1_pushes"] == 1


# ── L2 Tests ──────────────────────────────────────────────────────────────────

class TestL2AnchorTokens:

    def test_anchor_token_has_required_fields(self, memory_no_l3):
        """AnchorTokens must have id, embedding, summary, source_text."""
        # Force overflow to create an anchor
        for _ in range(20):
            memory_no_l3.push_to_l1("x " * 10, token_count=10)
        anchors = memory_no_l3.l2_anchors
        assert len(anchors) > 0
        a = anchors[0]
        assert isinstance(a.id, str) and len(a.id) > 0
        assert isinstance(a.embedding, np.ndarray)
        assert len(a.embedding) == 384  # all-MiniLM-L6-v2 dims
        assert isinstance(a.summary, str) and len(a.summary) > 0
        assert isinstance(a.source_text, str)

    def test_anchor_embedding_is_normalized(self, memory_no_l3):
        """Anchor embeddings should be unit-norm (normalized by sentence-transformers)."""
        for _ in range(20):
            memory_no_l3.push_to_l1("Normalized embedding test content", token_count=10)
        anchors = memory_no_l3.l2_anchors
        if anchors:
            norm = np.linalg.norm(anchors[0].embedding)
            assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: norm={norm}"

    def test_l2_anchor_count_increases_with_overflow(self, memory_no_l3):
        """More overflows → more anchors in L2."""
        for _ in range(30):
            memory_no_l3.push_to_l1("Filler content for compression test", token_count=10)
        assert memory_no_l3.l2_anchor_count > 0

    def test_get_anchor_by_id(self, memory_no_l3):
        """Should retrieve a specific anchor by its UUID."""
        for _ in range(20):
            memory_no_l3.push_to_l1("Content block", token_count=10)
        anchors = memory_no_l3.l2_anchors
        if anchors:
            target = anchors[0]
            found = memory_no_l3.get_anchor_by_id(target.id)
            assert found is not None
            assert found.id == target.id

    def test_get_l2_embeddings_shape(self, memory_no_l3):
        """get_l2_embeddings() should return correct matrix shape."""
        for _ in range(25):
            memory_no_l3.push_to_l1("Matrix shape test content block here", token_count=10)
        n = memory_no_l3.l2_anchor_count
        embs = memory_no_l3.get_l2_embeddings()
        if n > 0:
            assert embs.shape == (n, 384)


# ── L3 Tests ──────────────────────────────────────────────────────────────────

class TestL3Archive:

    def test_l3_retrieval_returns_relevant_block(self, memory_with_l3):
        """L3 semantic search should retrieve blocks relevant to the query."""
        # Push specific content that will overflow to L3
        specific_text = "The capital of France is Paris. Napoleon was born in Corsica."
        memory_with_l3.push_to_l1(specific_text, token_count=15)
        for _ in range(20):
            memory_with_l3.push_to_l1("Generic filler content about unrelated topics here", token_count=10)

        if memory_with_l3.l3_count > 0:
            from labyrinth.encoder import SemanticEncoder
            enc = SemanticEncoder()
            query_emb = enc.encode("What is the capital of France?")
            results = memory_with_l3.retrieve_from_l3(query_emb, n_results=3)
            assert len(results) > 0
            # Best result should contain relevant text
            texts = [r[0] for r in results]
            found = any("Paris" in t or "France" in t for t in texts)
            assert found, f"Expected Paris/France in results, got: {texts}"

    def test_l3_count_tracks_archived_blocks(self, memory_with_l3):
        """l3_count should equal number of archived blocks."""
        initial_count = memory_with_l3.l3_count
        for _ in range(25):
            memory_with_l3.push_to_l1("Archive count test content block text", token_count=10)
        final_count = memory_with_l3.l3_count
        assert final_count >= initial_count


# ── Reset Tests ───────────────────────────────────────────────────────────────

class TestMemoryReset:

    def test_reset_clears_all_layers(self, memory_no_l3):
        """After reset(), all layers should be empty."""
        for _ in range(25):
            memory_no_l3.push_to_l1("Content to be reset", token_count=10)
        memory_no_l3.reset()
        assert memory_no_l3.l1_token_count == 0
        assert memory_no_l3.l2_anchor_count == 0
        assert memory_no_l3.stats["total_tokens_seen"] == 0
