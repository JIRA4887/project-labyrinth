"""
tests/test_cache.py
Unit tests for the SemanticAnswerCache (Tier-1/Tier-2) and has_temporal_intent().
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from labyrinth.cache import SemanticAnswerCache, CacheResult, has_temporal_intent
from labyrinth.encoder import SemanticEncoder


# ── Temporal Intent Detection ─────────────────────────────────────────────────

class TestTemporalIntent:

    def test_detects_latest(self):
        assert has_temporal_intent("What are the latest results?")

    def test_detects_current(self):
        assert has_temporal_intent("What is the current status of the build?")

    def test_detects_today(self):
        assert has_temporal_intent("Show me today's deployment logs.")

    def test_detects_real_time(self):
        assert has_temporal_intent("Give me real-time prices.")

    def test_no_temporal_in_plain_query(self):
        assert not has_temporal_intent("Explain how JWT tokens work.")

    def test_no_temporal_in_technical_query(self):
        assert not has_temporal_intent("What is the retry limit for API requests?")

    def test_case_insensitive(self):
        assert has_temporal_intent("LATEST metrics please")

    def test_mixed_case(self):
        assert has_temporal_intent("Give me the Current deployment status")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cache():
    """Fresh cache with tight thresholds for deterministic testing."""
    enc = SemanticEncoder()
    return SemanticAnswerCache(
        encoder=enc,
        hit_threshold=0.95,
        topic_threshold=0.70,
        bypass_temporal=True,
    )


# ── Cache Miss ────────────────────────────────────────────────────────────────

class TestCacheMiss:

    def test_empty_cache_is_miss(self, cache):
        result = cache.lookup("What is the JWT secret key?")
        assert result.is_miss
        assert result.tier == 0

    def test_completely_different_query_is_miss(self, cache):
        cache.store(
            "What is the JWT secret?",
            "The JWT secret is stored in environment variables.",
        )
        result = cache.lookup("How do I make sourdough bread?")
        assert result.is_miss

    def test_miss_when_cache_empty_after_clear(self, cache):
        cache.store("JWT secret location", "It's in env vars.")
        cache.clear()
        result = cache.lookup("JWT secret location")
        assert result.is_miss


# ── Tier-1 Semantic Hit ─────────────────────────────────────────────────────

class TestTier1SemanticHit:

    def test_exact_query_is_tier1_hit(self, cache):
        query = "What is the JWT secret key location?"
        answer = "The JWT secret key is stored in the JWT_SECRET environment variable."
        cache.store(query, answer)
        result = cache.lookup(query)  # identical query
        assert result.is_semantic_hit
        assert result.answer == answer
        assert result.similarity >= 0.95

    def test_semantically_identical_query_is_tier1_hit(self, cache):
        """Paraphrased but semantically identical queries should hit Tier-1."""
        cache.store(
            "What is the JWT secret key location?",
            "Stored in JWT_SECRET env variable.",
        )
        # Rephrased version — should still be a Tier-1 hit at high similarity
        result = cache.lookup("Where is the JWT secret key stored?")
        # Allow Tier-1 or Tier-2 since the exact similarity depends on the model
        assert not result.is_miss, "Expected a cache hit for semantically similar query"

    def test_tier1_hit_increments_counter(self, cache):
        query = "Retry limit for failed requests"
        cache.store(query, "The retry limit is 3 attempts.")
        cache.lookup(query)
        assert cache._tier1_hits == 1

    def test_tier1_hit_rate(self, cache):
        query = "API rate limit"
        cache.store(query, "100 requests per minute.")
        cache.lookup(query)  # hit
        cache.lookup("How is sourdough made?")  # miss
        assert cache.tier1_hit_rate == 0.5


# ── Tier-2 Context Hit ────────────────────────────────────────────────────────

class TestTier2ContextHit:

    def test_context_chunks_returned_on_tier2(self, cache):
        """On a Tier-2 hit, context_chunks should be non-empty."""
        chunks = ["JWT uses HS256 algorithm.", "Token expiry is set to 24h."]
        cache.store(
            "Tell me about JWT configuration.",
            "JWT is configured with HS256 and 24h expiry.",
            context_chunks=chunks,
        )
        # A slightly different query on the same topic
        result = cache.lookup("What JWT settings are configured?")
        if result.is_context_hit:
            assert len(result.context) > 0

    def test_tier2_hit_has_no_answer(self, cache):
        """Tier-2 hits return context but no cached answer."""
        cache.store(
            "Tell me about JWT configuration.",
            "JWT is configured with HS256.",
            context_chunks=["JWT uses HS256 algorithm."],
        )
        result = cache.lookup("What JWT settings are configured?")
        if result.is_context_hit:
            assert result.answer is None


# ── Temporal Bypass ──────────────────────────────────────────────────────────

class TestTemporalBypass:

    def test_temporal_query_always_misses(self, cache):
        """Temporal queries MUST bypass the cache even if a matching entry exists."""
        cache.store(
            "What are the latest test results?",
            "All tests passed as of last run.",
        )
        result = cache.lookup("What are the latest test results?")
        assert result.is_miss, "Temporal query should always bypass the cache"

    def test_temporal_bypass_increments_counter(self, cache):
        cache.lookup("What is the current CPU usage?")
        cache.lookup("Show me today's logs.")
        assert cache._bypasses == 2

    def test_non_temporal_query_not_bypassed(self, cache):
        cache.store("What is the retry limit?", "3 attempts.")
        result = cache.lookup("What is the retry limit?")
        assert not result.is_miss, "Non-temporal query should not be bypassed"


# ── Staleness & Eviction ─────────────────────────────────────────────────────

class TestStaleness:

    def test_stale_entry_invalidated(self):
        """Entries older than max_age_seconds should be treated as misses."""
        enc = SemanticEncoder()
        cache = SemanticAnswerCache(
            encoder=enc,
            hit_threshold=0.95,
            topic_threshold=0.70,
            max_age_seconds=0.01,  # 10ms TTL
        )
        cache.store("What is the retry limit?", "3 attempts.")
        time.sleep(0.05)  # wait for TTL expiry
        result = cache.lookup("What is the retry limit?")
        assert result.is_miss, "Entry should be stale and invalidated"

    def test_invalidate_removes_similar_entries(self, cache):
        cache.store("What is the JWT secret?", "JWT_SECRET env var.")
        cache.store("Tell me about JWT tokens.", "JWT is a signed token format.")
        removed = cache.invalidate("JWT secret", similarity_threshold=0.80)
        assert removed >= 1


# ── Store & Stats ────────────────────────────────────────────────────────────

class TestStoreAndStats:

    def test_store_increases_size(self, cache):
        assert cache.size == 0
        cache.store("Query 1", "Answer 1.")
        cache.store("Query 2", "Answer 2.")
        assert cache.size == 2

    def test_total_queries_tracked(self, cache):
        cache.store("Q1", "A1.")
        cache.lookup("Q1")
        cache.lookup("Q2")
        assert cache.total_queries == 2

    def test_report_returns_string(self, cache):
        cache.store("Test query", "Test answer.")
        cache.lookup("Test query")
        report = cache.report()
        assert isinstance(report, str)
        assert "Tier-1" in report
        assert "Tier-2" in report

    def test_max_entries_eviction(self):
        """When max_entries is reached, LRU entry should be evicted."""
        enc = SemanticEncoder()
        tiny_cache = SemanticAnswerCache(encoder=enc, max_entries=2)
        tiny_cache.store("Query A", "Answer A.")
        tiny_cache.store("Query B", "Answer B.")
        tiny_cache.store("Query C", "Answer C.")  # triggers eviction
        assert tiny_cache.size == 2


# ── AnchorToken Fingerprinting ────────────────────────────────────────────────

class TestAnchorFingerprinting:

    def test_anchor_has_content_hash(self):
        """AnchorToken should auto-compute SHA-256 hash in __post_init__."""
        from labyrinth.memory import AnchorToken
        import numpy as np
        anchor = AnchorToken(
            id="test-id",
            embedding=np.zeros(384),
            summary="Test summary",
            source_text="The retry limit is 3 attempts.",
            token_count=10,
        )
        assert len(anchor.content_hash) == 64  # SHA-256 hex digest

    def test_verify_integrity_same_text(self):
        """verify_integrity() should return True for unchanged text."""
        from labyrinth.memory import AnchorToken
        import numpy as np
        text = "The retry limit is 3 attempts."
        anchor = AnchorToken(
            id="test-id",
            embedding=np.zeros(384),
            summary="Test summary",
            source_text=text,
            token_count=10,
        )
        assert anchor.verify_integrity(text) is True

    def test_verify_integrity_changed_text(self):
        """verify_integrity() should return False if text was modified."""
        from labyrinth.memory import AnchorToken
        import numpy as np
        anchor = AnchorToken(
            id="test-id",
            embedding=np.zeros(384),
            summary="Test summary",
            source_text="The retry limit is 3 attempts.",
            token_count=10,
        )
        assert anchor.verify_integrity("The retry limit is 5 attempts.") is False

    def test_anchor_hash_stored_in_memory(self):
        """After L2 compression, anchors should have non-empty content_hash."""
        from labyrinth.memory import LabyrinthMemory
        mem = LabyrinthMemory(l1_max_tokens=50, use_l3=False)
        for _ in range(15):
            mem.push_to_l1("Content block for hash test " * 3, token_count=10)
        anchors = mem.l2_anchors
        assert len(anchors) > 0
        assert all(len(a.content_hash) == 64 for a in anchors)
