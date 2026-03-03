"""
Project Labyrinth
=================
Breaking the Quadratic Barrier via Recursive Semantic Anchoring.

A drop-in memory management proxy for any OpenAI-compatible LLM API.

v2: Zero-Waste Agentic RAG enhancements — Semantic Answer Cache,
SHA-256 fingerprinting, Context Sufficiency, Temporal Bypass.

Usage:
    from labyrinth import LabyrinthProxy, SemanticAnswerCache

    proxy = LabyrinthProxy()
    cache_result, messages = proxy.ask("What is the JWT secret location?")
    if cache_result.is_semantic_hit:
        print(cache_result.answer)   # instant, $0 token cost
"""

from .memory import LabyrinthMemory, AnchorToken
from .encoder import SemanticEncoder
from .delta import DeltaProtocol
from .truth_loop import TruthLoop, TruthLoopEvent
from .cache import SemanticAnswerCache, CacheResult, CacheEntry, has_temporal_intent
from .proxy import LabyrinthProxy
from .backends import L3Backend, NumpyL3Backend, ChromaL3Backend, auto_select_backend

__version__ = "0.2.0"
__author__  = "Jitendra Rawat"
__license__ = "Apache-2.0"

__all__ = [
    # Core memory
    "LabyrinthMemory",
    "AnchorToken",
    "SemanticEncoder",
    # Protocols
    "DeltaProtocol",
    "TruthLoop",
    "TruthLoopEvent",
    # Cache (v2)
    "SemanticAnswerCache",
    "CacheResult",
    "CacheEntry",
    "has_temporal_intent",
    # Backends (v2)
    "L3Backend",
    "NumpyL3Backend",
    "ChromaL3Backend",
    "auto_select_backend",
    # Proxy
    "LabyrinthProxy",
]
