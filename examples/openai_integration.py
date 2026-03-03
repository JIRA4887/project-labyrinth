"""
examples/openai_integration.py
-------------------------------
Demonstrates Project Labyrinth integrated with the OpenAI API.

Run in dry-run mode (no API key needed):
    python examples/openai_integration.py --dry-run

Run with a real OpenAI API key:
    python examples/openai_integration.py --api-key sk-...

This script shows a side-by-side comparison of:
  - Standard OpenAI usage (full context every call)
  - Labyrinth-wrapped usage (compressed context + answer cache)
"""

from __future__ import annotations

import argparse
import time
import sys
import os
from typing import List, Dict

# ── Ensure labyrinth is importable from the project root ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from labyrinth import LabyrinthProxy
from labyrinth.delta import count_tokens


# ── Mock LLM (dry-run mode) ───────────────────────────────────────────────────

MOCK_RESPONSES = {
    "What is Project Labyrinth?":
        "Project Labyrinth is a drop-in memory proxy for LLMs that compresses "
        "conversation history using Recursive Semantic Anchoring, reducing token "
        "costs by 90%+ on long conversations.",
    "How does the Delta Update Protocol save tokens?":
        "The Delta Update Protocol sends only the semantic change since the last "
        "message, not the full history. This makes token cost O(1) per turn instead "
        "of O(n), giving linear rather than quadratic cost growth.",
    "Explain the Truth-Loop in simple terms.":
        "The Truth-Loop is a confidence monitor. If the AI is unsure about a "
        "compressed memory, it automatically retrieves the original raw text to "
        "double-check before answering. This prevents hallucinations from lossy "
        "compression.",
    "What is an Anchor Token?":
        "An Anchor Token is a compressed version of a text block: a 384-dimensional "
        "semantic vector plus a short summary. It is about 95% smaller than the "
        "original text but lets Labyrinth find the right raw block in L3 when needed.",
    "How do I use Labyrinth with my own chatbot?":
        "Install with `pip install labyrinth-memory`. Then wrap your API calls: "
        "`proxy = LabyrinthProxy(); result, messages = proxy.ask(query)`. "
        "If result.is_semantic_hit, return the cached answer. Otherwise call your "
        "LLM with the compressed messages list.",
}


def mock_llm(messages: List[Dict], query: str) -> tuple[str, int, int]:
    """Simulate an LLM response. Returns (response_text, prompt_tokens, completion_tokens)."""
    time.sleep(0.05)  # simulate network latency
    # Count tokens in the messages list
    prompt_text = " ".join(m.get("content", "") for m in messages)
    prompt_tokens = count_tokens(prompt_text) + count_tokens(query)
    response = MOCK_RESPONSES.get(query, f"[Mock response to: {query[:60]}]")
    completion_tokens = count_tokens(response)
    return response, prompt_tokens, completion_tokens


def real_llm(client, messages: List[Dict], query: str) -> tuple[str, int, int]:
    """Call the real OpenAI API. Returns (response_text, prompt_tokens, completion_tokens)."""
    messages_with_user = messages + [{"role": "user", "content": query}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_with_user,
    )
    text = response.choices[0].message.content
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens


# ── Simulation ────────────────────────────────────────────────────────────────

CONVERSATION = list(MOCK_RESPONSES.keys())

PRICE_PER_1K_INPUT  = 0.005   # GPT-4o input pricing ($/1K tokens)
PRICE_PER_1K_OUTPUT = 0.015   # GPT-4o output pricing ($/1K tokens)


def run_standard(llm_fn) -> dict:
    """Standard approach: full conversation history every call."""
    history: List[Dict] = []
    total_prompt, total_completion = 0, 0

    print("\n  [STANDARD] Full context every call")
    print("  " + "-" * 55)

    for i, query in enumerate(CONVERSATION):
        t0 = time.perf_counter()
        response, prompt_tok, completion_tok = llm_fn(history + [{"role": "user", "content": query}], query)
        latency = (time.perf_counter() - t0) * 1000

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})

        total_prompt += prompt_tok
        total_completion += completion_tok

        print(f"  T{i+1}: {query[:45]:<45}  prompt={prompt_tok:>5}  lat={latency:.0f}ms")

    cost = (total_prompt / 1000) * PRICE_PER_1K_INPUT + (total_completion / 1000) * PRICE_PER_1K_OUTPUT
    return {"total_prompt": total_prompt, "total_completion": total_completion, "cost": cost}


def run_labyrinth(llm_fn) -> dict:
    """Labyrinth approach: compressed context + semantic cache."""
    proxy = LabyrinthProxy(l1_max_tokens=400, use_l3=True)
    total_prompt, total_completion = 0, 0
    cache_hits = 0

    print("\n  [LABYRINTH] Compressed context + cache")
    print("  " + "-" * 55)

    for i, query in enumerate(CONVERSATION):
        t0 = time.perf_counter()

        cache_result, messages = proxy.ask(query)

        if cache_result.is_semantic_hit:
            # Tier-1 cache hit — zero LLM cost
            response = cache_result.answer
            prompt_tok = 0
            completion_tok = 0
            cache_hits += 1
            tag = "  ** CACHE HIT (Tier-1) — $0 cost **"
        else:
            response, prompt_tok, completion_tok = llm_fn(messages, query)
            proxy.push_assistant(response)
            proxy.store_answer(query, response)
            tag = ""

        latency = (time.perf_counter() - t0) * 1000
        total_prompt += prompt_tok
        total_completion += completion_tok

        print(f"  T{i+1}: {query[:45]:<45}  prompt={prompt_tok:>5}  lat={latency:.0f}ms{tag}")

    cost = (total_prompt / 1000) * PRICE_PER_1K_INPUT + (total_completion / 1000) * PRICE_PER_1K_OUTPUT
    return {
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "cost": cost,
        "cache_hits": cache_hits,
        "backend": proxy.memory.l3_backend_name,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Labyrinth vs Standard OpenAI integration demo")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Use mock LLM (no API key needed). Default: True.")
    parser.add_argument("--api-key", default=None, help="OpenAI API key for real LLM calls.")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Project Labyrinth — OpenAI Integration Comparison")
    print("=" * 60)

    if args.api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=args.api_key)
            llm_fn = lambda messages, query: real_llm(client, messages, query)
            print("  Mode: LIVE (using OpenAI API)")
        except ImportError:
            print("  Error: openai not installed. Run: pip install openai")
            sys.exit(1)
    else:
        print("  Mode: DRY-RUN (mock LLM, no API key needed)")
        print("  Tip:  Pass --api-key sk-... to use the real OpenAI API")
        llm_fn = mock_llm

    print(f"  Turns: {len(CONVERSATION)}")
    print(f"  Model pricing: ${PRICE_PER_1K_INPUT}/1K input, ${PRICE_PER_1K_OUTPUT}/1K output")

    std = run_standard(llm_fn)
    lab = run_labyrinth(llm_fn)

    # Report
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Standard':>12} {'Labyrinth':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Prompt tokens sent':<30} {std['total_prompt']:>12,} {lab['total_prompt']:>12,}")
    print(f"  {'Completion tokens':<30} {std['total_completion']:>12,} {lab['total_completion']:>12,}")
    print(f"  {'Total cost (USD)':<30} ${std['cost']:>11.4f} ${lab['cost']:>11.4f}")

    token_saving = (1 - lab["total_prompt"] / std["total_prompt"]) * 100 if std["total_prompt"] > 0 else 0
    cost_saving  = std["cost"] - lab["cost"]
    print()
    print(f"  Token reduction:     {token_saving:.1f}%")
    print(f"  Cost saved:          ${cost_saving:.4f}")
    print(f"  Cache hits (Tier-1): {lab.get('cache_hits', 0)} / {len(CONVERSATION)}")
    print(f"  L3 backend:          {lab.get('backend', 'N/A')}")
    print()
    print("  Labyrinth is production-ready. Integrate with 5 lines of code.")
    print()


if __name__ == "__main__":
    main()
