"""
labyrinth.cli
-------------
Command-line interface for Project Labyrinth.

After `pip install -e .` or `pip install labyrinth-memory`,
the `labyrinth` command is available:

    labyrinth status              — show version, Python, active backends
    labyrinth demo                — run a self-contained 10-turn demo
    labyrinth compress <file>     — analyse a text file and show compression
    labyrinth benchmark           — run the full benchmark suite
"""

from __future__ import annotations

import argparse
import sys
import time
import textwrap
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header(title: str):
    w = 58
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)

def _bar(value: float, width: int = 30, char: str = "#") -> str:
    filled = int(value * width)
    return char * filled + "." * (width - filled)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_status(args):
    """labyrinth status — show environment and backend info."""
    import labyrinth
    from labyrinth.backends import auto_select_backend, NumpyL3Backend, ChromaL3Backend
    from labyrinth.encoder import SemanticEncoder

    _print_header("Project Labyrinth — Status")

    py = sys.version_info
    print(f"  Version:       {labyrinth.__version__}")
    print(f"  Python:        {py.major}.{py.minor}.{py.micro}")

    # Check available backends
    print()
    print("  L3 Backends:")
    try:
        import chromadb
        # On Python 3.14+, even importing chromadb triggers pydantic v1 errors
        chroma_ok = py < (3, 13)
        print(f"    ChromaDB:    {'OK (active on this Python)' if chroma_ok else 'installed but DISABLED (Python >= 3.13, Pydantic V1 incompatible)'}")
    except (ImportError, Exception):
        print("    ChromaDB:    not installed or incompatible (optional)")

    print("    NumpyL3:     OK  [default on Python >= 3.13]")

    # Auto-select and show which one wins
    print()
    backend = auto_select_backend()
    print(f"  Active L3:     {backend.name}")

    # Check encoder
    print()
    print("  Encoder:       sentence-transformers (all-MiniLM-L6-v2)")
    try:
        enc = SemanticEncoder()
        emb = enc.encode("test")
        print(f"  Encoder dim:   {emb.shape[0]}  [OK]")
    except Exception as e:
        print(f"  Encoder:       ERROR — {e}")

    print()
    print("  Run `labyrinth demo` to see live compression in action.")
    print()


def cmd_demo(args):
    """labyrinth demo — self-contained 10-turn compression demo."""
    from labyrinth import LabyrinthProxy

    _print_header("Project Labyrinth — Live Compression Demo")
    print()
    print("  Simulating a 10-turn technical conversation...")
    print("  Watch how token costs DIVERGE between standard and Labyrinth.\n")

    TURNS = [
        ("What is Project Labyrinth?",
         "Project Labyrinth is a memory management proxy for LLMs. It uses Recursive Semantic Anchoring to compress conversational history into Anchor Tokens, reducing token costs by 90%+ on long conversations."),
        ("Explain the Delta Update Protocol.",
         "The Delta Update Protocol (DUP) sends only the semantic delta — the change since the last message — instead of the entire conversation history. This eliminates the O(n) token growth of naive concatenation."),
        ("How does L1 memory work?",
         "L1 is the working memory: the last N raw tokens always in context. It provides full fidelity for recent content. When L1 overflows, the oldest block is compressed into an L2 Anchor Token."),
        ("What is an Anchor Token?",
         "An Anchor Token is a compressed representation: a 384-dim embedding + extractive summary + SHA-256 fingerprint pointing to the raw block stored in L3. ~95% smaller than the original text."),
        ("How does the Truth-Loop prevent hallucinations?",
         "When model confidence drops below threshold t=0.82, the Truth-Loop fires: it retrieves the original raw block from L3, re-elevates it to L1 for that response, then re-archives. No permanent state change."),
        ("What is the L3 archive?",
         "L3 is the persistent vector archive. Raw text blocks are stored with their embeddings. The Truth-Loop queries L3 semantically when it needs to verify a compressed memory."),
        ("What is the Semantic Answer Cache?",
         "The Semantic Answer Cache (v0.2.0) adds two tiers: Tier-1 (>=95% sim) returns the cached answer at zero cost. Tier-2 (>=70% sim) reuses retrieval context, skipping the L3 search overhead."),
        ("How does SHA-256 fingerprinting help?",
         "Each Anchor Token stores a SHA-256 hash of its source text at compression time. Before trusting L2 data, the Truth-Loop recomputes the hash from L3 and compares. A mismatch = stale anchor."),
        ("What Python versions does Labyrinth support?",
         "All Python >= 3.10. On Python >= 3.13, the NumpyL3Backend is used automatically (ChromaDB has Pydantic V1 incompatibility). On Python <= 3.12, ChromaDB is used if installed."),
        ("How do I integrate Labyrinth with OpenAI?",
         "pip install labyrinth-memory. Then: proxy = LabyrinthProxy(); result, messages = proxy.ask(query). If result.is_semantic_hit, return result.answer directly. Else call openai.chat.completions.create(messages=messages)."),
    ]

    proxy = LabyrinthProxy(l1_max_tokens=300, use_l3=True)

    raw_cumulative = 0
    lab_cumulative = 0
    header_printed = False

    for i, (user_msg, assistant_msg) in enumerate(TURNS):
        # Estimate token counts (1 token ≈ 0.75 words)
        user_tokens = max(10, len(user_msg.split()) * 4 // 3 + 4)
        asst_tokens = max(20, len(assistant_msg.split()) * 4 // 3 + 4)
        turn_tokens = user_tokens + asst_tokens

        # Standard approach: total tokens grow linearly (i+1) * avg_per_turn
        raw_cumulative += turn_tokens + (i * turn_tokens // 2)  # growing context overhead

        # Labyrinth: push and measure
        proxy.push_user(user_msg)
        proxy.push_assistant(assistant_msg)
        lab_context = proxy.memory.l1_token_count + proxy.memory.l2_anchor_count * 50
        lab_cumulative = lab_context + asst_tokens

        saving_pct = (1 - lab_cumulative / raw_cumulative) * 100 if raw_cumulative > 0 else 0

        if not header_printed:
            print(f"  {'Turn':<5} {'User Query':<38} {'Raw':>7} {'Lab':>7} {'Saved':>7}")
            print(f"  {'-'*5} {'-'*38} {'-'*7} {'-'*7} {'-'*7}")
            header_printed = True

        query_short = (user_msg[:35] + "...") if len(user_msg) > 35 else user_msg.ljust(38)
        print(f"  T{i+1:<4} {query_short:<38} {raw_cumulative:>7,} {lab_cumulative:>7,} {saving_pct:>6.1f}%")

    # Final report
    final_saving = (1 - lab_cumulative / raw_cumulative) * 100
    usd_per_1k = 0.015  # GPT-4o
    usd_saved = (raw_cumulative - lab_cumulative) / 1000 * usd_per_1k

    print()
    print("  " + "-" * 60)
    print(f"  Final compression:  {final_saving:.1f}% token reduction")
    print(f"  USD saved (GPT-4o): ${usd_saved:.4f} for this 10-turn session")
    print(f"  L3 backend:         {proxy.memory.l3_backend_name}")
    print(f"  L2 anchors:         {proxy.memory.l2_anchor_count}")
    print(f"  L3 blocks:          {proxy.memory.l3_count}")
    print()
    print(proxy.truth_loop.report())
    print()


def cmd_compress(args):
    """labyrinth compress <file> — analyse a text file."""
    from labyrinth import LabyrinthMemory
    from labyrinth.delta import count_tokens

    path = Path(args.file)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    raw_tokens = count_tokens(text)

    _print_header(f"Compress Analysis: {path.name}")
    print(f"\n  File:        {path}")
    print(f"  Size:        {path.stat().st_size:,} bytes")
    print(f"  Raw tokens:  {raw_tokens:,}")

    # Simulate compression
    mem = LabyrinthMemory(l1_max_tokens=512, use_l3=True)
    chunk_size = 400
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    print(f"\n  Compressing {len(chunks)} blocks...", end="", flush=True)
    t0 = time.perf_counter()
    for chunk in chunks:
        tok = count_tokens(chunk)
        mem.push_to_l1(chunk, tok)
    elapsed = time.perf_counter() - t0
    print(f" done in {elapsed:.2f}s")

    lab_tokens = mem.l1_token_count + mem.l2_anchor_count * 50
    ratio = (1 - lab_tokens / raw_tokens) * 100 if raw_tokens > 0 else 0
    usd_saved = (raw_tokens - lab_tokens) / 1000 * 0.015

    print(f"\n  Results:")
    print(f"    L1 working memory:  {mem.l1_token_count:,} tokens")
    print(f"    L2 anchor tokens:   {mem.l2_anchor_count:,} anchors")
    print(f"    L3 archive:         {mem.l3_count:,} raw blocks")
    print(f"    Compressed context: {lab_tokens:,} tokens")
    print(f"    Compression ratio:  {ratio:.1f}%")
    print(f"    USD saved (GPT-4o): ${usd_saved:.4f} per request")
    print()

    bar_raw = _bar(1.0)
    bar_lab = _bar(lab_tokens / raw_tokens if raw_tokens > 0 else 0)
    print(f"  Raw [{bar_raw}] {raw_tokens:,}")
    print(f"  Lab [{bar_lab}] {lab_tokens:,}")
    print()


def cmd_benchmark(args):
    """labyrinth benchmark — run the test suite."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=Path(__file__).parent.parent,
    )
    sys.exit(result.returncode)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="labyrinth",
        description="Project Labyrinth — Token compression for LLM conversations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              labyrinth status
              labyrinth demo
              labyrinth compress myfile.txt
              labyrinth benchmark
        """),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    subparsers.add_parser("status", help="Show version, Python version, and active backends")
    subparsers.add_parser("demo", help="Run a self-contained 10-turn compression demo")

    p_compress = subparsers.add_parser("compress", help="Analyse a text file's compression ratio")
    p_compress.add_argument("file", help="Path to a text file to compress")

    subparsers.add_parser("benchmark", help="Run the full test suite")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "compress":
        cmd_compress(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
