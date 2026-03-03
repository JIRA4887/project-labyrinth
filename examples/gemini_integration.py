"""
examples/gemini_integration.py
-------------------------------
Demonstrates Project Labyrinth integrated with the Google Gemini API.

Run in dry-run mode (no API key needed):
    python examples/gemini_integration.py --dry-run
"""

import argparse
import time
import sys
import os
from typing import List, Dict

# Ensure labyrinth is importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from labyrinth import LabyrinthProxy
from labyrinth.delta import count_tokens

# Mock Responses
MOCK_RESPONSES = {
    "Compare Gemini's long context to Labyrinth.":
        "Gemini supports up to 2 million tokens of native context. However, "
        "processing 2M tokens is computationally heavy and slower. Labyrinth "
        "complements this by keeping the 'active' context small and only "
        "recalling specific blocks when needed, improving latency and cost.",
    "Does Labyrinth work with multi-modal LLMs?":
        "Currently, Labyrinth focus is on semantic text compression. For Gemini's "
        "multi-modal inputs, Labyrinth manages the text/conversation logic, "
        "while images/video are handled as standard payload references.",
}

def mock_gemini(messages: List[Dict], query: str):
    time.sleep(0.1)
    response = MOCK_RESPONSES.get(query, f"[Gemini mock response to: {query[:50]}]")
    prompt_tokens = count_tokens(str(messages)) + count_tokens(query)
    completion_tokens = count_tokens(response)
    return response, prompt_tokens, completion_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Project Labyrinth — Gemini Integration Demo")
    print("="*60)
    print("  Mode: DRY-RUN (Mock)")

    proxy = LabyrinthProxy(l1_max_tokens=300)
    
    for query in MOCK_RESPONSES.keys():
        print(f"\n  Query: {query}")
        result, messages = proxy.ask(query)
        
        if result.is_semantic_hit:
            response = result.answer
            print("  [CACHE HIT] Zero-token response.")
        else:
            # Note: Gemini SDK uses different message format (parts/text)
            # This mock demonstrates the logical flow.
            response, p_tok, c_tok = mock_gemini(messages, query)
            proxy.push_assistant(response)
            proxy.store_answer(query, response)
            print(f"  [LLM] Tokens: {p_tok} in / {c_tok} out")
        
        print(f"  Response: {response[:100]}...")

    print("\n" + proxy.report())

if __name__ == "__main__":
    main()
