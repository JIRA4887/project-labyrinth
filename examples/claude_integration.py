"""
examples/claude_integration.py
-------------------------------
Demonstrates Project Labyrinth integrated with the Anthropic Claude API.

Run in dry-run mode (no API key needed):
    python examples/claude_integration.py --dry-run

Run with a real Anthropic API key:
    python examples/claude_integration.py --api-key sk-ant-...
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

# Mock Responses for Dry-Run
MOCK_RESPONSES = {
    "Explain context window limits in Claude.":
        "Claude 3 and 3.5 models have a 200k token context window. While large, "
        "sending the full context every time is expensive. Labyrinth helps by "
        "compressing historical turns, keeping the prompt small and fast.",
    "How does Labyrinth handle long-term memory?":
        "Labyrinth uses a three-tier system: L1 (raw), L2 (semantic anchors), "
        "and L3 (persistent vector archive). This allows Claude to 'remember' "
        "details from millions of tokens ago without actually filling the context window.",
}

def mock_claude(messages: List[Dict], query: str):
    time.sleep(0.1)
    response = MOCK_RESPONSES.get(query, f"[Claude mock response to: {query[:50]}]")
    prompt_tokens = count_tokens(str(messages)) + count_tokens(query)
    completion_tokens = count_tokens(response)
    return response, prompt_tokens, completion_tokens

def real_claude(client, messages: List[Dict], query: str):
    # Convert OpenAI-style list if needed (LabyrinthProxy internal log is generic)
    # Anthropic expects 'role' and 'content' but uses 'user'/'assistant' roles
    # The system prompt should be passed separately in Anthropic SDK
    
    system_msg = ""
    claude_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_msg += msg["content"] + "\n"
        else:
            claude_messages.append({"role": msg["role"], "content": msg["content"]})
    
    claude_messages.append({"role": "user", "content": query})
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=system_msg,
        messages=claude_messages
    )
    
    return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Project Labyrinth — Claude Integration Demo")
    print("="*60)

    if args.api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=args.api_key)
        llm_fn = lambda msgs, q: real_claude(client, msgs, q)
        print("  Mode: LIVE (Anthropic API)")
    else:
        llm_fn = mock_claude
        print("  Mode: DRY-RUN (Mock)")

    proxy = LabyrinthProxy(l1_max_tokens=300)
    
    for query in MOCK_RESPONSES.keys():
        print(f"\n  Query: {query}")
        result, messages = proxy.ask(query)
        
        if result.is_semantic_hit:
            response = result.answer
            print("  [CACHE HIT] Zero-token response delivered.")
        else:
            response, p_tok, c_tok = llm_fn(messages, query)
            proxy.push_assistant(response)
            proxy.store_answer(query, response)
            print(f"  [LLM] Tokens: {p_tok} in / {c_tok} out")
        
        print(f"  Response: {response[:100]}...")

    print("\n" + proxy.report())

if __name__ == "__main__":
    main()
