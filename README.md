# Project Labyrinth
> Breaking the quadratic memory barrier for autonomous LLM agents.

Project Labyrinth achieves constant-time / linear cost context windows for large conversations using **Recursive Semantic Anchoring** and a **Delta Update Protocol**. 

## V2 Updates
- **Pluggable L3 Backends**: Native `NumpyL3Backend` built-in, no longer requires ChromaDB. Tested across **all new Python releases (including Python 3.14)**.
- **Semantic Answer Cache**: Tiered answer caching directly avoids redundant LLM inferences entirely. 
- **Production-ready CLI**: Instant verification directly via the simple `labyrinth` CLI.

## Quick Start

You can install `labyrinth-memory` to experiment without needing any external dependencies.

```bash
pip install labyrinth-memory
```

Test it immediately on your machine:

```bash
# View active backend and Python environments seamlessly:
labyrinth status

# Run a self-contained token-compression simulation demo 
# and see live USD/token savings for a 10-turn dialogue
labyrinth demo 

# Compress your own local files
labyrinth compress my_code.py
```

## Example Integration (OpenAI)

Integrating Labyrinth with standard language models takes ~3 lines of new code.

```python
import openai
from labyrinth import LabyrinthProxy

# 1. Initialize the caching memory proxy explicitly
proxy = LabyrinthProxy(l1_max_tokens=500, use_l3=True)

queries = [
    "What is the theory of relativity?",
    "What is the theory of relativity?", # Tier 1 cache hit!
]

for query in queries:
    # 2. Ask proxy to evaluate context & caching
    result, messages = proxy.ask(query)
    
    if result.is_semantic_hit:
        # 3a. Answer found in cache — zero cost
        response = result.answer
    else:
        # 3b. Pay for compressed tokens from OpenAI
        res = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        response = res.choices[0].message.content
        
        # 4. Push network response back to local memory & cache
        proxy.push_assistant(response)
        proxy.store_answer(query, response)

    print(response)
```

Check out the full [openai_integration.py](examples/openai_integration.py) to simulate API performance locally with Dry-Run mode (`python examples/openai_integration.py --dry-run`).

## Development
```bash
git clone https://github.com/JIRA4887/project-labyrinth.git
cd project-labyrinth
pip install -e .[dev,chroma]
labyrinth benchmark
```
