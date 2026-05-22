# RedHop examples — where it fits in a real RAG stack

```text
   your retriever  →  RedHop.build_context  →  your LLM
   (vector DB/BM25)    (this library)           (OpenAI/local/…)
```

**RedHop is NOT** a retriever, a vector DB, an agent framework, or a workflow
engine. **RedHop is a reasoning-preserving context optimization layer.** You
hand it the chunks your retriever returned plus a token budget; it assembles
the prompt context — pruning distractors, preserving the reasoning-critical
"second hop", and reporting exactly what it did.

All examples run **fully offline** on a tiny built-in corpus (no API key, no
vector DB). Set `OPENAI_API_KEY` to also see a real generation where noted.

## Run them

```bash
# A — basic integration: retrieval → build_context → generation
python examples/python/basic_rag.py

# D — strategy playground: every strategy side-by-side, with the contexts
python examples/python/strategy_playground.py

# B — context-economics dashboard (self-contained HTML, no JS deps)
python examples/dashboard/generate_dashboard.py && open examples/dashboard/dashboard.html
```

The first run compiles the Rust bridge once (`cargo build … --example
context_bridge --release`); subsequent runs are instant.

## The API

```python
import redhop

chunks = retriever.retrieve(query)              # your stack
ctx = redhop.build_context(
    query=query,
    retrieved_chunks=chunks,                    # strings, dicts, or LangChain Docs
    token_budget=12000,
    strategy="reasoning_preserving",            # the safe default
)
response = llm.generate(ctx.text)               # your stack
print(ctx.report)                               # observability ↓
```

```text
Context Optimization Report
───────────────────────────
Strategy: ReasoningPreserving

Input chunks:        8
Output chunks:       2
Tokens:              100 → 30  (-70%)
Distractors pruned:  6
Reasoning rescues:   1

Evidence density:    0.10 → 0.20
Retained evidence:   100%
...
```

Also available: `redhop.analyze_context(query, chunks)` — pure, non-destructive
diagnostics (what you have + what reasoning-preservation *would* rescue).

## How it works under the hood (and the path to `pip install redhop`)

These examples wrap the Rust engine through a thin JSON bridge (the
`context_bridge` example binary) — the minimal thing that makes
`redhop.build_context(...)` work today. The Python API here is exactly the one
that native wheels (`pip install redhop`, backed by pyo3) will expose; the
bridge is an implementation detail you won't see once wheels ship.

## Why a strategy matters (the evidence)

`strategy_playground.py` shows the core result live: aggressive
`distractor_filtered` **drops** the low-relevance second hop the multi-hop
answer needs, while `reasoning_preserving` keeps it *and* prunes distractors.
The full measured evidence is in [`docs/findings/`](../docs/findings/README.md)
(second-hop tax, n=300 end-to-end QA, the strategy benchmark).

## Coming next

- `examples/pdf_pipeline/` — enterprise PDF flow: extraction → ingestion
  diagnostics (OCR noise / duplicates / fragmentation) → context optimization
  → economics report.
