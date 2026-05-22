# RedHop (Python)

**Reasoning-preserving context optimization and retrieval observability for RAG systems.**

RedHop sits **between retrieval and generation**. You hand it the chunks your
retriever returned and a token budget; it assembles the prompt context —
pruning distractors, **preserving reasoning-critical "second-hop" evidence**,
and reporting exactly what it did. It is *not* a retriever, vector DB, agent
framework, or workflow engine.

```python
import redhop

chunks = retriever.retrieve(query)          # your stack
ctx = redhop.build_context(
    query=query,
    retrieved_chunks=chunks,                # list of dicts or strings
    strategy="reasoning_preserving",        # the safe default
    token_budget=12000,
)
response = llm.generate(ctx.text())         # your stack
print(ctx.report)                           # observability ↓
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

## Installation

```bash
pip install redhop          # alpha wheels (PyPI)
```

The wheel bundles the compiled Rust engine — no Rust toolchain needed to use it.

## Quickstart

`retrieved_chunks` accepts plain Python — a list of dicts (only `text` is
required) or strings:

```python
chunks = [
    {"id": "c1", "text": "...", "score": 0.82},
    {"id": "c2", "text": "..."},
    "a bare string also works",
]
ctx = redhop.build_context(query="...", retrieved_chunks=chunks)
print(ctx.text())              # the assembled prompt context
r = ctx.report
print(r.total_tokens, r.distractors_pruned, r.second_hop_rescue_count)
print(redhop.report_to_dict(r))   # full telemetry as a dict
```

## API surface

| Function | Purpose |
| -------- | ------- |
| `build_context(query, chunks, strategy, token_budget, ...)` | budget-aware assembly → `BuiltContext` |
| `filter_context(query, chunks, strategy, ...)` | filter junk, **no** budget truncation → `BuiltContext` |
| `analyze_context(query, chunks, ...)` | non-destructive diagnostics → `ContextReport` |
| `context_economics(query, chunks, ...)` | economics of a set as-is → `dict` |

`BuiltContext`: `.text()`, `.chunks`, `.report`.
`ContextReport`: `.strategy`, `.total_tokens`, `.distractors_pruned`,
`.second_hop_rescue_count`, `.evidence_density`, … ; `str(report)` is the
rendered report; `redhop.report_to_dict(report)` is the full telemetry.

## Strategies

| strategy | what it does | when |
| -------- | ------------ | ---- |
| `reasoning_preserving` *(default)* | keep query-relevant seeds **and** rescue low-relevance chunks linked to a seed; drop only unlinked junk | multi-hop / general; safe default |
| `distractor_filtered` | drop everything below a query-grounding bar | single-hop, or a *low* threshold only |
| `max_density` | greedily pack the densest chunks into the budget | single-hop / brutal budgets |
| `raw_topk` | keep retrieval order until the budget fills | baseline / no optimization |

### Why `reasoning_preserving` is the default — the second-hop tax

On multi-hop questions, the second hop (the evidence that *bridges* to the
answer) is **low-relevance-to-the-query by construction** — it connects through
a bridge entity, not the query terms. So every relevance-based operation
(aggressive distractor filtering, cross-encoder reranking, max-density pruning)
**drops it**. We measured this directly: a relevance filter keeps 96.8% of
second hops at threshold 0.05 but only **43.9% at 0.30**.

`reasoning_preserving` resists this: it keeps the query-relevant seeds, then
*rescues* low-relevance chunks that are linked to a seed (sharing the bridge
entity), dropping only true junk. End-to-end (n=300, generator = haiku) it
beat aggressive filtering with a CI-significant margin, and the gain was
causally localized to the rescued evidence.

> Transformers tolerate irrelevant context far better than they tolerate
> missing reasoning links. Premature removal of low-relevance reasoning
> evidence hurts more than the distractors do.

## Examples

```bash
python examples/basic_rag.py            # retrieval → build_context → generation
python examples/compare_strategies.py   # every strategy side-by-side
python examples/economics_demo.py       # context economics + analyze_context
```

## Evidence

Every default is grounded in a measured finding (with a falsified-hypotheses
registry) in the repo's evidence layer:

- Second-hop tax: `docs/findings/SECOND_HOP_TAX.md`
- Reasoning preservation (end-to-end QA): `docs/findings/REASONING_PRESERVATION.md`
- Context economics: `docs/findings/CONTEXT_ECONOMICS.md`
- Index: `docs/findings/README.md`

## Local development

```bash
# from python/, inside a virtualenv with maturin installed
pip install maturin
maturin develop --release      # builds the Rust engine + installs `redhop` editable
python -c "import redhop; print(redhop.__version__)"

maturin build --release        # produce a wheel in target/wheels/
```

Rust is the source of truth: this package is a thin pyo3 binding over the
`redhop-context` crate — no logic is duplicated in Python.
