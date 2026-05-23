<h1 align="center">RedHop</h1>

<p align="center"><b>Reasoning-preserving context optimization and retrieval observability for RAG systems.</b></p>

RedHop sits **between retrieval and generation**. You hand it the chunks your
retriever returned and a token budget; it assembles the prompt context —
pruning distractors, **preserving reasoning-critical "second-hop" evidence**,
and reporting exactly what it did. It is *not* a retriever, vector database,
agent framework, or workflow engine.

## Why RedHop exists

RAG stacks routinely "improve" context with relevance operations — aggressive
distractor filtering, cross-encoder reranking, max-density pruning. We measured
what those do to **multi-hop** questions and found a consistent failure:

> Transformers tolerate irrelevant context far better than they tolerate
> **missing reasoning links**. Premature removal of low-relevance reasoning
> evidence hurts more than the distractors do.

On a multi-hop question, the *second hop* (the evidence that bridges to the
answer) is **low-relevance-to-the-query by construction** — it connects through
a bridge entity, not the query terms. So relevance-based pruning silently drops
exactly the chunk the answer depends on. RedHop's default strategy keeps it, and
its reports make the tradeoff **visible and measurable** instead of silent.

This isn't a claim — it's measured, with confidence intervals and preserved
falsified hypotheses, in the [evidence layer](docs/findings/README.md).

## Quick example (Python)

Reason over a document — chunking, indexing, and retrieval are internal; you
think in documents and queries, not retrieval infrastructure:

```python
import redhop

doc = redhop.Document.from_text(text)         # bring your own parser/OCR
ctx = doc.context("Why did the proposed method fail?")
response = llm.generate(ctx.text())           # any provider; no lock-in
print(ctx.report)                             # what was retrieved/pruned, and why
```

Already have chunks? The low-level surface is still first-class:

```python
chunks = retriever.retrieve(query)            # your stack
ctx = redhop.build_context(
    query=query,
    retrieved_chunks=chunks,                  # list of dicts or strings
    strategy="auto",                          # size-gated: pass under headroom, prune under dilution
    token_budget=12000,
)
response = llm.generate(ctx.text())           # your stack
print(ctx.report)                             # observability ↓
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
```

## Context strategies, side by side

The same multi-hop retrieval, run through each strategy (`redhop compare`).
The answer needs the "British" second hop — aggressive filtering drops it:

```text
strategy                chunks   tokens   removed  rescued  distr  density  gold_ret  2nd_hop
raw_topk                8→8      100      0        0        0.88   0.10     1.00      ✓
distractor_filtered     8→1      17       7        0        0.00   0.29     0.50      ✗   ← dropped it
max_density             8→8      100      0        0        0.88   0.10     1.00      ✓
reasoning_preserving    8→2      30       6        1        0.50   0.20     1.00      ✓   ← kept + pruned
```

`distractor_filtered` removes all distractors but **taxes away the second hop**
(gold_ret 0.50). `reasoning_preserving` prunes 6 distractors *and* keeps the
reasoning-critical hop (it rescues low-relevance chunks linked to a kept seed).

| strategy | what it does | when |
| -------- | ------------ | ---- |
| `reasoning_preserving` *(default)* | keep query-relevant seeds **and** rescue low-relevance chunks linked to a seed; drop only unlinked junk | multi-hop / general |
| `distractor_filtered` | drop everything below a query-grounding bar | single-hop, or a *low* threshold only |
| `max_density` | greedily pack the densest chunks into the budget | single-hop / brutal budgets |
| `raw_topk` | keep retrieval order until the budget fills | baseline / no optimization |

## Findings (the evidence layer)

Every default exists because a specific failure was measured. Falsified
hypotheses are kept, not deleted — several of the strongest defaults came from
one. Full index: [docs/findings/](docs/findings/README.md).

| Finding | Status | Headline |
| ------- | ------ | -------- |
| [Second-hop tax](docs/findings/SECOND_HOP_TAX.md) | Confirmed (n=1327, CIs) | every relevance-based selection taxes the multi-hop second hop; a 0.30 filter keeps only 44% |
| [Reasoning preservation](docs/findings/REASONING_PRESERVATION.md) | Confirmed (n=300, CIs) | reasoning-preserving beats aggressive filtering end-to-end; gain causally localized to gold reachability |
| [Reranking limits](docs/findings/RERANKING_LIMITS.md) | **Falsified** | "a stronger reranker recovers missed recall" — uniform cross-encoder made recall *worse* |
| [Distractor robustness](docs/findings/DISTRACTOR_ROBUSTNESS.md) | Partially falsified | "distractor filtering is a free win" — net benefit is sign-unstable on multi-hop |
| [Context economics](docs/findings/CONTEXT_ECONOMICS.md) | Confirmed | distractors hurt & density helps on real LLM outputs (pooled −0.375 / +0.539) |

## Philosophy — bounded by design

RedHop optimizes **reasoning-completeness under finite attention**, not generic
retrieval scores. It is deliberately *not* a framework:

- **No** agents, planners, workflows, or orchestration DAGs.
- **No** graph traversal or query decomposition.
- **No** embedded LLM or vector DB — you bring those.
- **Observability-first**: every strategy emits a `ContextReport`.
- **Evidence-first**: APIs and defaults are grounded in measured failure
  geometry, with caveats and confidence intervals kept honest.

The library does one thing — allocate the context budget to the densest *and
most reasoning-complete* evidence — and reports what it did.

## Installation

**Python** (native wheel, no Rust toolchain needed to use it):

```bash
pip install redhop
```

**Rust**:

```toml
[dependencies]
redhop-context = "0.1"   # the core context API
```

## CLI

A thin, Unix-like eval/observability CLI (`cargo build -p redhop-cli`):

```bash
redhop compare --input retrieval.json \
  --strategies raw_topk,distractor_filtered,reasoning_preserving \
  --gold-ids c3,c7 --second-hop-id c7      # optional retention columns

redhop analyze-context context.json --query "..."   # Context Optimization Report
redhop benchmark --input labeled.json --budgets 250,800,12000   # JSON + CIs
redhop report results.json --html report.html       # render artifacts
```

See [crates/cli/README.md](crates/cli/README.md).

## Documentation

- **Docs site** (mdBook): `docs/book/` — `mdbook serve docs/book`
- **Retrieval & context tips** (start here): [docs/retrievaltips.md](docs/retrievaltips.md) — the operational laws and which API applies each
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Evidence layer**: [docs/findings/](docs/findings/README.md) ·
  **Benchmarks**: [benchmarks/](benchmarks/README.md) ·
  **Reports**: [reports/](reports/README.md)
- **Python**: [python/README.md](python/README.md) ·
  **Roadmap**: [ROADMAP.md](ROADMAP.md) · **FAQ**: [FAQ.md](FAQ.md)

## License

Apache-2.0.
