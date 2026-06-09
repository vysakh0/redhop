# RedHop vs LangChain vs LlamaIndex

A fair, reproducible comparison — including where RedHop *doesn't* win. We'd
rather you trust the numbers than the marketing.

> Looking for a different comparison? See
> [**RedHop's eval surface vs Ragas**](COMPARISON_RAGAS.md) for the
> answer-quality eval head-to-head (n=200 HotpotQA, r=+0.664 vs Ragas).

## TL;DR

- On answer quality, **RedHop is competitive with LlamaIndex and ahead of
  LangChain** — it holds its own with the category leaders. It is **not** a
  blowout, and we won't claim it is.
- RedHop's actual differentiation isn't raw retrieval quality — it's being a
  **bounded, interpretable, conditional context runtime**: it explains every
  decision, prunes only when it helps, and ships an evidence layer for every
  default. LangChain and LlamaIndex are broad orchestration/integration
  frameworks; RedHop is a focused layer that sits between your documents and the
  LLM.
- Pick RedHop when you want a small, observable, reasoning-preserving context step you
  can reason about. Pick the big frameworks when you want a large connector/agent
  ecosystem.

## Different category, on purpose

|  | RedHop | LangChain / LlamaIndex |
| --- | --- | --- |
| Scope | context assembly between docs and LLM | broad RAG + agent + integration frameworks |
| Retrievers/connectors | internal (BM25 today), not surfaced | dozens, user-wired |
| You think in | documents + queries | retrievers, indexes, query engines, chains |
| Decisions | explained (the Decision Report) | mostly opaque |
| Optimization | conditional (prune only under dilution) | usually fixed top-k |
| Claims | benchmark-backed; falsified hypotheses kept | varies |

RedHop is *not* a vector DB, parser, agent framework, or workflow engine — and
deliberately stays that way.

## The benchmark

Same documents, **BM25 retrieval for all three** (so we compare assembly, not
retrieval engines), same token budget (set below document size so selection
actually happens). Two datasets: **CUAD** (real contracts) and **HotpotQA**
(multi-hop). Two tiers: evidence retention (no LLM) and downstream answer
quality (gpt-4o-mini). Full method + caveats:
[docs/findings/FRAMEWORK_COMPARISON.md](findings/FRAMEWORK_COMPARISON.md).

**Evidence retention** (gold-evidence recall, ≥0.8, n=300):

| dataset | RedHop | LangChain | LlamaIndex |
| --- | --- | --- | --- |
| HotpotQA (multi-hop) | **77%** | 71% | 72% |
| CUAD (contracts) | 82% | 73% | **86%** |

**Answer quality** (gpt-4o-mini, F1, n=150):

| dataset | RedHop | LangChain | LlamaIndex |
| --- | --- | --- | --- |
| HotpotQA (F1 / EM) | **0.51** / 0.41 | 0.50 / 0.39 | 0.50 / **0.42** |
| CUAD (F1 / EM) | 0.34 / 0.17 | 0.25 / 0.11 | **0.35** / 0.16 |

## How to read this

- **RedHop leads multi-hop retention** and **ties LlamaIndex / beats LangChain on
  answers.** LlamaIndex edges RedHop on contract extraction. No single system
  dominates.
- **Retention is a loose proxy for answers** — RedHop's bigger retention lead
  shrinks to a near-tie on answer quality, because at a sensible budget every
  system gives the model enough to roughly tie. We report both so you can see it.
- These are BM25-vs-BM25 results; the frameworks' default *vector* retrievers
  aren't covered here.

## What you actually get with RedHop that you don't elsewhere

1. **A Decision Report for every call** — what it did, why, and *why it chose not
   to intervene*. No black box.
2. **Conditional optimization** — `Auto` prunes only when the context is large and
   diluted (where we measured it helps), and passes small contexts through
   untouched (where pruning is wash-to-harmful).
3. **An evidence layer** — every default traces to a measured finding, including
   the experiments that *failed*. See [docs/findings/](findings/README.md).
4. **A tiny, bounded surface** — `Document.from_text(...).context(query)`. No
   chains, agents, or vector-store wiring to learn.

## Reproduce it yourself

The benchmark lives in the repo — run it on your own data:

```bash
python3 -m venv bench/.venv
bench/.venv/bin/pip install rank-bm25 langchain-community llama-index-core llama-index-retrievers-bm25
bench/.venv/bin/pip install ./python
bench/.venv/bin/python bench/compare.py          # retention (free)
bench/.venv/bin/python bench/tier3.py --n 150    # answer quality (needs OPENROUTER_API_KEY)
```

## Honest caveats

- gpt-4o-mini only; one budget per dataset; two datasets. CUAD extraction F1 is
  low in absolute terms (hard task) — the *relative* ranking is the signal.
- LlamaIndex's contract edge is real and not yet explained (likely its node
  parsing / tokenization on legalese).
- RedHop's `reasoning_preserving` strategy does **not** beat plain top-k
  downstream — its value is the *runtime decisions and transparency*, not a
  magic optimizer. We say so because it's true.
