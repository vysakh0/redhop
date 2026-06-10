# RedHop vs LangChain vs LlamaIndex

A fair, reproducible comparison, including where RedHop *doesn't* win. We'd
rather you trust the numbers than the marketing.

> Looking for a different comparison? See
> [**RedHop's eval surface vs Ragas**](COMPARISON_RAGAS.md) for the
> answer-quality eval head-to-head (n=200 HotpotQA, r=+0.664 vs Ragas).

## TL;DR

- On answer quality, **RedHop is competitive with LlamaIndex and ahead of
  LangChain**. It holds its own with the category leaders. It is **not** a
  blowout, and we won't claim it is.
- RedHop's actual differentiation isn't raw retrieval quality. It's being a
  **bounded, interpretable, conditional context runtime**: it explains every
  decision, prunes only when it helps, and ships an evidence layer for every
  default. LangChain and LlamaIndex are broad orchestration/integration
  frameworks. RedHop is a focused layer that sits between your documents and the
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
| Claims | benchmark-backed, falsified hypotheses kept | varies |

RedHop is *not* a vector DB, parser, agent framework, or workflow engine, and
it deliberately stays that way.

## The benchmark

Same documents, **BM25 retrieval for all three** (so we compare assembly, not
retrieval engines), same token budget (set below document size so selection
actually happens). Two datasets: **CUAD** (real contracts) and **HotpotQA**
(multi-hop). Two tiers: evidence retention (no LLM) and downstream answer
quality (gpt-4o-mini). Full method + caveats:
[docs/findings/FRAMEWORK_COMPARISON.md](findings/FRAMEWORK_COMPARISON.md).

**Evidence retention** (gold-evidence recall, ≥0.8, n=300, latest rerun
2026-06-06, plus MuSiQue 2026-06-08):

| dataset | RedHop | LangChain | LlamaIndex |
| --- | --- | --- | --- |
| HotpotQA (multi-hop) | **80%** | 71% | 72% |
| MuSiQue (compositional multi-hop) | **22%** | 19% | 17% |
| CUAD (contracts, raw template query) | 82% | 73% | **86%** |

On MuSiQue, mean recall is essentially tied (0.59 vs 0.56). The ≥0.8 lead is
the durable part of the result. On both multi-hop datasets `raw_topk` matches
`reasoning_preserving`, so the edge comes from RedHop's chunking + BM25
defaults, not the assembly strategy
([MUSIQUE_MULTIHOP](findings/MUSIQUE_MULTIHOP.md)).

**Answer quality** (gpt-4o-mini, F1, n=150):

| dataset | RedHop | LangChain | LlamaIndex |
| --- | --- | --- | --- |
| HotpotQA (F1 / EM) | **0.51** / 0.41 | 0.50 / 0.39 | 0.50 / **0.42** |
| CUAD (F1 / EM) | 0.34 / 0.17 | 0.25 / 0.11 | **0.35** / 0.16 |

## How to read this

- **RedHop leads multi-hop retention** and **ties LlamaIndex / beats LangChain on
  answers.** LlamaIndex edges RedHop on contract extraction. No single system
  dominates.
- **Retention is a loose proxy for answers**: RedHop's bigger retention lead
  shrinks to a near-tie on answer quality, because at a sensible budget every
  system gives the model enough to roughly tie. We report both so you can see it.
- These are BM25-vs-BM25 results. The frameworks' default *vector* retrievers
  aren't covered here (see the hybrid head-to-head below for the dense-rerank
  comparison).

## CUAD in depth: fair preprocessing, and the recipe ladder

The 4-point raw-template gap to LlamaIndex is mechanism-known: **BM25
dilution from CUAD's fixed 24-word boilerplate**
([CUAD_RECALL_GAP](findings/CUAD_RECALL_GAP.md)).

**Fair-preprocessing result** (`bench/compare.py`, n=300, 2026-06-08):
applying `Stripper(boilerplate)` to *every* system's query before retrieval
lifts everyone:

| system | raw template | + same Stripper |
| --- | ---:| ---:|
| LlamaIndex | 86% | **94%** |
| RedHop | 82% | 88% |
| LangChain | 73% | 79% |

**LlamaIndex actually benefits more from the same Stripper than RedHop does.**
Its BM25 retriever is the stronger one on contract-extraction.

RedHop's own recipe ladder (controlled three-arm run, n=300,
[CUAD_CLAUSE_EXPANSION](findings/CUAD_CLAUSE_EXPANSION.md)):

| step | helper | retention | Δ |
| ---- | ------ | ---------:| -:|
| raw 24-word template | — | 81.3% | — |
| + strip the wrapper | `Stripper` | 87.7% | **+6.4** |
| + add workload synonyms | `Vocabulary` (34-key clause dict) | **90.7%** | **+3.0** |

What this does and doesn't show:

- A reproducible, in-process, audited path from 81.3% → 90.7% with the
  per-stage trail on the Decision Report. The `Stripper` primitive is
  reusable across any templated workload.
- **Not** "RedHop beats LlamaIndex by 4.7 points." The Vocabulary recipe was
  not applied to LlamaIndex, and given LlamaIndex's bigger lift from the
  Stripper step, an unmeasured-but-likely outcome is that it would match or
  exceed 90.7% with the same recipe. The retrieval engines are roughly
  comparable on contracts once preprocessing is held constant.
- On CUAD, BM25 + strip + expand is Pareto-optimal vs hybrid+cross-encoder
  (90.3% / ~2.5ms vs 89.0% / ~683ms in the 6-arm probe, run-to-run variance
  vs the 90.7% three-arm figure): the two paths are *substitutes*, not
  complements ([CUAD_HYBRID_RERANK](findings/CUAD_HYBRID_RERANK.md)).
- For single-doc extraction also set `strategy="raw_topk"`. It beats `auto`
  by ~4 points at every chunk size on CUAD.

## Hybrid head-to-head (same dense model on all three)

Apples-to-apples hybrid: identical bge-small embedder, n=100, post
pure-rerank fix in 0.3.1
([MULTIHOP_HYBRID_COMPETITORS](findings/MULTIHOP_HYBRID_COMPETITORS.md)):

| dataset (≥0.8 retention) | RedHop hybrid | LangChain | LlamaIndex |
| --- | ---:| ---:| ---:|
| HotpotQA | **81%** | 77% | 67% |
| MuSiQue | 34% | **39%** | 31% |

- Against RedHop's own BM25 default, hybrid is the one knob that moves
  multi-hop: HotpotQA ≥0.8 71% → 81%. The bottleneck is the
  lexical-vs-semantic gap on bridge passages, and only dense rerank pierces
  it ([MULTIHOP_HYBRID](findings/MULTIHOP_HYBRID.md)). Stripper and
  `candidate_k` tuning are measured no-ops there.
- The previously-published RedHop hybrid numbers (HotpotQA 83%, MuSiQue 26%)
  used RRF fusion, which buried compositional bridge passages. 0.3.1 replaced
  it with pure dense rerank (net −2 HotpotQA, +8 MuSiQue).
- RedHop's hybrid is currently **2–5× slower** than the competitors' hybrid:
  ORT-CPU vs PyTorch-MPS, plus eager index-time embedding and ~60% more
  chunks at the 128-token default. Known, mechanism-attributed, unfixed.
  See [HYBRID_LATENCY_PROFILE](findings/HYBRID_LATENCY_PROFILE.md).

## What you actually get with RedHop that you don't elsewhere

1. **A Decision Report for every call**: what it did, why, and *why it chose not
   to intervene*. No black box.
2. **Conditional optimization**: `Auto` prunes only when the context is large and
   diluted (where we measured it helps), and passes small contexts through
   untouched (where pruning is wash-to-harmful).
3. **An evidence layer**: every default traces to a measured finding, including
   the experiments that *failed*. See [docs/findings/](findings/README.md).
4. **A tiny, bounded surface**: `Document.from_text(...).context(query)`. No
   chains, agents, or vector-store wiring to learn.

## Reproduce it yourself

The benchmark lives in the repo. Run it on your own data:

```bash
python3 -m venv bench/.venv
bench/.venv/bin/pip install rank-bm25 langchain-community llama-index-core llama-index-retrievers-bm25
bench/.venv/bin/pip install ./python
bench/.venv/bin/python bench/compare.py          # retention (free)
bench/.venv/bin/python bench/tier3.py --n 150    # answer quality (needs OPENROUTER_API_KEY)
```

## Honest caveats

- gpt-4o-mini only, one budget per dataset. CUAD extraction F1 is low in
  absolute terms (hard task). The *relative* ranking is the signal. The
  answer-quality table predates the 2026-06-06 retention rerun.
- MuSiQue absolute retention is low for *all three* systems (22% / 19% / 17%).
  Compositional multi-hop is largely unsolved at the single-pass BM25 tier,
  and we'd rather show that than hide it.
- LlamaIndex's raw-template contract edge is real and now explained: BM25
  boilerplate dilution ([CUAD_RECALL_GAP](findings/CUAD_RECALL_GAP.md)). The
  chunk-fragmentation hypothesis was falsified
  ([CUAD_CHUNK_FRAGMENTATION_NULL](findings/CUAD_CHUNK_FRAGMENTATION_NULL.md)).
- RedHop's `reasoning_preserving` strategy does **not** beat plain top-k
  downstream: its value is the *runtime decisions and transparency*, not a
  magic optimizer. We say so because it's true.
