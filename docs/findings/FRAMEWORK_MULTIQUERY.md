# Multi-query-per-document pattern — RedHop vs LangChain vs LlamaIndex

> **Status:** **Confirmed.** Three honest findings from a 100-query
> chatbot/KB-style benchmark on CUAD:
> 1. **RedHop has the fastest cold start** (1ms p50 to index a contract).
> 2. **LangChain has the fastest warm queries** (0.3ms p50, vs RedHop's
>    3.3ms — Tantivy's analyzer + scoring overhead is ~10× the simpler
>    BM25Okapi path).
> 3. **LlamaIndex has the best retention** (83% ≥0.8 vs RedHop 77%,
>    LangChain 67%) — their 256-token SentenceSplitter chunking
>    happens to fit CUAD contract shape.

## TL;DR

CUAD multi-query, n=10 contracts × 10 questions each = 100 queries,
budget=2000 tok, candidate_k=40:

| system | cold p50 | warm p50 | warm p99 | total wall-clock | mean recall | **≥0.8** |
|---|---:|---:|---:|---:|---:|---:|
| **redhop[topk]** | **1ms** | 3.3ms | 11ms | 439ms | 0.87 | 77% |
| **langchain** | 4ms | **0.3ms** | 1ms | **118ms** | 0.83 | 67% |
| **llamaindex** | 17ms | 0.5ms | 2ms | 476ms | 0.91 | **83%** |

All BM25-only (no dense rerank, no model download). Same metric as
`bench/compare.py` (gold-span word recall on the CUAD answer span).

## Why this probe exists

[FRAMEWORK_COMPARISON](FRAMEWORK_COMPARISON.md) measures
one-query-per-document — build an index, ask a question, throw it
away. That's the right benchmark for stateless "summarize this PDF"
jobs, but it MISSES the most common production pattern:

- Chat apps with the same document in context across a conversation
- Knowledge-base lookups: build the index once, then answer everything
- Support agents iterating on the same ticket
- Long document-review sessions

All of those build the index **once** and run **many** queries against
it. The numbers that matter become:
- **Cold cost**: index-build time (paid once)
- **Warm cost**: per-query time after the index exists (paid M times)
- **Total wall-clock** for the realistic pattern

`bench/compare.py` would tell you a multi-query workload is "fast"
because per-call latency looks low — but that hides the
build-then-query asymmetry. This probe separates them.

## What the numbers say, sharp

### 1. RedHop's cold start is the fastest

1ms p50 to index a CUAD contract (~30-50 KB of text). Tantivy's BM25
indexer is competitive with anything. LangChain's BM25Okapi at 4ms is
fine; LlamaIndex at 17ms pays for the SentenceSplitter chunking step
plus the BM25 setup.

For "build once, query many" patterns where cold start happens
per-document (e.g. one Document per chat session), RedHop's cold
advantage is meaningful: at 1000 docs you save ~16 seconds vs LlamaIndex.

### 2. LangChain's warm queries are 10× faster than RedHop's

0.3ms vs 3.3ms p50. The cause is implementation: RedHop uses Tantivy
with full stemming/analyzer per query (Snowball Porter2, Unicode
normalization, stopword handling); LangChain uses BM25Okapi from
`rank_bm25` which is essentially a TF-IDF hash table lookup.

**For most workloads this gap doesn't matter** — 3.3ms warm queries
are still way below human reaction time. But if you're doing 10,000
warm queries against the same index (a busy KB), RedHop's cumulative
latency is ~33s vs LangChain's ~3s. The wall-clock total in this
probe (439ms vs 118ms for 100 queries) is the visible artifact.

Two paths if this matters for your workload:

- **Use the `raw_topk` strategy** (already the default for `Document`
  configurations that don't request `reasoning_preserving`) — RedHop's
  assembly overhead is part of the 3.3ms; raw_topk minimizes it.
- **Accept the trade.** RedHop's analyzer + stemming is what makes
  `"highlight"` match `"highlighted"`, which is part of why retention
  is better on the templated-workload regime. The 3ms is the price.

### 3. LlamaIndex wins retention on CUAD

83% ≥0.8 vs RedHop 77% vs LangChain 67%. The likely cause: LlamaIndex's
SentenceSplitter at 256-token chunks fits CUAD's clause-paragraph
shape better than RedHop's 128-token default
([MULTIHOP_CONSTANT_CHUNKING](MULTIHOP_CONSTANT_CHUNKING.md) showed
the chunker is the lever).

This is consistent with the
[FRAMEWORK_COMPARISON](FRAMEWORK_COMPARISON.md) one-shot CUAD result
(LlamaIndex 86% > RedHop 82%) — LlamaIndex's contract retention
advantage holds in the multi-query pattern too. RedHop closes the gap
with `Stripper + Vocabulary` on the query side, which is workload-
specific authoring effort.

## What this changes

- **`bench/compare_multiquery.py`** is now part of the suite. Users
  evaluating RedHop for chatbot/KB patterns have a measurement that
  matches their workload.
- **The "fast" claim is now precise.** RedHop has the best cold start,
  LangChain has the best warm-query throughput, LlamaIndex has the
  best contract retention. "Just fast" doesn't survive the multi-
  query measurement; the right tier matters.

## Honest limits

- **One dataset (CUAD).** Hotpot/MuSiQue don't have multi-question-
  per-doc structure cleanly available. CUAD-shape is templated legal
  QA; results may shift on a different multi-query workload (chat
  histories, customer-support ticket lifecycles, etc.).
- **n=10 docs × m=10 queries = 100 queries.** Smaller sample than
  `bench/compare.py`'s n=300, but the warm-query measurement requires
  setting up new docs which limits how big we can make it without
  long bench runs.
- **BM25-only.** The dense/hybrid path isn't measured in this probe.
  Under hybrid, RedHop's index-build becomes ~200ms (embedding cost)
  while LangChain/LlamaIndex hybrid amortize embeddings differently.
  That's the `compare.py` + competitor-probe territory.
- **No statistical CIs.** Single-run timings can vary ±10% on a
  laptop; the cross-system ordering is stable across re-runs but the
  precise numbers shift.

## Reproduce

```bash
bench/.venv/bin/python bench/compare_multiquery.py
```

Raw run: [`reports/framework_comparison_multiquery_2026-06-08.txt`](../../reports/framework_comparison_multiquery_2026-06-08.txt).

## See also

- [FRAMEWORK_COMPARISON](FRAMEWORK_COMPARISON.md) — one-query-per-doc
  numbers (the stateless pattern).
- [MULTIHOP_CONSTANT_CHUNKING](MULTIHOP_CONSTANT_CHUNKING.md) — why
  LlamaIndex's chunker wins on CUAD (the chunker is the lever).
- [CHUNK_GRANULARITY](CHUNK_GRANULARITY.md) — earlier finding on
  chunk-size impact.
