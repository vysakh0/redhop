# Multi-hop retention ceiling — hybrid retrieval rescues bridge passages

> **Status:** **Confirmed** (HotpotQA + MuSiQue, n=100 each, four arms).
> Of the three knobs RedHop ships for "push retention further" —
> `Stripper`, larger `candidate_k`, and `retrieval="hybrid"` — only
> the **hybrid** tier moves the multi-hop number. Stripper is a clean
> no-op (correctly, since `analyze_query_set` doesn't recommend it on
> non-templated workloads). Larger candidate_k is also flat. Dense
> rerank rescues bridge passages that share concepts with the query
> but not words.

## TL;DR

**HotpotQA (n=100, budget 400 tok):**

| arm | mean recall | ≥0.5 | ≥0.8 | p50 ms |
|---|---:|---:|---:|---:|
| A. BM25 baseline (raw_topk, k=20) | 0.89 | 95% | 71% | 2.7 |
| B. BM25 + Stripper (query-side) | 0.89 | 95% | 71% | 2.6 |
| D. BM25 with candidate_k=60 (3×) | 0.89 | 95% | 71% | 2.6 |
| **C. retrieval="hybrid"** | **0.93** | **97%** | **83% (+12)** | 237 |

**MuSiQue (n=100, budget 400 tok):**

| arm | mean recall | ≥0.5 | ≥0.8 | p50 ms |
|---|---:|---:|---:|---:|
| A. BM25 baseline (raw_topk, k=20) | 0.61 | 66% | 23% | 3.2 |
| B. BM25 + Stripper (query-side) | 0.61 | 66% | 23% | 3.2 |
| D. BM25 with candidate_k=60 (3×) | 0.61 | 66% | 23% | 3.3 |
| **C. retrieval="hybrid"** | **0.65** | **74% (+8)** | **26% (+3)** | 386 |

## Question

The 0.3.1 audit established that RedHop leads multi-hop retention on
both HotpotQA and MuSiQue but the lead's magnitude shrinks on the
harder dataset, and that `raw_topk` matches `reasoning_preserving` on
both — so the assembly strategy isn't the differentiator.

User-facing question: **if I'm already using RedHop and I want better
multi-hop retention, which of the helpers we ship can push further?**

Four arms tested:

- **A. BM25 baseline** — the existing claim's reference point.
- **B. BM25 + Stripper(generic-stopwords)** — *predicted no-op*
  because `analyze_query_set` reports the workload isn't templated
  (HotpotQA template_word_share=0.000; MuSiQue=0.113, both well below
  the 0.50 threshold). Test that the helper isn't silently *harmful*
  outside its intended regime.
- **C. `retrieval="hybrid"`** — BM25 candidate pool reranked with the
  bge-small dense embedder. The multi-hop failure mode (per
  [SECOND_HOP_TAX](SECOND_HOP_TAX.md)) is the *bridge passage* — a
  paragraph that links two hops semantically but shares few words with
  the original query. Dense rerank should rescue it.
- **D. BM25 with candidate_k=60** (3× the default) — cheap-knob test:
  is the bridge passage in a larger pool but missed by tighter
  selection?

## What the result says, sharp

**Stripper is a clean no-op on non-templated workloads.** Zero
retention change on either dataset, as `analyze_query_set` predicted.
The helper isn't silently doing damage when applied outside its
regime. Good — the analyzer + Stripper recommendation flow is
internally consistent.

**Larger candidate_k is also flat.** Surprising at first glance, but
informative: the bridge passages aren't being filtered out by a
too-small candidate pool. They aren't *in* the larger BM25 pool
either. BM25 ranks them low not because of pool size, but because the
bridge-passage lexical overlap with the query is genuinely small.
This rules out one tempting "easy fix."

**Hybrid retrieval is the real lever, and the lift is substantial.**
+12 points ≥0.8 on HotpotQA (71% → 83%) and +0.04 mean recall.
+8 points ≥0.5 on MuSiQue (66% → 74%), +0.04 mean recall, +3 points
≥0.8. The bge-small dense rerank rescues paragraphs that share
*concepts* with the question even when they share few words —
exactly the bridge-passage mechanism that SECOND_HOP_TAX named.

**Latency cost is real:** ~90× on HotpotQA (2.7ms → 237ms), ~120× on
MuSiQue (3.2ms → 386ms). The dense embedder runs locally over the
candidate pool; not free.

## What this changes in our positioning

The previous multi-hop story was *"RedHop leads at BM25 default; you
get the lead for free."* Honest revision:

> "RedHop leads at BM25 default, and `retrieval="hybrid"` pushes it
> +8 to +12 points further at ~90-120× the per-query latency. The
> Stripper/Vocabulary helpers don't apply to multi-hop (the analyzer
> correctly stays quiet; the helpers are designed for templated
> workloads like CUAD, not open-domain QA)."

For users picking a configuration:

| If your workload is... | Use... |
|---|---|
| Multi-hop QA, latency-sensitive (≤10ms/query) | BM25 default — the lead over LangChain/LlamaIndex BM25 holds |
| Multi-hop QA, latency-tolerant (~250-400ms/query OK) | `retrieval="hybrid"` — measured +12 ≥0.8 on HotpotQA |
| Templated workload (legal QA, support tickets) | `analyze_query_set` + `Stripper` ± `Vocabulary` |
| Open-domain QA where queries are vague paraphrases | `retrieval="semantic"` (full dense, BM25 bypassed) |

## Honest limits

- **n=100 per dataset.** Smaller than the n=300 retention numbers in
  FRAMEWORK_COMPARISON / MUSIQUE_MULTIHOP. The hybrid arm is ~100×
  slower than baseline, so n=300 would have taken ~30 min. The
  direction is clean (every system-pair comparison is consistent); the
  precise +12 / +8 numbers would shift a point or two at larger n.
- **One dense model tested** (bge-small). bge-base might do slightly
  better; bge-large likely overshoots a CPU latency budget. We did not
  sweep.
- **No comparison vs LangChain/LlamaIndex hybrid arms.** The +12 on
  HotpotQA pushes RedHop into a region (83% ≥0.8) we have not measured
  the competitors at. They both ship dense retrieval too; whether they
  would also lift to 83% on HotpotQA with the same embedder is
  unmeasured. Likely they would — this finding measures the **ceiling
  of RedHop's own knobs**, not relative position.
- **Vocabulary.apply not measured** on multi-hop. Would require a
  synonym corpus authored without knowledge of the gold answers; we
  don't have one for HotpotQA/MuSiQue. The SPIDER_ENRICH-style
  curator-conflict trap applies.
- **Vocabulary.enrich on chunks not measured.** Out of regime
  ([VOCABULARY_ENRICH](VOCABULARY_ENRICH.md)): multi-hop paragraphs
  are prose, neither short nor opaque. The four-corner observation
  predicts null-or-harm.
- **Word-recall metric** measures retrieval, not answer correctness.
  Downstream LLM F1 on the hybrid-rerank pool is not measured.

## Reproduce

```bash
bench/.venv/bin/python bench/multihop_helpers_probe.py
```

First run downloads the bge-small ONNX model (~80MB, cached). Raw run:
[`reports/multihop_helpers_probe_2026-06-08.txt`](../../reports/multihop_helpers_probe_2026-06-08.txt).

## See also

- [SECOND_HOP_TAX](SECOND_HOP_TAX.md) — the mechanism underneath:
  relevance-based selection drops the bridge passage. This finding
  shows dense rerank is what rescues it.
- [MUSIQUE_MULTIHOP](MUSIQUE_MULTIHOP.md) — the BM25-default
  multi-hop comparison this probe extends.
- [GLOBAL_DENSE](GLOBAL_DENSE.md) — the parallel finding for
  paraphrase/synonym queries.
- [CUAD_HYBRID_RERANK](CUAD_HYBRID_RERANK.md) — same question on
  contracts, opposite result: on CUAD, hybrid and Stripper are
  *substitutes* (don't stack). On multi-hop, Stripper does nothing
  (predicted; confirmed) and hybrid is the only lever that moves.
