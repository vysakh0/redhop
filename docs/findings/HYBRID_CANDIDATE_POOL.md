# Hybrid `candidate_pool` is not a lever — leave the default at 50

> **Status: negative result, confirmed.** Swept
> `candidate_pool ∈ {10, 25, 50, 100, 200, 500}` on `retrieval="hybrid"`
> across three workloads (CUAD / HotpotQA / MuSiQue, n=100 each).
> Retention is **flat** within ±2pt noise on every workload — including
> CUAD, where the pool genuinely constrains retrieval (mean 96 chunks
> per doc, max 343). Latency moves <10% on small corpora and doesn't
> monotonically scale with pool on CUAD.
>
> **The default of 50 stays.** This is a "we measured, the knob is
> noise" finding, not a "flip the default" finding (different shape
> from [RAW_ANALYZER](RAW_ANALYZER.md), which was a measured win at the
> default).

## Why this probe ran

The raw-analyzer flip raised an obvious follow-up: where else does
RedHop have a defaulted-on heuristic that nobody has actually
measured? `candidate_pool` (the BM25 first-stage pool size that gets
dense-reranked in hybrid mode) was a strong candidate — defaulted to
50, shipped since 0.3.0, never swept.

The prior MULTIHOP_HYBRID probe measured a similar parameter
(`candidate_k` on the lexical-only retriever, 20 vs 60) and found it
flat. But that's a different param — the lexical top-K with no dense
rerank afterwards. The hybrid `candidate_pool` controls how many
candidates dense rerank gets to look at, and dense rerank is the
mechanism MULTIHOP_HYBRID identified as the +8 to +12 retention lever
on multi-hop. It was plausible the pool size would matter here even
though it didn't for lexical alone.

The intuition both ways:
- **Too small** → BM25 doesn't surface the bridge passage at all →
  dense rerank never sees it → rescue impossible
- **Too big** → dense embeds N more chunks per query → latency grows
  linearly with no gain once the bridge is reliably in the pool

We didn't know which side `candidate_pool=50` lived on. So: sweep.

## The first run measured the wrong thing

Initial probe ran on HotpotQA + MuSiQue only (n=100 each, pool sweep
{10, 25, 50, 100, 200, 500}). Result: completely flat retention on
both workloads.

Then I checked chunks-per-doc:

| Workload | chunks/doc (min, median, max) |
|---|---|
| HotpotQA | 6, 10, 25 |
| MuSiQue | 11, 15, 21 |

So `candidate_pool=50` was ≥ corpus size on essentially every query in
those workloads. The pool wasn't constraining retrieval — we measured
"flat" because the parameter never bound. Like measuring whether a
60-mph speed limit matters on a road where the cars top out at 40.

## The second run — CUAD

CUAD contracts are real-sized documents:

| Workload | chunks/doc (min, median, max) |
|---|---|
| **CUAD** | **3, ?, 343** (mean 96) |

This is the regime where pool genuinely constrains: pool=50 covers
only ~half the corpus for the larger contracts; pool=10 covers a small
fraction of even median-sized contracts.

### Results

| candidate_pool | mean recall | ≥0.5 | ≥0.8 | p50 ms |
|---:|---:|---:|---:|---:|
| 10 | 0.80 | 83% | 64% | 1718 |
| 25 | 0.78 | 82% | 61% | 1680 |
| **50** (default) | 0.80 | 83% | 63% | 1559 |
| 100 | 0.80 | 82% | 63% | 1451 |
| 200 | 0.78 | 81% | 60% | 1359 |
| 500 | 0.79 | 82% | 61% | 1411 |

Retention is flat within ±2pt noise across a 50× sweep. pool=10 has
the highest ≥0.8 (64%); pool=200 has the lowest (60%) — both within
n=100 noise. There is no signal here.

Latency on CUAD doesn't even monotonically grow with pool — pool=200
is the *fastest* point at 1359ms. The expected linear-in-pool cost of
dense embedding is dwarfed by per-document fixed costs (chunking,
BM25 index build over 96+ chunks) on this workload.

## Why is it flat? The same mechanism MULTIHOP_HYBRID named

The bridge passage is in one of two states:

1. **BM25 surfaces it in the top ~10.** Any pool size retrieves it;
   dense rerank can promote it; pool growth past 10 is wasted work.
2. **BM25 can't surface it in the top 500.** The bridge shares so few
   lexical tokens with the query that even a 500-deep BM25 pool misses
   it. Dense rerank can't help because dense rerank only sees what
   BM25 surfaced.

There isn't a meaningful third bucket where the bridge sits at BM25
rank 11-500 — i.e., where the pool size actually matters. The
distribution of "lexical overlap between query and bridge passage" is
bimodal in practice: either it's a decent BM25 match (in the top 10)
or it's almost nothing (off the cliff entirely). Growing the pool
mostly adds distractors, not bridges.

This generalizes the MULTIHOP_HYBRID finding ("larger candidate_k is
also flat" — for lexical alone) to the hybrid mode.

## What this changes

- **Default of 50 stays.** We measured it, it's neither over- nor
  under-allocated in any meaningful way.
- **Don't tune this param.** Users who see `candidate_pool` in the API
  and think "bigger is better, set it to 200 just in case" are wasting
  effort (and on small corpora, a tiny amount of latency).
- **Confirms RedHop's hybrid story.** The +8 to +12 retention lift on
  multi-hop from `retrieval="hybrid"` is from the dense reranker
  rescuing bridge passages that BM25 *already surfaced low*, not from
  expanding the pool. The default pool is large enough; the bottleneck
  is BM25's lexical reach, not the pool size.

## Honest limits

- **Three workloads only.** CUAD (templated legal), HotpotQA (2-hop
  Wiki), MuSiQue (compositional 2-4 hop Wiki). A code-search workload
  or a workload with very long documents (>500 chunks where pool=50 is
  a tiny fraction) could in principle behave differently. Probability
  that they invalidate the conclusion: low, because the mechanism
  argument doesn't depend on workload — it depends on BM25 having
  bimodal lexical-overlap behavior, which seems robust.
- **n=100 per pool.** Bigger n would tighten the ±2pt error band, but
  the direction is clear (flat).
- **Single retriever pair.** BM25 + bge-small dense. A different dense
  model (e5-base, gte-base) might surface bridge passages differently,
  but the *pool* still wouldn't matter — it'd be the dense model's
  ability to score lexically-distant passages that changed.
- **Single fusion strategy.** We use pure dense rerank (the 0.3.1
  default, see [HYBRID_LATENCY_PROFILE](HYBRID_LATENCY_PROFILE.md)),
  not RRF. RRF might respond differently to pool size, but RRF is
  off-default for the buried-bridge reason.

## Reproduce

```bash
bench/.venv/bin/python bench/hybrid_candidate_pool_sweep.py
```

Raw run: [`reports/hybrid_candidate_pool_sweep_2026-06-08.txt`](../../reports/hybrid_candidate_pool_sweep_2026-06-08.txt).

## See also

- [MULTIHOP_HYBRID](MULTIHOP_HYBRID.md) — the original probe that
  found hybrid is the +8/+12 multi-hop lever and that larger BM25
  `candidate_k` (lexical-only) is flat. This finding extends that to
  the hybrid pool.
- [RAW_ANALYZER](RAW_ANALYZER.md) — the audit's positive result
  (default was wrong, flipped). Contrast: a defaulted-on heuristic
  that *was* hurting, vs this one which is just inert.
- [HYBRID_LATENCY_PROFILE](HYBRID_LATENCY_PROFILE.md) — why hybrid is
  expensive (dense embedding dominates) and why the pool size matters
  less than you'd think for latency.
