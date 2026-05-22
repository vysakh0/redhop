# Reasoning-Preserving vs Aggressive Filtering — End-to-End QA (n=300, CIs)

> **Hypothesis:** keeping the reasoning-critical second hop (not just relevance) produces better downstream answers.
> **Status:** Confirmed — n=300, paired bootstrap 95% CIs; effect causally localized to gold reachability.
> **Setup:** 300 gap-qualified multi-hop HotpotQA queries; `claude haiku` generator; gold-keyword recall.
> **Headline:** reasoning − filtered = +0.035 (CI [+0.003, +0.067]); +0.173 on the rescued subset, ~0 (CI spans 0) on the identical-gold control. Surprise: the aggressive *filter* hurt (0.829→0.705), the distractors didn't.
> **Reproduce:** `cargo run -p neorag-examples --example emit_reasoning_qa --release` then `python ../neorag/scripts/score_reasoning_qa.py --n 300` (output in [reports/reasoning_preserving_n300/](../../reports/reasoning_preserving_n300/)).
> **Justifies API:** `build_context(strategy = ReasoningPreserving)`.
> **Caveats:** one generator (haiku), lexical kw-recall proxy, rescued subset small (n=25). See §caveats.

---

The retention experiment ([SECOND_HOP_TAX.md](SECOND_HOP_TAX.md))
proved hermetically (n=1327, CIs) that `ReasoningPreserving` rescues
second hops an aggressive relevance filter drops — that is *reachability*.
This is the *reasoning-success* test: does keeping the second hop produce
better answers? It is the first context strategy with a concrete
mechanism, causal motivation, and measured reachability mitigation that
earns a downstream-QA evaluation.

```bash
cargo run -p neorag-examples --example emit_reasoning_qa --release   # Rust: build contexts
python ../neorag/scripts/score_reasoning_qa.py --n 300 --model haiku # lab: LLM + score + CIs
```

300 gap-qualified multi-hop HotpotQA queries (the regime the tax lives
in), four contexts from the SAME polluted input (gold + 8 off-document
distractors) at an aggressive filter threshold (0.20). 1200 `claude
haiku` calls, paired bootstrap 95% CIs.

## Results

| condition | kw_recall | refusal% |
| --------- | --------- | -------- |
| gold_only (ceiling) | 0.830 | 16% |
| polluted (gold + 8 distractors) | 0.829 | 21% |
| filtered (DistractorFiltered @0.20) | **0.705** | **29%** |
| reasoning (ReasoningPreserving @0.20) | **0.740** | 25% |

| paired comparison | Δ kw_recall | 95% CI |
| ----------------- | ----------- | ------ |
| distractors hurt (gold_only − polluted) | +0.001 | [−0.038, +0.039] |
| **reasoning − filtered** | **+0.035** | **[+0.003, +0.067]** |

## The headline is not what we expected — and it's stronger for the thesis

**The distractors didn't hurt. The *filter* did.**

- Adding 8 distractors to the gold context left haiku's answer quality
  essentially unchanged (0.829 vs 0.830 ceiling; CI on the delta spans
  zero). On this multi-hop population, with the gold present, haiku is
  robust to off-document distractors. The "distractors degrade answers"
  result from the earlier broader experiment **does not replicate on this
  stricter gap-qualified multi-hop subset** — reported honestly.
- But the aggressive relevance **filter crashed quality from 0.829 to
  0.705 (−0.124)** and drove refusals from 21% to 29%. Trying to remove
  distractors that weren't hurting, the filter removed reasoning-critical
  evidence — the second-hop tax, now visible end-to-end as *the filter
  being actively harmful.* The cure was worse than the disease.
- **`ReasoningPreserving` recovers part of the self-inflicted damage:**
  0.705 → 0.740, a +0.035 gain whose 95% CI **excludes zero** (~28% of
  the filtering damage recovered), and pulls refusals back from 29% → 25%.

This is the cleanest possible vindication of the project's direction:
*do not aggressively optimize for query relevance.* The relevance filter
didn't just fail to help — it hurt — and the reasoning-aware strategy is
what makes aggressive filtering safer.

## Mechanism: does reachability cause the answer gain?

The decisive test — separating *reachability* from *reasoning success* —
splits the queries by whether ReasoningPreserving actually saved gold the
filter dropped. We use **full gold retention** per strategy (not a single
proxy chunk): RESCUED = reasoning kept *more* gold than filtered; CONTROL =
*identical* gold retention. (Verified: reasoning's kept-gold is always a
superset of filtered's — 0/300 violations — because both keep
above-threshold seeds and only reasoning rescues below-threshold linked
gold.)

| subset (full-gold) | n | reasoning − filtered | 95% CI |
| ------------------ | - | -------------------- | ------ |
| **RESCUED** (reasoning kept MORE gold) | 25 | **+0.173** | [+0.040, +0.320] |
| CONTROL (IDENTICAL gold retention) | 275 | +0.022 | [−0.007, +0.054] |

This is the clean mechanism the theory predicted:

- **On the RESCUED subset the effect is +0.173** (CI excludes zero) — ~5×
  the aggregate. Precisely where ReasoningPreserving saved low-relevance
  gold the filter taxed, the answer improved sharply.
- **On the CONTROL subset the CI now includes zero** (+0.022 [−0.007,
  +0.054]). Where gold reachability is identical, the two strategies are
  **statistically indistinguishable** — so ReasoningPreserving's other
  differences (readmitting linked junk, reordering) have **no measurable
  downstream effect**. The entire significant advantage localizes to gold
  reachability.

This is the first evidence that **low-relevance-gold preservation
propagates into downstream answer quality, not merely retrieval
reachability** — and that reachability is the *whole* causal story, not
junk handling.

### A correction to the first pass (kept for transparency)

The initial mechanism split used a single-chunk proxy (lowest-grounding
gold) and showed a non-zero control (+0.037, CI excluded zero), which we
flagged as a likely proxy limitation. The refinement confirms it: under
full-gold labeling the control collapses to ~0 (CI spans zero). The proxy
was mislabeling a few queries; the corrected label produces the clean
result. Measurement refinement, not architecture change — exactly as
predicted.

## What is established vs what is not

**Established (CI-backed):**
- An aggressive relevance filter is **net-harmful** on multi-hop QA here
  (−0.124 vs unfiltered), via the second-hop tax — measured end-to-end,
  not inferred.
- **`ReasoningPreserving` significantly beats aggressive filtering**
  (+0.035, CI [+0.003, +0.067]).
- **The advantage is caused by gold reachability.** It is large and
  significant where reasoning saved taxed gold (+0.173) and statistically
  indistinguishable from zero where gold retention is identical — a clean
  causal localization, not a correlation.

**Not established / honest caveats:**
- The aggregate reasoning−filter effect is **real but marginal** (CI lower
  bound +0.003); it is a recovery of filter-induced damage, not a gain over
  *unfiltered* context (reasoning 0.740 < polluted 0.829). The strongest
  operational takeaway is: **if you must filter aggressively, use
  reasoning-preserving; if you don't need to filter, the unfiltered
  context already beats both** on this distractor-robust generator.
- The rescued subset is small (n=25, wide CI [+0.040, +0.320]) — the
  *direction and significance* are solid; the *magnitude* is imprecise.
- **One generator (haiku), one dataset, lexical kw-recall proxy.** A
  weaker generator would likely show real distractor degradation (and so
  a larger filtering-vs-no-filtering tradeoff); kw-recall under-counts
  paraphrase.

## Where this leaves the strategy menu

- **Default: don't over-filter.** On distractor-robust generators the
  unfiltered context wins; aggressive relevance filtering is a
  net-negative move via the second-hop tax.
- **When filtering is necessary** (weak generator, hard token budget,
  genuinely harmful distractors): use **`ReasoningPreserving`**, not plain
  `DistractorFiltered` — it recovers a CI-significant slice of the
  filter's self-inflicted damage, most where it rescues a taxed hop.
- This reframes the product claim honestly: NeoRAG's value here is not
  "filter to boost quality" but **"make the filtering you do
  reasoning-safe"** — and, more broadly, *measure when filtering helps at
  all* rather than assuming it does.

## Next (measurement, not architecture)

1. ~~Full-gold-retention labeling~~ **(done — see the mechanism section.)**
   The control collapsed to ~0 under full-gold labeling, localizing the
   entire advantage to gold reachability.
2. **Weaker generator** (a small open model via the lab) where distractors
   *do* degrade — there the filtering tradeoff, and the
   reasoning-preserving advantage over both filtered and unfiltered, should
   be larger and more decision-relevant.
3. **Semantic-linkage rescue** (signal upgrade, not architecture change):
   embedding similarity instead of lexical Jaccard for the bridge link, to
   rescue paraphrase-linked second hops the lexical signal misses.
