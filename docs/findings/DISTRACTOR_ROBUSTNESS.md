# End-to-End QA: Do Distractors Hurt, and Does Filtering Help?

The correlation→causation closer for context economics. RedHop (Rust)
built, per HotpotQA query, four contexts from KNOWN gold chunks +
controlled off-topic distractor injection; the Python lab called
`claude haiku` on each and scored answer quality (gold-keyword recall)
+ refusal rate.

```bash
cargo run -p redhop-examples --example emit_qa_contexts --release   # Rust: build contexts
python ../neorag/scripts/score_context_qa.py --n 30                 # lab: LLM + score
```

Filter safety (Rust side): distractor filtering kept **96% of gold
chunks (69/72)** while removing 94 injected distractors.

## Results — and a sign flip that is itself the finding

| condition | kw_recall (n=20) | kw_recall (n=30) | refusal% (n=30) |
| --------- | ---------------- | ---------------- | --------------- |
| gold_only (clean ceiling) | 0.924 | 0.916 | 10% |
| polluted_4 (gold + 4 distractors) | 0.871 | 0.883 | 10% |
| polluted_8 (gold + 8 distractors) | 0.891 | 0.883 | 17% |
| filtered_8 (polluted_8, filtered) | **0.841** | **0.903** | 13% |

| derived | n=20 | n=30 |
| ------- | ---- | ---- |
| distractor degradation (gold_only − polluted_8) | +0.033 | +0.033 |
| filtering net effect (filtered_8 − polluted_8) | **−0.050** | **+0.020** |

**The filtering effect flipped sign between n=20 and n=30.** Ten extra
queries reversed the conclusion. That is the single most important
result here, and it disciplines every claim below.

## What is robust vs what is not

### Robust: distractors degrade generated answers (causal)

`gold_only − polluted_8 = +0.033` on **both** runs. Adding off-topic
distractors to a context that already contains the gold lowers answer
quality and **raises refusals (10% → 17%)**. This causally confirms
Experiment B's correlation (`distractor_ratio → answer kw-recall` =
−0.375 across 5 real LLM runs): distractors don't just correlate with
worse retrieval metrics, they *cause* worse generated answers. The
effect is modest — `claude haiku` is fairly robust to a few
distractors — but consistent and directional.

### NOT resolved: whether distractor *filtering* nets positive

`filtered_8 − polluted_8` was **−0.050 at n=20 and +0.020 at n=30**.
The sign flipped. At effect sizes of ~0.02–0.05 and n=20–30, this is
**within sample noise** — the experiment cannot say whether safe
distractor filtering is a net win or loss at this scale. The n=30 run's
"filtering recovers 60% of the degradation" is *not* a claim I'll stand
behind, because the n=20 run said the opposite.

### Why it's a wash: the second-hop tax

The mechanism is visible in the safety check. The filter kept 96% of
gold but dropped 4% (3/72 chunks) — and on multi-hop QA the dropped
chunks are the **low-query-relevance second hops** (same geometry as the
cross-encoder and max-density findings). So:

> distractor-removal benefit  ≈  second-hop-loss cost  ⇒  net ≈ 0  ⇒
> the result is dominated by sample noise.

If the filter dropped *zero* gold (a perfect filter), the gold_only
ceiling (0.916) shows the headroom is real — distractor removal *would*
help. The filter can't realize it because it cannot tell a
low-relevance distractor from a low-relevance second hop.

## The honest correction to last turn's product framing

Last turn the working hypothesis was "safe distractor filtering is the
strongest immediate productizable feature." This experiment **tempers
that**:

- The *premise* (distractors hurt) is confirmed end-to-end. ✓
- But *filtering's net benefit on multi-hop is unproven at this scale*,
  and the second-hop tax makes it a wash here. ✗ (for multi-hop)
- It is likely net-positive on **single-hop** workloads, where gold is
  query-relevant and the filter won't drop it — but that's an
  untested extrapolation, flagged as such.

So the productizable claim narrows honestly to: **distractor filtering
is safe (96% gold retention) and removes demonstrably harmful content,
but its end-to-end answer-quality benefit on multi-hop QA is within
noise until the second hop is preserved.** Second-hop preservation moves
from "research frontier" to "the binding constraint on whether the
flagship feature actually pays off."

## The reranking-limits geometry, a fourth appearance

1. ExpandTopK — can't reach the second hop.
2. Cross-encoder rerank — demotes it.
3. Max-density pruning — drops it.
4. **Distractor filtering — drops 4% of it, enough to cancel the
   distractor-removal benefit.**

Every relevance-based operation taxes the second hop. Even the "safe"
absolute-threshold filter, at a low 0.10 cutoff, catches the lowest-
relevance gold. The safety margin is "how far the second hop's relevance
sits above the threshold" — and on adversarial multi-hop it sits close.

## Methodological lesson (worth keeping)

Small-sample LLM evaluation is unstable: a 50-point swing in the
filtering verdict came from 10 queries. Any future end-to-end claim
needs **hundreds of queries and reported confidence intervals**, not
20–30. The robust findings here (distractors degrade; the second-hop
tax) survive because they're either consistent across both runs or
mechanistically grounded; the filtering net-effect does not, and is
reported as unresolved.

## Honest limits

- **n=20–30, sign-unstable.** The headline caveat.
- **kw-recall is a coarse answer-quality proxy** (gold-keyword presence);
  it under-counts paraphrase and over-counts incidental term matches.
- **`claude haiku`, single generator.** A weaker model would likely show
  larger distractor degradation (less robust to noise).
- **HotpotQA, adversarially multi-hop.** The second-hop tax is worst-case
  here; single-hop QA would likely show clean filtering wins.
- **Controlled off-topic injection**, not natural retrieval distractors
  (HotpotQA's same-topic distractors don't clear a lexical filter — see
  the first emit attempt — which is itself why natural-distribution
  filtering barely fires).

## Next (measurement, large-n)

1. **Large-n rerun (≥300 queries) with CIs** to resolve the filtering
   sign. This is the prerequisite for any product claim about filtering.
2. **Single-hop workload** (e.g. a non-multi-hop QA set) to test the
   "filtering wins when gold is query-relevant" extrapolation.
3. **Second-hop-preserving filter** (the research frontier): a filter
   that removes off-topic chunks while protecting low-relevance chunks
   that are *entity-linked* to higher-relevance evidence. If it lifts
   filtered_8 toward the gold_only ceiling, the flagship feature is
   validated. Strictly a measurement-gated build, not a speculative one.
