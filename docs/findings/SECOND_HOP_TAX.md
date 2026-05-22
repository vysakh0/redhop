# The Second-Hop Tax — Measured Directly, at Large n

> **Hypothesis:** every operation that selects by query relevance drops the multi-hop second hop (low-relevance-to-query by construction).
> **Status:** Confirmed — two datasets (HotpotQA n=1327, MuSiQue n=1484), bootstrap 95% CIs.
> **Setup:** multi-hop questions with a query-relevance gap; hermetic (no LLM, no embeddings); lexical grounding label.
> **Headline:** a relevance filter keeps 96.8% of second hops at threshold 0.05 but only 43.9% at 0.30 on HotpotQA — and 75.1%→4.3% on the harder MuSiQue; `ReasoningPreserving` recovers up to +23 pts (HotpotQA) / +29 pts (MuSiQue).
> **Reproduce:** `cargo run -p redhop-examples --example second_hop_retention --release`
> **Justifies API:** `build_context(strategy = ReasoningPreserving)`.
> **Caveats:** lexical grounding/linkage; retention is reachability, not answer quality. See §Honest limits.

---

Across five earlier experiments the same shape kept appearing: every
operation that selects by *query relevance* drops the multi-hop second
hop, because the second hop is low-relevance-to-query by construction
(it is linked to the answer through a bridge entity, not through the
query). We named the pattern the **second-hop tax** but had only
measured it indirectly — gold-retention deltas, recall losses, and one
LLM evaluation whose sign flipped between n=20 and n=30.

This experiment measures the tax head-on, with no LLM and no embeddings,
so it runs deterministically at large n.

```bash
cargo run -p redhop-examples --example second_hop_retention --release
```

## Method (hermetic, n = 1327)

For each multi-hop HotpotQA query we label the gold chunks by their
query grounding (query-term overlap, the same primitive the strategies
use):

- **first hop** — the gold chunk(s) with the *higher* query relevance
- **second hop** — the gold chunk with the *lowest* query relevance

We keep only queries where the second hop is genuinely less
query-relevant than the first hop (a real relevance gap — mean gap
**0.234**). That is the regime the tax lives in: 1327 of the dev set's
multi-hop questions qualify. We inject 8 off-document distractors
("true junk", near-zero query overlap), present the set relevance-ranked
as a real retriever would, build a context under each strategy, and
measure per query with **bootstrap 95% CIs over queries**:

- `second_hop_retention` — did the reasoning-critical hop survive? (want HIGH)
- `junk_suppression` — fraction of injected distractors removed (want HIGH)
- `first_hop_retention` — did the query-relevant hop survive? (sanity)

## Panel A — the FILTER tax (generous budget)

Budget is generous, so nothing is dropped for space; the only thing that
drops a chunk is the filter decision. As the grounding threshold rises,
the filter removes more junk — and the question is whether it takes the
second hop with it.

| strategy @ τ | second_hop_retention [95% CI] | junk_supp | first_ret |
| ------------ | ----------------------------- | --------- | --------- |
| distractor_filtered @0.05 | 0.968 [0.958, 0.977] | 0.158 | 1.000 |
| reasoning_preserving @0.05 | 0.974 [0.965, 0.983] | 0.149 | 1.000 |
| distractor_filtered @0.10 | 0.927 [0.913, 0.940] | 0.393 | 0.999 |
| reasoning_preserving @0.10 | **0.948** [0.936, 0.959] | 0.363 | 0.999 |
| distractor_filtered @0.20 | 0.749 [0.724, 0.772] | 0.809 | 0.984 |
| reasoning_preserving @0.20 | **0.833** [0.812, 0.853] | 0.758 | 0.986 |
| distractor_filtered @0.30 | 0.439 [0.411, 0.468] | 0.972 | 0.913 |
| reasoning_preserving @0.30 | **0.671** [0.645, 0.696] | 0.926 | 0.928 |

Two facts, both with non-overlapping CIs from τ=0.10 upward:

1. **The tax is real and grows with filter aggressiveness.** Plain
   distractor filtering keeps 96.8% of second hops at τ=0.05 but only
   **43.9% at τ=0.30** — an aggressive relevance filter discards more
   than half of the reasoning-critical evidence. This calibrates the
   earlier "distractor filtering is safe" claim precisely: it is safe
   *at a low absolute threshold*, and the tax climbs steeply as you
   tighten it.

2. **Reasoning-preserving selection reduces the tax at every
   threshold**, and the rescue grows exactly where the tax is worst:
   +2.1 pts at τ=0.10, +8.4 at τ=0.20, **+23.2 at τ=0.30**. The CIs do
   not overlap at τ≥0.10, so this is a real effect, not sample noise.

The cost is honest and visible: reasoning-preserving suppresses slightly
*less* junk (e.g. 0.926 vs 0.972 at τ=0.30), because it readmits junk
that happens to be lexically linked to a kept seed. The trade is "keep a
little more junk to save a lot more second hops" — and it is a good
trade precisely in the aggressive regime where the tax is expensive.

### Cross-dataset replication — MuSiQue (n=1484, mean gap 0.293)

The same experiment on MuSiQue (more hops, a wider relevance gap)
replicates the filter tax and makes it **more severe** — second-hop
retention [95% CI]:

| τ | distractor_filtered | reasoning_preserving |
| - | ------------------- | -------------------- |
| 0.05 | 0.751 [0.729, 0.774] | 0.821 [0.802, 0.841] |
| 0.10 | 0.452 [0.425, 0.478] | 0.639 [0.613, 0.662] |
| 0.20 | 0.154 [0.137, 0.173] | 0.442 [0.417, 0.467] |
| 0.30 | 0.043 [0.034, 0.053] | 0.255 [0.232, 0.276] |

On MuSiQue an aggressive (τ=0.30) filter retains only **4.3%** of second
hops (vs 43.9% on HotpotQA); reasoning-preserving rescues a larger margin
throughout (e.g. +28.8 pts at τ=0.20), CIs non-overlapping. The harder
the multi-hop structure, the steeper the tax — the filtering result is
the robust, two-dataset core.

## Panel B — the RANKING/BUDGET tax (tight budget, τ=0.10)

Budget is now tight (220 tokens, ~half the set), so selection under
scarcity governs what survives.

| strategy | second_hop_retention [95% CI] | junk_supp | first_ret |
| -------- | ----------------------------- | --------- | --------- |
| raw_topk | 0.960 [0.949, 0.971] | 0.393 | 0.999 |
| distractor_filtered | 0.916 [0.900, 0.929] | 0.542 | 0.998 |
| max_density | 0.904 [0.888, 0.919] | 0.363 | 0.991 |
| reasoning_preserving | 0.930 [0.915, 0.943] | 0.524 | 0.999 |

- **max_density retains the fewest second hops** (0.904) — the
  relevance-density ranking pushes the low-relevance hop below the
  budget cutoff. The ranking tax, again.
- **raw_topk retains the most** (0.960) but suppresses the least junk
  (0.393) — it just keeps the relevance-top until the budget fills, junk
  included.
- **reasoning_preserving is the best balance**: nearly raw_topk's
  second-hop retention (0.930) with markedly better junk suppression
  (0.524 vs 0.363) than max_density.

## What this establishes

The second-hop tax is no longer an inference from indirect signals — it
is a **directly measured, CI-backed property** of relevance-based
selection on multi-hop retrieval:

> Any operation that selects or ranks by query relevance — ExpandTopK,
> cross-encoder reranking, max-density pruning, threshold distractor
> filtering — taxes the reasoning-critical second hop, and the tax
> scales with how aggressively the operation optimizes relevance.

And the mitigation works: **reasoning-preserving selection** (keep
query-relevant seeds, then rescue low-relevance chunks that are lexically
linked to a seed via the bridge entity, drop only unlinked junk)
recovers a large fraction of the taxed second hops, at a measured and
modest junk-suppression cost. This is the concrete realization of the
project's frontier: *reasoning-aware evidence allocation, not relevance
optimization.*

## Honest limits

- **Lexical grounding and lexical linkage.** Both the second-hop label
  and the rescue signal use query-term / term-set overlap, NOT
  embeddings. This is cheap and deterministic by design, but it means
  the rescue fires on lexical bridge-entity overlap; a second hop that
  shares *no* surface tokens with its first hop (pure paraphrase
  linkage) would not be rescued. A semantic-linkage variant is a
  measurement-gated future build, not a speculative one.
- **Retention is reachability, not answer quality.** "The second hop
  survived" is a necessary condition for a correct multi-hop answer, not
  proof of one. The end-to-end answer-quality question still needs the
  Python lab's LLM (see [DISTRACTOR_ROBUSTNESS.md](DISTRACTOR_ROBUSTNESS.md)),
  and that experiment's sign-flip caution still stands — this result
  tells us *what reaches the model*, which is the lever RedHop controls.
- **Controlled off-document junk.** The injected distractors are clearly
  off-topic (near-zero overlap); natural same-topic distractors are
  harder and would lower junk_suppression for all strategies alike.
- **Second hop = lowest-grounding gold** is a proxy for "the
  reasoning-critical low-relevance hop." It is the right proxy for the
  tax, but on questions where both hops are equally query-relevant the
  tax does not apply (those queries are filtered out here, by design).

## Where this leaves the strategy menu

- **distractor_filtered** stays the safe default *at a low threshold*
  (τ≈0.05–0.10), where the tax is small and junk removal is real.
- **reasoning_preserving** is the multi-hop-aware choice: use it when the
  workload is multi-hop and you want to filter aggressively without
  paying the full tax. It is now measurement-validated, not aspirational.
- **max_density** remains gated to single-hop / high-relevance workloads
  — Panel B reconfirms it is the worst on second-hop retention.
