# The Dense-Rerank Ceiling — 0.80 is the second-hop tax, and no fixed knob breaks it

> **Hypothesis:** dense local rerank plateaus at recall@3 ≈ 0.80 because of the **second-hop tax** (the bridge passage is non-relevant to the query, so a bi-encoder ranks it below the top-3); a reasoning-preserving *linkage rescue* — keep dense's reliable top hit, then promote pool candidates linked to it (the `link_strength` Jaccard ReasoningPreserving uses) — should recover the missed second hop without a model upgrade or agentic loop.
> **Status:** Confirmed (the ceiling *is* the second-hop tax) / **Falsified** (no *fixed-knob* rescue beats dense). The recoverable headroom is real but only reachable per-query — the "no cheap escalation trigger" law reappears on the dense substrate.
> **Setup:** global HotpotQA pool (3,957 deduped paragraphs), 400 queries — **100% multi-hop** (≥2 gold supporting facts) — BM25 top-50 → BGE-small dense rerank, recall@3, lexical/semantic split at overlap median 0.857. Rescue arm: slot-1 = dense top-1; rank the rest by `dense_cos + β·link_strength(seed, candidate)`; β=0 ≡ dense.
> **Headline:** dense = 0.801; best *fixed* β = 0.802 (noise), then monotone harm. 148 queries have a gold **in the pool** but missed by dense@3; an **oracle per-query β recovers 26 (18%)** of them with zero hurt — but no single global β extracts that without equal collateral demotion.
> **Reproduce:** `cargo run -p redhop-examples --example semantic_reasoning_rerank --features onnx --release`. Raw: `reports/semantic_reasoning_rerank.txt`.
> **Caveats:** lexical linkage (Jaccard); recall@3 is reachability not answer quality; HotpotQA bridge-style multi-hop. The oracle 18% is an upper bound, not an achievable policy.

---

## Why this experiment

[LOCAL_RERANK](LOCAL_RERANK.md) showed dense local rerank matches *global* dense (recall@3 ≈
0.80) at a fraction of the cost. The natural next question: is 0.80 a real
ceiling, or is there headroom reachable **with what we have** — no bigger
embedder, and (by standing project constraint) **no agentic / iterative
multi-hop retrieval**?

The diagnosis going in: every query here is multi-hop, and recall@3 needs *both*
supporting facts in the top-3. The first hop is query-relevant (dense nails it);
the **second hop is the bridge passage that connects only through the first hop**
— low query relevance by construction — so a relevance-scoring bi-encoder ranks
it low. This is the [SECOND_HOP_TAX](SECOND_HOP_TAX.md) on a dense substrate. If so, the
on-thesis fix is **reasoning-preserving rescue** (the ReasoningPreserving mechanism):
seed on dense's reliable hit, then promote candidates *linked to the seed* rather
than to the query.

## Result

Recall@3 by arm (β = weight on the linkage bonus; β=0 ≡ dense):

| arm | lexical | semantic | ALL |
| --- | ------- | -------- | --- |
| **dense** (baseline) | 0.808 | 0.795 | **0.801** |
| dense + rescue β=0.25 | 0.811 | 0.795 | 0.802 |
| dense + rescue β=0.5 | 0.806 | 0.785 | 0.795 |
| dense + rescue β=1 | 0.793 | 0.783 | 0.787 |
| dense + rescue β=2 | 0.759 | 0.768 | 0.764 |
| dense + rescue β=4 | 0.725 | 0.715 | 0.720 |

At any **fixed** β the rescue does not beat dense: best is +0.001 (noise) at
β=0.25, then it degrades monotonically.

**Second-hop recovery diagnostic** — of the **148** queries with a gold *in the
pool* but missed by dense@3 (confirming the cap is ranking, not pool recall):

- **oracle per-query β recovers 26 (18%)** with **0 hurt**;
- but those 26 are only reachable by choosing β *per query*. At a fixed global β,
  promoting them costs collateral demotions elsewhere — netting ~zero.

## Interpretation

1. **0.80 is the second-hop tax.** 100% of queries are multi-hop; 148 misses have
   the gold sitting in the pool, ranked below 3 — exactly the bridge passage a
   relevance bi-encoder demotes. The ceiling is structural, not noise dense could
   squeeze out. A *stronger embedder* would sharpen the first hop and barely move
   the second — wrong axis.
2. **The linkage signal is real but not cheaply addressable.** When dense's top-1
   is the correct first hop, boosting its links rescues the bridge; when top-1 is
   *wrong*, boosting its links pulls in more wrong-chain chunks. A fixed β cannot
   tell the two apart — so the 18% recoverable headroom is gated behind
   **per-query adaptivity**. This is the same impasse as the falsified escalation
   triggers in [SEMANTIC_MISMATCH](SEMANTIC_MISMATCH.md) (Phase 3), now reproduced on the dense path:
   **the information is present; no fixed, query-agnostic knob extracts it.**
3. **Breaking 0.80 requires crossing a line we've drawn.** The recoverable headroom
   needs either *per-query adaptivity* (a trigger/classifier — ML/architecture
   creep) or *iterative conditioning* (MDR-style: re-encode `[query + hop-1]` and
   retrieve again — the agentic/iterative line ruled out by the standing
   no-architecture constraint). Both would work; both are deliberately off the table.

## Failure cases / honest limits

- Linkage is lexical Jaccard; a semantic linkage signal might shift the oracle
  ceiling, but the *fixed-knob* impasse (good rescue vs bad-chain amplification)
  would remain — it is a routing problem, not a similarity-metric problem.
- recall@3 is reachability; we did not run Tier-3 answers here (the ceiling is a
  retrieval result). The single most promising *single-shot* idea left —
  re-encoding `[query + dense-top-1]` once with the existing BGE (MDR's
  conditioning minus the loop) — is **untested** and is the natural next probe.

## What changed afterward

- Confirms the dense path's plateau is the [SECOND_HOP_TAX](SECOND_HOP_TAX.md), not an
  embedding-quality problem — so "buy a bigger embedder" is not the lever.
- Establishes that **no fixed-knob reasoning rescue improves dense local rerank**,
  reinforcing the conservative-default posture: don't bolt a lexical rescue onto
  the dense reranker expecting a free lift.
- Frames the only remaining levers honestly (per-query adaptivity or iterative
  conditioning) so the trade-off against the no-architecture constraint is
  explicit. The lightweight-tier counterpart: [SEMANTIC_ZERO_DEP](SEMANTIC_ZERO_DEP.md).

## Update — 2026-06-06 — MDR single-pass falsified as a uniform policy, but the per-query signal is real

The finding flagged ONE remaining single-shot probe that stays inside the
no-iteration constraint:

> The single most promising single-shot idea left — re-encoding
> `[query + dense-top-1]` once with the existing BGE (MDR's conditioning
> minus the loop) — is **untested** and is the natural next probe.

We measured it on the SAME setup as the headline (3957 paragraphs, 400
queries, BM25 top-50 → BGE-small dense, recall@3). The augmented text
was the literal "MDR minus the loop": `[query] [dense_top_1.text]`,
re-embedded once with the same BGE call, then cosine over the same
BM25 pool to produce a new top-3. Two variants:

- `mdr_pure` — take the top-3 of the MDR ranking outright.
- `mdr_seed` — anchor-preserving: keep dense's top-1 in slot 1, fill
  slots 2-3 from the MDR ranking (skipping the anchor). Never gives up
  dense's reliable first hit.

### Headline numbers

| arm        | lexical | semantic |     ALL |
| ---------- | ------: | -------: | ------: |
| dense      |   0.804 |    0.803 | **0.804** |
| mdr_pure   |   0.764 |    0.741 | **0.752** |
| mdr_seed   |   0.764 |    0.741 | **0.752** |

The dense baseline reproduces the headline finding's 0.801 (here 0.804;
within sampling noise; same code path, same model, same data).

**Both MDR variants regress dense by Δ ≈ −0.05.** Single-pass re-encode
is not shippable as a uniform policy.

### But the per-query diagnostic tells a richer story

On the **147 queries** with a gold *in the pool but missed by dense@3*
(the recoverable set — same diagnostic the headline used):

- `mdr_pure` recovered **35 (24%)**, hurt **9 (6%)**, net **+26**.
- `mdr_seed` recovered **35 (24%)**, hurt **9 (6%)**, net **+26**.

That **24% recovery rate exceeds the oracle linkage-rescue's 18%** from
the headline finding's table. The MDR re-encode IS finding second-hop
chunks the linkage Jaccard signal can't, and the false-positive rate on
this subset (9/147 = 6%) is genuinely low.

### Why the global recall drops anyway

The +26 net on the recoverable subset is overwhelmed by **collateral
damage on the easy queries**. The 253 queries dense@3 already hit
correctly (400 − 147 = 253) are where the augmented embedding
`[query + dense_top_1]` drifts AWAY from "what the query was asking" —
the seed text dilutes the query intent, so the MDR ranking can drop a
correct chunk out of slot 1-3 in favor of seed-related-but-irrelevant
material.

Crudely: 0.752 vs 0.804 globally = ~21 queries lost on aggregate; +26
recovered on the recoverable subset; ~47 hurt on the easy subset (each
losing one of two gold slots). Numbers match the pattern.

### What this means: the THIRD same-shaped result in a row

This is the same per-query gating pattern that:

- [RERANKING_LIMITS](RERANKING_LIMITS.md): CE helps 12% / hurts 17% — uniform CE is
  net-negative, oracle gate would extract +0.051. (And the
  2026-06-06 update there falsified the kind-label gate in both
  directions.)
- This finding's β-rescue: linkage rescue oracle recovers 18% of
  recoverable-but-missed queries at zero hurt — but no fixed global β
  extracts it.
- **MDR single-pass (here): recovers 24% of recoverable misses, but
  uniform application costs more on the easy subset than it gains on
  the hard one.**

Three different mechanisms (cross-encoder, linkage, MDR re-encode),
three different oracle headrooms (12-24% of recoverable), one universal
failure mode: **the policy gate that decides "fire on this query, skip
on that one" doesn't exist, and the question-type / static-knob
substitutes don't work.** The information is there; the routing isn't.

### What this opens / closes

- **Closes** MDR single-pass as a *uniform* runtime feature. The
  −0.05 global cost is unambiguous; this isn't sample noise.
- **Strongly reinforces** the open problem: the actual bottleneck on
  every escalation lever measured to date is per-query gating, not
  the lever's mechanism.
- **Opens** the possibility of MDR re-encode as the action on the
  "fire on this one" branch of a future gate — but that's contingent
  on the gate existing and predicting the right queries, which is
  the open work. Building MDR-as-policy without that gate is strictly
  worse than not building it.

### Honest limits

- Single 400-item run, no CI. The −0.05 delta is large enough not to
  be noise at this n; the +24% subset recovery is large enough to be
  signal (vs the linkage oracle's 18%, both n≈148). Magnitudes are
  approximate but directions are decisive.
- Augmented text is literal concatenation; we did NOT try structured
  prompting (`[query] [SEP] [hop1]`, asymmetric "query:" / "passage:"
  prefixes, etc.). BGE-small-en-v1.5 wasn't trained on those, so a
  large lift from prompting is unlikely — but worth a brief A/B if
  CE-gating ever becomes a priority again, since the marginal cost is
  the same single re-encode.
- BGE-small only. A bigger encoder (`bge-base`, larger context) might
  use the augmented text more productively — but that's an encoder
  upgrade, not a fix to the gating problem this finding repeatedly
  surfaces.
