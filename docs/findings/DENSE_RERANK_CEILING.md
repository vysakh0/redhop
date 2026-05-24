# The Dense-Rerank Ceiling — 0.80 is the second-hop tax, and no fixed knob breaks it

> **Hypothesis:** dense local rerank plateaus at recall@3 ≈ 0.80 because of the **second-hop tax** (the bridge passage is non-relevant to the query, so a bi-encoder ranks it below the top-3); a reasoning-aware *linkage rescue* — keep dense's reliable top hit, then promote pool candidates linked to it (the `link_strength` Jaccard ReasoningPreserving uses) — should recover the missed second hop without a model upgrade or agentic loop.
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
on-thesis fix is **reasoning-aware rescue** (the ReasoningPreserving mechanism):
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
