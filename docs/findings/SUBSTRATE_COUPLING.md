# BGE Recalibration — Genuine Reduction vs Threshold Drift

**Question.** The prior experiment found BGE reduced interventions but
degraded classifier calibration under hashing-tuned thresholds. Was the
reduced escalation a *genuine* substrate effect, or did *threshold drift*
merely suppress it? Recalibrate for BGE and find out.

```bash
cargo run -p redhop-examples --example bge_recalibration --features onnx --release
```

## Result — a precise structural finding, not a clean win/loss

### 1. The drift, quantified

BGE's `semantic_grounding` over the top-4 BM25 results:

| p10 | p25 | p50 | p75 | p90 |
| --- | --- | --- | --- | --- |
| 0.796 | 0.802 | 0.823 | 0.843 | 0.865 |

**100% of queries score ≥ 0.75** (the hashing-tuned `easy` threshold).
BGE's grounding distribution is tightly packed in 0.80–0.87 — far above
where the hashing baseline sat. The threshold is *useless* for BGE: it
never discriminates. This is the drift, measured exactly.

### 2. Recalibrating the easy threshold is a no-op (the surprise)

Sweeping `easy_min_semantic_grounding` from 0.75 to 0.99:

| easy_sem | interv | useful | lift | rerank/q | ECE |
| -------- | ------ | ------ | ---- | -------- | --- |
| 0.75 | 28% | 53% | 0.062 | 0.000 | 0.277 |
| 0.85 | 28% | 53% | 0.062 | 0.000 | 0.269 |
| 0.99 | 28% | 53% | 0.062 | 0.000 | 0.270 |

**Flat.** Raising the threshold (which suppresses "Easy"
classifications) does not change controller behavior at all. That is
structurally informative:

> Intervention is gated by the **distractor / ambiguous detection
> rules** and the **policy thresholds on `p_distractor` / `p_ambiguous`**
> — *not* by the easy classification. Suppressing "Easy" reshuffles
> softmax mass, but if the distractor/ambiguous *rules* don't fire
> (their diagnostics — distractor_ratio, centroid_dispersion,
> score_entropy — don't cross their thresholds), `p_distractor` and
> `p_ambiguous` stay below the policy thresholds and no intervention
> happens.

The classifier's ECE "drift" (0.193 → 0.277) is real, but it is **largely
decoupled from intervention behavior**, because the actions depend on
diagnostics other than `semantic_grounding`.

### 3. At matched operating points, the substrate barely moves economics

| metric | hashing@0.75 | BGE@recal |
| ------ | ------------ | --------- |
| intervention rate | 30% | 28% |
| useful % | 50% | 53% |
| **mean recall lift** | **0.062** | **0.062** |
| rerank calls/query | 0.017 | 0.000 |
| mean harmful lift | 0.000 | 0.000 |
| wasted interventions | 9 | 8 |
| ECE | 0.193 | 0.277 |

**Recall lift is identical (0.062).** BGE intervenes marginally less
(28% vs 30%) with marginally higher precision (53% vs 50%) and zero
reranking — a small genuine efficiency edge that *survives*
recalibration (so it isn't pure drift), but it's marginal.

## The honest answer to the central question

> Does stronger semantic retrieval genuinely reduce reranking need, or
> did threshold drift merely suppress escalation?

**Neither, cleanly — and the reason is the deepest result here:**

In this configuration the embedding substrate is in the **sensing path
only** (it feeds the semantic-tier diagnostics), not the **action path**.
The retriever is BM25 (ignores embeddings) and the escalation reranker is
lexical (ignores embeddings). So:

- **Recall lift is substrate-independent (0.062 both)** — it comes
  entirely from BM25 + the embedding-blind actions (ExpandTopK re-runs
  BM25; lexical rerank reorders by term overlap).
- The substrate changes only *which* queries get classified into which
  regime — and that effect is too weak, on these diagnostics, to move
  the economics.
- "Reduced reranking need" was never really exercised: the controller
  reranks ~never on both substrates anyway (it prefers cheap ExpandTopK).

**The systems truth: the embedding substrate must be in the *action
path* — driving retrieval (dense) or reranking (semantic / cross-
encoder) — to affect retrieval economics. As a pure *sensing* upgrade,
a better embedder sharpens diagnostics but doesn't change what the
controller can *do*, because the actions don't use embeddings.**

This *refines* the prior "calibration coupling" finding rather than
overturning it:
- Substrate ↔ classifier-*labels* are coupled (ECE drifts). ✓ (still true)
- Substrate ↔ controller-*actions* are **weakly** coupled here, because
  the actions are embedding-blind. The label drift does not propagate to
  behavior.

## What this validates (again)

The conservative controller's **zero-harm property holds on both
substrates and at every recalibration point** (harmful lift = 0.000
everywhere). Substrate transition and threshold recalibration never
destabilized it. That is the robustness the architecture was designed
for.

## Honest limits

- **60-item sample, single run, no CI.** The 0.062-vs-0.062 identity is
  exact at this sample; treat the marginal deltas (28% vs 30%) as
  directional, not significant.
- **Easy-threshold recalibration only.** I recalibrated the threshold
  that *drifted*; a full recalibration would also re-tune the
  distractor/ambiguous thresholds. But the no-op result shows the easy
  threshold isn't the lever, so that's the right thing to have isolated.
- **Sensing-only substrate.** The whole point of the finding: to test
  reranking-need reduction, the substrate must be in the action path.

## The clean next experiment (now well-defined)

To actually answer "does stronger retrieval reduce reranking need," put
the substrate in the **action path**:

1. **Dense retrieval arm** — let BGE drive the retriever (not BM25), so
   recall genuinely differs by substrate (the bakeoff's +99% recall is
   the headroom). Then measure whether the controller needs to escalate
   less because first-stage retrieval is already strong.
2. **Semantic / cross-encoder escalation** — make `EscalateReranker`
   use a real semantic reranker, so "reranking need" is a meaningful,
   substrate-sensitive quantity.

Both put embedding quality where it can change the controller's
*options*, not just its *perceptions*. That's where the reranking-need
question can actually be settled.
