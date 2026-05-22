# Adaptive Controller on a Real Semantic Substrate

**Central question.** Does stronger semantic retrieval (BGE-small) sharpen
the adaptive controller — improve intervention precision and reduce
unnecessary escalation — versus the hashing baseline?

**Experiment.** Run the adaptive threshold sweep twice on the same
HotpotQA sample (60 items), **holding the candidate retriever constant
(BM25)** and varying only the embedding backend that feeds the
semantic-tier diagnostics. Because BM25 ignores embeddings, static recall
is identical across arms; any difference in controller behavior is
attributable to diagnostic sharpness alone.

```bash
cargo run -p redhop-examples --example adaptive_real_vs_hashing \
    --features onnx --release
```

## Results (60 HotpotQA items, BM25 retrieval, top-k=4)

| metric | hashing | BGE-small | Δ |
| ------ | ------- | --------- | - |
| intervention rate | 42% | 35% | **−7 pts** |
| useful % (precision) | 52% | 52% | 0 |
| mean recall lift | 0.096 | 0.079 | −0.017 |
| mean rerank calls/query | 0.008 | 0.000 | −0.008 |
| rerank compute avoided | 99% | 100% | +1 pt |
| mean useful lift | 0.432 | 0.425 | ≈0 |
| **mean harmful lift** | **0.000** | **0.000** | **0** |
| wasted interventions | 42 | 36 | −6 |
| classifier accuracy | 21% | 12% | **−9 pts** |
| ECE (calibration) | 0.176 | 0.283 | **+0.107** |

## The honest two-part reading

This is **not** a clean "BGE makes everything better." It's more
instructive than that.

### Part 1 — the controller shifts to a less-aggressive operating point (good, mostly)

Under BGE, semantic grounding reads *higher* (real embeddings produce
more confident query↔chunk cosines than the lexical hashing trick). So
more queries clear the "Easy" bar, the controller intervenes **less**
(42% → 35%), reranks essentially **never** (0.008 → 0.000), and wastes
**fewer** interventions (42 → 36). Precision (useful%) and safety
(harmful lift = 0) are **preserved**. On the "do less, stay safe" axis,
BGE is a clear win: ~7 points less escalation at identical precision and
zero downside.

### Part 2 — but classifier calibration *degrades*, and that's the real lesson

`classifier accuracy` drops 21% → 12% and `ECE` worsens 0.176 → 0.283.
The recall lift also shrinks (0.096 → 0.079) — the controller skips not
just wasted interventions but some *useful* ones too.

Why: **the regime-classifier thresholds were calibrated for the hashing
substrate.** BGE's grounding-score distribution is different (shifted
higher, differently shaped), so the same thresholds land at a different,
*miscalibrated* operating point — over-classifying "Easy," under-firing
on queries that would have benefited. The ECE jump is the direct
measurement of that drift.

**The deepest operational insight here is the coupling:**

> Embedding substrate and controller calibration are *not*
> independent. You cannot swap the embedder and keep the thresholds.
> Each substrate needs its own threshold calibration.

This is exactly consistent with the calibration-discipline thesis from
`REAL_WORKLOAD.md` (thresholds are workload-specific) — now
extended: they are *substrate*-specific too.

## What survived, what's confirmed, what's actionable

**Survived:** the entire adaptive architecture runs end-to-end on a real
semantic substrate — no crashes, sane metrics, the conservative policy's
zero-harm property holds on both substrates. The system is no longer
hypothetical.

**Confirmed:** the controller is robust to classifier miscalibration on
the *outcome* axis. Despite accuracy dropping to 12% against the
heuristic regime labels, intervention precision (useful%) and safety
(harm = 0) held — because misclassified regimes still route to
similar/safe actions. The policy degrades gracefully, it doesn't break.

**Actionable:** before deploying BGE (or any real embedder), **recalibrate
the regime thresholds for that substrate.** The threshold sweep
(`ThresholdSweep`) is exactly the tool — re-run it with BGE diagnostics
and pick the operating point off the BGE reliability diagram, not the
hashing one. The un-recalibrated BGE run above is the "what not to do":
it leaves recall lift on the table (0.079 vs a likely-higher recalibrated
value) because it inherited the wrong thresholds.

## Does better retrieval reduce the need for reranking?

**On this evidence: yes, but for a subtle reason.** BGE pushes the
controller to rerank less (0.008 → 0.000 calls/query) — but that's
because higher semantic-grounding reads classify more queries as "Easy"
under the *inherited* thresholds, not necessarily because reranking
became less valuable. The clean version of this experiment requires the
recalibration step: with BGE-tuned thresholds, measure whether the
controller still finds fewer queries that *need* escalation. The
un-recalibrated result is suggestive (less reranking, same precision, no
harm) but the calibration drift means it's not yet the definitive answer.

## Honest limits

- **Un-recalibrated BGE thresholds.** The headline weakness and the
  headline finding. The next run must recalibrate.
- **Heuristic regime labels.** "Classifier accuracy" is measured against
  the HotpotQA `(level, type)` → regime heuristic, itself a proxy. The
  accuracy *drop* is real signal about distribution shift, but the
  absolute numbers shouldn't be over-read.
- **60-item sample, BM25 retrieval, lexical reranker.** The reranker is
  the cheap lexical one (the ONNX cross-encoder isn't wired into this
  run); rerank-savings figures are about *whether* the controller fires
  reranking, not cross-encoder latency.
- **CPU, single run.** No statistical CI on the deltas; a full-dev-set
  run with bootstrap would tighten them.

## Next (clean version of this experiment)

1. **Recalibrate.** Run `ThresholdSweep` under BGE diagnostics, read the
   BGE reliability diagram, pick BGE-specific thresholds. Re-run the
   comparison with each substrate at *its own* calibrated operating
   point. That isolates "does better sensing reduce needed escalation"
   from "thresholds drifted."
2. **Wire the ONNX cross-encoder** into `EscalateReranker` and repeat, so
   rerank-savings become real cross-encoder-latency savings.
3. **Dense retrieval arm.** Optionally let BGE drive *retrieval* (not just
   diagnostics) to measure the combined effect — separate experiment,
   separate confound.
