# Report: cross-model end-to-end replication (reasoning preservation)

Does the reasoning-preservation finding hold across model families? Same n=300
gap-qualified multi-hop HotpotQA contexts (built by `emit_reasoning_qa` under the
**validated signal** — stopword removal + Snowball stemming; aggressive filter
threshold 0.20), scored by different generators via the `claude` CLI / OpenRouter.

Full interpretation: [docs/findings/REASONING_PRESERVATION.md](../../docs/findings/REASONING_PRESERVATION.md)
and the companion paper (`../neorag/paper2`).

- **Date:** 2026-05-23
- **n:** 300 × 4 conditions per model
- **Scored by:** `python ../neorag/scripts/score_reasoning_qa.py --n 300 --model <id>`

## Per-model summary

| model | gold_only | polluted | filtered | reasoning | distractors hurt (gold−poll) | filter harm (filt−poll) | rescued subset (reason−filt) | control |
| ----- | --------- | -------- | -------- | --------- | ---------------------------- | ----------------------- | ---------------------------- | ------- |
| openai/gpt-4o-mini | 0.730 | 0.714 | 0.606 | 0.621 | +0.016 (ns) | **−0.108** | **+0.218** [.077,.372] (n=26) | −0.003 (ns) |
| meta-llama/llama-3.3-70b | 0.699 | 0.648 | 0.586 | 0.603 | **+0.051** [.018,.084] (sig) | **−0.062** | +0.154 [.000,.346] (n=26) | +0.003 (ns) |
| claude-haiku *(prior signal)* | 0.830 | 0.829 | 0.705 | 0.740 | +0.001 (ns) | −0.124 | +0.173 [.040,.320] | +0.022 (ns) |

(haiku is on the *prior* signal — not apples-to-apples until re-run on these contexts.)

## Reading (3 models)

**Robust across all three families:** aggressive filtering is **net-harmful** on
multi-hop (filter−polluted = −0.062 to −0.124) — even on Llama-3.3-70B, where
distractors *do* bite. The "cure worse than the disease" holds regardless of model.

**Generator-dependent (the honest nuance):** whether distractors *alone* hurt
depends on model strength. Frontier models (haiku, gpt-4o-mini) are nearly inert
to off-document distractors (gold ≈ polluted, ns); **Llama-3.3-70B is measurably
hurt (+0.051, CI excludes 0).** So the asymmetry is not "distractors are
harmless" — it is "**missing reasoning evidence hurts *more* than irrelevant
context**": on Llama, distractors cost 0.051, but filtering them away cost 0.062
*more* on top — the reasoning-evidence loss (~0.11) still dominates.

**Causal mechanism replicates in direction on all three:** the reasoning−filter
gain concentrates in the rescued subset (+0.15 to +0.22) and is ~0 on the
identical-retention control. Significance is strong on haiku/gpt-4o-mini and
borderline on Llama (CI lower bound 0.000). The *aggregate* reasoning−filter
delta is dilution-limited everywhere (rescue fires on ~9% of queries).

## Files
- `gpt-4o-mini_n300.txt`, `llama-3.3-70b_n300.txt` — raw scorer output.

## Pending
- haiku re-run on these (new-signal) contexts, for a fully apples-to-apples 3-model §5.2.
