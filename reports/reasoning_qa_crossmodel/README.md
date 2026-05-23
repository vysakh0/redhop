# Report: cross-model end-to-end replication (reasoning preservation)

Does the reasoning-preservation finding hold across model families? Same n=300
gap-qualified multi-hop HotpotQA contexts (built by `emit_reasoning_qa` under the
**validated signal** — stopword removal + Snowball stemming; aggressive filter
threshold 0.20), scored by different generators via the `claude` CLI / OpenRouter.

Full interpretation: [docs/findings/REASONING_PRESERVATION.md](../../docs/findings/REASONING_PRESERVATION.md)
and the companion paper (`../neorag/paper2`).

- **Date:** 2026-05-23
- **n:** 300 × 4 conditions per model
- **Scored by:** `python python/eval/score_reasoning_qa.py --n 300 --model <id>`

## Per-model summary (all on the new signal, apples-to-apples)

| model | tier | gold_only | polluted | filtered | reasoning | distractors hurt (gold−poll) | filter harm (filt−poll) | rescued subset (reason−filt) | control |
| ----- | ---- | --------- | -------- | -------- | --------- | ---------------------------- | ----------------------- | ---------------------------- | ------- |
| claude-haiku | frontier | 0.817 | 0.822 | 0.676 | 0.681 | −0.005 (ns) | **−0.146** | **+0.231** [.038,.423] (n=26) | −0.017 (ns) |
| openai/gpt-4o-mini | frontier | 0.730 | 0.714 | 0.606 | 0.621 | +0.016 (ns) | **−0.108** | **+0.218** [.077,.372] (n=26) | −0.003 (ns) |
| qwen/qwen3.5-flash | non-frontier | 0.743 | 0.698 | 0.589 | 0.604 | **+0.045** [.011,.082] (sig) | **−0.109** | **+0.205** [.064,.359] (n=26) | −0.003 (ns) |
| meta-llama/llama-3.3-70b | non-frontier | 0.699 | 0.648 | 0.586 | 0.603 | **+0.051** [.018,.084] (sig) | **−0.062** | +0.154 [.000,.346] (n=26) | +0.003 (ns) |

## Reading (4 models)

**Robust across all four families:** aggressive filtering is **net-harmful** on
multi-hop (filter−polluted = −0.062 to −0.146) — even on the non-frontier models
where distractors *do* bite. The "cure worse than the disease" holds regardless
of model.

**The split is by model *tier*, not age (the honest nuance):** whether distractors
*alone* hurt depends on robustness. **Frontier models (haiku, gpt-4o-mini) are
inert** to off-document distractors (gold ≈ polluted, ns); **both non-frontier
models are measurably hurt** — and this includes the *recent* qwen3.5-flash
(+0.045, sig), not just the older Llama-3.3-70B (+0.051, sig). So distractor-
sensitivity is a property of model strength, not an artifact of one old model. The
asymmetry is not "distractors are harmless" — it is "**missing reasoning evidence
hurts *more* than irrelevant context**": on the sensitive models distractors cost
~0.05, but filtering them away costs ~0.06–0.11 *more* on top.

**Causal mechanism replicates in direction on all four:** the reasoning−filter
gain concentrates in the rescued subset (+0.15 to +0.23) and is ~0 on the
identical-retention control. Significance is strong on haiku/gpt-4o-mini/qwen and
borderline on Llama (CI lower bound 0.000). The *aggregate* reasoning−filter
delta is dilution-limited everywhere (rescue fires on ~9% of queries).

## Scope note
This table is the **small-context** regime (gold + 8 distractors, generous budget).
At **large** contexts (~30k tokens) a different effect dominates — see
[docs/findings/CONTEXT_DILUTION.md](../../docs/findings/CONTEXT_DILUTION.md): there
pruning *recovers* accuracy and the bridge-aware advantage washes out.

## Files
- `haiku_newsignal_n300.txt`, `gpt-4o-mini_n300.txt`, `llama-3.3-70b_n300.txt` — raw scorer output.
- qwen3.5-flash raw output: [`../qwen3.5-flash_n300.txt`](../qwen3.5-flash_n300.txt).
