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
| openai/gpt-4o-mini | 0.730 | 0.714 | 0.606 | 0.621 | +0.016 (ns) | **−0.108** | **+0.218** [.077,.372] (n=26) | −0.003 (ns, n=274) |

(haiku, on the *prior* signal, is in `../reasoning_preserving_n300/`; not
apples-to-apples until re-run on these contexts.)

## Reading

The two core claims replicate on GPT-4o-mini (a different family from haiku):
- **distractors are inert** on a strong model (gold ≈ polluted, CI spans 0);
- **aggressive filtering is net-harmful** (−0.108, refusals 13%→27%);
- the **causal mechanism is clean and cross-family**: where reasoning-preservation
  rescued gold the filter dropped, answers improved **+0.218** (CI excludes 0);
  where retention was identical, no difference.

Honest caveat: the *aggregate* reasoning−filter delta is small and not
significant for GPT-4o-mini (+0.016), because rescue fires on ~9% of queries
(26/300) and washes out over the rest. The mechanism (rescued subset) is the
robust, travelling result; the aggregate is dilution-limited.

## Files
- `gpt-4o-mini_n300.txt` — raw scorer output.

## Pending
- A common enterprise open model (Llama-3.x-70B) on the same contexts.
- haiku re-run on these (new-signal) contexts, for an apples-to-apples §5.2.
