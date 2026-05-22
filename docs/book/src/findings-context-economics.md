# Context economics

> Full doc: [`docs/findings/CONTEXT_ECONOMICS.md`](https://github.com/redhop/redhop/blob/main/docs/findings/CONTEXT_ECONOMICS.md)

**Hypothesis.** The premise behind context optimization: distractors hurt and
answer-bearing evidence density helps — and this holds on *real* LLM outputs,
not just retrieval metrics.

**Experiment.** Two parts. (A) Pearson correlations of retrieval
`distractor_ratio` and `answer_span_density` against measured answer quality,
across four generator models and two datasets (HotpotQA, MuSiQue). (B)
Token-efficiency curves: `build_context` over real BGE dense retrieval, sweeping
the token budget across strategies.

**Result.**

| | pooled correlation with answer quality |
| --- | --- |
| distractor ratio | **−0.375** (negative everywhere, −0.18 to −0.47) |
| evidence density | **+0.539** (positive everywhere, +0.37 to +0.70) |

Model-independent: distractors hurt and density helps across every
generator/dataset. The strategy sweep reproduced the second-hop tax a third way
— **max-density pruning drops the second hop** (gold 0.623 vs 0.819 at a tight
budget) because it ranks by query-relevance density.

**Caveats.** Density/distractor metrics are lexical (query-term overlap).
HotpotQA is adversarially multi-hop, so the max-density gold-drop is worst-case.
Gold retention is a reachability proxy, not end-to-end answer quality (that's the
[filtering](./findings-filtering-failures.md) finding's domain).

**Implications.** The empirical foundation for optimizing context by evidence
density and distractor suppression — *but* relative-ranking pruning
(max-density) is recall-risky on multi-hop, while absolute-threshold distractor
removal at a low bar is the safe, free win. This is the menu RedHop's strategies
implement, and the `ContextReport` surfaces these economics for every assembly.
