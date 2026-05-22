# Second-hop tax

> Full doc: [`docs/findings/SECOND_HOP_TAX.md`](https://github.com/redhop/redhop/blob/main/docs/findings/SECOND_HOP_TAX.md)

**Hypothesis.** Every operation that selects by query relevance drops the
multi-hop second hop, because the second hop is low-relevance-to-query by
construction (it connects through a bridge entity, not the query).

**Experiment.** Hermetic (no LLM, no embeddings), so it runs deterministically at
large n. For each multi-hop HotpotQA query with a query-relevance gap (n=1327),
label the lowest-grounding gold chunk as the second hop, inject off-document
distractors, and measure per strategy with bootstrap 95% CIs: second-hop
retention vs junk suppression, across a filter-threshold sweep.

**Result.** A relevance filter keeps **96.8%** of second hops at threshold 0.05
but only **43.9% at 0.30** — the tax scales with filter aggressiveness.
`reasoning_preserving` beats the plain filter at every threshold, and the gap
widens where the tax is worst (+23 pts of retention at threshold 0.30), with
non-overlapping CIs. The cost: it suppresses slightly less junk (it readmits
junk lexically linked to a seed).

**Caveats.** Lexical grounding/linkage (not embeddings). Retention is
*reachability*, not answer correctness. Controlled off-document distractors, not
natural same-topic ones. The "second hop = lowest-grounding gold" label is a
proxy for the reasoning-critical hop.

**Implications.** This is the law behind RedHop's default. Any
relevance-ranking operation inherits the multi-hop blind spot; the safe move is
removing what is *clearly* irrelevant (a low absolute threshold) and preserving
orthogonal-but-linked evidence. It unifies the
[reranking](./findings-reranking-limits.md) and
[filtering](./findings-filtering-failures.md) failures.
