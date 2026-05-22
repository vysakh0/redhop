# FAQ

A short version; the full list is in
[`FAQ.md`](https://github.com/redhop/redhop/blob/main/FAQ.md).

**How is it different from LangChain / LlamaIndex?** Those are frameworks that
orchestrate retrieval, agents, and generation. RedHop is not a framework — it
does one thing (context assembly + observability) and composes *under* them.

**How is it different from a reranker?** A reranker scores by query relevance.
RedHop's central finding is that the multi-hop second hop is low-relevance to
the query *by construction*, so relevance reranking demotes it. RedHop preserves
it. See [Reranking Limits](./findings-reranking-limits.md).

**Does it call an LLM or embedding model?** No. The default lexical strategies
need neither. Embeddings are optional (only `redundancy_pruned` uses them). You
own the LLM and embedder.

**Which strategy should I use?** `reasoning_preserving` (default) for
general/multi-hop; `distractor_filtered` at a *low* threshold for single-hop;
`max_density` only for brutal budgets on single-hop; `raw_topk` is the baseline.

**Is it production-ready?** Alpha. APIs are stable and findings reproducible, but
it isn't published yet and the bindings aren't battle-tested at scale. Pin
versions.

**Does it claim to "solve reasoning" or beat SOTA?** No. It makes one measured
tradeoff visible — relevance optimization can remove reasoning-critical evidence
— and preserves it while still pruning junk. Gains are modest, bounded, and
reported with their limits.
