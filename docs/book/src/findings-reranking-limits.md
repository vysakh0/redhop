# Reranking limits

> Full doc: [`docs/findings/RERANKING_LIMITS.md`](https://github.com/redhop/redhop/blob/main/docs/findings/RERANKING_LIMITS.md)

**Hypothesis.** Dense retrieval misses the orthogonal second hop; a
cross-encoder re-scoring a *wider* net should recover it — its geometry should
match the failure.

**Status: FALSIFIED** — and the falsification is one of the project's deepest
results.

**Experiment.** 60 HotpotQA items, dense BGE retrieval, wide net (top-20) →
top-4. Compare static dense (no CE), uniform CE rerank, selective CE, and an
oracle that fires CE only when it helps.

**Result.**

| strategy | recall@4 | CE calls |
| -------- | -------- | -------- |
| static dense (no CE) | **0.732** | 0 |
| uniform CE | 0.704 | 60 |
| oracle (CE only when it helps) | **0.783** | **7** |

Uniform cross-encoder reranking made recall **worse (−0.029)**. CE helped 12% of
queries and **hurt 17%**. A cross-encoder scores *query↔passage relevance*, so on
multi-hop it confidently **demotes** the low-query-relevance second hop below the
cutoff — exactly the wrong move.

**Caveats.** 60-item sample, single run, no CI (the 12%-help / 17%-hurt split is
the robust qualitative finding). ms-marco MiniLM cross-encoder. HotpotQA is
adversarially multi-hop; on single-hop/comparison workloads CE would likely
help.

**Implications.** No query-passage reranker — lexical, semantic, or
cross-encoder — can recover the second hop, because it's low-relevance *by
definition*. But the oracle shows selective escalation has real headroom
(+0.051 at 8.5× lower cost): the value is in firing CE *only* where it helps, not
uniformly. This reinforced the [second-hop tax](./findings-second-hop-tax.md).
