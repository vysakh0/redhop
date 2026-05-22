# Philosophy

RedHop optimizes **reasoning-completeness under finite attention**, not generic
retrieval scores. Four ideas define it.

## 1. Relevance ≠ reasoning usefulness

A retriever (or reranker) ranks chunks by *query relevance*. But on multi-hop
questions, the evidence that completes the reasoning chain is often **not
relevant to the query** — it's relevant to a *bridge entity* the query never
mentions.

> *Who was the nationality of the inventor of the miners' safety lamp?*
> Hop 1: "the safety lamp was invented by Humphry Davy" (query-relevant).
> Hop 2: "Humphry Davy was British" (relevant to *Davy*, not to the query).

Hop 2 is what you actually need, and it scores low on query relevance. This is
the **second-hop tax**.

## 2. Why aggressive filtering fails

Because the second hop is low-relevance by construction, any operation that
prunes by relevance taxes it. We measured this directly: a relevance filter
keeps 96.8% of second hops at threshold 0.05 but only **43.9% at 0.30**. The
"cure" (filtering distractors) can be worse than the disease — see
[Filtering Failures](./findings-filtering-failures.md) and
[Reranking Limits](./findings-reranking-limits.md).

The mechanism is general: transformers are robust to a few irrelevant chunks,
but fragile to a *missing* reasoning link. So RedHop's `reasoning_preserving`
strategy keeps query-relevant seeds **and** rescues low-relevance chunks that
are linked to a seed (sharing the bridge entity), dropping only unlinked junk.

## 3. Observability matters

Most RAG stacks "stuff the top-k and hope" — there is no record of what the
context actually contained or what optimization did to it. RedHop emits a
[`ContextReport`](./observability.md) for every assembly: tokens removed,
distractors pruned, reasoning rescues, evidence density, estimated waste. The
invisible becomes visible.

## 4. Evidence-first, bounded by design

Every default exists because a specific failure was **measured**, and the record
is kept honestly — including [falsified hypotheses](./findings.md). Caveats and
confidence intervals are part of the product.

RedHop is deliberately *not* a framework. It has:

- **No** agents, planners, workflows, or orchestration DAGs.
- **No** graph traversal or query decomposition.
- **No** embedded LLM or vector DB.

It does one thing — allocate the context budget to the densest *and most
reasoning-complete* evidence — and reports what it did.
