# RedHop

**Reasoning-preserving context optimization and retrieval observability for RAG systems.**

RedHop sits **between retrieval and generation**. You hand it the chunks your
retriever returned and a token budget; it assembles the prompt context —
pruning distractors, **preserving reasoning-critical "second-hop" evidence**,
and reporting exactly what it did.

It is *not* a retriever, vector database, agent framework, or workflow engine.
You keep your retriever and your LLM; RedHop optimizes the context in between
and makes the tradeoffs **visible and measurable**.

```python
import redhop

ctx = redhop.build_context(
    query=query,
    retrieved_chunks=chunks,          # list of dicts or strings
    strategy="reasoning_preserving",  # the safe default
    token_budget=12000,
)
response = llm.generate(ctx.text())
print(ctx.report)                     # Context Optimization Report
```

## The one idea

On a multi-hop question, the chunk that *bridges* to the answer (the "second
hop") is **low-relevance-to-the-query by construction** — it connects through a
bridge entity, not the query terms. So relevance-based operations (aggressive
filtering, reranking, max-density pruning) tend to drop exactly the evidence the
answer needs.

> Transformers tolerate irrelevant context far better than they tolerate
> missing reasoning links.

RedHop's default keeps the second hop while still pruning junk, and reports the
tradeoff instead of leaving it silent. The claim is **measured**, with
confidence intervals and preserved falsified hypotheses — see [Findings](./findings.md).

## Where to go next

- [Philosophy](./philosophy.md) — why relevance ≠ reasoning usefulness.
- [Quickstart](./quickstart.md) — install and run in a minute.
- [Context Strategies](./strategies.md) — what each strategy does and when.
- [Findings](./findings.md) — the evidence behind every default.
