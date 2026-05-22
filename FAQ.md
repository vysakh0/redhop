# FAQ

### What is RedHop, in one sentence?

A reasoning-preserving context optimization and retrieval observability library:
it assembles the prompt context from your retrieved chunks under a token budget,
preserving reasoning-critical evidence, and reports what it did.

### Where does it sit in my stack?

Between retrieval and generation: `retriever → redhop.build_context → LLM`. You
keep your vector DB / BM25 retriever and your LLM; RedHop optimizes the context
in the middle.

### How is it different from LangChain / LlamaIndex / Haystack?

Those are frameworks that orchestrate retrieval, agents, and generation. RedHop
is not a framework — it does one thing (context assembly + observability) and
composes under them. You can call `build_context` on the chunks any of them
return.

### How is it different from a reranker?

A reranker scores chunks by **query relevance**. RedHop's central finding is that
on multi-hop questions the reasoning-critical "second hop" is *low-relevance to
the query by construction*, so relevance reranking demotes/drops it. RedHop
preserves it. (See the [reranking-limits finding](docs/findings/RERANKING_LIMITS.md)
— where a uniform cross-encoder made recall *worse*.)

### Does RedHop call an LLM or embedding model?

No. It needs neither to run its default lexical strategies. Embeddings are
optional (only `redundancy_pruned` uses them, if you attach them to chunks). You
own the LLM and the embedder.

### What is the "second-hop tax"?

On a multi-hop question, the chunk that bridges to the answer is relevant to a
*bridge entity*, not the query — so it has low query relevance by construction.
Every relevance-based operation (aggressive filtering, reranking, max-density
pruning) tends to drop it. We measured this directly (n=1327, CIs): a relevance
filter keeps 96.8% of second hops at threshold 0.05 but only **43.9% at 0.30**.
Full detail: [SECOND_HOP_TAX.md](docs/findings/SECOND_HOP_TAX.md).

### Which strategy should I use?

`reasoning_preserving` (the default) for general/multi-hop. `distractor_filtered`
at a *low* threshold is safe for single-hop. `max_density` only when the budget
is brutal and the workload is single-hop. `raw_topk` is the no-op baseline.

### Is it production-ready?

Alpha. The APIs are stable and the findings are reproducible, but it has not
been published to PyPI/crates.io yet and the bindings haven't been battle-tested
at scale. Use it, file issues, but pin versions.

### Why should I trust the findings?

Because the discipline is visible: every claim has a hypothesis, a reproduce
command, confidence intervals where applicable, and honest caveats — and
**falsified hypotheses are kept**, not hidden. See the
[evidence layer](docs/findings/README.md) and its falsified-hypotheses registry.

### Does RedHop claim to "solve reasoning" or beat SOTA?

No. It makes a specific, measured tradeoff visible: relevance optimization can
remove reasoning-critical evidence, and RedHop preserves it while still pruning
junk. The gains are modest, bounded, and reported with their limits.

### What languages/bindings are supported?

Rust (the source of truth) and Python (pyo3/maturin). A CLI ships for
evaluation. npm/`napi` bindings are on the roadmap.
