# FAQ

### What is RedHop, in one sentence?

A reasoning-aware context runtime for RAG: hand it a document and a question, and it
chunks, retrieves, and allocates the context your model should actually see — pruning
distractors while preserving reasoning-critical evidence — with citations and a
Decision Report explaining what it did.

### Where does it sit in my stack?

Between your documents and the LLM: `document → redhop → LLM`. RedHop owns retrieval
(BM25 by default; optional dense/hybrid — no vector DB) and context assembly; you bring
the documents and the LLM. Already have chunks from your own retriever? Feed them
directly via `Document.from_chunks([...])` or the low-level `redhop.build_context(...)`.

### How is it different from LangChain / LlamaIndex / Haystack?

Those are frameworks that orchestrate retrieval, agents, and generation. RedHop is not a
framework — it does one thing (turn a document + query into the right context, and
explain the decision) and composes under them. You can also call `build_context` on the
chunks any of them return.

### How is it different from a reranker?

A reranker scores chunks by **query relevance**. RedHop's central finding is that on
multi-hop questions the reasoning-critical "second hop" is *low-relevance to the query by
construction*, so relevance reranking demotes/drops it — RedHop's default preserves it.
(RedHop *does* offer an optional cross-encoder reranker via `rerank="cross-encoder"` when
you want one; see the [reranking-limits finding](docs/findings/RERANKING_LIMITS.md) for
where a uniform cross-encoder made recall *worse*.)

### Does RedHop call an LLM or an embedding model?

It never calls an LLM. The default lexical (BM25) tier needs **no model at all**. The
optional `hybrid` / `semantic` tiers and `rerank="cross-encoder"` download a small ONNX
model on first use (cached) — free, local, no API key, no vector DB. You own the LLM.

### What is the "second-hop tax"?

On a multi-hop question, the chunk that bridges to the answer is relevant to a *bridge
entity*, not the query — so it has low query relevance by construction. Every
relevance-based operation (aggressive filtering, reranking, max-density pruning) tends to
drop it. We measured this directly (n=1327, CIs): a relevance filter keeps 96.8% of
second hops at threshold 0.05 but only **43.9% at 0.30**. Full detail:
[SECOND_HOP_TAX.md](docs/findings/SECOND_HOP_TAX.md).

### Which strategy should I use?

`reasoning_preserving` (the default) for general/multi-hop. `distractor_filtered` at a
*low* threshold is safe for single-hop. `max_density` only when the budget is brutal and
the workload is single-hop. `raw_topk` is the no-op baseline. Or `auto` to size-gate it.

### Which retrieval tier?

`lexical` (BM25, the default — no model, great for keyword-dense docs). `hybrid` for
semantic search that scales across many files. `semantic` for highest recall when the
question and answer share no words. See [Retrieval options](https://redhop.dev).

### Is it production-ready?

Alpha (0.x). Published to PyPI (`pip install redhop`), with npm shipping alongside and
crates.io to follow. The APIs are stable and the findings reproducible, but it's young
and not yet battle-tested at scale — use it, file issues, pin versions.

### Why should I trust the findings?

Because every claim is reproducible: each has a hypothesis, a reproduce command,
confidence intervals where applicable, and honest caveats — and **falsified hypotheses
are kept**, not hidden. See the [evidence layer](docs/findings/README.md) and its
falsified-hypotheses registry.

### Does RedHop claim to "solve reasoning" or beat SOTA?

No. It makes a specific, measured tradeoff visible: relevance optimization can remove
reasoning-critical evidence, and RedHop preserves it while still pruning junk. It's a
targeted improvement, reported with its limits — not a claim to beat SOTA.

### What languages/bindings are supported?

Rust (the source of truth), Python (`pip install redhop`), and Node.js
(`npm install redhop`). A CLI ships for evaluation.
