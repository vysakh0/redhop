# The Lightweight-Semantic Frontier — how far retrieval gets without a transformer

> **Hypothesis:** semantic recall (the gap BM25 leaves on paraphrase / low-overlap queries) can be recovered *without* a transformer encoder — either from corpus co-occurrence structure (a term-graph) or from pretrained *static* embeddings (token lookup + mean-pool, no inference runtime).
> **Status:** Partially confirmed. A zero-dep corpus-graph **local reranker** gives a real, bounded lift over BM25 (semantic recall@3 0.49→0.56, 0.59 with idf-weighted dims). But it — and external static embeddings — plateau in the **same ~0.56–0.59 band**, far below contextual BGE (0.80). Three sub-hypotheses falsified: MaxSim late-interaction, RM3 pseudo-relevance feedback, and "static embeddings are a lighter dense replacement."
> **Setup:** global HotpotQA pool (3,957 deduped paragraphs, text-only), 400 multi-hop queries, BM25 top-50 candidate pool, recall@3, split lexical vs semantic at the query↔gold overlap median (0.857). Zero-dep arms tested in the `semantic-bm25` library (pure-std BM25 + PPMI term graph); static embeddings tested via Model2Vec / SentenceTransformer over the identical pool.
> **Headline:** the zero-dep tier and the tiny-dep tier hit the **same ceiling** — both are non-contextual, so neither substitutes for the transformer on semantic-mismatch queries. The best free option is your own corpus-graph reranker, which a 30 MB pretrained static table does **not** beat.
> **Reproduce:** `cargo run -p redhop-examples --example export_semantic_pool --release` then in `../semantic-bm25` `cargo run --release --example eval -- --format beir --corpus <pool>/<slice> --graph second-order --mode local --kcand 50`; static: `cargo run -p redhop-examples --example export_rerank_pool --release` then `bench/.venv/bin/python python/eval/static_rerank.py`. Raw: `reports/static_rerank.txt`.
> **Caveats:** HotpotQA multi-hop, lexical-grounding split, recall@3 (reachability, not answer quality). Two BM25 implementations (Tantivy in RedHop, std in semantic-bm25) give ~0.49–0.51 semantic R@3 — consistent. Numbers are directions, not promises.

---

## Why this experiment

[CONTEXT_DILUTION](CONTEXT_DILUTION.md) and [SEMANTIC_MISMATCH](SEMANTIC_MISMATCH.md) established that dense
retrieval helps on semantic-heavy queries and that [LOCAL_RERANK](LOCAL_RERANK.md) recovers
~96% of global dense's gains — but at the cost of an ONNX runtime + a 133 MB
model. The open question was whether the *semantic* part could be had **without
the transformer dependency at all**, two ways:

1. **From the corpus itself** — a term–term co-occurrence graph (PPMI), used to
   compare query and document *context profiles* (second-order / distributional
   similarity, à la Schütze; PPMI ≈ implicit LSA, Levy & Goldberg). Pure `std`,
   no model. (The `semantic-bm25` library.)
2. **From pretrained static embeddings** — Model2Vec / static-retrieval models:
   a token→vector lookup table, mean-pooled, with **no inference runtime** (just
   a float table + tokenizer).

Both are tested as **rerankers over the same BM25 top-50 pool** the dense study
used, so every number is directly comparable to BGE's 0.80.

## Result 1 — corpus-graph second-order rerank (zero dependency)

Lexical topology first (BM25 prunes the corpus to 50), distributional refinement
second (reorder the pool by cosine of second-order context profiles — each term
represented by its row-normalized PPMI neighbors). Recall@3, semantic slice
(vanilla BM25 baseline 0.493):

| zero-dep method | lexical | semantic | ALL |
| --------------- | ------- | -------- | --- |
| BM25 (no rerank) | 0.663 | 0.493 | 0.575 |
| second-order **expansion** (expand query, re-run BM25) | 0.687 | 0.522 | 0.601 |
| second-order **local rerank** (centroid cosine) | 0.661 | 0.563 | 0.610 |
| second-order local rerank **+ idf-weighted dims** | 0.635 | 0.589 | 0.611 |
| tf-idf-cosine control (rerank, **no graph**) | — | 0.502 | — |

Reading:
- **Reranking beats expansion** on the semantic slice (0.56 vs 0.52): the gold is
  usually already in the BM25 pool, ranked low — reorder it rather than widen the
  query. Same lesson as [LOCAL_RERANK](LOCAL_RERANK.md), now with a zero-dep scorer.
- **The graph is the lever, not the cosine.** The tf-idf-cosine control (rerank
  by cosine but with *no* graph neighbors) reaches only 0.502 ≈ BM25; adding the
  second-order context profile is what lifts it to 0.563.
- **Second-order is the best graph mode** (1-hop; propagating further over an
  already-distributional graph over-diffuses), edging PPMI-expansion and bridge.
- **idf-weighted dims is a tilt, not a free win:** +0.026 semantic but −0.026
  lexical (now below BM25), a wash on ALL. A knob for semantic-dominant corpora,
  not a safe default. Plain centroid is the safe always-on (lexical-neutral).

## Result 2 — static pretrained embeddings (tiny dependency)

Same BM25 pool, reranked by static embeddings (recall@3):

| method | lexical | semantic | ALL | dependency |
| ------ | ------- | -------- | --- | ---------- |
| BM25 (pool order) | 0.684 | 0.505 | 0.591 | none |
| potion-retrieval-32M (Model2Vec) | 0.578 | 0.536 | 0.556 | numpy + ~30 MB table |
| static-retrieval-mrl-en-v1 | 0.585 | 0.556 | 0.570 | torch + table |
| **BGE-small (ONNX)** | 0.808 | 0.795 | **0.801** | ort + 133 MB |

Both static models land at **~0.56–0.57 ALL — below plain BM25 (0.591)** — and
*hurt* the lexical slice (they disturb BM25's already-good ranking). The recon
estimate of ~0.70 (from MTEB *averages*) did not transfer to this short-question
multi-hop setting.

## The frontier (the point of the doc)

| tier | dependency | semantic R@3 | ALL R@3 |
| ---- | ---------- | ------------ | ------- |
| zero-dep (corpus-graph second-order rerank) | none | 0.56–0.59 | ~0.61 |
| tiny-dep (pretrained static embeddings) | ~30 MB table (+torch) | 0.54–0.56 | ~0.57 |
| contextual (BGE-small) | ort + 133 MB | **0.80** | **0.80** |

The zero-dep and tiny-dep tiers **converge on the same band** — and it is not a
coincidence. Both are **non-contextual / lookup-based**: a static table and a
corpus co-occurrence graph are the same class of trick (distributional, no
in-context encoding), so they hit the same ceiling. **BGE's advantage is
contextualization** — the transformer encoding query and passage *in context* —
which no lookup method, externally pretrained or corpus-derived, replicates.

Two consequences:
- **You already have the best lightweight semantic reranker available:** the
  zero-dep corpus-graph reranker is *as good as* a pretrained static table, with
  *zero* dependency. Adding the static-embedding dependency buys nothing over it.
- **There is no free lunch in the middle.** To keep BGE-quality semantic recall
  you need the transformer runtime; to go light you accept ~0.56–0.61.

## Falsified sub-hypotheses (preserved)

- **MaxSim late-interaction over corpus-graph vectors** — *hypothesis:* per-query-term
  best-match (ColBERT-style) beats the centroid by rewarding a single bridge term.
  *Result:* semantic R@3 **fell** 0.563→0.531. On 2–4-term questions over sparse
  graph vectors, per-term max rewards any doc with one loosely-related term — less
  discriminative than the aggregated centroid. Late interaction needs many query
  tokens; questions don't have them.
- **RM3 pseudo-relevance feedback** — *hypothesis:* classical PRF lifts recall.
  *Result:* **monotonically harmful** (semantic R@3 0.493→0.406 at λ=0.5; λ=1.0
  reproduces BM25 exactly). PRF assumes high first-pass precision; here BM25 R@1
  ≈ 0.31, so the feedback model is built on distractors and amplifies wrong terms.
  Same geometry as [SECOND_HOP_TAX](SECOND_HOP_TAX.md) / [RERANKING_LIMITS](RERANKING_LIMITS.md).
- **Static embeddings as a lighter dense replacement** — *hypothesis (from recon):*
  ~0.70 R@3 at a fraction of the dependency. *Result:* 0.56–0.57, below BM25 and
  far below BGE. Falsified for this workload.

## What changed afterward

- Confirms RedHop's default should stay **BM25 lexical-first**, with the
  ReasoningPreserving / dilution machinery on top — *not* a forced embedding dep.
- A zero-dep "semantic refinement" option (corpus-graph second-order **local
  rerank**, plain centroid, always-on) is the right shape for a lightweight
  semantic tier *if* one is offered — it is lexical-neutral and free. It does
  **not** replace dense on semantic-mismatch workloads.
- Closes the "can we get dense-quality cheaply" question: **no** — the gap is
  contextualization, and lightweight tiers (graph or static) share one ceiling.
  See [DENSE_RERANK_CEILING](DENSE_RERANK_CEILING.md) for where the dense path itself plateaus.

Full lexical↔semantic boundary mapping: [SEMANTIC_MISMATCH](SEMANTIC_MISMATCH.md). The dense
local-rerank architecture this complements: [LOCAL_RERANK](LOCAL_RERANK.md).
