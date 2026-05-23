# Local Rerank — semantic recall without a vector index

> **The result:** BM25 prunes the corpus to a candidate pool; dense reranks
> **only that pool** (local), never the whole corpus (global). On natural
> HotpotQA this **matches global dense on both retrieval recall and downstream
> answers** — recovering ~96% of dense's gains — at a fraction of the compute and
> with **no global ANN / vector index**.
> **Status:** Confirmed, Tier-1 (recall) + Tier-3 (answers), on a 3,957-paragraph
> global corpus, n=400. Conditional on BM25 *candidate* recall@K (the honest cap).
> **Why it matters:** it's the evidence-backed way to give RedHop semantic recall
> without taking on a vector database — and it needs **no escalation trigger**
> (which the [trigger study](SEMANTIC_MISMATCH.md) showed don't exist cheaply).
> **Reproduce:** `cargo run -p redhop-examples --example semantic_local_rerank --features onnx --release`
> then `python python/eval/score_local_rerank.py`. Raw:
> `reports/semantic_local_rerank*.txt`.

---

## The idea

Treat lexical and semantic retrieval as **different operators**, not equal score
systems:

| stage | responsibility |
| ----- | -------------- |
| **BM25 (lexical)** | candidate *topology* — cheaply prune the whole corpus to ~K |
| **dense (semantic)** | local *refinement* — reorder only those K by meaning |

So semantics becomes **local neighborhood refinement over a lexically-anchored
pool**, not global semantic replacement. The dense model only ever touches K
items, never the corpus.

## Results (global HotpotQA, 3,957 paragraphs, n=400; K_cand=50, top-3)

**Retrieval recall@3:**

| subset | bm25 | global dense | **local rerank** | hybrid |
| ------ | ---- | ------------ | ---------------- | ------ |
| lexical-friendly | 0.68 | 0.80 | **0.81** | 0.77 |
| semantic-heavy | 0.50 | 0.80 | 0.79 | 0.67 |
| ALL | 0.59 | 0.80 | **0.80** | 0.72 |

**Downstream answers (gpt-4o-mini, F1 / EM):**

| subset | bm25 | global dense | **local rerank** | hybrid |
| ------ | ---- | ------------ | ---------------- | ------ |
| lexical-friendly | 0.47 / 0.39 | 0.57 / 0.47 | **0.59 / 0.47** | 0.56 / 0.47 |
| semantic-heavy | 0.27 / 0.21 | 0.52 / 0.42 | 0.50 / 0.40 | 0.38 / 0.30 |
| ALL | 0.37 / 0.29 | **0.54 / 0.44** | **0.54 / 0.43** | 0.47 / 0.38 |

Local rerank **equals global dense** on recall (0.80) and answers (0.54 F1),
**beats naive hybrid** (no fusion poisoning — dense reorders rather than votes
against BM25), and recovers the +0.23 F1 semantic-heavy gain over BM25.

## Why it works — the crux

**BM25 recall@50 is 0.94 even on semantic-heavy queries** (0.99 on lexical). The
gold passage is almost always *in* the candidate pool — BM25 just ranks it low
(recall@3 = 0.50). Dense doesn't need to search the corpus; it only **reorders the
pool**. Of the 165 queries where global dense beat BM25@3, local rerank recovered
**158 (96%)**; only 4% had gold outside the top-50 (where global dense is needed).

## Economics

| | global dense | local rerank |
| - | ------------ | ------------ |
| search per query | 0.505 ms over 3,957 | **0.019 ms over 50** |
| vector index / ANN | required (whole corpus) | **none** — BM25 prunes |
| dense touches | the corpus | **K candidates** |

The embedding model still runs — on the K candidates (query-time) or the corpus
(precomputed once). Local rerank's win is **bounding dense to K and dropping the
global ANN**, not eliminating the model. For huge corpora, BM25 (which scales
trivially) does the corpus-wide work and dense stays local.

## The honest cap

This works because BM25 has decent **candidate** recall — there's *partial*
lexical overlap to surface gold at depth K. On **pure-synonym** mismatch (the
controlled probe, where gold shares ~0 query terms), BM25's pool wouldn't contain
gold and local rerank degrades toward BM25 — only global dense reaches the
residual. So local rerank is the right default *when first-stage candidate recall
is high*, which holds on natural data but not on adversarial paraphrase.

## Direction

> **Lexical topology first → local semantic refinement second → runtime economics
> always visible.** No global ANN, no escalation trigger.

This is RedHop's retrieval-runtime direction, evidence-backed. The product step is
to offer it as a `Document` retrieval mode (BM25 top-K → local dense rerank →
`build_context`), behind the onnx feature — semantic recall without a vector store.

Treat this as a **measured retrieval-runtime result, not a new algorithm.** The
full journey that produced it (boundary mapping, the failed escalation triggers)
is in [SEMANTIC_MISMATCH.md](SEMANTIC_MISMATCH.md).
