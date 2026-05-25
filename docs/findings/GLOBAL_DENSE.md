# Global Dense vs Local Rerank on Semantic Mismatch (no LLM)

> **Hypothesis:** local rerank (BM25 prune → dense reorder) is enough; a global
> dense pass over every chunk isn't worth adding.
> **Status:** **Falsified for paraphrase/synonym-heavy corpora.** When the answer
> shares no terms with the query, BM25's pool never contains it, so local rerank
> *cannot* recover it — global dense does. Adding a bounded global-dense tier is
> justified; it stays the right *default* only on lexically-overlapping data
> (see [LOCAL_RERANK](LOCAL_RERANK.md), where local ≈ global on HotpotQA).
> **Setup:** the controlled semantic-mismatch probe (`data/semantic_mismatch.json`)
> — 25 queries, each with a GOLD passage (semantically right, low lexical overlap),
> a TRAP (high lexical overlap, wrong meaning — a BM25 attractor), and distractors;
> all 100 passages pooled into one corpus. Metric: is GOLD retrieved (recall@1 /
> recall@3), no LLM. Embedder: BGE-small (int8) via the `retrieval="dense"` /
> `"rerank"` tiers. Global dense = exact cosine over **all** chunks, **no ANN**.
> **Headline:** recall@1 — lexical **20%**, local rerank **32%**, **global dense 88%**;
> recall@3 — 20% / 32% / **96%**. Local rerank barely beats BM25 because it inherits
> BM25's pool; global dense scores every chunk. On the `control` category (lexical
> overlap present) all three retrieve 4/4 — global doesn't hurt the easy case.
> **Reproduce:** `bench/.venv/bin/python bench/semantic_modes.py`. Raw output in
> [reports/semantic_modes.txt](../../reports/semantic_modes.txt).
> **Caveats:** an adversarial probe by construction (engineered low-overlap golds +
> lexical traps), n=25; small bounded pool (100 passages); single embedder. It shows
> *where* global wins, not a general retrieval-quality ranking.

---

## Results

**recall@1 (GOLD retrieved):**

| mode | paraphrase | legal_syn | reformul | low_overlap | control | ALL |
| ---- | ---------- | --------- | -------- | ----------- | ------- | --- |
| lexical (BM25) | 0/6 | 0/6 | 1/5 | 0/4 | 4/4 | **20%** |
| local rerank   | 1/6 | 1/6 | 1/5 | 1/4 | 4/4 | **32%** |
| global dense   | 6/6 | 3/6 | 5/5 | 4/4 | 4/4 | **88%** |

**recall@3:** lexical 20% · local rerank 32% · **global dense 96%**.

## Reading

- **Local rerank is bottlenecked by BM25.** It can only reorder what BM25 surfaced;
  on paraphrase/synonymy the gold passage shares no terms with the query, so it's
  absent from the pool and no reranking recovers it (32% ≈ BM25's 20%).
- **Global dense fixes exactly that** — it cosines the query against *every* chunk,
  so a lexically-disjoint answer is still reachable (88–96%).
- **It's not free, and not always needed.** On lexically-overlapping data (the
  `control` slice here; HotpotQA in [LOCAL_RERANK](LOCAL_RERANK.md)) local rerank
  already matches global. Global costs O(N) cosine per query and only makes sense for
  **bounded** corpora — at scale you'd want a real ANN/vector store, which RedHop
  deliberately isn't.

## What changed afterward

- **Shipped `RetrievalMode::Dense` / `retrieval="dense"`** — global, exact brute-force
  cosine over all cached chunk embeddings, no BM25 prune, no ANN. Implemented in
  `LocalRerankRetriever::global()` (`crates/retrieval/src/local_rerank.rs`), wired
  through `redhop-document` and the Python binding.
- **Guidance / defaults unchanged:** `lexical` (BM25) is the default; `rerank` for
  semantic recall on normal data; **`dense` for paraphrase/synonym-heavy bounded
  corpora** where BM25 misses the answer entirely. The default stays lexical because
  most corpora overlap lexically and global is O(N) per query.
- **Open:** measure global vs local on a *natural* (non-adversarial) corpus to size
  the everyday gap; pick a recall/latency budget where `dense` auto-escalates.
