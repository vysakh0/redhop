# Indexing & Query Speed vs LangChain / LlamaIndex (no LLM)

> **Hypothesis:** the `Document` runtime is faster than the big Python frameworks
> on the real `from_text → context` path — and "even dense rerank is faster."
> **Status:** **Largely falsified as a speed claim.** Compared *like-for-like within
> the same retrieval scenario*, RedHop has **no setup-speed moat**: lexical-vs-lexical
> all three index in well under a second; semantic-vs-semantic RedHop's local rerank
> is **slower** to set up (ONNX embedding). The earlier "RedHop ~0.02s vs ~7s" gap was
> RedHop's **lexical default** vs the frameworks' **vector default** — a *defaults*
> difference, not an engine win. RedHop's one real query-time edge: in the semantic
> scenario its warm query is cheaper (cosine over a small BM25 pool vs full vector search).
> **Setup:** real CUAD contracts concatenated to ~14k / ~38k / ~189k tokens. Two
> scenarios, like-for-like within each — **lexical** (BM25 retriever on all three, no
> embeddings) and **semantic** (embeddings on all three). Each framework uses its own
> splitter, BM25 retriever, and in-memory vector store; embeddings are the same model
> family (e5-small-v2) everywhere — ONNX for RedHop's rerank, PyTorch
> (sentence-transformers) for LC/LI. CPU only, single machine (10 cores). PDF parsing
> excluded. Metric: **time-to-first-answer** and **warm per-query**.
> **Headline:** lexical — RedHop / LangChain / LlamaIndex all ≤0.25s to first answer at
> ~189k tokens (no moat). Semantic — all embed every chunk; RedHop rerank **51s** vs
> LangChain **7.6s** / LlamaIndex **6.5s** to set up, but **4.3ms** warm vs **16–18ms**.
> **Reproduce:** `HF_HUB_OFFLINE=1 bench/.venv/bin/python bench/speed_compare.py`
> (needs `bench/models/e5-small-onnx`). Raw output in
> [reports/speed_vs_frameworks.txt](../../reports/speed_vs_frameworks.txt).
> **Caveats:** in-memory stores (no FAISS/ANN tuning); single model + machine; ONNX vs
> PyTorch is a runtime difference, not a controlled variable; RedHop's per-call does
> more than raw retrieval (pruning + Decision Report) so its warm-ms isn't a pure
> retrieval-latency number.

---

## Why this eval

The claim "RedHop is faster than LangChain/LlamaIndex, and even dense rerank is
faster because it only touches a small pool" needed checking against the real
`contract.pdf` path. Two corrections fell out: (1) what dense rerank actually does
(below), and (2) the first cut compared RedHop's lexical default to the frameworks'
*vector* default — apples to oranges. This version compares **like-for-like within
each scenario**.

## Scenario 1 — LEXICAL (BM25 on all three, no embeddings)

Time to first answer / warm per-query:

| size | RedHop (BM25) | LangChain (BM25) | LlamaIndex (BM25) |
| ---- | ------------- | ---------------- | ----------------- |
| ~14k tok  | 0.00s / 1.0ms | 0.01s / 0.1ms | 0.02s / 0.1ms |
| ~38k tok  | 0.01s / 1.1ms | 0.01s / 0.1ms | 0.05s / 0.1ms |
| ~189k tok | 0.02s / 1.0ms | 0.05s / 0.7ms | 0.25s / 0.1ms |

**No speed moat.** All three index in well under a second with no embedding step;
RedHop is competitive but not faster. On *raw* warm retrieval the Python BM25
retrievers are actually quicker (0.1–0.7ms) than RedHop's ~1ms — though RedHop's
call also prunes and produces a Decision Report, so it's doing more per query.

## Scenario 2 — SEMANTIC (embeddings on all three, e5-small)

| size | RedHop (dense rerank) | LangChain (vector) | LlamaIndex (vector) |
| ---- | --------------------- | ------------------ | ------------------- |
| ~14k tok  | 1.9s / 5.4ms  | 0.44s / 14.1ms | 0.29s / 5.5ms |
| ~38k tok  | 8.2s / 4.2ms  | 1.50s / 7.1ms  | 1.01s / 7.3ms |
| ~189k tok | 51.3s / 4.3ms | 7.57s / 16.4ms | 6.53s / 18.4ms |

**RedHop is slower to set up, faster per warm query.** All three embed every chunk
once; RedHop's ONNX embedding is ~4× slower per chunk than the frameworks' PyTorch
path, so setup is far slower (51s vs ~6.5–7.6s at 189k; int8 quantization → ~27s at a
4× smaller model). But once indexed, RedHop's warm query (cosine over the ~50-chunk
BM25 pool) is ~4ms vs ~16–18ms for full vector search — its only clean speed edge.

## Reading

- **Speed is not the moat.** Like-for-like, RedHop neither dominates nor collapses;
  it's competitive lexically and a mixed bag semantically (slower setup, faster warm).
  The honest pitch is the rest of the runtime — bounded API, conditional pruning,
  Decision Report, no vector infra — *not* raw speed.
- **The real "no-embedding" advantage is a default/DX point, not an engine win.**
  RedHop's default is lexical, so out of the box it's queryable instantly with no
  embedding step; a typical vector-RAG quickstart embeds every chunk first. Use BM25
  in LangChain/LlamaIndex too and the gap disappears.
- **Dense rerank is a recall feature, not a speed feature.** It embeds the whole
  document once (see Correction); pick it for semantic recall, accept the setup cost.

## Correction this eval forced

Dense rerank does **not** "only embed the candidate pool." `LocalRerankRetriever::index`
(`crates/retrieval/src/local_rerank.rs`) embeds **all** chunks once and caches them;
`candidate_pool` controls only how many BM25 candidates get cosine-*scored* per query.
Still no ANN/vector index (cached vectors + exact cosine over a small pool), so the
"no vector DB at any tier" claim holds; the "no embedding" claim does not.

## What changed afterward

1. **ONNX intra-op threads fix (shipped).** `OnnxEmbedder::load` never set
   `with_intra_threads`, so ORT ran ~single-threaded. Setting it to the core count cut
   rerank embed-all ~1.4× (189k: 70s → 51s) and, with the release build, dropped warm
   queries (BM25 9.6ms → 1.0ms; rerank 13ms → 4.3ms). (`crates/embeddings/src/onnx.rs`.)
2. **int8 quantization is the recommended rerank model trade** — embed-all 50.5s →
   26.6s at a 4× smaller model (133MB → 34MB).
3. **Stop marketing speed as a differentiator.** The site copy was corrected from a
   misleading lexical-vs-vector comparison to honest per-scenario numbers; the pitch is
   the runtime, not the clock.
4. **Open:** ORT CPU embedding is ~4× slower than torch here (graph/op tuning, a faster
   embedder, or a GPU/CoreML EP could close it) — but speed isn't the strategic claim.
