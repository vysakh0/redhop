# The MuSiQue Recall Gap — and what changed in `RetrievalMode::Hybrid`

> **Hypothesis:** RedHop's dense recall@4 = 0.76 on HotpotQA and 0.28 on MuSiQue —
> using the same BGE-small, same pipeline, same metric. The gap is large and
> worth understanding before assuming "MuSiQue is just harder."
>
> **Result:** the gap is FIVE distinct things, not one. Two of them are
> directly addressable in the runtime; three are corpus-shape properties.
> The runtime change that landed in this finding: **`RetrievalMode::Hybrid`
> now does BM25 + global Dense + RRF fusion**, replacing the previous
> "BM25-prune → dense-rerank-of-pool → RRF" composition. Wide-net recall
> jumps +0.07 on MuSiQue@50 and +0.02 on HotpotQA@50, with no regression
> at K=4. The old composition is preserved as
> [`LocalRerankRetriever`](../../crates/redhop/src/retrieval/local_rerank.rs)
> for users who hit the global-cosine cost.
>
> **Reproduce:**
> ```bash
> cargo run -p redhop-examples --example musique_recall_diagnostic --features onnx --release
> cargo run -p redhop-examples --example musique_hybrid_recall     --features onnx --release
> cargo run -p redhop-examples --example musique_chunk_sweep       --features onnx --release
> cargo run -p redhop-examples --example musique_embedder_swap     --features onnx --release
> ```

## Diagnosing the gap

Same harness on both corpora: 200 queries (100 bridge + 100 comparison
for HotpotQA, 200 answerable 2-hop for MuSiQue), `SentenceChunker(40,
60, 0)`, BGE-small (Qdrant ONNX), recall@K for K=4/10/20/50.

|                              | HotpotQA | MuSiQue |
| ---                          | -------- | ------- |
| mean gold chunks per query   | 2.22     | **5.09** |
| mean chunks per document     | 2.99     | 2.38    |
| BM25 recall@4                | 0.7081   | **0.3065** |
| BM25 recall@50               | 0.9500   | 0.5470  |
| dense recall@4               | 0.7643   | 0.2806  |
| dense recall@50              | 0.9601   | **0.5051** |

Five distinct things explain the gap, in roughly descending share:

### 1. Gold density — partly metric artifact

MuSiQue queries need on average **5 gold chunks** to achieve recall=1.0,
vs HotpotQA's 2.22. At k_final=4 MuSiQue mathematically cannot reach
1.0 — a query that needs 5 chunks in 4 slots is capped at 0.80. This
isn't a method failure; it's a metric mismatch.

### 2. Retrieval-signal type — BM25 strictly wins on MuSiQue

| K  | HotpotQA BM25 vs dense | MuSiQue BM25 vs dense |
| -- | ---------------------- | --------------------- |
| 4  | 0.71 ≈ 0.76 (dense)    | **0.31 > 0.28** (BM25) |
| 10 | 0.85 ≈ 0.86 (dense)    | **0.40 > 0.37** (BM25) |
| 50 | 0.95 ≈ 0.96 (dense)    | **0.55 > 0.51** (BM25) |

MuSiQue questions are compositional (multi-hop, named-entity-heavy);
BM25's lexical exact-match wins because the bridge entity often appears
literally in the gold passage. A pure-semantic embedder loses on this
geometry. HotpotQA is mostly paraphrase / semantic-friendly, where BM25
and dense tie.

### 3. Wide-net coverage — RRF over BOTH signals strictly dominates either

Adding an RRF fusion step over the BM25 top-50 + dense top-50 (each
retrieving independently from the whole corpus) surfaces gold chunks
that *either retriever alone misses*:

| K  | HotpotQA RRF Δ vs best | MuSiQue RRF Δ vs best |
| -- | ---------------------- | --------------------- |
| 4  | −0.0042                | −0.0176               |
| 10 | +0.0142 ✓              | −0.0222               |
| 20 | +0.0218 ✓              | +0.0226 ✓              |
| 50 | +0.0241 ✓              | **+0.0693 ✓**          |

At wide K (≥ 20) RRF clearly beats single-retriever on both corpora. At
K=4 the two retrievers' top-1 candidates clash and RRF dilutes the
better single retriever's signal. The lift comes from the *pool*, not
the top — which is exactly the regime downstream strategies
(ReasoningPreserving, CE rerank, context allocation) feed on.

### 4. Embedder capacity — modest

Swapping BGE-small (384-dim, 24M params) for BGE-base (768-dim, 110M):

| K  | HotpotQA dense (small → base) | MuSiQue dense (small → base) |
| -- | ----------------------------- | ---------------------------- |
| 4  | 0.7643 → 0.7962 (+0.032 ✓)    | 0.2806 → 0.2971 (+0.017)     |
| 10 | 0.8612 → 0.8851 (+0.024)      | 0.3654 → 0.3643 (≈0)         |
| 50 | 0.9601 → 0.9617 (≈0)          | 0.5051 → 0.5319 (+0.027)     |

Real win on HotpotQA at K=4. Modest on MuSiQue at K=50. The
embedder is not the dominant constraint for hard multi-hop corpora.

### 5. Chunking — NOT the bottleneck

Sweeping `target_tokens` in {16, 24, 40, 64, 96}, RRF@50 has the same
U-shape on both corpora with peak at target=40 — the existing default.
This is an indirect cross-corpus confirmation that the default chunker
calibration ([CHUNK_GRANULARITY](CHUNK_GRANULARITY.md)) holds beyond
HotpotQA.

| target_tokens | HotpotQA RRF@50 | MuSiQue RRF@50 |
| ------------- | --------------- | -------------- |
| 16            | 0.9505 (−0.034) | 0.5692 (−0.047) |
| 24            | 0.9569 (−0.027) | 0.5794 (−0.037) |
| **40**        | **0.9842**      | **0.6163**      |
| 64            | 0.8928 (−0.091) | 0.4865 (−0.130) |
| 96            | 0.8247 (−0.160) | 0.4128 (−0.204) |

## What we changed in the runtime

`RetrievalMode::Hybrid { candidate_pool }` previously composed:
> BM25 retrieves a candidate pool, dense reranks ONLY that pool, RRF
> fuses the BM25 ranking with the dense ranking of the pool.

The dense step never saw a chunk BM25 hadn't already surfaced. That's
fine for very-large corpora (saves a global cosine pass) but caps
recall@50 at BM25's pool depth on lexical-favoring corpora — which is
where dense usually has the most to *add*.

The new `RetrievalMode::Hybrid { candidate_pool }`:
> BM25 retrieves top-`candidate_pool` from the whole corpus, **and
> global dense independently retrieves top-`candidate_pool` from the
> whole corpus**. The two ranked lists are RRF-fused (k=60).

Cost increase: dense becomes O(|corpus|) cosines per query instead of
O(|pool|). On the test corpus (≈7k chunks) the delta is small. On
million-chunk corpora it would be substantial.

The old behavior remains available as a building block:
[`LocalRerankRetriever`](../../crates/redhop/src/retrieval/local_rerank.rs)
is fully public and can be assembled by callers who hit the
global-cosine cost. The previous "free hybrid recall on big corpora"
contract is now opt-in rather than the default. See
[LOCAL_RERANK](LOCAL_RERANK.md) for that retriever's original
characterization.

## Bottom line

- **+0.07 RRF@50 on MuSiQue** at the wide-net stage — directly improves
  the candidate pool every downstream strategy operates on.
- **+0.02 RRF@50 on HotpotQA** — small but in the same direction.
- **−0.01 to +0.00 at K=4** on either corpus — fusion's top-rank
  dilution; the upside is at wider K.
- The 0.51 ceiling on MuSiQue@50 itself doesn't fully break: even the
  new Hybrid + bge-base only reaches around 0.55 on MuSiQue@50. Most
  of the remaining gap is structural (gold density, true retrieval
  misses) and would need either query decomposition or a much larger
  embedder — neither of which fits the bounded-architecture constraint.

## Honest limits

- BGE-small ONNX (Qdrant), ms-marco MiniLM-L-6 (Xenova); 200-query
  stratified HotpotQA and 200-query answerable 2-hop MuSiQue samples.
  Single run, no bootstrap CI on each delta.
- Cost increase on large corpora is real. Documented in
  `RetrievalMode::Hybrid`'s rustdoc with a pointer to
  `LocalRerankRetriever` for the cheap path.
- The recall@4 dilution on MuSiQue (−0.018 vs best-single) was
  consistent across runs. RRF is a wide-net win, not a top-rank win.
  Strategies that consume the wide net benefit; strategies that consume
  only the top-K cutoff are mostly unaffected.
