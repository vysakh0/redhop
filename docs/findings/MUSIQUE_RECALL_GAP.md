# The MuSiQue Recall Gap — diagnostic + falsified runtime refactor

> **Hypothesis:** RedHop's dense recall@4 = 0.76 on HotpotQA and 0.28 on
> MuSiQue — using the same BGE-small, same pipeline, same metric. Worth
> understanding before assuming "MuSiQue is just harder."
>
> **Result:** the gap is FIVE distinct things, not one. Two looked
> addressable via a runtime change (refactor `RetrievalMode::Hybrid`
> to use full-pool RRF over BM25 + global dense, replacing the
> previous BM25-prune-then-dense-rerank composition). The refactor was
> built end-to-end on `feature/hybrid-full-pool-rrf` and an honest A/B
> measurement **falsified the shipping decision**: the recall lift the
> first measurement promised did not carry through to the user-facing
> `candidate_k = 20` cutoff, and the new composition introduced a small
> regression at K=4 on HotpotQA. The branch stays as a research record;
> main keeps the existing Hybrid behavior.
>
> **Reproduce:**
> ```bash
> # Diagnostic + sub-measurements (chunking, embedder, fusion):
> cargo run -p redhop-examples --example musique_recall_diagnostic --features onnx --release
> cargo run -p redhop-examples --example musique_hybrid_recall     --features onnx --release
> cargo run -p redhop-examples --example musique_chunk_sweep       --features onnx --release
> cargo run -p redhop-examples --example musique_embedder_swap     --features onnx --release
> # Direct A/B that falsified the runtime refactor:
> cargo run -p redhop-examples --example hybrid_old_vs_new         --features onnx --release
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

Five distinct contributors:

### 1. Gold density — partly metric artifact

MuSiQue queries need on average **5 gold chunks** to reach recall=1.0,
vs HotpotQA's 2.22. At k_final=4 MuSiQue mathematically cannot reach
1.0 — a query that needs 5 chunks in 4 slots is capped at 0.80. Part
of the visible gap is a metric mismatch, not a method failure.

### 2. Retrieval-signal type — BM25 strictly wins on MuSiQue

| K  | HotpotQA BM25 vs dense | MuSiQue BM25 vs dense |
| -- | ---------------------- | --------------------- |
| 4  | 0.71 ≈ 0.76 (dense)    | **0.31 > 0.28** (BM25) |
| 10 | 0.85 ≈ 0.86 (dense)    | **0.40 > 0.37** (BM25) |
| 50 | 0.95 ≈ 0.96 (dense)    | **0.55 > 0.51** (BM25) |

Compositional, named-entity-heavy questions favor lexical exact-match;
HotpotQA is mostly paraphrase / semantic-friendly.

### 3. Wide-net coverage — RRF over BOTH signals dominates either at K≥20

Adding an RRF fusion step over the BM25 top-50 + dense top-50 (each
retrieving independently from the whole corpus) surfaces gold chunks
that *either retriever alone misses*:

| K  | HotpotQA RRF Δ vs best | MuSiQue RRF Δ vs best |
| -- | ---------------------- | --------------------- |
| 4  | −0.0042                | −0.0176               |
| 10 | +0.0142 ✓              | −0.0222               |
| 20 | +0.0218 ✓              | +0.0226 ✓              |
| 50 | +0.0241 ✓              | **+0.0693 ✓**          |

RRF beats single-retriever at wide K; at K=4 the two retrievers' top-1
candidates clash and RRF dilutes the better single retriever's signal.

### 4. Embedder capacity — modest

Swapping BGE-small (384-dim) for BGE-base (768-dim): real win on
HotpotQA@4 (+0.032), small on MuSiQue@50 (+0.027). Embedder size is
not the dominant constraint.

### 5. Chunking — NOT the bottleneck

`target_tokens` sweep peaks at 40 (the existing default) on BOTH
corpora. Indirectly confirms the
[CHUNK_GRANULARITY](CHUNK_GRANULARITY.md) calibration generalizes.

## The runtime refactor that didn't ship

Findings (2) and (3) suggested a concrete runtime change: refactor
`RetrievalMode::Hybrid { candidate_pool }` from its current composition

> BM25 retrieves a candidate pool, dense reranks ONLY that pool, RRF
> fuses the BM25 ranking with the dense ranking of the same pool.

to a full-pool RRF

> BM25 retrieves top-`candidate_pool` from the whole corpus, AND global
> dense independently retrieves top-`candidate_pool` from the whole
> corpus. The two ranked lists are RRF-fused (k=60).

The refactor was built on `feature/hybrid-full-pool-rrf` (commit
`c81ffbe`): rustdoc rewritten, `Document::ensure_indexed()` restructured
to build + index BM25 and global Dense sub-retrievers in place and
wrap them in `HybridRetriever`, `HybridRetriever::embeddings()`
override added so `Document::embedded_chunks()` keeps working through
the composition, README + node docs + CHANGELOG updated. 403 tests
passed; fmt + clippy `--workspace --all-targets -D warnings` clean.

### The A/B that falsified the ship decision

The original `musique_hybrid_recall` experiment compared *three
retrievers*: BM25 alone, dense alone, RRF of the two. It did NOT
compare the old and new Hybrid compositions directly. Before merging,
we added `hybrid_old_vs_new.rs` — same corpus, same queries, same
chunker, only the Hybrid composition differs:

|                | @4         | @10        | @20        | @50        |
| -------------- | ---------- | ---------- | ---------- | ---------- |
| **HotpotQA**   |            |            |            |            |
| old (LocalRerank) | 0.7714  | 0.8703     | 0.9322     | 0.9500     |
| new (BM25+Dense+RRF) | 0.7602 | 0.8778  | 0.9338     | 0.9842     |
| Δ              | **−0.0113** | +0.0075   | +0.0017    | +0.0342    |
| **MuSiQue**    |            |            |            |            |
| old (LocalRerank) | 0.2927  | 0.3821     | 0.4758     | 0.5470     |
| new (BM25+Dense+RRF) | 0.2889 | 0.3833  | 0.4832     | 0.6144     |
| Δ              | −0.0038    | +0.0012    | +0.0074    | **+0.0674** |

The pre-registered ship criterion: NEW beats OLD at the default
`candidate_k = 20` by Δ ≥ +0.02 on at least one corpus without
regressing the other by more than 0.01.

**Both criteria fail.** HotpotQA Δ@20 = +0.0017 (tie); MuSiQue Δ@20 =
+0.0074 (marginal, below ship bar). And the K=4 numbers are net
negative on the majority workload (HotpotQA −0.011), with no
compensating gain on MuSiQue.

The wide-K wins are real (MuSiQue@50 +0.067, HotpotQA@50 +0.034) but
the default `Document.context()` flow consumes `candidate_k = 20`
candidates and then truncates further inside the assembler. Most users
never see the K=50 regime. The runtime cost (global cosine per query
on the whole corpus, vs O(|pool|) before) is paid by every query
regardless.

### Outcome

The refactor stays on `feature/hybrid-full-pool-rrf` as a research
record. **Main keeps the existing `RetrievalMode::Hybrid` semantics**
(BM25 prune → dense rerank of pool → RRF fuse).

What the falsification specifically establishes:

1. The "+0.07 RRF@50 on MuSiQue" claim from the first measurement was
   *technically true* (RRF beats single retrievers at K=50) but
   misleading as a runtime improvement (the new composition doesn't
   strictly dominate the old one at user-facing K).
2. The old Hybrid (`LocalRerankRetriever`) is approximately optimal
   at `candidate_k = 20` on both corpora. Replacing it with a more
   expensive composition for marginal +0.007 lift is the wrong call.
3. Future work that wants to push past this should either change the
   downstream cutoff (raise `candidate_k`) — where the wide-net wins
   live — or attack a different lever entirely (query decomposition,
   bigger embedder).

## Bottom line for the evidence layer

- **Five-way decomposition** of the cross-corpus recall gap (above)
  stands as a finding. It's the right way to think about
  HotpotQA-vs-MuSiQue-shaped differences in any future retrieval work.
- **The chunker default (40, 60)** is calibration-verified on a second
  out-of-distribution corpus (MuSiQue), strengthening the
  `CHUNK_GRANULARITY` finding.
- **The Hybrid runtime change is closed** for the foreseeable future.
  Anybody tempted to refactor `RetrievalMode::Hybrid` to full-pool RRF
  needs to read this finding first; the A/B already shows the answer.
- **The A/B harness** (`hybrid_old_vs_new.rs`) is the falsification
  test future-us needs. Re-runnable with the same env vars as the
  diagnostic.

## Honest limits

- BGE-small ONNX (Qdrant), ms-marco MiniLM-L-6 (Xenova); 200-query
  stratified HotpotQA and 200-query answerable 2-hop MuSiQue samples.
  Single run, no bootstrap CI on each delta.
- The A/B was at `candidate_pool = 50` and recall reported at the
  retriever level (before any assembly). The user-facing recall after
  budget truncation could differ — but if anything, truncation makes
  the wide-K wins LESS visible, not more. The ship-decision direction
  doesn't change.
- The cost increase (global cosine per query) was not directly
  benchmarked; the architectural cost is real on million-chunk corpora
  even if it didn't show up as a latency delta on the 7k-chunk test.
