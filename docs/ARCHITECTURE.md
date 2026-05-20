# NeoRAG Architecture

## What NeoRAG is

NeoRAG is a Rust library for **retrieval infrastructure**: chunking,
retrieval, reranking, and diagnostics. It does not generate text, embed
text, or store vectors persistently — those concerns are pushed to the
caller through trait boundaries. The library's contribution is the
*orchestration* layer between them and the *diagnostics* engine that makes
retrieval quality observable.

## What NeoRAG is not

- Not an LLM framework. There is no model in this repository.
- Not an agent framework. There is no planner, no tool dispatch.
- Not a vector database. `FlatVectorIndex` is a correctness baseline; real
  ANN is delegated to whatever the user plugs in.
- Not a graph retrieval system. We treat semantic topology as a research
  topic, not an architectural commitment.

## Layering

```
┌──────────────────────────────────────────────────────────────┐
│  neorag-pipeline           NeoRAG facade + builder           │
└────────┬──────────┬──────────┬──────────┬───────────────────-┘
         │          │          │          │
         ▼          ▼          ▼          ▼
   ┌─────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────┐
   │chunking │ │retrieval  │ │reranking │ │diagnostics   │
   └────┬────┘ └─────┬─────┘ └─────┬────┘ └──────┬───────┘
        │            │             │             │
        └────────────▼─────────────▼─────────────▼
                     neorag-core (traits + types)
                                ▲
                                │
                  neorag-storage (ChunkStore, VectorIndex)
```

Every box above the `neorag-core` line depends on the trait surface
defined there. Crates do not depend on each other's implementations.

## Trait surface

`neorag-core` defines six pluggable abstractions:

| Trait                | Owns                                          |
| -------------------- | --------------------------------------------- |
| `TokenizerBackend`   | Token counting, sentence segmentation, truncation. |
| `Chunker`            | `Document → Vec<Chunk>`.                      |
| `EmbeddingProvider`  | `&[String] → Vec<Embedding>`.                 |
| `VectorIndex`        | Add + nearest-neighbor search over embeddings. |
| `Retriever`          | `Query → Vec<RetrievalResult>` + ingest.       |
| `Reranker`           | Reorder candidate `Vec<RetrievalResult>`.      |
| `DiagnosticsEngine`  | `(Query, &[RetrievalResult]) → DiagnosticsReport`. |

This is the entire contract a caller has to understand. The pipeline
facade composes these; downstream language bindings (`pyo3`, `napi-rs`)
will expose them by name.

## Data flow

```
Document(s)
    │  chunker.chunk_batch
    ▼
Vec<Chunk>                   ← optionally enriched with Embedding
    │  retriever.index
    ▼
[retriever state]
    │  retriever.retrieve(query, candidate_k)
    ▼
Vec<RetrievalResult>          ← carries score + ScoreBreakdown
    │  reranker.rerank (optional)
    ▼
Vec<RetrievalResult>          ← reordered, top_k items
    │  diagnostics.diagnose
    ▼
DiagnosticsReport
```

Hybrid retrieval fans out the query to several sub-retrievers in parallel
(`futures::join_all`) and fuses with **Reciprocal Rank Fusion** by default.
RRF is the right pick for heterogeneous score distributions because it is
rank-based and scale-free; weighted-sum fusion with per-list min-max
normalization is available when scores are commensurable.

## Why these design choices

### Embeddings are not in this repository

We do not bundle any model. Forcing the embedding model into the library
ties users to a single quality/latency/cost point and pulls in heavy
runtime dependencies (ONNX, candle, …). The `EmbeddingProvider` trait is
async and batch-friendly so any of those can plug in cleanly.

### Diagnostics are first-class

Retrieval failure modes are observable from text alone — you do not need
the LLM to know whether you served it a context full of distractors.
`neorag-diagnostics` computes six metrics on every query without any
model dependence. `DefaultDiagnosticsEngine` also emits *warnings* with
machine-readable codes (`low_lexical_grounding`,
`high_distractor_ratio`, `retrieval_saturated`) intended for monitoring,
alerting, and adaptive routing decisions.

### Chunking is core

Chunk boundaries determine evidence density and topical purity, both of
which dominate the diagnostics metrics that matter. `AdaptiveChunker`
exists as the long-term home for evidence-aware chunking: today it
combines sentence segmentation with a Jaccard cohesion gate; future work
adds topic-purity scoring, embedding-based cohesion, and entropy/surprisal
boundary detection (see roadmap in `crates/chunking/src/adaptive.rs`).

### Tantivy for BM25

Lexical retrieval is a solved problem with Tantivy: production-quality
analyzers, fast scoring, and an in-memory RAM directory for embeddable
use. We use it as a building block, not a foundation — `Bm25Retriever` is
just an implementation of `Retriever`.

### Flat ANN as the default

`FlatVectorIndex` does exact cosine over unit-normalized vectors. At a
few tens of thousands of vectors this is faster than the round-trip into
an external ANN library would be, and it is *correct* by construction.
Higher-throughput ANN (`usearch`, `hnswlib-rs`) plugs in by implementing
`VectorIndex`.

## Performance

NeoRAG uses `tokio` for I/O concurrency and `rayon` for CPU-bound batch
work (chunking, multi-document indexing). The hybrid retriever fans out
sub-retrievers in parallel by default. Tantivy indexing happens on
blocking workers via `tokio::task::spawn_blocking` so the runtime is not
starved.

The benchmark suite (`crates/benchmarks/benches/*.rs`) covers chunking
throughput and BM25 retrieval latency; run with `cargo bench`.

## Future bindings (Phase 6)

`Chunk`, `Document`, `Query`, `RetrievalResult`, and `DiagnosticsReport`
are all `Serialize + Deserialize`. The intent is that:

- `neorag-python-bindings` (`pyo3 + maturin`) exposes `NeoRAG`,
  `NeoRAGBuilder`, and the trait-shaped factory functions as a Python
  class hierarchy, with chunkers/retrievers selectable by string name.
- `neorag-node-bindings` (`napi-rs`) does the same for Node.

The data model already crosses FFI cleanly; the wrappers themselves are
mostly mechanical.

## What we explicitly avoided

- Fake-AI boundary detection in chunking. Adaptive chunking ships a
  conservative lexical-cohesion gate today and roadmaps the rest.
- Speculative topology systems, knowledge-graph retrieval, or
  semantic-continuity heuristics. Those are research, not infrastructure.
- LLM integrations. Once retrieval returns, NeoRAG is done; whatever
  comes after is the caller's problem.
