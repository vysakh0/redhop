# RedHop Architecture

## What RedHop is

RedHop is a Rust library for **retrieval infrastructure**: chunking,
retrieval, reranking, and diagnostics. It does not generate text, embed
text, or store vectors persistently. Those concerns are pushed to the
caller through trait boundaries. The library's contribution is the
*orchestration* layer between them and the *diagnostics* engine that makes
retrieval quality observable.

## What RedHop is not

- Not an LLM framework. There is no model in this repository.
- Not an agent framework. There is no planner, no tool dispatch.
- Not a vector database. `FlatVectorIndex` is a correctness baseline. Real
  ANN is delegated to whatever the user plugs in.
- Not a graph retrieval system. We treat semantic topology as a research
  topic, not an architectural commitment.

## Layering

The published Rust crate is `redhop`. It is one consolidated crate
organized as modules. Sibling crates handle the things that don't
belong inside the published surface (the adaptive controller, the
research/calibration tooling, the bindings).

```
                ┌─────────────────────────────────────────┐
                │  redhop  (the published Rust crate)     │
                │                                          │
                │   document   context   analyzer   files  │ ←- public API
                │   chunking   retrieval reranking embeddings│ ←- semantic-/files-gated
                │   storage    core (traits + types)        │
                └─────────────────────────────────────────┘
                              ▲           ▲
                              │           │
                ┌─────────────┴─────┐ ┌───┴──────────────────────────┐
                │ redhop-pipeline   │ │ redhop-py    (pyo3 bindings) │
                │ redhop-diagnostics│ │ redhop-node  (napi bindings) │
                │ redhop-orchestration│└──────────────────────────────┘
                │ redhop-observability│
                │ redhop-calibration │ ←- research / eval / metrics
                │ redhop-benchmarks  │
                │ redhop-cli         │ ←- thin CLI
                │ redhop-examples    │ ←- finding-reproduction
                └────────────────────┘
```

Inside `redhop`, every non-`core` module depends only on the trait
surface defined in `core`. Modules do not depend on each other's
implementations. The sibling crates depend on `redhop` and use only its
public surface (no `pub(crate)` reach-ins).

## Trait surface

`redhop::core` defines six pluggable abstractions:

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
facade composes these. The language bindings (`pyo3`, `napi-rs`) expose
them by name.

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
rank-based and scale-free. Weighted-sum fusion with per-list min-max
normalization is available when scores are commensurable.

## Why these design choices

### Embeddings are not in this repository

We do not bundle any model. Forcing the embedding model into the library
ties users to a single quality/latency/cost point and pulls in heavy
runtime dependencies (ONNX, candle, …). The `EmbeddingProvider` trait is
async and batch-friendly so any of those can plug in cleanly.

### Diagnostics are first-class

Retrieval failure modes are observable from text alone: the LLM is not
needed to know whether a context is full of distractors.
`redhop-diagnostics` computes six metrics on every query without any
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
use. We use it as a building block, not a foundation: `Bm25Retriever` is
just an implementation of `Retriever`.

### Flat ANN as the default

`FlatVectorIndex` does exact cosine over unit-normalized vectors. At a
few tens of thousands of vectors this is faster than the round-trip into
an external ANN library would be, and it is *correct* by construction.
Higher-throughput ANN (`usearch`, `hnswlib-rs`) plugs in by implementing
`VectorIndex`.

## Performance

RedHop uses `tokio` for I/O concurrency and `rayon` for CPU-bound batch
work (chunking, multi-document indexing). The hybrid retriever fans out
sub-retrievers in parallel by default. Tantivy indexing happens on
blocking workers via `tokio::task::spawn_blocking` so the runtime is not
starved.

The benchmark suite (`crates/benchmarks/benches/*.rs`) covers chunking
throughput and BM25 retrieval latency. Run with `cargo bench`.

## Language bindings

`Chunk`, `Document`, `Query`, `RetrievalResult`, and `DiagnosticsReport`
are all `Serialize + Deserialize`, which makes them cross FFI cleanly:

- **`redhop-py`** (`pyo3 + maturin`) ships the Python wheel. It exposes
  `Document.from_text` / `from_chunks` / `from_file` / `from_bytes` /
  `from_folder` (with `language=`, `strategy=`, `retrieval=` kwargs),
  `Document.context` / `analyze` / `n_files` / `skipped_files`, plus
  the top-level `build_context` / `filter_context` / `analyze_context`
  / `context_economics` / `grounding_score` / `link_strength`.
- **`redhop-node`** (`napi-rs`) ships the npm package. Mirrors the
  Python surface: `Document.fromText` / `fromChunks` / `fromFile` /
  `fromBytes` / `fromFolder` (with `language` option),
  `Document.context` / `analyze` / `chunkCount` / `nFiles` /
  `skippedFiles`, plus top-level `groundingScore` / `linkStrength`.

The bindings wrap the consolidated `redhop` crate directly (no
parallel implementations).

## What we explicitly avoided

- Fake-AI boundary detection in chunking. Adaptive chunking ships a
  conservative lexical-cohesion gate today and roadmaps the rest.
- Speculative topology systems, knowledge-graph retrieval, or
  semantic-continuity heuristics. Those are research, not infrastructure.
- LLM integrations. Once retrieval returns, RedHop is done. Whatever
  comes after is the caller's problem.
