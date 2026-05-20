# NeoRAG

**Adaptive evidence retrieval infrastructure for LLM applications.**

NeoRAG is *not* an LLM framework, agent framework, vector database, or
graph-retrieval engine. It is a Rust library that chunks, retrieves, reranks,
and **diagnoses** retrieval — leaving the choice of LLM, embedding model, and
vector store entirely to the caller.

## Philosophy

Transformer QA quality primarily depends on:

- answer-bearing **evidence density**,
- **lexical grounding** between query and evidence,
- **distractor suppression** in the retrieved set,

and *not* on retrieval topology, semantic continuity, or graph trajectories.
NeoRAG is designed around that observation. It optimizes evidence quality,
makes retrieval failure modes observable, and treats diagnostics as
first-class output rather than an afterthought.

## What's in the box

| Crate                | Purpose                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| `neorag-core`        | Traits and data types (`Chunker`, `Retriever`, `Reranker`, …) and error types. |
| `neorag-chunking`    | `FixedChunker`, `SentenceChunker`, `AdaptiveChunker` + a default tokenizer. |
| `neorag-retrieval`   | `Bm25Retriever` (Tantivy), `DenseRetriever`, `HybridRetriever` + RRF / weighted-sum fusion. |
| `neorag-reranking`   | `ScoreFusionReranker`, `LexicalGroundingReranker`, `EvidenceDensityReranker`. |
| `neorag-diagnostics` | Six retrieval-quality metrics + `DefaultDiagnosticsEngine` with configurable warnings. |
| `neorag-storage`     | `ChunkStore` and `FlatVectorIndex` (exact-cosine baseline; ANN is pluggable via `VectorIndex`). |
| `neorag-pipeline`    | `NeoRAG` facade + builder composing the above.                          |
| `neorag-benchmarks`  | Criterion benchmarks.                                                   |
| `neorag-examples`    | Runnable examples.                                                      |

## Quick start

```rust
use std::sync::Arc;
use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
use neorag_core::{Document, TokenizerBackend};
use neorag_pipeline::NeoRAG;
use neorag_retrieval::Bm25Retriever;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = Arc::new(SentenceChunker::new(tok, 80, 120, 0)?);
    let retriever = Arc::new(Bm25Retriever::new()?);

    let mut rag = NeoRAG::builder()
        .with_chunker(chunker)
        .with_retriever(retriever)
        .build()?;

    rag.ingest(vec![Document::new("doc1", "Tokio is an async runtime for Rust.")]).await?;

    let results = rag.retrieve("rust async runtime", 5).await?;
    let report = rag.diagnose(&"rust async runtime".into(), &results)?;
    println!("confidence: {:?}", report.retrieval_confidence);
    Ok(())
}
```

Two runnable examples ship with the repo:

```
cargo run -p neorag-examples --example quickstart
cargo run -p neorag-examples --example diagnostics
```

## Diagnostics

NeoRAG ships six metrics, each in `[0, 1]`:

- **`lexical_grounding`** — average query/chunk term overlap. Predicts reader hallucination.
- **`chunk_purity`** — intra-chunk sentence cohesion. Diagnoses bad chunker boundaries.
- **`answer_density`** — fraction of retrieved tokens matching query terms.
- **`distractor_ratio`** — fraction of results below a per-chunk grounding cutoff *(lower is better)*.
- **`retrieval_saturation`** — tail/head term overlap. `1.0` means more `top_k` won't help.
- **`evidence_concentration`** — score peakedness. `1.0` is a clean single peak.
- **`retrieval_confidence`** — composite scalar summary.

`DefaultDiagnosticsEngine` also emits machine-readable **warnings** with
configurable thresholds (`low_lexical_grounding`, `high_distractor_ratio`,
`retrieval_saturated`).

## Extensibility

Every subsystem is a trait. To plug in a custom embedding model, ANN index,
or remote retriever, implement the relevant trait from `neorag-core` and pass
your type to the builder. Nothing in the library is closed.

## Roadmap

- **Phase 6 — language bindings.** `pyo3 + maturin` and `napi-rs` wrappers
  over the same trait surface. The data types in `neorag-core` are all
  `Serialize + Deserialize` precisely to keep the FFI boundary cheap.
- **Phase 7 — adaptive chunking enrichment.** Topic-purity scoring,
  embedding-based cohesion gating, entropy/surprisal boundary detection.
  Behind feature flags so the default chunker doesn't regress.
- **Phase 8 — high-throughput ANN.** `usearch` / `hnswlib-rs` backends behind
  `VectorIndex`. The flat baseline stays as a correctness reference.
- **Phase 9 — cross-encoder reranking.** Re-uses the existing `Reranker`
  trait; only the implementation crate is new.

See `docs/ARCHITECTURE.md` for a deeper architectural tour.

## License

Apache-2.0.
