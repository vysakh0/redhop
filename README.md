# RedHop

**Reasoning-preserving context optimization and retrieval observability for RAG systems.**

RedHop is *not* an agent framework, a workflow engine, a vector database, a
graph runtime, or a universal retrieval orchestrator. It is a Rust library that
sits **between retrieval and generation**: you hand it the chunks your retriever
returned and a token budget, and it assembles the prompt context — pruning
distractors, **preserving reasoning-critical bridge evidence**, and reporting
exactly what it did. The choice of LLM, embedding model, and vector store stays
entirely with the caller.

## Philosophy

RedHop does not optimize generic retrieval scores. It optimizes
**reasoning-preserving context allocation under finite context budgets.**

Many conventional relevance optimizations — aggressive distractor
filtering, cross-encoder reranking, max-density pruning, "more neighbors"
expansion — were *empirically found to damage multi-hop reasoning* by
removing low-relevance **bridge evidence** (the second hop, which is
relevant to the bridge entity, not the query). The recurring measurement:

> Transformers tolerate irrelevant context far better than they tolerate
> missing reasoning links. Premature removal of low-relevance reasoning
> evidence hurts more than the distractors do.

So RedHop's APIs and defaults are grounded in **measured reasoning-failure
geometry**, not generic retrieval assumptions. It still optimizes
answer-bearing evidence density, lexical grounding, and distractor
suppression — but never at the cost of dropping reasoning-critical bridge
evidence, and it makes retrieval failure modes observable as first-class
output.

## Benchmark philosophy & the evidence layer

Every default and API exists because a specific failure was measured. That
record is permanent and reproducible in the **[evidence layer](docs/findings/README.md)**:

- **[docs/findings/](docs/findings/README.md)** — hypothesis → result → mechanism for each finding, with a falsified-hypotheses registry (we *keep* the priors that failed; several of RedHop's strongest defaults came from one).
- **[benchmarks/](benchmarks/README.md)** — the reproducible harnesses that regenerate every claim.
- **[reports/](reports/README.md)** — captured raw outputs (e.g. the n=300 causal reasoning-preservation run).

The discipline — measure aggressively, let hypotheses fail, extract the
real mechanism — is itself part of the design.

## What's in the box

| Crate                | Purpose                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| `redhop-core`        | Traits and data types (`Chunker`, `Retriever`, `Reranker`, …) and error types. |
| `redhop-chunking`    | `FixedChunker`, `SentenceChunker`, `AdaptiveChunker` + a default tokenizer. |
| `redhop-retrieval`   | `Bm25Retriever` (Tantivy), `DenseRetriever`, `HybridRetriever` + RRF / weighted-sum fusion. |
| `redhop-reranking`   | `ScoreFusionReranker`, `LexicalGroundingReranker`, `EvidenceDensityReranker`. |
| `redhop-diagnostics` | Six retrieval-quality metrics + `DefaultDiagnosticsEngine` with configurable warnings. |
| `redhop-storage`     | `ChunkStore` and `FlatVectorIndex` (exact-cosine baseline; ANN is pluggable via `VectorIndex`). |
| `redhop-context`     | Finite-attention context assembly: `build_context` + strategies (incl. `ReasoningPreserving`, which resists the [second-hop tax](docs/findings/SECOND_HOP_TAX.md) and beats aggressive filtering [end-to-end](docs/findings/REASONING_PRESERVATION.md)) + economics readout. |
| `redhop-cli`         | Thin eval/observability CLI (`redhop compare` / `analyze-context` / `benchmark` / `report`). See [crates/cli](crates/cli/README.md). |
| `redhop-pipeline`    | `RedHop` facade + builder composing the above.                          |
| `redhop-benchmarks`  | Criterion benchmarks.                                                   |
| `redhop-examples`    | Runnable examples.                                                      |

## Quick start

```rust
use std::sync::Arc;
use redhop_chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop_core::{Document, TokenizerBackend};
use redhop_pipeline::RedHop;
use redhop_retrieval::Bm25Retriever;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = Arc::new(SentenceChunker::new(tok, 80, 120, 0)?);
    let retriever = Arc::new(Bm25Retriever::new()?);

    let mut rag = RedHop::builder()
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
cargo run -p redhop-examples --example quickstart
cargo run -p redhop-examples --example diagnostics
```

## Diagnostics

RedHop ships six metrics, each in `[0, 1]`:

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
or remote retriever, implement the relevant trait from `redhop-core` and pass
your type to the builder. Nothing in the library is closed.

## Roadmap

- **Phase 6 — language bindings.** `pyo3 + maturin` and `napi-rs` wrappers
  over the same trait surface. The data types in `redhop-core` are all
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
