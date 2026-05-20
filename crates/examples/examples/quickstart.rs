//! Minimal end-to-end NeoRAG example.
//!
//! Run with: `cargo run -p neorag-examples --example quickstart`

use std::sync::Arc;

use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
use neorag_core::{Document, TokenizerBackend};
use neorag_pipeline::NeoRAG;
use neorag_retrieval::Bm25Retriever;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = Arc::new(SentenceChunker::new(tok, 80, 120, 0)?);
    let retriever = Arc::new(Bm25Retriever::new()?);

    let mut rag = NeoRAG::builder()
        .with_chunker(chunker)
        .with_retriever(retriever)
        .build()?;

    let docs = vec![
        Document::new(
            "rust-book",
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             It accomplishes these goals by being memory safe without using garbage collection.",
        ),
        Document::new(
            "tokio-docs",
            "Tokio is an asynchronous runtime for the Rust programming language. \
             It provides the building blocks needed for writing network applications. \
             It gives the flexibility to target a wide range of systems.",
        ),
        Document::new(
            "django-docs",
            "Django is a high-level Python web framework that encourages rapid development \
             and clean, pragmatic design.",
        ),
    ];

    rag.ingest(docs).await?;

    let query = "rust async runtime";
    let results = rag.retrieve(query, 3).await?;

    println!("Query: {query}");
    println!("Components: {:?}", rag.component_names());
    println!();
    println!("Top results:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  {}. [{:.3}] {}  ::  {}",
            i + 1,
            r.score.value,
            r.chunk.source,
            r.chunk.text.chars().take(80).collect::<String>()
        );
    }

    let report = rag.diagnose(&query.into(), &results)?;
    println!();
    println!("Diagnostics:");
    println!("  lexical_grounding:      {:?}", report.lexical_grounding);
    println!("  chunk_purity:           {:?}", report.chunk_purity);
    println!("  answer_density:         {:?}", report.answer_density);
    println!("  distractor_ratio:       {:?}", report.distractor_ratio);
    println!("  evidence_concentration: {:?}", report.evidence_concentration);
    println!("  retrieval_saturation:   {:?}", report.retrieval_saturation);
    println!("  retrieval_confidence:   {:?}", report.retrieval_confidence);
    for w in &report.warnings {
        println!("  ⚠ {} — {}", w.code, w.message);
    }

    Ok(())
}
