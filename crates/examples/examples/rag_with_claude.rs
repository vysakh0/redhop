//! End-to-end RedHop demo that hands its retrieved evidence to a real LLM
//! via the local `claude` CLI.
//!
//! Run with:
//!     cargo run -p redhop-examples --example rag_with_claude
//!     cargo run -p redhop-examples --example rag_with_claude -- "your question here"
//!
//! RedHop is responsible for chunking + retrieval + reranking + diagnostics.
//! The LLM only sees the assembled prompt; it has no other access to the
//! corpus, which is exactly the contract RedHop was designed around.

use std::process::Command;
use std::sync::Arc;

use redhop_chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop_core::{Document, Query, TokenizerBackend};
use redhop_pipeline::RedHop;
use redhop_reranking::LexicalGroundingReranker;
use redhop_retrieval::Bm25Retriever;

fn corpus() -> Vec<Document> {
    vec![
        Document::new(
            "rust-lang",
            "Rust is a systems programming language focused on safety, speed, and \
             concurrency. It achieves memory safety without garbage collection by \
             enforcing ownership and borrowing rules at compile time. Rust was \
             originally designed by Graydon Hoare at Mozilla Research starting in 2006.",
        ),
        Document::new(
            "tokio",
            "Tokio is an asynchronous runtime for the Rust programming language. \
             It provides the building blocks needed for writing network applications, \
             from clients and servers to scalable production systems. Tokio uses a \
             work-stealing scheduler to distribute tasks across worker threads.",
        ),
        Document::new(
            "tantivy",
            "Tantivy is a full-text search engine library written in Rust. It is \
             inspired by Apache Lucene and is designed to be embeddable. Tantivy uses \
             BM25 scoring by default and supports faceted search, range queries, and \
             approximate phrase queries.",
        ),
        Document::new(
            "ownership",
            "Rust's ownership system is the feature that most distinguishes it from \
             other systems languages. Every value has a single owner; when the owner \
             goes out of scope, the value is dropped. References can borrow values \
             either immutably or mutably, but never both simultaneously.",
        ),
        Document::new(
            "async-await",
            "Async functions in Rust return values that implement the Future trait. \
             These futures are inert until polled by an executor. The async/await \
             syntax was stabilized in Rust 1.39, in November 2019, after several years \
             of community design work.",
        ),
        Document::new(
            "django",
            "Django is a high-level Python web framework that encourages rapid \
             development and clean, pragmatic design. It follows the model-template-views \
             architectural pattern and includes an ORM, an authentication system, and \
             an admin interface out of the box.",
        ),
        Document::new(
            "postgres",
            "PostgreSQL is a powerful, open-source object-relational database system \
             with over 35 years of active development. It supports both SQL and JSON \
             querying, has strong ACID semantics, and uses MVCC for concurrent \
             transactions without read locks.",
        ),
    ]
}

fn build_prompt(query: &str, results: &[redhop_core::RetrievalResult]) -> String {
    let mut s = String::new();
    s.push_str(
        "You will answer a question using ONLY the evidence chunks below. \
         If the evidence does not contain the answer, say so plainly. \
         Cite chunks by their [source] tag. Keep the answer under three sentences.\n\n",
    );
    s.push_str("=== EVIDENCE ===\n");
    for (i, r) in results.iter().enumerate() {
        s.push_str(&format!(
            "[{}] source={} score={:.3}\n{}\n\n",
            i + 1,
            r.chunk.source,
            r.score.value,
            r.chunk.text
        ));
    }
    s.push_str("=== QUESTION ===\n");
    s.push_str(query);
    s
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let query = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "When was async/await stabilized in Rust?".to_string());

    // Build the pipeline: sentence chunker + BM25 + lexical-grounding rerank.
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = Arc::new(SentenceChunker::new(tok, 60, 90, 0)?);
    let retriever = Arc::new(Bm25Retriever::new()?);
    let reranker = Arc::new(LexicalGroundingReranker::default());

    let mut rag = RedHop::builder()
        .with_chunker(chunker)
        .with_retriever(retriever)
        .with_reranker(reranker)
        .with_candidate_k(16)
        .build()?;

    rag.ingest(corpus()).await?;

    let q = Query::new(&query);
    let results = rag.retrieve(q.clone(), 4).await?;
    let report = rag.diagnose(&q, &results)?;

    println!("================ RedHop ================");
    println!("query:      {query}");
    println!("components: {:?}", rag.component_names());
    println!();
    println!("retrieved evidence ({} chunks):", results.len());
    for (i, r) in results.iter().enumerate() {
        println!(
            "  [{}] score={:.3} source={}  ::  {}",
            i + 1,
            r.score.value,
            r.chunk.source,
            r.chunk.text.chars().take(90).collect::<String>()
        );
    }
    println!();
    println!("diagnostics:");
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

    // Hand the prompt to the LLM.
    let prompt = build_prompt(&query, &results);
    println!();
    println!("================ Claude ================");
    let out = Command::new("claude")
        .args(["-p", &prompt, "--model", "haiku"])
        .output()?;
    if !out.status.success() {
        eprintln!("claude exited with status {}", out.status);
        eprintln!("stderr:\n{}", String::from_utf8_lossy(&out.stderr));
        std::process::exit(1);
    }
    print!("{}", String::from_utf8_lossy(&out.stdout));
    Ok(())
}
