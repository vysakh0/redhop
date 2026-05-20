//! Phase 8 demo: the closed adaptive loop, end to end.
//!
//! Runs four queries (aligned / paraphrase / wrong_overlap / sparse) through
//! the adaptive orchestrator and prints the full action history for each.
//! The example demonstrates the three Phase 8 falsification claims live:
//!
//!   1. Easy queries take exactly one terminal Stop action and zero
//!      retrieval mutation.
//!   2. DistractorHeavy queries escalate the reranker once and the action
//!      record carries a measurable actual_gain.
//!   3. Sparse queries Abstain immediately, so a downstream LLM is told
//!      "evidence insufficient" rather than fed hallucination fuel.
//!
//! Run with:
//!     cargo run -p neorag-examples --example adaptive_loop

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
use neorag_core::{
    Chunk, ChunkId, DiagnosticsEngine, Document, Embedding, Query, RerankerLevel,
    Result as CoreResult, RetrievalResult, Retriever, TokenizerBackend,
};
use neorag_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use neorag_orchestration::RuleBasedClassifier;
use neorag_pipeline::NeoRAG;
use neorag_reranking::LexicalGroundingReranker;
use neorag_retrieval::Bm25Retriever;

/// A retriever wrapper that re-attaches embeddings after the base
/// retriever's call. BM25 indexes do not persist arbitrary binary blobs,
/// so the embedding field on a [`Chunk`] is lost when retrieved through
/// Tantivy. For workflows where the semantic-tier diagnostics matter,
/// the caller keeps a side cache and rehydrates the chunk after retrieval.
/// This pattern is what production users will end up implementing; we
/// demonstrate it here as a small wrapper rather than embedding it in
/// the BM25 crate.
struct EmbedAttachingRetriever {
    inner: Arc<dyn Retriever>,
    by_id: HashMap<ChunkId, Embedding>,
}
impl EmbedAttachingRetriever {
    fn new(inner: Arc<dyn Retriever>, chunks: &[Chunk]) -> Self {
        let by_id = chunks
            .iter()
            .filter_map(|c| c.embedding.clone().map(|e| (c.id.clone(), e)))
            .collect();
        Self { inner, by_id }
    }
}
#[async_trait]
impl Retriever for EmbedAttachingRetriever {
    async fn retrieve(&self, q: &Query, top_k: usize) -> CoreResult<Vec<RetrievalResult>> {
        let mut results = self.inner.retrieve(q, top_k).await?;
        for r in &mut results {
            if r.chunk.embedding.is_none() {
                if let Some(e) = self.by_id.get(&r.chunk.id) {
                    r.chunk.embedding = Some(e.clone());
                }
            }
        }
        Ok(results)
    }
    async fn index(&mut self, _c: &[Chunk]) -> CoreResult<()> {
        // The inner retriever owns indexing; this wrapper only re-attaches
        // embeddings on the way out.
        Ok(())
    }
    fn name(&self) -> &'static str {
        "embed_attaching"
    }
}

const DIM: usize = 128;

fn embed(text: &str) -> Embedding {
    const TOPIC_WEIGHT: f32 = 4.0;
    const NOISE_START: usize = 10;
    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
        "this", "that", "with", "as", "be", "by", "on", "at", "it",
    ];
    const TOPIC_FELINE: &[&str] = &[
        "cat", "cats", "kitten", "kittens", "feline", "felines", "purr", "purrs",
        "mews", "tabby",
    ];
    const TOPIC_RUNTIME: &[&str] = &[
        "tokio", "executor", "executors", "scheduler", "schedulers", "future",
        "futures", "async", "runtime", "runtimes", "await",
    ];
    const TOPIC_DATABASE: &[&str] = &[
        "postgres", "postgresql", "database", "databases", "sql", "transaction",
        "transactions", "acid", "row", "rows",
    ];

    fn hash_word(w: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in w.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    let mut v = vec![0f32; DIM];
    for w in text.split(|c: char| !c.is_alphanumeric()) {
        let w = w.to_lowercase();
        if w.is_empty() || STOPWORDS.contains(&w.as_str()) {
            continue;
        }
        if TOPIC_FELINE.contains(&w.as_str()) {
            v[0] += TOPIC_WEIGHT;
        } else if TOPIC_RUNTIME.contains(&w.as_str()) {
            v[1] += TOPIC_WEIGHT;
        } else if TOPIC_DATABASE.contains(&w.as_str()) {
            v[2] += TOPIC_WEIGHT;
        } else {
            let slot = NOISE_START + (hash_word(&w) as usize) % (DIM - NOISE_START);
            v[slot] += 1.0;
        }
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    Embedding(v)
}

fn embed_chunks(chunks: Vec<Chunk>) -> Vec<Chunk> {
    chunks
        .into_iter()
        .map(|c| {
            let e = embed(&c.text);
            c.with_embedding(e)
        })
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;

    let docs = vec![
        Document::new(
            "cats",
            "The tabby cat purrs and mews. Cats nap in the sun. Tabbies stalk mice.",
        ),
        Document::new(
            "tokio",
            "Tokio is an async runtime. The Tokio executor schedules futures across workers.",
        ),
        Document::new(
            "postgres",
            "PostgreSQL provides ACID transactions. Postgres supports SQL and stores rows on disk.",
        ),
    ];
    let chunks = embed_chunks(neorag_core::Chunker::chunk_batch(&chunker, &docs)?);

    let mut bm25 = Bm25Retriever::new()?;
    neorag_core::Retriever::index(&mut bm25, &chunks).await?;
    let retriever: Arc<dyn Retriever> =
        Arc::new(EmbedAttachingRetriever::new(Arc::new(bm25), &chunks));

    let lexical: Arc<dyn DiagnosticsEngine> = Arc::new(DefaultDiagnosticsEngine::new());
    let semantic: Arc<dyn DiagnosticsEngine> = Arc::new(SemanticDiagnosticsEngine::new());
    let diagnostics = Arc::new(LayeredDiagnosticsEngine::lexical_and_semantic(
        lexical, semantic,
    ));

    let rag = NeoRAG::builder()
        .with_chunker(Arc::new(chunker))
        .with_retriever(retriever)
        .with_diagnostics(diagnostics)
        .with_classifier(Arc::new(RuleBasedClassifier::new()))
        .with_reranker_at(
            RerankerLevel::Lexical,
            Arc::new(LexicalGroundingReranker::default()),
        )
        .build()?;

    let queries = [
        ("aligned       ", "tokio async runtime"),
        ("paraphrase    ", "feline kitten purrs"),
        ("wrong_overlap ", "cat purrs frequently and softly"),
        ("sparse        ", "ancient roman aqueducts"),
    ];

    println!("Phase 8 adaptive loop demo — the controller is CONSERVATIVE by design.");
    println!("Most queries will result in zero retrieval mutation; that is the point.");
    println!("See crates/orchestration/tests/adaptive_falsification.rs for the cases");
    println!("where strong signal makes the controller intervene.\n");

    for (label, text) in queries {
        let query = Query::new(text).with_embedding(embed(text));
        let state = rag.adaptive_run(query).await?;

        let intervened = state
            .history
            .iter()
            .any(|t| !t.action.is_terminal());

        println!("================ {label} :: {text} ================");
        println!(
            "regime           = {}  p={:.2}",
            state
                .regime
                .as_ref()
                .map(|r| r.argmax.code())
                .unwrap_or("none"),
            state
                .regime
                .as_ref()
                .map(|r| r.p(r.argmax))
                .unwrap_or(0.0)
        );
        println!("iterations       = {}", state.iteration);
        println!("intervened       = {}", intervened);
        println!("abstained        = {}", state.abstained());
        println!(
            "final candidates = {}  (top_k = {})",
            state.candidates.len(),
            state.current_top_k
        );
        println!("reranker level   = {}", state.reranker_level);
        println!("history:");
        for (i, t) in state.history.iter().enumerate() {
            print!(
                "  [{i}] iter={} action={:<18}",
                t.iteration,
                t.action.code()
            );
            print!(
                " expected={:.3} actual={:>7} latency_ms={}",
                t.expected_gain,
                t.actual_gain
                    .map(|g| format!("{g:+.3}"))
                    .unwrap_or_else(|| "n/a".into()),
                t.latency_ms
            );
            print!(
                " cost=(retr={}, rerank={}, Δchunks={:+})",
                t.cost.retrieval_calls, t.cost.rerank_calls, t.cost.chunks_delta
            );
            println!();
            println!("       rationale: {}", t.rationale);
        }
        println!();
    }
    Ok(())
}
