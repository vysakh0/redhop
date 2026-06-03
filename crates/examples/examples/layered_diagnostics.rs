//! Layered diagnostics demo.
//!
//! Builds a small corpus, embeds chunks with a deterministic topic-bucket
//! hasher (so the example is hermetic — no model dependency), retrieves with
//! BM25, and runs the *layered* diagnostics engine
//! (`DefaultDiagnosticsEngine` + `SemanticDiagnosticsEngine`).
//!
//! The corpus is constructed so two queries demonstrate the paraphrase
//! regime the semantic tier was designed to catch:
//!
//! - "feline kitten purrs" — no lexical overlap with corpus that uses
//!   "cat" / "tabby"; semantic tier catches it.
//! - "ancient roman aqueducts" — no overlap of any kind; both tiers
//!   correctly mark it as sparse.
//!
//! Run with:
//!     cargo run -p redhop-examples --example layered_diagnostics

use std::sync::Arc;

use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    Chunk, DiagnosticsEngine, Document, Embedding, Query, Retriever, TokenizerBackend,
};
use redhop::retrieval::Bm25Retriever;
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};

const DIM: usize = 128;

/// The same topic-bucket embedder used by the semantic-tier falsification test.
fn embed(text: &str) -> Embedding {
    const TOPIC_WEIGHT: f32 = 4.0;
    const NOISE_START: usize = 10;
    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "this", "that",
        "with", "as", "be", "by", "on", "at", "it",
    ];
    const TOPIC_FELINE: &[&str] = &[
        "cat", "cats", "kitten", "kittens", "feline", "felines", "purr", "purrs", "mews", "tabby",
    ];
    const TOPIC_RUNTIME: &[&str] = &[
        "tokio",
        "executor",
        "executors",
        "scheduler",
        "schedulers",
        "future",
        "futures",
        "async",
        "runtime",
        "runtimes",
        "await",
    ];
    const TOPIC_DATABASE: &[&str] = &[
        "postgres",
        "postgresql",
        "database",
        "databases",
        "sql",
        "transaction",
        "transactions",
        "acid",
        "row",
        "rows",
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

fn embed_chunks(mut chunks: Vec<Chunk>) -> Vec<Chunk> {
    for c in &mut chunks {
        c.embedding = Some(embed(&c.text));
    }
    chunks
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;

    let docs = vec![
        Document::new(
            "cats",
            "The tabby cat purrs and mews. Cats nap in the sun. Tabbies stalk mice and birds.",
        ),
        Document::new(
            "tokio",
            "Tokio is an async runtime. The Tokio executor schedules futures across worker threads.",
        ),
        Document::new(
            "postgres",
            "PostgreSQL provides ACID transactions. Postgres supports SQL and stores rows on disk.",
        ),
    ];

    let chunks = embed_chunks(redhop::core::Chunker::chunk_batch(&chunker, &docs)?);

    let mut bm25 = Bm25Retriever::new()?;
    bm25.index(&chunks).await?;

    // Layered diagnostics: lexical + semantic.
    let lexical: Arc<dyn DiagnosticsEngine> = Arc::new(DefaultDiagnosticsEngine::new());
    let semantic: Arc<dyn DiagnosticsEngine> = Arc::new(SemanticDiagnosticsEngine::new());
    let diagnostics = LayeredDiagnosticsEngine::lexical_and_semantic(lexical, semantic);

    for (label, text) in &[
        ("aligned", "tabby cat purrs"),
        ("paraphrase", "feline kitten purrs"),
        ("wrong_overlap", "cat purrs frequently and softly"),
        ("sparse", "ancient roman aqueducts"),
    ] {
        let mut query = Query::new(*text).with_embedding(embed(text));
        query.top_k = Some(3);

        // Inject embeddings on retrieved chunks too. (BM25 itself ignores
        // embeddings; the chunk metadata round-trips through the index, but
        // not the embedding, so we look them up from the original chunks.)
        let results = bm25.retrieve(&query, 3).await?;
        let results = attach_embeddings(results, &chunks);

        let report = diagnostics.diagnose(&query, &results)?;
        println!("================ {label}: {text} ================");
        for (i, r) in results.iter().enumerate() {
            println!(
                "  [{}] score={:.3} source={}  ::  {}",
                i + 1,
                r.score.value,
                r.chunk.source,
                r.chunk.text.chars().take(80).collect::<String>()
            );
        }
        println!("  -- lexical tier --");
        println!(
            "    lexical_grounding:        {:?}",
            report.lexical_grounding
        );
        println!("    chunk_purity:             {:?}", report.chunk_purity);
        println!("    answer_density:           {:?}", report.answer_density);
        println!(
            "    distractor_ratio:         {:?}",
            report.distractor_ratio
        );
        println!(
            "    retrieval_confidence:     {:?}",
            report.retrieval_confidence
        );
        println!("  -- semantic tier --");
        println!(
            "    semantic_grounding:       {:?}",
            report.semantic_grounding
        );
        println!(
            "    semantic_redundancy:      {:?}",
            report.semantic_redundancy
        );
        println!(
            "    centroid_dispersion:      {:?}",
            report.centroid_dispersion
        );
        println!(
            "    semantic_distractor_ratio:{:?}",
            report.semantic_distractor_ratio
        );
        if !report.warnings.is_empty() {
            println!("  warnings:");
            for w in &report.warnings {
                println!("    ⚠ {} — {}", w.code, w.message);
            }
        }
        println!();
    }

    Ok(())
}

fn attach_embeddings(
    mut results: Vec<redhop::core::RetrievalResult>,
    indexed: &[Chunk],
) -> Vec<redhop::core::RetrievalResult> {
    for r in &mut results {
        if let Some(c) = indexed.iter().find(|c| c.id == r.chunk.id) {
            r.chunk.embedding = c.embedding.clone();
        }
    }
    results
}
