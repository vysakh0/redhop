//! Phase 7 demo: regime classification with full audit trace.
//!
//! Runs four queries against a small corpus through the layered diagnostics
//! and the rule-based regime classifier. For each query it prints the full
//! `ClassificationTrace` — features, thresholds, rules fired, and the
//! per-regime probability distribution — so you can see exactly *why* the
//! classifier reached its verdict.
//!
//! Run with:
//!     cargo run -p redhop-examples --example regime_classification

use std::sync::Arc;

use redhop_chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop_core::{
    Chunk, DiagnosticsEngine, Document, Embedding, Query, RegimeClassifier, RetrievalRegime,
    TokenizerBackend,
};
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use redhop_orchestration::RuleBasedClassifier;
use redhop_pipeline::RedHop;
use redhop_retrieval::Bm25Retriever;

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
    let chunks = embed_chunks(redhop_core::Chunker::chunk_batch(&chunker, &docs)?);

    let mut bm25 = Bm25Retriever::new()?;
    redhop_core::Retriever::index(&mut bm25, &chunks).await?;

    let lexical: Arc<dyn DiagnosticsEngine> = Arc::new(DefaultDiagnosticsEngine::new());
    let semantic: Arc<dyn DiagnosticsEngine> = Arc::new(SemanticDiagnosticsEngine::new());
    let diagnostics = Arc::new(LayeredDiagnosticsEngine::lexical_and_semantic(
        lexical, semantic,
    ));
    let classifier: Arc<dyn RegimeClassifier> = Arc::new(RuleBasedClassifier::new());

    let rag = RedHop::builder()
        .with_chunker(Arc::new(chunker))
        .with_retriever(Arc::new(bm25))
        .with_diagnostics(diagnostics)
        .with_classifier(classifier.clone())
        .build()?;

    let queries = [
        ("aligned", "tokio async runtime"),
        ("paraphrase", "feline kitten purrs"),
        ("wrong_overlap", "cat purrs frequently and softly"),
        ("sparse", "ancient roman aqueducts"),
    ];

    for (label, text) in queries {
        let query = Query::new(text).with_embedding(embed(text));
        let mut state = rag.retrieve_with_state(query, 4).await?;
        // Re-attach embeddings for the semantic tier (BM25 strips them).
        for r in &mut state.candidates {
            if let Some(c) = chunks.iter().find(|c| c.id == r.chunk.id) {
                r.chunk.embedding = c.embedding.clone();
            }
        }
        // Reclassify against the embedding-bearing chunks.
        let diag = diagnostics_for(&state.candidates, &state.query)?;
        let conf = redhop_orchestration::compute_confidence(&state.candidates);
        let dist = classifier.classify(&diag, &conf);

        println!("================ {label}: {text} ================");
        println!(
            "argmax = {}  (distribution entropy = {:.3} nats)",
            dist.argmax,
            dist.entropy()
        );
        println!("probabilities:");
        for r in RetrievalRegime::all() {
            println!("  {:<18} {:.3}", r.code(), dist.p(*r));
        }
        println!("rules fired:");
        for f in &dist.trace.rules_fired {
            println!(
                "  [{:<28}] → {:<18} weight={:.2}",
                f.rule, f.regime, f.weight
            );
            println!("      {}", f.justification);
        }
        println!("raw per-regime scores (pre-softmax):");
        for (r, s) in &dist.trace.raw_scores {
            println!("  {:<18} {:.3}", r.code(), s);
        }
        println!();
    }

    Ok(())
}

fn diagnostics_for(
    candidates: &[redhop_core::RetrievalResult],
    query: &Query,
) -> anyhow::Result<redhop_core::DiagnosticsReport> {
    let lexical = DefaultDiagnosticsEngine::new();
    let semantic = SemanticDiagnosticsEngine::new();
    let l = lexical.diagnose(query, candidates)?;
    let s = semantic.diagnose(query, candidates)?;
    Ok(l.merge(s))
}
