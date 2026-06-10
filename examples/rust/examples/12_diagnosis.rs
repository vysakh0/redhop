//! 12 · Diagnosis — when retrieval looks weak, the Decision Report tells you why.
//!
//! Real-world scenario:
//!   A support team is wiring up Q&A over a policy doc. A user asks
//!   "how long do I have to cancel and get my money back?" and gets
//!   an empty answer. The doc uses *refund* and *termination*, not
//!   *cancel* and *money back*, so BM25 has nothing to match.
//!
//! What this demonstrates:
//!   - `report.diagnosis` populated on every `context()` call.
//!   - Layer-2 facts: `query_terms`, `zero_match_terms`, `term_stats`
//!     computed against the corpus vocabulary.
//!   - The closed hints registry: one bounded hint per documented
//!     failure shape, each citing the finding that justifies it.
//!   - A healthy query produces zero hints.
//!
//! Run:
//!   cargo run --example 12_diagnosis

use redhop::{Chunk, ChunkId, Document, HintCode, TokenCount};

fn corpus() -> Vec<Chunk> {
    vec![
        Chunk::new(
            ChunkId::new("a"),
            "Refund Policy. Refunds are available within thirty days of purchase.",
            "policy.md",
            TokenCount(11),
        ),
        Chunk::new(
            ChunkId::new("b"),
            "Termination for convenience. Either party may terminate this agreement.",
            "policy.md",
            TokenCount(10),
        ),
        Chunk::new(
            ChunkId::new("c"),
            "Governing Law. This agreement is governed by the laws of California.",
            "policy.md",
            TokenCount(11),
        ),
    ]
}

fn main() -> redhop::Result<()> {
    let mut doc = Document::from_chunks(corpus())?;

    // 1. Healthy query: facts populated, no hints.
    let healthy = doc.context("refund policy thirty days")?;
    let d = &healthy.report.diagnosis;
    println!("Healthy query:");
    println!("  query_terms            = {:?}", d.query_terms);
    println!("  corpus_stats_available = {}", d.corpus_stats_available);
    println!("  zero_match_terms       = {:?}", d.zero_match_terms);
    println!("  hints                  = {}", d.hints.len());
    println!();

    // 2. Vocabulary-mismatch query: VocabMismatch hint fires.
    let paraphrase = doc.context("How long do I have to cancel and get my money back?")?;
    let d = &paraphrase.report.diagnosis;
    println!("Vocabulary-mismatch query:");
    println!("  query_terms            = {:?}", d.query_terms);
    println!("  zero_match_terms       = {:?}", d.zero_match_terms);
    println!("  empty_context          = {}", d.empty_context);
    for hint in &d.hints {
        let label = match hint.code {
            HintCode::EmptyContext => "empty_context",
            HintCode::VocabMismatch => "vocab_mismatch",
            HintCode::LowConfidence => "low_confidence",
            HintCode::LowDiscriminationQuery => "low_discrimination_query",
            HintCode::UnderdeterminedQuery => "underdetermined_query",
            _ => "unknown",
        };
        println!("  hint {:?}", label);
        println!("    evidence  : {}", hint.evidence);
        println!("    message   : {}", hint.message);
    }
    println!();

    // 3. Rendered report (the same data, human-readable).
    let rendered = paraphrase.report.render(None);
    if let Some(idx) = rendered.find("Query diagnosis") {
        println!("Rendered report (excerpt):");
        println!("{}", &rendered[idx..]);
    }

    Ok(())
}
