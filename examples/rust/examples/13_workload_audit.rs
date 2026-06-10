//! 13 · Workload audit — point RedHop's diagnostics at your existing pipeline.
//!
//! Real-world scenario:
//!   A team already has a retrieval pipeline (their own Tantivy BM25
//!   over contracts, in this sketch). They are not ready to migrate.
//!   They want to know, across their last few hundred production
//!   queries, *why* retrieval sometimes fails, and which single knob
//!   the data says to reach for first.
//!
//! What this demonstrates:
//!   - The bring-your-own-retrieval (BYO) loop: caller-supplied
//!     chunks via `redhop::analyze_context(query, chunks)`. RedHop
//!     never owns the retriever; it observes what the retriever
//!     returned.
//!   - Workload-level aggregation via `redhop::summarize_diagnoses`.
//!     One focus recommendation per workload, with a finding citation.
//!   - Layer 1 (BYO) vs Layer 2 (full corpus diagnosis via
//!     `Document::from_chunks`).
//!
//! Run:
//!   cargo run --example 13_workload_audit

use redhop::{
    analyze_context, summarize_diagnoses, Chunk, ChunkId, ContextConfig, Document, Query,
    RetrievalMethod, RetrievalResult, Score, TokenCount,
};

const CORPUS: &[&str] = &[
    "Refund Policy. Refunds are available within thirty days of purchase.",
    "Termination for convenience. Either party may terminate this agreement.",
    "Governing Law. This agreement is governed by the laws of California.",
    "Limitation of Liability. The cap is twelve months of fees.",
    "Confidentiality. Each party shall keep the other party's information confidential.",
];

/// Stand-in for "your existing retriever". Word overlap, top-k.
fn external_search(query: &str, k: usize) -> Vec<&'static str> {
    let q_terms: std::collections::HashSet<String> =
        query.to_lowercase().split_whitespace().map(String::from).collect();
    let mut scored: Vec<(i32, &'static str)> = CORPUS
        .iter()
        .map(|text| {
            let score = text
                .to_lowercase()
                .split_whitespace()
                .filter(|w| q_terms.contains(*w))
                .count() as i32;
            (score, *text)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.into_iter().take(k).map(|(_, t)| t).collect()
}

fn build_queries() -> Vec<&'static str> {
    let mut q = Vec::new();
    for _ in 0..6 {
        q.extend_from_slice(&[
            "how do I cancel and get my money back",
            "when can I quit this contract",
            "what is the cap on damages",
            "who keeps secrets",
        ]);
    }
    for _ in 0..4 {
        q.extend_from_slice(&[
            "refund policy",
            "termination for convenience",
            "governing law",
            "limitation of liability cap",
        ]);
    }
    q
}

fn main() -> redhop::Result<()> {
    let queries = build_queries();
    let cfg = ContextConfig::default();

    // ── Layer 1: BYO retrieval ────────────────────────────────────────
    let mut layer1_reports = Vec::new();
    for (qi, q) in queries.iter().enumerate() {
        let texts = external_search(q, 3);
        let retrieved: Vec<RetrievalResult> = texts
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                let chunk = Chunk::new(
                    ChunkId::new(format!("{}-{}", qi, i)),
                    t,
                    "external",
                    TokenCount(t.split_whitespace().count()),
                );
                RetrievalResult::new(
                    chunk,
                    Score {
                        value: 1.0,
                        method: RetrievalMethod::External,
                    },
                )
            })
            .collect();
        let q = Query::new(*q);
        layer1_reports.push(analyze_context(&q, &retrieved, &cfg));
    }
    println!("── Layer 1: observe what your retriever returned ──");
    println!("{}", summarize_diagnoses(&layer1_reports).render());

    // ── Layer 2: also point RedHop at the same corpus, once ──────────
    let corpus_chunks: Vec<Chunk> = CORPUS
        .iter()
        .enumerate()
        .map(|(i, t)| {
            Chunk::new(
                ChunkId::new(i.to_string()),
                *t,
                "corpus",
                TokenCount(t.split_whitespace().count()),
            )
        })
        .collect();
    let mut doc = Document::from_chunks(corpus_chunks)?;
    let mut layer2_reports = Vec::new();
    for q in &queries {
        layer2_reports.push(doc.context(q)?.report);
    }
    println!("\n── Layer 2: same queries against an in-memory corpus index ──");
    println!("{}", summarize_diagnoses(&layer2_reports).render());

    Ok(())
}
