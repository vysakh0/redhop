//! Tier-2 performance/memory hardening: how the `Document` runtime scales with
//! document size. Concatenates CUAD contracts into one large document at
//! increasing sizes and measures, locally and deterministically:
//!
//!   - chunk time      (Document::from_text — lazy chunking)
//!   - index time      (first context() call — lazy in-memory BM25 build)
//!   - steady query latency (subsequent context() calls), p50 / p95
//!   - chunk count and total tokens at each size
//!
//! Peak memory: run the whole sweep under the OS profiler to capture RSS, e.g.
//!   /usr/bin/time -l cargo run -p redhop-examples --example eval_document_scaling --release
//! (look for "maximum resident set size"; dominated by the largest document).
//!
//! Validates the "local-first, lightweight, large-document" claim. No vector
//! infra, no network, no LLM.
//!
//! Run:  cargo run -p redhop-examples --example eval_document_scaling --release

use std::time::Instant;

use redhop_document::Document;
use serde::Deserialize;

#[derive(Deserialize)]
struct Cuad {
    data: Vec<Contract>,
}
#[derive(Deserialize)]
struct Contract {
    paragraphs: Vec<Paragraph>,
}
#[derive(Deserialize)]
struct Paragraph {
    context: String,
    qas: Vec<Qa>,
}
#[derive(Deserialize)]
struct Qa {
    question: String,
}

fn p(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)]
}

fn main() -> anyhow::Result<()> {
    let path = std::env::var("REDHOP_CUAD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("cuad/cuad_sample.json"));
    let cuad: Cuad = serde_json::from_str(&std::fs::read_to_string(&path)?)?;

    let contexts: Vec<&String> = cuad
        .data
        .iter()
        .flat_map(|c| c.paragraphs.iter().map(|p| &p.context))
        .collect();
    let queries: Vec<String> = cuad
        .data
        .iter()
        .flat_map(|c| {
            c.paragraphs
                .iter()
                .flat_map(|p| p.qas.iter().map(|q| q.question.clone()))
        })
        .take(25)
        .collect();

    let sizes = [1usize, 5, 15, 30, contexts.len().min(50)];
    println!("Document scaling (local, no LLM) — {}", path.display());
    println!(
        "  pooled contracts available: {}   queries/size: {}\n",
        contexts.len(),
        queries.len()
    );
    println!(
        "  {:>9}  {:>8}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}",
        "contracts", "tokens", "chunks", "chunk_ms", "index_ms", "q_p50_ms", "q_p95_ms"
    );

    for &n in &sizes {
        let text = contexts
            .iter()
            .take(n)
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let t0 = Instant::now();
        let mut doc = Document::from_text("scale", &text)?;
        let chunk_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let total_tokens = doc.total_tokens();
        let n_chunks = doc.len();

        // First query forces the lazy index build.
        let ti = Instant::now();
        let _ = doc.context(&queries[0])?;
        let index_ms = ti.elapsed().as_secs_f64() * 1000.0;

        // Steady-state query latency over the rest.
        let mut lat: Vec<f64> = Vec::new();
        for q in queries.iter().skip(1) {
            let t = Instant::now();
            let _ = doc.context(q)?;
            lat.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!(
            "  {:>9}  {:>8}  {:>7}  {:>9.1}  {:>9.1}  {:>9.2}  {:>9.2}",
            n,
            total_tokens,
            n_chunks,
            chunk_ms,
            index_ms,
            p(&lat, 0.5),
            p(&lat, 0.95)
        );
    }
    println!("\n(For peak RSS: run this under `/usr/bin/time -l`.)");
    Ok(())
}
