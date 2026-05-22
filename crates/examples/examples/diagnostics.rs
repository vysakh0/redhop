//! Demonstrates the diagnostics engine on a deliberately-bad retrieval, then
//! a clean one, side by side. Useful as a smoke test that the warnings are
//! firing for the right reasons.
//!
//! Run with: `cargo run -p redhop-examples --example diagnostics`

use std::sync::Arc;

use redhop_chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop_core::{Document, Query, TokenizerBackend};
use redhop_pipeline::RedHop;
use redhop_retrieval::Bm25Retriever;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = Arc::new(SentenceChunker::new(tok, 40, 60, 0)?);
    let retriever = Arc::new(Bm25Retriever::new()?);
    let mut rag = RedHop::builder()
        .with_chunker(chunker)
        .with_retriever(retriever)
        .build()?;

    rag.ingest(vec![
        Document::new("a", "cats nap in the sun. cats stalk mice. cats purr."),
        Document::new("b", "dogs run. dogs bark. dogs fetch."),
        Document::new("c", "the weather today is sunny and warm."),
        Document::new("d", "breakfast was great this morning."),
    ])
    .await?;

    // Query for something that *isn't* in the corpus — expect warnings.
    let q_bad = Query::new("rust async runtime");
    let r_bad = rag.retrieve(q_bad.clone(), 4).await?;
    let d_bad = rag.diagnose(&q_bad, &r_bad)?;
    println!("--- bad query ---");
    print_report(&d_bad);

    // Query for something that IS in the corpus.
    let q_good = Query::new("cats nap mice");
    let r_good = rag.retrieve(q_good.clone(), 3).await?;
    let d_good = rag.diagnose(&q_good, &r_good)?;
    println!("--- good query ---");
    print_report(&d_good);

    Ok(())
}

fn print_report(r: &redhop_core::DiagnosticsReport) {
    println!("  lexical_grounding:      {:?}", r.lexical_grounding);
    println!("  chunk_purity:           {:?}", r.chunk_purity);
    println!("  answer_density:         {:?}", r.answer_density);
    println!("  distractor_ratio:       {:?}", r.distractor_ratio);
    println!("  evidence_concentration: {:?}", r.evidence_concentration);
    println!("  retrieval_saturation:   {:?}", r.retrieval_saturation);
    println!("  retrieval_confidence:   {:?}", r.retrieval_confidence);
    for w in &r.warnings {
        println!("  ⚠ {} — {}", w.code, w.message);
    }
    println!();
}
