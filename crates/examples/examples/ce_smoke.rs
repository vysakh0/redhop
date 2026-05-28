//! Smoke test: load the ms-marco cross-encoder and verify it scores
//! a relevant passage above an irrelevant one. Run-path verification
//! for OnnxCrossEncoder (analogous to the BGE bakeoff for the embedder).
//!
//! Run: cargo run -p redhop-examples --example ce_smoke --features onnx --release

use redhop::core::{
    Chunk, ChunkId, Query, Reranker, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenCount,
};
use redhop::reranking::OnnxCrossEncoder;

const CE_MODEL: &str =
    "/Users/vysakh/projects/neorag/models/ms-marco-MiniLM-L-6-v2/onnx/model.onnx";
const CE_TOKENIZER: &str =
    "/Users/vysakh/projects/neorag/models/ms-marco-MiniLM-L-6-v2/tokenizer.json";

fn cand(id: &str, text: &str) -> RetrievalResult {
    RetrievalResult {
        chunk: Chunk::new(
            ChunkId::new(id),
            text,
            "doc",
            TokenCount(text.split_whitespace().count()),
        ),
        score: Score {
            value: 1.0,
            method: RetrievalMethod::Dense,
        },
        breakdown: ScoreBreakdown::default(),
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let ce = OnnxCrossEncoder::load(CE_MODEL, CE_TOKENIZER, 256)?;
    let query = Query::new("What is the capital of France?");
    let candidates = vec![
        cand(
            "rel",
            "Paris is the capital and most populous city of France.",
        ),
        cand("irrel", "The mitochondrion is the powerhouse of the cell."),
        cand(
            "partial",
            "France is a country in Western Europe with many cities.",
        ),
    ];
    let ranked = ce.rerank(&query, candidates, 3).await?;
    println!("cross-encoder ranking (most relevant first):");
    for (i, r) in ranked.iter().enumerate() {
        println!(
            "  {}. [{:.3}] {}  ::  {}",
            i + 1,
            r.score.value,
            r.chunk.id,
            r.chunk.text
        );
    }
    assert_eq!(
        ranked[0].chunk.id.as_str(),
        "rel",
        "relevant passage should rank first"
    );
    println!("\nOK: cross-encoder run-path verified (relevant passage ranked #1).");
    Ok(())
}
