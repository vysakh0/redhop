//! Using the opt-in semantic tier: `Document` with `RetrievalMode::Dense`.
//!
//! The default `Document` is BM25 (zero deps). For semantic-heavy queries you
//! inject an embedder and switch to `Dense` — the model cosines the query
//! against *every* chunk (exact brute force, no ANN), so a paraphrase answer
//! that shares no terms with the query is still reachable. The ONNX dependency
//! lives here, at the construction site, not in the library.
//!
//! Run:  cargo run -p redhop-examples --example document_dense --features onnx --release

use std::sync::Arc;

use redhop::core::EmbeddingProvider;
use redhop::document::{Document, DocumentConfig, RetrievalMode};
use redhop::embeddings::{EmbedderConfig, OnnxEmbedder};

const DIM: usize = 384;
const DEFAULT_MODEL: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/onnx/model.onnx";
const DEFAULT_TOKENIZER: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/tokenizer.json";

const TEXT: &str = "The employee was terminated for cause and a severance review followed. \
    The annual budget review was approved by the board after a long discussion. \
    The cafeteria introduced a new vegetarian menu on Fridays. \
    Quarterly revenue rose twelve percent year over year.";

const QUERY: &str = "why did the employee leave the company?";

fn main() -> anyhow::Result<()> {
    let model = std::env::var("REDHOP_BGE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let tokenizer =
        std::env::var("REDHOP_BGE_TOKENIZER").unwrap_or_else(|_| DEFAULT_TOKENIZER.into());
    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &model,
        &tokenizer,
        EmbedderConfig::bge(DIM),
    )?);

    let cfg = DocumentConfig {
        target_tokens: 16,
        max_tokens: 32,
        retrieval_mode: RetrievalMode::Dense,
        ..Default::default()
    };
    let mut doc = Document::from_text_with("hr.txt", TEXT, cfg)?.with_embedder(embedder);

    // "leave the company" shares no terms with "terminated for cause", so BM25
    // alone would miss it. Global dense cosines every chunk and surfaces the
    // termination chunk anyway. The measured benefit is in
    // docs/findings/GLOBAL_DENSE.md (synonym-mismatch recall@1 20%→88%).
    let ctx = doc.context(QUERY)?;
    println!("query: {QUERY}\n--- assembled context ---\n{}", ctx.text());
    Ok(())
}
