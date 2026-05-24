//! Using the opt-in semantic tier: `Document` with `RetrievalMode::DenseRerank`.
//!
//! The default `Document` is BM25 (zero deps). For semantic-heavy queries you
//! inject an embedder and switch the mode — BM25 prunes to a candidate pool, a
//! dense model reorders only that pool (no ANN). The ONNX dependency lives here,
//! at the construction site, not in the library.
//!
//! Run:  cargo run -p redhop-examples --example document_rerank --features onnx --release

use std::sync::Arc;

use redhop_core::EmbeddingProvider;
use redhop_document::{Document, DocumentConfig, RetrievalMode};
use redhop_embeddings::{EmbedderConfig, OnnxEmbedder};

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
        retrieval_mode: RetrievalMode::DenseRerank { candidate_pool: 50 },
        ..Default::default()
    };
    let mut doc = Document::from_text_with("hr.txt", TEXT, cfg)?.with_embedder(embedder);

    // This is an API-usage demo. The query shares the term "employee" with two
    // chunks (the termination and the budget "review"); the dense stage reorders
    // the lexical pool toward the termination chunk. The *measured* semantic-recall
    // benefit (dense reordering a large BM25 pool) is in docs/findings/LOCAL_RERANK.md;
    // note dense rerank only helps when BM25's pool already contains the chunk —
    // pure zero-overlap synonyms need a global dense index, by design.
    let ctx = doc.context(QUERY)?;
    println!("query: {QUERY}\n--- assembled context ---\n{}", ctx.text());
    Ok(())
}
