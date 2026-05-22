//! Input JSON schemas and conversions shared by the CLI subcommands.
//!
//! The retrieval input is intentionally simple — the JSON your retriever
//! already produces:
//!
//! ```json
//! {
//!   "query": "…",                       // optional; --query overrides
//!   "chunks": [
//!     {"id": "c1", "text": "…",
//!      "source": "doc.pdf", "token_count": 42, "embedding": [..]}
//!   ]
//! }
//! ```
//! Only `text` is required per chunk.

use anyhow::Context as _;
use redhop_context::ContextStrategy;
use redhop_core::{
    Chunk, ChunkId, Embedding, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown, TokenCount,
};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkInput {
    #[serde(default)]
    pub id: Option<String>,
    pub text: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub token_count: Option<usize>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RetrievalInput {
    #[serde(default)]
    pub query: Option<String>,
    pub chunks: Vec<ChunkInput>,
}

impl RetrievalInput {
    /// Load from a file path, or from stdin if `path` is `-`.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let raw = if path == "-" {
            use std::io::Read;
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        } else {
            std::fs::read_to_string(Path::new(path))
                .with_context(|| format!("reading {path}"))?
        };
        serde_json::from_str(&raw).with_context(|| format!("parsing {path} as retrieval JSON"))
    }

    pub fn to_results(&self) -> Vec<RetrievalResult> {
        self.chunks
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let tok = c
                    .token_count
                    .unwrap_or_else(|| c.text.split_whitespace().count().max(1));
                let mut chunk = Chunk::new(
                    ChunkId::new(c.id.clone().unwrap_or_else(|| format!("c{i}"))),
                    c.text.clone(),
                    c.source.clone().unwrap_or_else(|| "input".into()),
                    TokenCount(tok),
                );
                if let Some(e) = &c.embedding {
                    chunk = chunk.with_embedding(Embedding::from(e.clone()));
                }
                RetrievalResult {
                    chunk,
                    score: Score { value: 1.0, method: RetrievalMethod::Dense },
                    breakdown: ScoreBreakdown::default(),
                }
            })
            .collect()
    }
}

/// Parse a strategy name (the same spellings the Python shim/bridge use).
pub fn parse_strategy(s: &str) -> anyhow::Result<ContextStrategy> {
    Ok(match s.trim() {
        "raw_topk" => ContextStrategy::RawTopK,
        "distractor_filtered" => ContextStrategy::DistractorFiltered,
        "redundancy_pruned" => ContextStrategy::RedundancyPruned,
        "max_density" => ContextStrategy::MaxDensity,
        "reasoning_preserving" => ContextStrategy::ReasoningPreserving,
        other => anyhow::bail!(
            "unknown strategy '{other}' (expected: raw_topk, distractor_filtered, \
             redundancy_pruned, max_density, reasoning_preserving)"
        ),
    })
}
