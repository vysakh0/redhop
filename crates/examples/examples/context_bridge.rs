//! JSON bridge: the entry point the Python `neorag` shim calls.
//!
//! Reads one request from stdin, runs `analyze_context` (before) and
//! `build_context` (after) through the public API, and writes the assembled
//! context text, the `ContextReport` telemetry, and a pretty rendering to
//! stdout. This is the minimal thing that makes `neorag.build_context(...)`
//! work from Python today; pyo3 bindings are the future packaging path.
//!
//! Request (stdin):
//!   {
//!     "query": "...",
//!     "chunks": [{"id":"c1","text":"...","source":"doc","token_count":12,
//!                 "embedding":[...]}, ...],   // source/token_count/embedding optional
//!     "token_budget": 12000,
//!     "strategy": "reasoning_preserving",     // raw_topk|distractor_filtered|
//!                                             // redundancy_pruned|max_density|
//!                                             // reasoning_preserving
//!     "distractor_min_grounding": 0.10,       // optional
//!     "link_min_jaccard": 0.12,               // optional
//!     "redundancy_max_cosine": 0.92,          // optional
//!     "mode": "build"                          // "build" | "analyze"; default build
//!   }
//!
//! Response (stdout): { "text": "...", "report": {...}, "rendered": "..." }
//!
//! Run:  echo '<json>' | cargo run -q -p neorag-examples --example context_bridge --release

use std::io::Read;

use neorag_context::{analyze_context, build_context, ContextConfig, ContextStrategy};
use neorag_core::{Chunk, ChunkId, Embedding, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown, TokenCount};
use serde::Deserialize;

#[derive(Deserialize)]
struct ChunkIn {
    id: String,
    text: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    token_count: Option<usize>,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct Request {
    query: String,
    chunks: Vec<ChunkIn>,
    #[serde(default = "default_budget")]
    token_budget: usize,
    #[serde(default = "default_strategy")]
    strategy: String,
    #[serde(default = "default_dmg")]
    distractor_min_grounding: f32,
    #[serde(default = "default_link")]
    link_min_jaccard: f32,
    #[serde(default = "default_redundancy")]
    redundancy_max_cosine: f32,
    #[serde(default = "default_mode")]
    mode: String,
}

fn default_budget() -> usize { 8192 }
fn default_strategy() -> String { "reasoning_preserving".into() }
fn default_dmg() -> f32 { 0.10 }
fn default_link() -> f32 { 0.12 }
fn default_redundancy() -> f32 { 0.92 }
fn default_mode() -> String { "build".into() }

fn parse_strategy(s: &str) -> anyhow::Result<ContextStrategy> {
    Ok(match s {
        "raw_topk" => ContextStrategy::RawTopK,
        "distractor_filtered" => ContextStrategy::DistractorFiltered,
        "redundancy_pruned" => ContextStrategy::RedundancyPruned,
        "max_density" => ContextStrategy::MaxDensity,
        "reasoning_preserving" => ContextStrategy::ReasoningPreserving,
        other => anyhow::bail!("unknown strategy: {other}"),
    })
}

fn main() -> anyhow::Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let req: Request = serde_json::from_str(&input)?;

    let cfg = ContextConfig {
        token_budget: req.token_budget,
        strategy: parse_strategy(&req.strategy)?,
        distractor_min_grounding: req.distractor_min_grounding,
        link_min_jaccard: req.link_min_jaccard,
        redundancy_max_cosine: req.redundancy_max_cosine,
    };

    let query = Query::new(&req.query);
    let retrieved: Vec<RetrievalResult> = req
        .chunks
        .into_iter()
        .map(|c| {
            let tok = c.token_count.unwrap_or_else(|| c.text.split_whitespace().count().max(1));
            let mut chunk = Chunk::new(
                ChunkId::new(c.id),
                c.text,
                c.source.unwrap_or_else(|| "input".into()),
                TokenCount(tok),
            );
            if let Some(e) = c.embedding {
                chunk = chunk.with_embedding(Embedding::from(e));
            }
            RetrievalResult {
                chunk,
                score: Score { value: 1.0, method: RetrievalMethod::Dense },
                breakdown: ScoreBreakdown::default(),
            }
        })
        .collect();

    // Always compute the "before" view so the rendering can show deltas.
    let before = analyze_context(&query, &retrieved, &cfg);

    let out = if req.mode == "analyze" {
        serde_json::json!({
            "text": "",
            "report": before,
            "rendered": before.render(None),
        })
    } else {
        let ctx = build_context(&query, &retrieved, &cfg);
        serde_json::json!({
            "text": ctx.text(),
            "report": ctx.report,
            "rendered": ctx.report.render(Some(&before)),
        })
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
