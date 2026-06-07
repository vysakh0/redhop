//! # RedHop
//!
//! A **reasoning-preserving context runtime** for RAG. Hand it a document and a
//! question; it chunks, retrieves, and allocates the context the model should
//! actually see — and returns a **Decision Report** explaining what it kept,
//! what it dropped, and why. Plus citations back to the source. No vector
//! database, no LLM, all in-process.
//!
//! This is the published Rust crate — every public surface lives here.
//! Internally it is organized as modules ([`core`], [`chunking`],
//! [`retrieval`], [`context`], [`document`]; plus `embeddings`, `reranking`,
//! `files` under their feature flags); the most-used types are re-exported
//! at the crate root so the short path just works.
//!
//! ```no_run
//! # fn main() -> redhop::Result<()> {
//! let mut doc = redhop::Document::from_text("doc", "…long document text…")?;
//! let ctx = doc.context("Why did the proposed method fail?")?;
//! let _prompt = ctx.text();   // feed to any LLM provider — no lock-in
//! let _report = &ctx.report;  // what was retrieved/pruned, and why
//! # Ok(()) }
//! ```
//!
//! ## Loading documents
//!
//! With the `files` feature, parse a file straight to a [`Document`] (PDF, DOCX,
//! PPTX, XLSX, or text/code) — chunked, indexed, with per-chunk citations:
//!
//! ```no_run
//! # fn main() -> redhop::Result<()> {
//! # #[cfg(feature = "files")]
//! # {
//! let mut doc = redhop::read_file("contract.pdf")?;
//! let ctx = doc.context("What is the governing law?")?;
//! for c in &ctx.chunks {
//!     // c.source / c.metadata["page"|"heading"|"line"] → cite the evidence
//! }
//! # }
//! # Ok(()) }
//! ```
//!
//! ## Feature flags
//!
//! - `files` — built-in parsers + `read_file`/`read_bytes`.
//! - `semantic` — the bundled ONNX embedding backend (`embeddings` module)
//!   and cross-encoder reranker (`reranking` module) for dense/hybrid
//!   retrieval; inject the embedder with [`Document::with_embedder`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// ── Modules (the consolidated workspace; each was its own crate pre-0.2) ────
pub mod analyzer;
pub mod chunking;
pub mod context;
pub mod core;
pub mod document;
pub mod retrieval;
pub mod rewrite;
pub mod storage;

#[cfg(feature = "semantic")]
pub mod embeddings;
#[cfg(feature = "files")]
pub mod files;
#[cfg(feature = "semantic")]
pub mod reranking;

// ── High-level surface re-exports ───────────────────────────────────────────
pub use crate::document::{Document, DocumentConfig, RetrievalMode, Section};

// Query-side diagnostics for templated workloads. Grounded in
// docs/findings/CUAD_RECALL_GAP.md (the mechanism), docs/findings/CUAD_PRF_NULL.md
// (what doesn't help), and docs/findings/QUERY_SET_ANALYZER.md (the
// cross-workload probe that validated the heuristic thresholds).
pub use crate::analyzer::{
    analyze_query_set, DilutionCost, QuerySetReport,
};

// Query-side rewrite primitives. Compile boilerplate / glossaries once,
// route every rewrite through the same `QueryRewrite` seam, and land the
// per-stage trail in `ContextReport::query_rewrites` so every change is
// auditable. Replaces the old `drop_template_terms` / `expand_query_terms`
// functions from 0.2.x (deleted in 0.3.0 — see the redesign notes in
// `crate::rewrite` for the three flaws that motivated the move).
pub use crate::rewrite::{
    apply_chain as apply_query_rewrites, Vocabulary, QueryRewrite, RewriteRecord,
    RewriteResult, Stripper,
};

// The built context + its telemetry, and the lower-level context entry points.
pub use crate::context::{
    analyze_context, build_context, context_economics, filter_context, grounding_score,
    link_strength, AutoDecision, BuiltContext, ContextConfig, ContextReport, ContextStrategy,
};
// In-process evaluation — score a `BuiltContext` against optional gold signals,
// no LLM judge required. See `crate::context::eval` for the rationale.
pub use crate::context::eval::{evaluate, EvalGold, EvalReport};

// Core types you handle directly.
pub use crate::core::{
    Chunk, ChunkId, Embedding, Error, Query, Result, RetrievalMethod, RetrievalResult, Score,
    ScoreBreakdown, TokenCount,
};

/// Pluggable abstractions for advanced use — custom retrievers, embedders,
/// chunkers, or tokenizers behind the same contracts RedHop uses internally.
pub mod traits {
    pub use crate::core::{Chunker, EmbeddingProvider, Retriever, TokenizerBackend};
}

mod load;
pub use load::{
    chunks, citations, retrieval_from_str, strategy_from_str, text, Citation, FolderOptions,
    LoadOptions,
};
#[cfg(feature = "files")]
pub use load::{
    read_bytes, read_bytes_with, read_file, read_file_with, read_folder, read_folder_with,
};
