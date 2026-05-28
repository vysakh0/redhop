//! # RedHop
//!
//! A **reasoning-aware context runtime** for RAG. Hand it a document and a
//! question; it chunks, retrieves, and allocates the context the model should
//! actually see — and returns a **Decision Report** explaining what it kept,
//! what it dropped, and why. Plus citations back to the source. No vector
//! database, no LLM, all in-process.
//!
//! This is the published Rust crate — every public surface lives here.
//! Internally it is organized as modules ([`core`], [`chunking`],
//! [`retrieval`], [`context`], [`document`], optionally [`embeddings`],
//! [`reranking`], [`files`]); the most-used types are re-exported at the
//! crate root so the short path just works.
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
//! # #[cfg(feature = "files")]
//! # fn main() -> redhop::Result<()> {
//! let mut doc = redhop::read_file("contract.pdf")?;
//! let ctx = doc.context("What is the governing law?")?;
//! for c in &ctx.chunks {
//!     // c.source / c.metadata["page"|"heading"|"line"] → cite the evidence
//! }
//! # Ok(()) }
//! ```
//!
//! ## Feature flags
//!
//! - `files` — built-in parsers + [`read_file`]/[`read_bytes`].
//! - `semantic` — the bundled ONNX embedding backend ([`embeddings`]) and
//!   cross-encoder reranker ([`reranking`]) for dense/hybrid retrieval;
//!   inject the embedder with [`Document::with_embedder`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// ── Modules (the consolidated workspace; each was its own crate pre-0.2) ────
pub mod chunking;
pub mod context;
pub mod core;
pub mod document;
pub mod retrieval;
pub mod storage;

#[cfg(feature = "files")]
pub mod files;
#[cfg(feature = "semantic")]
pub mod embeddings;
#[cfg(feature = "semantic")]
pub mod reranking;

// ── High-level surface re-exports ───────────────────────────────────────────
pub use crate::document::{Document, DocumentConfig, RetrievalMode, Section};

// The built context + its telemetry, and the lower-level context entry points.
pub use crate::context::{
    analyze_context, build_context, context_economics, filter_context, grounding_score,
    link_strength, AutoDecision, BuiltContext, ContextConfig, ContextReport, ContextStrategy,
};

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
pub use load::{chunks, citations, text, Citation, FolderOptions, LoadOptions};
#[cfg(feature = "files")]
pub use load::{
    read_bytes, read_bytes_with, read_file, read_file_with, read_folder, read_folder_with,
};
