//! # RedHop
//!
//! A **reasoning-aware context runtime** for RAG. Hand it a document and a
//! question; it chunks, retrieves, and allocates the context the model should
//! actually see — and returns a **Decision Report** explaining what it kept,
//! what it dropped, and why. Plus citations back to the source. No vector
//! database, no LLM, all in-process.
//!
//! This crate is the high-level façade: it re-exports the [`Document`] surface
//! and the types you handle, over the focused `redhop-*` crates. The defaults are
//! evidence-backed (see the benchmarks/findings), so the short path just works.
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
//! - `semantic` — the bundled ONNX embedding backend (see [`embeddings`]) for the
//!   dense/hybrid retrieval tiers; inject it with [`Document::with_embedder`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// ── High-level surface ──────────────────────────────────────────────────────
pub use redhop_document::{Document, DocumentConfig, RetrievalMode, Section};

// The built context + its telemetry, and the lower-level context entry points.
pub use redhop_context::{
    analyze_context, build_context, context_economics, filter_context, grounding_score,
    link_strength, BuiltContext, ContextConfig, ContextReport, ContextStrategy,
};

// Core types you handle directly.
pub use redhop_core::{
    Chunk, ChunkId, Embedding, Error, Query, Result, RetrievalMethod, RetrievalResult, Score,
    ScoreBreakdown, TokenCount,
};

/// Pluggable abstractions for advanced use — custom retrievers, embedders,
/// chunkers, or tokenizers behind the same contracts RedHop uses internally.
pub mod traits {
    pub use redhop_core::{Chunker, EmbeddingProvider, Retriever, TokenizerBackend};
}

/// Built-in document parsers (PDF/DOCX/PPTX/XLSX + text/code/markdown). Requires
/// the `files` feature; this is what powers [`read_file`] / [`read_bytes`].
#[cfg(feature = "files")]
pub mod files {
    pub use redhop_files::{extract, extract_bytes, ExtractError, ExtractedDoc, Section};
}

/// The bundled ONNX embedding backend + model registry for the semantic/hybrid
/// tiers. Requires the `semantic` feature. Build an embedder and inject it via
/// [`Document::with_embedder`].
#[cfg(feature = "semantic")]
pub mod embeddings {
    pub use redhop_embeddings::*;
}

/// Parse a file on disk into a ready-to-query [`Document`] (default config):
/// text/code, Markdown, PDF, DOCX, PPTX, XLSX. The file path is tracked as each
/// chunk's source, with page/heading/line for citations. Requires the `files`
/// feature. For custom chunking/retrieval config, use [`files::extract`] +
/// [`Document::from_sources_with`].
#[cfg(feature = "files")]
pub fn read_file(path: impl AsRef<std::path::Path>) -> Result<Document> {
    let doc = redhop_files::extract(path).map_err(|e| Error::Other(e.to_string()))?;
    build_from_extracted(doc)
}

/// Parse already-in-memory bytes into a [`Document`] — the on-ramp for cloud
/// object storage (S3 / GCS / Azure Blob), HTTP downloads, or DB blobs. `name`
/// (e.g. `"contract.pdf"`) selects the parser by extension and becomes the
/// citation source. Requires the `files` feature.
#[cfg(feature = "files")]
pub fn read_bytes(data: &[u8], name: &str) -> Result<Document> {
    let doc = redhop_files::extract_bytes(data, name).map_err(|e| Error::Other(e.to_string()))?;
    build_from_extracted(doc)
}

#[cfg(feature = "files")]
fn build_from_extracted(doc: redhop_files::ExtractedDoc) -> Result<Document> {
    let sections: Vec<Section> = doc
        .sections
        .into_iter()
        .map(|s| Section {
            text: s.text,
            page: s.page,
            heading: s.heading,
            line: s.line,
        })
        .collect();
    Document::from_sources_with(vec![(doc.source, sections)], DocumentConfig::default())
}
