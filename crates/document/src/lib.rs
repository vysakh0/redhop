//! # redhop-document
//!
//! The **reasoning-aware document context runtime** — RedHop's high-level
//! surface. You have documents and need reasoning; you should not have to
//! think about retrievers, vector stores, query engines, or ANN infrastructure.
//!
//! ```no_run
//! # fn demo() -> redhop_core::Result<()> {
//! use redhop_document::Document;
//!
//! let mut doc = Document::from_text("report.txt", "…long document text…")?;
//! let ctx = doc.context("Why did the proposed method fail?")?;
//! // feed `ctx.text()` to any LLM provider yourself — no lock-in.
//! let _prompt = ctx.text();
//! let _report = &ctx.report;  // what was retrieved/pruned and why
//! # Ok(()) }
//! ```
//!
//! ## What this layer owns (and what it does not)
//!
//! It **owns**: chunking, internal indexing/retrieval, context allocation,
//! reasoning-safe optimization, observability, and token economics. It
//! **does not own** document parsing/OCR (bring your own text — PyMuPDF,
//! Marker, Unstructured, …) and it is **not** an orchestration / agent /
//! workflow runtime.
//!
//! Internally it retrieves (BM25 today), but retrieval is an implementation
//! detail: the user mental model is *documents + reasoning*, not *retrieval
//! infrastructure*. This crate is pure layering over [`redhop_chunking`],
//! [`redhop_retrieval`], and [`redhop_context`] — no new logic, no new
//! architecture tower.
//!
//! The default context policy is [`ContextStrategy::Auto`]: do nothing under
//! headroom, prune under dilution, preserve bridge evidence, and report every
//! decision. See `docs/findings/CONTEXT_DILUTION.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::Arc;

use redhop_chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop_context::{
    analyze_context, build_context, BuiltContext, ContextConfig, ContextReport, ContextStrategy,
};
use redhop_core::{
    Chunk, Chunker, Document as SourceDoc, Query, Result, RetrievalResult, Retriever,
    TokenizerBackend,
};
use redhop_retrieval::Bm25Retriever;
use tokio::runtime::{Builder, Runtime};

/// Tuning for a [`Document`]'s internal chunking, retrieval, and context
/// policy. Every field has an evidence-backed default; users reason about
/// documents, not knobs.
#[derive(Debug, Clone)]
pub struct DocumentConfig {
    /// Target tokens per chunk (sentence-budgeted chunking).
    pub target_tokens: usize,
    /// Hard cap on tokens per chunk.
    pub max_tokens: usize,
    /// Sentences of overlap between adjacent chunks.
    pub overlap_sentences: usize,
    /// How many candidate chunks to retrieve before context assembly.
    pub candidate_k: usize,
    /// Context-assembly policy. Defaults to [`ContextStrategy::Auto`].
    pub context: ContextConfig,
}

impl Default for DocumentConfig {
    fn default() -> Self {
        Self {
            target_tokens: 256,
            max_tokens: 512,
            overlap_sentences: 1,
            candidate_k: 20,
            // The runtime's philosophy: size-gated, conservative, observable.
            context: ContextConfig { strategy: ContextStrategy::Auto, ..Default::default() },
        }
    }
}

/// A document you reason over. Holds its chunks and a lazily-built internal
/// index; `context()` and `analyze()` retrieve candidates and hand them to the
/// context runtime. Retrieval is an internal detail — never surfaced.
pub struct Document {
    chunks: Vec<Chunk>,
    cfg: DocumentConfig,
    rt: Runtime,
    // Lazily built on first query so construction is cheap (goal: lazy
    // chunk/index init). `Bm25Retriever` is in-memory (Tantivy RAM index).
    retriever: Option<Bm25Retriever>,
}

impl Document {
    /// Build a document from raw text, chunking it with the default policy.
    /// Bring your own parser/OCR — this layer takes text, not PDFs.
    pub fn from_text(source: impl Into<String>, text: impl Into<String>) -> Result<Self> {
        Self::from_text_with(source, text, DocumentConfig::default())
    }

    /// Build from raw text with an explicit [`DocumentConfig`].
    pub fn from_text_with(
        source: impl Into<String>,
        text: impl Into<String>,
        cfg: DocumentConfig,
    ) -> Result<Self> {
        let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
        let chunker =
            SentenceChunker::new(tok, cfg.target_tokens, cfg.max_tokens, cfg.overlap_sentences)?;
        let chunks = chunker.chunk(&SourceDoc::new(source, text))?;
        Self::from_chunks_with(chunks, cfg)
    }

    /// Build from chunks you already produced (your own chunker/parser).
    pub fn from_chunks(chunks: Vec<Chunk>) -> Result<Self> {
        Self::from_chunks_with(chunks, DocumentConfig::default())
    }

    /// Build from chunks with an explicit [`DocumentConfig`].
    pub fn from_chunks_with(chunks: Vec<Chunk>, cfg: DocumentConfig) -> Result<Self> {
        if chunks.is_empty() {
            return Err(redhop_core::Error::InvalidConfig(
                "cannot build a Document with no chunks — the text was empty or produced no \
                 chunks. Pass non-empty text to `from_text`, or chunks to `from_chunks`."
                    .into(),
            ));
        }
        // A current-thread runtime is enough: the internal retriever's work is
        // CPU-bound (Tantivy on a blocking worker); we only block_on it.
        let rt = Builder::new_current_thread().build()?;
        Ok(Self { chunks, cfg, rt, retriever: None })
    }

    /// Number of chunks the document holds.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the document has no chunks.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Assemble the reasoning context for a query: retrieve candidates from
    /// the internal index, then allocate them under the context policy.
    /// Returns the prompt context plus a [`ContextReport`] of what it did.
    pub fn context(&mut self, query: &str) -> Result<BuiltContext> {
        let results = self.retrieve(query, self.cfg.candidate_k)?;
        Ok(build_context(&Query::new(query), &results, &self.cfg.context))
    }

    /// Like [`Document::context`] but overriding how many candidates to
    /// retrieve before assembly.
    pub fn context_with_k(&mut self, query: &str, candidate_k: usize) -> Result<BuiltContext> {
        let results = self.retrieve(query, candidate_k)?;
        Ok(build_context(&Query::new(query), &results, &self.cfg.context))
    }

    /// Diagnose the retrieval for a query **without** modifying anything:
    /// distractor load, evidence density, second-hop candidates, and (for the
    /// Auto policy) whether it would prune. Pure observability.
    pub fn analyze(&mut self, query: &str) -> Result<ContextReport> {
        let results = self.retrieve(query, self.cfg.candidate_k)?;
        Ok(analyze_context(&Query::new(query), &results, &self.cfg.context))
    }

    fn ensure_indexed(&mut self) -> Result<()> {
        if self.retriever.is_none() {
            let mut r = Bm25Retriever::new()?;
            self.rt.block_on(r.index(&self.chunks))?;
            self.retriever = Some(r);
        }
        Ok(())
    }

    fn retrieve(&mut self, query: &str, k: usize) -> Result<Vec<RetrievalResult>> {
        self.ensure_indexed()?;
        let q = Query::new(query);
        let retriever = self.retriever.as_ref().expect("indexed above");
        self.rt.block_on(retriever.retrieve(&q, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "The safety lamp was invented by Humphry Davy. \
        Humphry Davy was born in Penzance, Cornwall, England, and was a chemist. \
        Photosynthesis converts sunlight into chemical energy in green plants. \
        The Eiffel Tower is located in Paris, France. \
        Rust is a systems programming language focused on memory safety.";

    #[test]
    fn from_text_chunks_and_retrieves() {
        let mut doc = Document::from_text_with(
            "doc",
            TEXT,
            DocumentConfig { target_tokens: 8, max_tokens: 16, ..Default::default() },
        )
        .unwrap();
        assert!(doc.len() >= 2, "text should produce multiple chunks");
        let ctx = doc.context("Where was the safety lamp inventor born?").unwrap();
        assert!(!ctx.text().is_empty());
        // The retrieved+assembled context should reach the answer evidence.
        assert!(ctx.text().to_lowercase().contains("penzance"));
    }

    #[test]
    fn analyze_is_non_destructive_and_reports_a_decision() {
        let mut doc = Document::from_text_with(
            "doc",
            TEXT,
            DocumentConfig { target_tokens: 8, max_tokens: 16, ..Default::default() },
        )
        .unwrap();
        let report = doc.analyze("safety lamp inventor nationality").unwrap();
        assert!(report.n_input_chunks > 0);
        // Auto on a small retrieval → passthrough (no pruning).
        assert_eq!(report.strategy, ContextStrategy::RawTopK);
    }

    #[test]
    fn empty_document_errors_clearly() {
        let err = Document::from_chunks(vec![]).map(|_| ()).unwrap_err().to_string();
        assert!(err.contains("no chunks"), "unhelpful error: {err}");
        // Whitespace-only text produces no chunks → same clear error.
        assert!(Document::from_text("doc", "   \n  ").is_err());
    }

    #[test]
    fn from_chunks_skips_chunking() {
        let chunks = vec![
            Chunk::new(
                redhop_core::ChunkId::new("a"),
                "Humphry Davy was British.",
                "doc",
                redhop_core::TokenCount(4),
            ),
            Chunk::new(
                redhop_core::ChunkId::new("b"),
                "The safety lamp was invented by Humphry Davy.",
                "doc",
                redhop_core::TokenCount(8),
            ),
        ];
        let mut doc = Document::from_chunks(chunks).unwrap();
        assert_eq!(doc.len(), 2);
        let ctx = doc.context("who invented the safety lamp").unwrap();
        assert!(!ctx.text().is_empty());
    }
}
