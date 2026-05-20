//! # neorag-pipeline
//!
//! Top-level facade composing chunking, retrieval, optional reranking, and
//! diagnostics into a single ergonomic API.
//!
//! Most users should interact with NeoRAG through [`NeoRAG`] and its builder
//! [`NeoRAGBuilder`]; everything else in the workspace is reachable from here
//! through the underlying traits, which keeps every component swappable
//! without forking this facade.
//!
//! ## Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
//! use neorag_core::{Document, TokenizerBackend};
//! use neorag_pipeline::NeoRAG;
//! use neorag_retrieval::Bm25Retriever;
//! # async fn run() -> anyhow::Result<()> {
//! let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
//! let chunker = Arc::new(SentenceChunker::new(tok.clone(), 256, 384, 0)?);
//! let retriever = Bm25Retriever::new()?;
//! let mut rag = NeoRAG::builder()
//!     .with_chunker(chunker)
//!     .with_retriever(Arc::new(retriever))
//!     .build()?;
//! rag.ingest(vec![Document::new("doc1", "hello world")]).await?;
//! let results = rag.retrieve("hello", 5).await?;
//! let report = rag.diagnose(&"hello".into(), &results)?;
//! # let _ = report;
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::Arc;

use neorag_core::{
    Chunker, DiagnosticsEngine, DiagnosticsReport, Document, Error, Query, Reranker, Result,
    RetrievalResult, Retriever,
};
use neorag_diagnostics::DefaultDiagnosticsEngine;

/// Builder for [`NeoRAG`].
///
/// The builder enforces the *required* components (chunker, retriever) at
/// `build()` time. Optional components (reranker, custom diagnostics engine)
/// default to sensible no-ops.
pub struct NeoRAGBuilder {
    chunker: Option<Arc<dyn Chunker>>,
    retriever: Option<Arc<dyn Retriever>>,
    reranker: Option<Arc<dyn Reranker>>,
    diagnostics: Option<Arc<dyn DiagnosticsEngine>>,
    candidate_k: usize,
}

impl Default for NeoRAGBuilder {
    fn default() -> Self {
        Self {
            chunker: None,
            retriever: None,
            reranker: None,
            diagnostics: None,
            candidate_k: 32,
        }
    }
}

impl NeoRAGBuilder {
    /// Set the chunker (required).
    pub fn with_chunker(mut self, c: Arc<dyn Chunker>) -> Self {
        self.chunker = Some(c);
        self
    }

    /// Set the retriever (required).
    ///
    /// The retriever is wrapped in an `Arc` because [`NeoRAG`] needs shared
    /// ownership: indexing and retrieval may race in concurrent callers.
    /// If you also need mutable access for indexing, keep a separate handle
    /// to a `Bm25Retriever` (etc.) outside the facade and call its
    /// [`Retriever::index`] directly before wrapping it.
    pub fn with_retriever(mut self, r: Arc<dyn Retriever>) -> Self {
        self.retriever = Some(r);
        self
    }

    /// Optional: attach a reranker that runs over the retriever's output.
    pub fn with_reranker(mut self, r: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(r);
        self
    }

    /// Optional: replace the default diagnostics engine.
    pub fn with_diagnostics(mut self, d: Arc<dyn DiagnosticsEngine>) -> Self {
        self.diagnostics = Some(d);
        self
    }

    /// Optional: how many candidates to pull from the retriever before
    /// passing into the reranker. Ignored when no reranker is configured.
    pub fn with_candidate_k(mut self, k: usize) -> Self {
        self.candidate_k = k.max(1);
        self
    }

    /// Finalize the configuration and build the facade.
    pub fn build(self) -> Result<NeoRAG> {
        let chunker = self
            .chunker
            .ok_or(Error::MissingComponent("chunker"))?;
        let retriever = self
            .retriever
            .ok_or(Error::MissingComponent("retriever"))?;
        let diagnostics = self
            .diagnostics
            .unwrap_or_else(|| Arc::new(DefaultDiagnosticsEngine::new()));
        Ok(NeoRAG {
            chunker,
            retriever,
            reranker: self.reranker,
            diagnostics,
            candidate_k: self.candidate_k,
        })
    }
}

/// The top-level NeoRAG facade.
///
/// Holds a chunker, a retriever, optionally a reranker, and a diagnostics
/// engine. All components are accessed through traits, so users can swap
/// any of them — including replacing the entire retriever with a remote
/// service that implements [`Retriever`] — without touching this struct.
pub struct NeoRAG {
    chunker: Arc<dyn Chunker>,
    retriever: Arc<dyn Retriever>,
    reranker: Option<Arc<dyn Reranker>>,
    diagnostics: Arc<dyn DiagnosticsEngine>,
    candidate_k: usize,
}

impl NeoRAG {
    /// Start building a new pipeline.
    pub fn builder() -> NeoRAGBuilder {
        NeoRAGBuilder::default()
    }

    /// Ingest a batch of documents: chunk them, then hand the chunks to the
    /// retriever's own index path.
    ///
    /// Note: this requires a `&mut self` cast through `Arc::get_mut`-style
    /// gymnastics if the retriever is shared; in practice we keep a clone
    /// of the inner `Arc<dyn Retriever>` and call `index` on it through a
    /// small trick — see implementation.
    pub async fn ingest(&mut self, docs: Vec<Document>) -> Result<()> {
        let chunks = self.chunker.chunk_batch(&docs)?;
        // The Retriever::index signature takes `&mut self`. We obtain a
        // unique mutable view by routing through `Arc::get_mut`, which is
        // only sound if no other clones of `self.retriever` exist. In the
        // typical builder-then-use lifecycle that is the case; concurrent
        // callers wishing to index while retrieving should construct their
        // own retriever and call `index` on it directly before sharing.
        let retriever = Arc::get_mut(&mut self.retriever).ok_or_else(|| {
            Error::Storage(
                "retriever is shared via Arc and cannot be indexed through the facade; call index() on it directly before sharing"
                    .into(),
            )
        })?;
        retriever.index(&chunks).await
    }

    /// Run a query through retrieval (and optional reranking).
    pub async fn retrieve(
        &self,
        query: impl Into<Query>,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        let query = query.into();
        let k = if self.reranker.is_some() {
            self.candidate_k.max(top_k)
        } else {
            top_k
        };
        let candidates = self.retriever.retrieve(&query, k).await?;
        if let Some(rr) = &self.reranker {
            rr.rerank(&query, candidates, top_k).await
        } else {
            Ok(candidates)
        }
    }

    /// Compute diagnostics for a query and its results.
    pub fn diagnose(
        &self,
        query: &Query,
        results: &[RetrievalResult],
    ) -> Result<DiagnosticsReport> {
        self.diagnostics.diagnose(query, results)
    }

    /// Names of the configured components, for logging / diagnostics.
    pub fn component_names(&self) -> ComponentNames {
        ComponentNames {
            chunker: self.chunker.name(),
            retriever: self.retriever.name(),
            reranker: self.reranker.as_ref().map(|r| r.name()),
            diagnostics: self.diagnostics.name(),
        }
    }
}

/// Names of the components in a [`NeoRAG`] pipeline.
#[derive(Debug, Clone)]
pub struct ComponentNames {
    /// Chunker name.
    pub chunker: &'static str,
    /// Retriever name.
    pub retriever: &'static str,
    /// Reranker name, if any.
    pub reranker: Option<&'static str>,
    /// Diagnostics engine name.
    pub diagnostics: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
    use neorag_core::TokenizerBackend;
    use neorag_retrieval::Bm25Retriever;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn end_to_end_bm25() {
        rt().block_on(async {
            let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
            let chunker = Arc::new(SentenceChunker::new(tok.clone(), 16, 24, 0).unwrap());
            let retriever = Arc::new(Bm25Retriever::new().unwrap());

            let mut rag = NeoRAG::builder()
                .with_chunker(chunker)
                .with_retriever(retriever)
                .build()
                .unwrap();

            rag.ingest(vec![
                Document::new(
                    "tokio",
                    "Tokio is an asynchronous runtime for Rust. It powers async applications.",
                ),
                Document::new(
                    "django",
                    "Django is a high-level Python web framework. It encourages rapid development.",
                ),
            ])
            .await
            .unwrap();

            let results = rag.retrieve("rust async runtime", 3).await.unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].chunk.source, "tokio");

            let report = rag
                .diagnose(&Query::new("rust async runtime"), &results)
                .unwrap();
            assert!(report.lexical_grounding.is_some());
            assert!(report.retrieval_confidence.is_some());
        });
    }

    #[test]
    fn build_fails_without_required_components() {
        let r = NeoRAG::builder().build();
        assert!(r.is_err());
    }
}
