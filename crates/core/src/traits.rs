//! Core traits.
//!
//! Every pluggable subsystem in NeoRAG is defined here as a trait. Concrete
//! implementations (in `neorag-chunking`, `neorag-retrieval`, …) implement
//! these and the [`pipeline`] crate composes them.
//!
//! [`pipeline`]: ../../neorag_pipeline/index.html

use crate::types::{
    Chunk, DiagnosticsReport, Document, Embedding, Query, RetrievalResult, Sentence, TokenCount,
};
use crate::Result;
use async_trait::async_trait;

/// Counts tokens and segments text.
///
/// The same backend is reused by chunkers (for token-budget enforcement) and
/// by retrievers (for query tokenization in lexical search).
///
/// Implementations must be `Send + Sync` so they can be shared across worker
/// threads in parallel chunking/indexing pipelines.
pub trait TokenizerBackend: Send + Sync {
    /// Number of tokens in `text`.
    fn count_tokens(&self, text: &str) -> Result<TokenCount>;

    /// Sentence-segment `text`. Returns sentences with byte offsets so callers
    /// can reassemble groups without copying the source string repeatedly.
    fn split_sentences(&self, text: &str) -> Result<Vec<Sentence>>;

    /// Truncate `text` so that it fits within `max_tokens` tokens.
    ///
    /// Default implementation does a coarse, repeated byte-trim using
    /// `count_tokens`; subclasses with cheap per-token offsets should
    /// override.
    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> Result<String> {
        if max_tokens == 0 {
            return Ok(String::new());
        }
        let mut s = text.to_string();
        while self.count_tokens(&s)?.value() > max_tokens && !s.is_empty() {
            // Drop ~10% of bytes each iteration; we only need a *fit*, not
            // perfection. Real backends override this with a token-aware path.
            let cut = (s.len() as f32 * 0.9) as usize;
            let cut = cut.min(s.len().saturating_sub(1));
            // Find the nearest char boundary at or below `cut`.
            let mut boundary = cut;
            while boundary > 0 && !s.is_char_boundary(boundary) {
                boundary -= 1;
            }
            s.truncate(boundary);
        }
        Ok(s)
    }
}

/// Splits documents into retrievable chunks.
///
/// Chunkers are the most consequential component for retrieval quality:
/// chunk boundaries determine evidence density and topical purity. NeoRAG
/// expects multiple implementations (fixed, sentence, adaptive) and treats
/// the choice as a first-class configuration knob.
pub trait Chunker: Send + Sync {
    /// Chunk a single document.
    fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>>;

    /// Chunk multiple documents.
    ///
    /// Default implementation iterates sequentially; implementations that
    /// can parallelize cheaply (e.g. via rayon) should override.
    fn chunk_batch(&self, docs: &[Document]) -> Result<Vec<Chunk>> {
        let mut out = Vec::new();
        for d in docs {
            out.extend(self.chunk(d)?);
        }
        Ok(out)
    }

    /// Human-readable name, used in diagnostics.
    fn name(&self) -> &'static str;
}

/// Produces embeddings for chunks and queries.
///
/// NeoRAG ships no model itself; this trait exists so callers can plug in
/// `fastembed-rs`, an ONNX model, a remote API, or anything else.
///
/// Asynchronous because most realistic providers do network I/O.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of texts.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>>;

    /// Dimensionality of vectors produced by this provider.
    fn dim(&self) -> usize;

    /// Human-readable name, used in diagnostics.
    fn name(&self) -> &'static str;
}

/// A low-level approximate-nearest-neighbor index.
///
/// Separate from [`Retriever`] so that storage backends (HNSW, IVF, flat
/// brute-force, external services) can be swapped without touching the
/// retriever above them.
pub trait VectorIndex: Send + Sync {
    /// Add a vector keyed by its chunk id.
    fn add(&mut self, id: crate::types::ChunkId, vector: Embedding) -> Result<()>;

    /// Top-`k` nearest neighbors to `query`, returned as `(chunk_id, score)`
    /// pairs in descending order of score.
    fn search(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Result<Vec<(crate::types::ChunkId, f32)>>;

    /// Number of vectors stored.
    fn len(&self) -> usize;

    /// True iff the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Retrieves chunks for a query.
///
/// Retrievers are the workhorse of the system. NeoRAG provides BM25,
/// dense-vector, hybrid, and adapter implementations; users can also attach
/// their own.
///
/// Asynchronous so retrievers backed by remote services (Vespa, Pinecone,
/// HTTP rerankers) compose naturally.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Run the query, return at most `top_k` results.
    async fn retrieve(&self, query: &Query, top_k: usize) -> Result<Vec<RetrievalResult>>;

    /// Ingest a batch of chunks into whatever underlying index this retriever
    /// owns. May be a no-op for adapter retrievers that wrap an external
    /// service.
    async fn index(&mut self, chunks: &[Chunk]) -> Result<()>;

    /// Human-readable name, used in diagnostics.
    fn name(&self) -> &'static str;
}

/// Reorders retrieval results using additional signal.
///
/// Rerankers are intentionally separated from retrievers so the same
/// retrieval stage can be paired with different reranking strategies, and so
/// future cross-encoder models can plug in without touching the candidate
/// retrieval path.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank `candidates` for `query`, returning at most `top_k` results.
    async fn rerank(
        &self,
        query: &Query,
        candidates: Vec<RetrievalResult>,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>>;

    /// Human-readable name, used in diagnostics.
    fn name(&self) -> &'static str;
}

/// Computes a [`DiagnosticsReport`] for a query/result pair.
///
/// Diagnostics engines are deliberately stateless with respect to the
/// retrieval that produced their input — they only see what the retriever
/// returned. This keeps them swappable and testable in isolation.
pub trait DiagnosticsEngine: Send + Sync {
    /// Compute diagnostics for a query and its retrieval results.
    fn diagnose(&self, query: &Query, results: &[RetrievalResult]) -> Result<DiagnosticsReport>;

    /// Human-readable name, used in diagnostics.
    fn name(&self) -> &'static str;
}

/// Maps a retrieval state's *measurements* (diagnostics + confidence) to a
/// soft distribution over [`RetrievalRegime`][rrg]s.
///
/// Classifiers are intentionally narrow: they see the same observations the
/// human dashboard sees and nothing else. They do *not* read the chunk text
/// or the query embedding directly — only the metrics the diagnostics
/// engines produced. This is what makes them swappable and what keeps
/// later learned versions auditable: the input contract is small and
/// stable.
///
/// Implementations must populate the returned distribution's
/// [`ClassificationTrace`][trc] — interpretability is a hard requirement,
/// not optional.
///
/// [rrg]: crate::state::RetrievalRegime
/// [trc]: crate::state::ClassificationTrace
pub trait RegimeClassifier: Send + Sync {
    /// Produce a regime distribution from the given diagnostics and
    /// confidence inputs.
    fn classify(
        &self,
        diagnostics: &crate::types::DiagnosticsReport,
        confidence: &crate::state::ConfidenceProfile,
    ) -> crate::state::RegimeDistribution;

    /// Human-readable name, used in diagnostics.
    fn name(&self) -> &'static str;
}
