//! Library-wide error type.

use thiserror::Error;

/// Result alias used throughout NeoRAG.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type for the NeoRAG library.
///
/// Concrete implementations (Tantivy-backed BM25, HF tokenizers, etc.) wrap
/// their own failures into one of these variants so that downstream callers —
/// and language bindings — see a stable surface.
#[derive(Debug, Error)]
pub enum Error {
    /// A required component (retriever, tokenizer, …) was not configured.
    #[error("missing component: {0}")]
    MissingComponent(&'static str),

    /// Configuration was structurally valid but semantically wrong
    /// (e.g. `chunk_size = 0`, or hybrid weights summing to zero).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Tokenization failed.
    #[error("tokenization error: {0}")]
    Tokenization(String),

    /// Chunking failed.
    #[error("chunking error: {0}")]
    Chunking(String),

    /// A retriever failed.
    #[error("retrieval error: {0}")]
    Retrieval(String),

    /// A reranker failed.
    #[error("reranking error: {0}")]
    Reranking(String),

    /// A storage/index backend failed.
    #[error("storage error: {0}")]
    Storage(String),

    /// Embedding provider failed.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Dimension mismatch between query and indexed embeddings.
    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Dimension expected by the index.
        expected: usize,
        /// Dimension observed on the input.
        got: usize,
    },

    /// I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Catch-all for anything that hasn't earned its own variant yet.
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Convenience constructor for ad-hoc messages.
    pub fn msg<S: Into<String>>(s: S) -> Self {
        Self::Other(s.into())
    }
}
