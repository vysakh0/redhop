//! Core types, traits, and errors.
//!
//! Foundational shapes that the rest of the crate composes through:
//! [`Chunk`], [`Query`], [`RetrievalResult`], the [`Chunker`] /
//! [`Retriever`] / [`Reranker`] / [`EmbeddingProvider`] traits, and the
//! [`Error`] / [`Result`] alias.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod traits;
pub mod types;

pub use error::{Error, Result};
pub use traits::{
    Chunker, EmbeddingProvider, Reranker, Retriever, TokenizerBackend, VectorIndex,
};
pub use types::{
    Chunk, ChunkId, ChunkMetadata, Document, Embedding, Query, RetrievalMethod, RetrievalResult,
    Score, ScoreBreakdown, Sentence, TokenCount,
};
