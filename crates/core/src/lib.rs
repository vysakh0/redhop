//! # redhop-core
//!
//! Foundational types, traits, and error definitions for the RedHop retrieval
//! infrastructure.
//!
//! This crate intentionally contains **no implementation logic** beyond what
//! is required to define the shape of the system. All concrete chunkers,
//! retrievers, rerankers, diagnostics engines, and storage backends live in
//! their own crates and depend on the abstractions defined here.
//!
//! Architectural intent:
//!
//! - The library is *retrieval infrastructure*, not an LLM framework.
//! - Embeddings are pluggable; RedHop ships abstractions, not models.
//! - Diagnostics are first-class, not an afterthought.
//! - Everything is composed through `trait` boundaries so callers can swap any
//!   subsystem without touching the rest of the stack.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod state;
pub mod traits;
pub mod types;

pub use error::{Error, Result};
pub use state::{
    AbstainReason, ActionCost, Budget, ClassificationTrace, ConfidenceProfile,
    RegimeDistribution, RerankerLevel, RetrievalAction, RetrievalRegime, RetrievalState, RuleFire,
    StopReason, TakenAction,
};
pub use traits::{
    Chunker, DiagnosticsEngine, EmbeddingProvider, RegimeClassifier, Reranker, Retriever,
    TokenizerBackend, VectorIndex,
};
pub use types::{
    Chunk, ChunkId, ChunkMetadata, DiagnosticsReport, DiagnosticsWarning, Document, Embedding,
    Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown, Sentence, TokenCount,
};
