//! # neorag-retrieval
//!
//! Retrieval engines built on top of [`neorag_core::Retriever`].
//!
//! Three concrete retrievers are provided:
//!
//! - [`bm25::Bm25Retriever`] — lexical retrieval backed by Tantivy.
//! - [`dense::DenseRetriever`] — dense vector retrieval over a pluggable
//!   [`VectorIndex`].
//! - [`hybrid::HybridRetriever`] — composition of an arbitrary number of
//!   sub-retrievers with rank-based fusion (RRF) by default.
//!
//! Score-level fusion utilities are in [`fusion`].
//!
//! [`VectorIndex`]: neorag_core::VectorIndex

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bm25;
pub mod dense;
pub mod fusion;
pub mod hybrid;

pub use bm25::Bm25Retriever;
pub use dense::DenseRetriever;
pub use fusion::{reciprocal_rank_fusion, weighted_sum_fusion, FusionStrategy};
pub use hybrid::HybridRetriever;
