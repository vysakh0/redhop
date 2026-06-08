//! # redhop-retrieval
//!
//! Retrieval engines built on top of [`crate::core::Retriever`].
//!
//! Four concrete retrievers are provided:
//!
//! - [`bm25::Bm25Retriever`] — lexical retrieval backed by Tantivy.
//! - [`dense::DenseRetriever`] — dense vector retrieval over a pluggable
//!   [`VectorIndex`].
//! - [`hybrid::HybridRetriever`] — composition of an arbitrary number of
//!   sub-retrievers with rank-based fusion (RRF) by default.
//! - [`local_rerank::LocalRerankRetriever`] — BM25 prune + local dense rerank
//!   of the candidate pool (no ANN); embedder-agnostic.
//!
//! Score-level fusion utilities are in [`fusion`].
//!
//! [`VectorIndex`]: crate::core::VectorIndex

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bm25;
pub mod dense;
pub mod fusion;
pub mod hybrid;
pub mod local_rerank;

pub use bm25::Bm25Retriever;
// `pub` helper that returns a concrete `TextAnalyzer` — used by
// `crate::analyzer::SnowballAnalyzer::build_text_analyzer` to keep all the
// generic intermediate Tantivy types private to bm25.rs.
pub use bm25::build_redhop_pipeline;
pub use bm25::build_raw_pipeline;
pub use dense::DenseRetriever;
pub use fusion::{reciprocal_rank_fusion, weighted_sum_fusion, FusionStrategy};
pub use hybrid::HybridRetriever;
pub use local_rerank::LocalRerankRetriever;
