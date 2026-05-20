//! # neorag-reranking
//!
//! Rerankers that take a candidate list and reorder it using a secondary
//! signal. Rerankers compose: a `HybridRetriever` can be followed by a
//! lexical-grounding reranker, which can be followed by a cross-encoder.
//!
//! Today's implementations are all text-only and deterministic — no model
//! dependence. The traits and signatures, however, are exactly the shape a
//! future cross-encoder reranker will need.
//!
//! - [`ScoreFusionReranker`] — recombines per-stage scores already present in
//!   each result's `ScoreBreakdown`. Useful after a hybrid retriever that
//!   stored both lexical and dense contributions.
//! - [`LexicalGroundingReranker`] — boosts candidates whose chunks share
//!   more query terms.
//! - [`EvidenceDensityReranker`] — boosts candidates whose chunks have a
//!   higher per-token query-term density (denser evidence per token of
//!   context).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod evidence_density;
pub mod lexical;
pub mod score_fusion;

pub use evidence_density::EvidenceDensityReranker;
pub use lexical::LexicalGroundingReranker;
pub use score_fusion::ScoreFusionReranker;
