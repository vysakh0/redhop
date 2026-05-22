//! # redhop-diagnostics
//!
//! Retrieval-quality diagnostics — *first-class*, not optional.
//!
//! Diagnostics in RedHop are split into two tiers:
//!
//! - **Lexical tier** ([`DefaultDiagnosticsEngine`]) — computed from text
//!   alone. Cheap, deterministic, no embedding model required. Catches
//!   missing query-term grounding, off-topic distractors, evidence
//!   saturation, and concentration.
//! - **Semantic tier** ([`SemanticDiagnosticsEngine`]) — computed from
//!   embeddings already present on the [`Query`] and [`Chunk`]s.
//!   Closes the paraphrase blind spot of the lexical tier without
//!   pulling in any model dependency: it consumes the embeddings the
//!   dense retriever already produces.
//!
//! [`LayeredDiagnosticsEngine`] composes both tiers into a single unified
//! [`DiagnosticsReport`]. This is the recommended production setup once a
//! pipeline has dense embeddings in flight.
//!
//! The two-tier split exists for a specific reason: it lets retrieval
//! diagnose itself even when embeddings are absent (BM25-only deployments,
//! cold start, OCR'd corpora), and adds *strictly more* signal when
//! embeddings are available. No tier overwrites the other.
//!
//! [`Query`]: redhop_core::Query
//! [`Chunk`]: redhop_core::Chunk

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod engine;
pub mod ingestion;
pub mod layered;
pub mod metrics;
pub mod semantic;

pub use engine::{DefaultDiagnosticsEngine, DiagnosticsThresholds};
pub use ingestion::{diagnose_ingestion, IngestionReport, IngestionThresholds};
pub use layered::LayeredDiagnosticsEngine;
pub use semantic::{SemanticDiagnosticsConfig, SemanticDiagnosticsEngine};
