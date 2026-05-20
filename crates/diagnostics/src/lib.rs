//! # neorag-diagnostics
//!
//! Retrieval-quality diagnostics — *first-class*, not optional.
//!
//! The premise: retrieval failure is almost never "the right chunk wasn't
//! in the index". It is "the right chunk was *there* but buried under
//! distractors, or the chunk had too little lexical grounding for the
//! reader model to anchor on, or the top-k saturated on near-duplicates."
//! Each of those failure modes is observable post-retrieval, before any LLM
//! is invoked. This module provides the primitives.
//!
//! Six metrics are computed (each in `[0, 1]`, higher is better unless
//! noted):
//!
//! - [`metrics::lexical_grounding`] — average query-term overlap with the
//!   retrieved chunks. Low values predict reader hallucination.
//! - [`metrics::chunk_purity`] — average per-chunk topical coherence,
//!   estimated as intra-chunk sentence-term overlap. Low values mean
//!   chunks straddle topic boundaries (a chunker problem).
//! - [`metrics::answer_density`] — fraction of retrieved tokens that are
//!   query-relevant (proxy for answer-bearing evidence density).
//! - [`metrics::distractor_ratio`] — fraction of retrieved chunks whose
//!   per-chunk grounding is below a confidence threshold. *Lower is
//!   better*; the metric is reported as a 0..1 fraction.
//! - [`metrics::retrieval_saturation`] — does the tail of results add new
//!   information, or just rehash the head? `1.0` means saturated (no new
//!   information at the bottom).
//! - [`metrics::evidence_concentration`] — how peaked the top scores are.
//!   `1.0` means a single dominant result, `0.0` means a flat plateau.
//!
//! [`DefaultDiagnosticsEngine`] runs all six.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod engine;
pub mod metrics;

pub use engine::DefaultDiagnosticsEngine;
