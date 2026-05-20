//! # neorag-orchestration
//!
//! Retrieval-state observation and regime classification.
//!
//! This is the **Phase 7 home** for everything that *observes* retrieval
//! state without mutating it. Phase 8 will add the action engine and the
//! adaptive orchestrator in this same crate; for now it ships:
//!
//! - [`confidence::compute_confidence`] — derives a
//!   [`ConfidenceProfile`][cp] from a list of [`RetrievalResult`][rr]s.
//! - [`classifier::RuleBasedClassifier`] — interpretable, threshold-driven
//!   regime classifier with full audit traces.
//!
//! ## Why a separate crate
//!
//! Three reasons:
//!
//! 1. Diagnostics observe per-result quality; orchestration observes
//!    *state*. The conceptual layer is different and the dependencies are
//!    different — this crate touches neither chunkers nor retrievers.
//! 2. The forthcoming `AdaptiveOrchestrator`, `Actuator`, and policy
//!    implementations will share this crate's home, but should not pull
//!    `neorag-diagnostics` deeper into the dependency graph.
//! 3. Bindings (Python, Node) will frequently want orchestration alone
//!    when they're wrapping an external retriever — pulling Tantivy via
//!    `neorag-retrieval` would be wasteful.
//!
//! [cp]: neorag_core::ConfidenceProfile
//! [rr]: neorag_core::RetrievalResult

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod classifier;
pub mod confidence;

pub use classifier::{ClassifierThresholds, RuleBasedClassifier};
pub use confidence::compute_confidence;
