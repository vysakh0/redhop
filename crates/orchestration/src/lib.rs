//! # redhop-orchestration
//!
//! Retrieval-state observation, regime classification, policy, and the
//! adaptive controller that actions them.
//!
//! Ships:
//!
//! - [`confidence::compute_confidence`] — derives a
//!   [`redhop::core::ConfidenceProfile`] from a list of
//!   [`redhop::core::RetrievalResult`]s.
//! - [`classifier::RuleBasedClassifier`] — interpretable,
//!   threshold-driven regime classifier with full audit traces.
//! - [`policy::ConservativeRulePolicy`] — the bounded policy that turns
//!   regime probabilities into [`policy::PolicyDecision`]s.
//! - [`actuator::DefaultActuator`] — the work-doer behind a
//!   [`actuator::Actuator`] trait so the orchestrator stays testable
//!   against mocks.
//! - [`orchestrator::AdaptiveOrchestrator`] — the iteration loop that
//!   ties diagnostics → classifier → policy → actuator together under a
//!   bounded budget.
//!
//! ## Why a separate crate
//!
//! Two reasons:
//!
//! 1. Diagnostics observe per-result quality; orchestration observes
//!    *state*. The conceptual layer is different and the dependencies are
//!    different — this crate touches neither chunkers nor retrievers
//!    directly, only the [`redhop::core::Retriever`] trait through
//!    [`actuator::Actuator`].
//! 2. Bindings (Python, Node) that wrap an external retriever can pull
//!    orchestration alone without dragging in Tantivy through
//!    `redhop`'s retrieval layer.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod actuator;
pub mod classifier;
pub mod confidence;
pub mod orchestrator;
pub mod policy;

pub use actuator::{ActuationOutcome, Actuator, DefaultActuator};
pub use classifier::{ClassifierThresholds, RuleBasedClassifier};
pub use confidence::compute_confidence;
pub use orchestrator::AdaptiveOrchestrator;
pub use policy::{ConservativeRulePolicy, Policy, PolicyDecision, PolicyThresholds};
