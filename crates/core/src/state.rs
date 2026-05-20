//! Retrieval state and regime classification types.
//!
//! These types are the **read-only backbone** of NeoRAG's adaptive layer.
//! They appear in core because every later subsystem — the policy engine,
//! the orchestrator, future learned policies — works against them as a
//! pluggable surface. By keeping the types here we ensure that
//! `neorag-orchestration`, `neorag-diagnostics`, and any user-built
//! classifier all agree on shape.
//!
//! ## Design constraints
//!
//! Three constraints shape every type in this module:
//!
//! 1. **Interpretability is non-negotiable.** Every classification carries a
//!    [`ClassificationTrace`] recording the features it saw, the thresholds
//!    it applied, and the rules that fired. A user holding a
//!    [`RegimeDistribution`] should be able to answer "why did you call
//!    this query `Saturated`?" without re-running anything.
//! 2. **Probabilistic, not categorical.** [`RegimeDistribution`] is a soft
//!    mass function over regimes. The `argmax` is a convenience accessor,
//!    not the source of truth. This is what lets later phases hedge — a
//!    `{Saturated: 0.6, Easy: 0.3}` posterior gets a different action than
//!    `{Saturated: 0.99}` even though both have the same argmax.
//! 3. **Empty defaults are legal.** Every `Option` and every `Vec` defaults
//!    to "we did not measure that"; the consumer treats `None` as honest
//!    silence rather than failure.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::types::{DiagnosticsReport, Query, RetrievalResult};

/// The five canonical retrieval regimes.
///
/// Each regime maps to a different *kind* of failure (or success) and, in
/// later phases, to a different action. The set was chosen from empirical
/// observation rather than first principles — see the project's HotpotQA
/// and judge-model traces for the calibration data behind these labels.
///
/// `BTreeMap` and `Ord` derivations exist so a probability map keyed on
/// regime serializes in deterministic order, which matters for snapshot
/// testing and FFI round-trips.
#[derive(
    Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize,
)]
pub enum RetrievalRegime {
    /// High grounding, high concentration, low saturation. Retrieval is
    /// done; stop iterating.
    Easy,
    /// High redundancy / saturation. Top-k is rehashing the same evidence;
    /// more retrieval will not help.
    Saturated,
    /// High distractor ratio (lexical or semantic). The candidate set is
    /// contaminated by off-topic hits.
    DistractorHeavy,
    /// Flat score distribution, high centroid dispersion, modest grounding.
    /// The retriever is hedging across semantically distinct candidates.
    Ambiguous,
    /// Low grounding across both tiers. The corpus probably does not
    /// contain the answer.
    Sparse,
}

impl RetrievalRegime {
    /// Stable machine-readable code, suitable for logs and FFI.
    pub fn code(self) -> &'static str {
        match self {
            Self::Easy => "easy",
            Self::Saturated => "saturated",
            Self::DistractorHeavy => "distractor_heavy",
            Self::Ambiguous => "ambiguous",
            Self::Sparse => "sparse",
        }
    }

    /// All regimes, in display order.
    pub fn all() -> &'static [RetrievalRegime] {
        &[
            Self::Easy,
            Self::Saturated,
            Self::DistractorHeavy,
            Self::Ambiguous,
            Self::Sparse,
        ]
    }
}

impl std::fmt::Display for RetrievalRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

/// A confidence *profile* — three views on top-k score peakedness plus a
/// composite scalar.
///
/// One scalar confidence is dashboard candy and hides regime-specific
/// failure modes. The profile here is the smallest set of orthogonal
/// confidence signals we found empirically useful:
///
/// - **`decision_margin`** answers *"is the top-1 clearly the top-1?"*
/// - **`score_entropy`** answers *"is the whole distribution peaked or flat?"*
/// - **`posterior_concentration`** answers *"how much mass does the top-1 carry?"*
/// - **`aggregate`** is the composite for the dashboard view; not the source of truth.
///
/// All fields are in `[0, 1]` where higher is more confident.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfidenceProfile {
    /// `(score[0] − score[1]) / (|score[0]| + ε)`, clamped to `[0, 1]`.
    /// `None` when the result set has fewer than two items.
    pub decision_margin: Option<f32>,
    /// Shannon entropy of the softmax-normalized score distribution,
    /// divided by `log(k)` so `0` means a clean peak and `1` means a flat
    /// plateau. `None` when there are fewer than two items.
    pub score_entropy: Option<f32>,
    /// Mass on the top-1 after softmax normalization; equivalent to
    /// `1 − (rest of mass)`.
    pub posterior_concentration: Option<f32>,
    /// Composite scalar in `[0, 1]`; convenience for plotting. Defined as
    /// `posterior_concentration * (1 − score_entropy)` when both are
    /// available.
    pub aggregate: Option<f32>,
}

/// A record of which rule contributed how much to a regime classification.
///
/// Every classifier-driven regime decision in NeoRAG accumulates one
/// `RuleFire` per rule that contributed. Together with
/// [`ClassificationTrace::features`] and
/// [`ClassificationTrace::thresholds`] they fully reconstruct *why* the
/// classifier reached its verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleFire {
    /// Stable rule name (`snake_case`). Stable across versions so tests
    /// and dashboards can pin to specific rules.
    pub rule: String,
    /// Which regime this rule's contribution went to.
    pub regime: RetrievalRegime,
    /// How much this rule added to that regime's score before softmax.
    pub weight: f32,
    /// Human-readable justification, with concrete numbers from the
    /// inputs. Suitable for embedding in error messages or audit logs.
    pub justification: String,
}

/// Full audit trail of one classification.
///
/// The trace is always populated when a classifier is invoked — never
/// `Option<ClassificationTrace>`, never elided. Interpretability is a
/// hard design constraint, not a debug feature.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationTrace {
    /// Raw feature values the classifier inspected, keyed by feature
    /// name. Includes `None` features as absent keys rather than null
    /// values so the JSON form stays small.
    pub features: BTreeMap<String, f32>,
    /// Threshold values that were in effect. Different
    /// [`crate::traits::RegimeClassifier`] implementations may use
    /// different thresholds; recording them here makes classifications
    /// reproducible across configurations.
    pub thresholds: BTreeMap<String, f32>,
    /// Every rule that fired, in evaluation order.
    pub rules_fired: Vec<RuleFire>,
    /// The pre-softmax regime scores. Lets downstream code recompute the
    /// distribution with a different temperature if needed.
    pub raw_scores: BTreeMap<RetrievalRegime, f32>,
}

/// A probability mass function over [`RetrievalRegime`]s.
///
/// `probabilities` sums to (approximately) `1.0`. `argmax` is the regime
/// with the highest probability; callers wanting hedged policies should
/// inspect the full map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeDistribution {
    /// Per-regime probability mass.
    pub probabilities: BTreeMap<RetrievalRegime, f32>,
    /// Highest-mass regime.
    pub argmax: RetrievalRegime,
    /// Full audit trail.
    pub trace: ClassificationTrace,
}

impl RegimeDistribution {
    /// Probability mass on a specific regime, defaulting to `0.0`.
    pub fn p(&self, r: RetrievalRegime) -> f32 {
        self.probabilities.get(&r).copied().unwrap_or(0.0)
    }

    /// Entropy of the regime distribution (nats), useful as a meta-signal:
    /// when the classifier itself is uncertain (high entropy), policies
    /// should prefer conservative actions.
    pub fn entropy(&self) -> f32 {
        let mut h = 0.0f32;
        for &p in self.probabilities.values() {
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    }
}

/// The state object that flows through the adaptive retrieval loop.
///
/// In Phase 7 this struct is *populated but inert* — no part of NeoRAG
/// mutates it after creation. Phase 8 adds the action/budget/history
/// fields and the orchestrator that mutates them in place.
///
/// Construction goes through [`RetrievalState::new`] or
/// [`RetrievalState::builder`]; fields are public so consumers can read
/// them directly without going through getters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalState {
    /// The query that produced this state.
    pub query: Query,
    /// Zero-indexed iteration counter. Phase 7 only ever sees `0`.
    pub iteration: u32,
    /// Current working set of retrieval results.
    pub candidates: Vec<RetrievalResult>,
    /// Diagnostics computed against `candidates`.
    pub diagnostics: DiagnosticsReport,
    /// Score-based confidence signals derived from `candidates`.
    pub confidence: ConfidenceProfile,
    /// Optional regime classification. `None` when no classifier was
    /// configured.
    pub regime: Option<RegimeDistribution>,
}

impl RetrievalState {
    /// Construct a new state from already-computed components.
    pub fn new(
        query: Query,
        candidates: Vec<RetrievalResult>,
        diagnostics: DiagnosticsReport,
        confidence: ConfidenceProfile,
    ) -> Self {
        Self {
            query,
            iteration: 0,
            candidates,
            diagnostics,
            confidence,
            regime: None,
        }
    }

    /// Attach a regime classification.
    pub fn with_regime(mut self, regime: RegimeDistribution) -> Self {
        self.regime = Some(regime);
        self
    }

    /// Convenience: the headline regime label, if any.
    pub fn regime_label(&self) -> Option<RetrievalRegime> {
        self.regime.as_ref().map(|r| r.argmax)
    }
}
