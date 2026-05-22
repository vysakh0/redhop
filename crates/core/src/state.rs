//! Retrieval state and regime classification types.
//!
//! These types are the **read-only backbone** of RedHop's adaptive layer.
//! They appear in core because every later subsystem — the policy engine,
//! the orchestrator, future learned policies — works against them as a
//! pluggable surface. By keeping the types here we ensure that
//! `redhop-orchestration`, `redhop-diagnostics`, and any user-built
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
/// Every classifier-driven regime decision in RedHop accumulates one
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

// ─────────────────────────────────────────────────────────────────────
// Phase 8: actions, budget, taken-action history
// ─────────────────────────────────────────────────────────────────────

/// Reranker tiers in escalation order. `None < Lexical < Semantic <
/// CrossEncoder`.
///
/// The ordering is meaningful: the orchestrator only escalates *up* the
/// ladder, never down, and the [`RuleBasedClassifier`][rbc] uses the
/// current level to decide whether escalation is still possible.
///
/// [rbc]: ../../redhop_orchestration/struct.RuleBasedClassifier.html
#[derive(
    Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize,
)]
pub enum RerankerLevel {
    /// No reranker applied.
    None,
    /// Cheap lexical reranker (term overlap / grounding).
    Lexical,
    /// Embedding-based reranker (cosine over chunk vectors).
    Semantic,
    /// Heavy cross-encoder. Not implemented in Phase 8; reserved.
    CrossEncoder,
}

impl RerankerLevel {
    /// Stable machine-readable code.
    pub fn code(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Lexical => "lexical",
            Self::Semantic => "semantic",
            Self::CrossEncoder => "cross_encoder",
        }
    }

    /// Next-higher tier, if any.
    pub fn escalate(self) -> Option<RerankerLevel> {
        match self {
            Self::None => Some(Self::Lexical),
            Self::Lexical => Some(Self::Semantic),
            Self::Semantic => Some(Self::CrossEncoder),
            Self::CrossEncoder => None,
        }
    }
}

impl std::fmt::Display for RerankerLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

/// Why the controller decided to stop iterating.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum StopReason {
    /// Easy regime: retrieval looks done.
    Confident,
    /// Saturated regime: more retrieval will not help.
    SaturatedNoBenefit,
    /// No regime fired with enough probability to act on.
    NoSignal,
    /// Iteration budget exhausted.
    BudgetExhausted,
    /// Last action did not improve evidence quality.
    NoImprovement,
}

impl StopReason {
    /// Stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Confident => "confident",
            Self::SaturatedNoBenefit => "saturated_no_benefit",
            Self::NoSignal => "no_signal",
            Self::BudgetExhausted => "budget_exhausted",
            Self::NoImprovement => "no_improvement",
        }
    }
}

/// Why the controller decided to abstain — i.e. emit the result set with
/// an explicit "evidence insufficient" flag rather than letting a downstream
/// LLM hallucinate over it.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum AbstainReason {
    /// Sparse regime: corpus likely does not contain the answer.
    Sparse,
    /// Grounding signals stayed low through every iteration attempted.
    PersistentLowGrounding,
}

impl AbstainReason {
    /// Stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::PersistentLowGrounding => "persistent_low_grounding",
        }
    }
}

/// The action catalog the policy may choose from.
///
/// **Phase 8 intentionally restricts the action space to four members.**
/// Query rewriting, chunk mutation, and graph traversal are deliberately
/// excluded until the four below are empirically validated. The space is
/// designed to be conservative — every action either terminates (Stop /
/// Abstain) or performs exactly one bounded retrieval mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalAction {
    /// Terminate iteration. The current candidates are the final evidence.
    Stop {
        /// Why we stopped.
        reason: StopReason,
    },
    /// Terminate and emit a structured "insufficient evidence" outcome.
    Abstain {
        /// Why we abstained.
        reason: AbstainReason,
    },
    /// Re-retrieve with a larger top-k.
    ExpandTopK {
        /// Current top-k.
        from: usize,
        /// New top-k.
        to: usize,
    },
    /// Apply the next reranker tier to the current candidates.
    EscalateReranker {
        /// Current reranker level.
        from: RerankerLevel,
        /// Target reranker level.
        to: RerankerLevel,
    },
}

impl RetrievalAction {
    /// Stable machine-readable code, suitable for logs and FFI.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Stop { .. } => "stop",
            Self::Abstain { .. } => "abstain",
            Self::ExpandTopK { .. } => "expand_top_k",
            Self::EscalateReranker { .. } => "escalate_reranker",
        }
    }

    /// True iff this action terminates the loop.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Stop { .. } | Self::Abstain { .. })
    }
}

/// Bookkeeping of work performed by a single action.
///
/// These counters are essential for later policy evaluation — without
/// them you cannot decide whether `EscalateReranker` is paying for itself
/// on the workloads where it fires.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionCost {
    /// Number of additional retrieval calls this action triggered.
    pub retrieval_calls: u32,
    /// Number of reranker calls this action triggered.
    pub rerank_calls: u32,
    /// Net change in candidate-set size (`new − old`); may be negative
    /// when reranking trims tail candidates.
    pub chunks_delta: i32,
}

/// A complete record of one action's effect on the retrieval state.
///
/// `TakenAction` is the unit of evidence the policy-evaluation layer
/// later consumes. Each record holds the policy's *prediction*
/// (`expected_gain`), the controller's *measurement*
/// (`actual_gain`), and enough context to recompute either offline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakenAction {
    /// The action that was applied.
    pub action: RetrievalAction,
    /// Zero-indexed iteration at which this action ran.
    pub iteration: u32,
    /// The policy's predicted gain in evidence quality, in `[0, 1]`.
    /// Terminal actions report `0.0`.
    pub expected_gain: f32,
    /// Measured gain after the action ran, in `[-1, 1]`. `None` for
    /// terminal actions where there is no "after" to measure against.
    pub actual_gain: Option<f32>,
    /// Diagnostics observed before the action ran.
    pub pre_diagnostics: DiagnosticsReport,
    /// Diagnostics observed after the action ran. `None` for terminal
    /// actions.
    pub post_diagnostics: Option<DiagnosticsReport>,
    /// Wall-clock time the action spent.
    pub latency_ms: u64,
    /// Work performed by the action.
    pub cost: ActionCost,
    /// The policy's human-readable rationale for choosing this action.
    pub rationale: String,
}

/// Compute budget for one adaptive retrieval session.
///
/// Defaults are deliberately tight: 3 iterations, 50 top-k cap, 2 rerank
/// calls. The whole point of the adaptive controller is to do *less*
/// work, not more — a profligate budget invites runaway iteration on
/// genuinely sparse queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    /// Iterations the orchestrator is allowed to run.
    pub max_iterations: u32,
    /// Hard cap on top-k after expansion.
    pub max_top_k: usize,
    /// Hard cap on reranker invocations.
    pub max_rerank_calls: u32,
    /// Remaining iterations. Mutated by the orchestrator.
    pub remaining_iterations: u32,
    /// Remaining reranker calls. Mutated by the orchestrator.
    pub remaining_rerank_calls: u32,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            max_top_k: 50,
            max_rerank_calls: 2,
            remaining_iterations: 3,
            remaining_rerank_calls: 2,
        }
    }
}

impl Budget {
    /// Construct a budget with custom limits; `remaining_*` start full.
    pub fn new(max_iterations: u32, max_top_k: usize, max_rerank_calls: u32) -> Self {
        Self {
            max_iterations,
            max_top_k,
            max_rerank_calls,
            remaining_iterations: max_iterations,
            remaining_rerank_calls: max_rerank_calls,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────

/// The state object that flows through the adaptive retrieval loop.
///
/// Phase 7 introduced the read-only fields (`query`, `candidates`,
/// `diagnostics`, `confidence`, `regime`); Phase 8 added `history`,
/// `budget`, and the current `reranker_level`. The orchestrator mutates
/// these in place across iterations.
///
/// Fields are public so consumers can read them directly without going
/// through getters; mutation is the orchestrator's job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalState {
    /// The query that produced this state.
    pub query: Query,
    /// Zero-indexed iteration counter.
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
    /// Full audit trail of actions the orchestrator has taken.
    #[serde(default)]
    pub history: Vec<TakenAction>,
    /// Current top-k in effect for retrieval.
    #[serde(default)]
    pub current_top_k: usize,
    /// Current reranker tier in effect.
    #[serde(default = "default_reranker_level")]
    pub reranker_level: RerankerLevel,
    /// Compute budget; counters mutate with each iteration.
    #[serde(default)]
    pub budget: Budget,
}

fn default_reranker_level() -> RerankerLevel {
    RerankerLevel::None
}

impl RetrievalState {
    /// Construct a new state from already-computed components.
    pub fn new(
        query: Query,
        candidates: Vec<RetrievalResult>,
        diagnostics: DiagnosticsReport,
        confidence: ConfidenceProfile,
    ) -> Self {
        let n = candidates.len();
        Self {
            query,
            iteration: 0,
            candidates,
            diagnostics,
            confidence,
            regime: None,
            history: Vec::new(),
            current_top_k: n,
            reranker_level: RerankerLevel::None,
            budget: Budget::default(),
        }
    }

    /// Attach a regime classification.
    pub fn with_regime(mut self, regime: RegimeDistribution) -> Self {
        self.regime = Some(regime);
        self
    }

    /// Override the budget for this session.
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Convenience: the headline regime label, if any.
    pub fn regime_label(&self) -> Option<RetrievalRegime> {
        self.regime.as_ref().map(|r| r.argmax)
    }

    /// True iff the orchestrator terminated this session via `Abstain`.
    pub fn abstained(&self) -> bool {
        self.history
            .last()
            .map(|t| matches!(t.action, RetrievalAction::Abstain { .. }))
            .unwrap_or(false)
    }

    /// The terminal action, if the session has terminated.
    pub fn terminal_action(&self) -> Option<&RetrievalAction> {
        self.history
            .last()
            .map(|t| &t.action)
            .filter(|a| a.is_terminal())
    }
}
