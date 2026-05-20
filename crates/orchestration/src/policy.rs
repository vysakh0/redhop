//! Adaptive retrieval policy.
//!
//! A *policy* maps a [`RetrievalState`] to a [`PolicyDecision`] — an
//! action plus the policy's expected gain and its rationale. Policies are
//! pure functions of state. The [`Actuator`][act] separately applies the
//! decision; that split is what lets us replay, compare, and later learn
//! over historical decisions without re-running retrieval.
//!
//! ## Phase 8 policy is deliberately conservative
//!
//! Three design principles, in order of importance:
//!
//! 1. **Default to doing nothing.** The orchestrator's purpose is to
//!    suppress over-actuation. Every action must clear a non-trivial
//!    probability bar before the policy will choose it; if no rule is
//!    confident enough, the policy chooses
//!    `Stop { reason: NoSignal }`.
//! 2. **One intervention of each kind per session.** Once the
//!    orchestrator has escalated the reranker once, the policy will not
//!    request another escalation in the same session — repeated
//!    escalation has not earned its complexity on our traces. The same
//!    rule applies to `ExpandTopK`. The budget caps in
//!    [`Budget`][bud] provide a second line of defense.
//! 3. **Action space is small and bounded.** Only `Stop`,
//!    `Abstain`, `ExpandTopK`, `EscalateReranker`. No query rewriting,
//!    no chunk mutation, no branching. Bigger action spaces will arrive
//!    only when this one demonstrably falls short.
//!
//! [act]: crate::actuator::Actuator
//! [bud]: neorag_core::Budget

use neorag_core::{
    AbstainReason, RetrievalAction, RetrievalRegime, RetrievalState, StopReason,
};

/// Output of a policy decision.
#[derive(Debug, Clone)]
pub struct PolicyDecision {
    /// The action to apply.
    pub action: RetrievalAction,
    /// Policy-predicted improvement in evidence quality `[0, 1]`.
    /// Terminal actions are `0.0`.
    pub expected_gain: f32,
    /// Human-readable rationale; goes into [`TakenAction::rationale`].
    ///
    /// [`TakenAction::rationale`]: neorag_core::TakenAction::rationale
    pub rationale: String,
}

/// Pluggable policy trait.
pub trait Policy: Send + Sync {
    /// Decide what to do given the current state.
    fn decide(&self, state: &RetrievalState) -> PolicyDecision;

    /// Human-readable name; surfaced in logs and FFI.
    fn name(&self) -> &'static str;
}

/// Tunable probability thresholds for the conservative policy.
///
/// Every threshold is visible and recorded in the policy's rationale
/// when a rule fires, mirroring the interpretability discipline used by
/// [`crate::classifier::RuleBasedClassifier`].
#[derive(Debug, Clone)]
pub struct PolicyThresholds {
    /// Minimum `p(Easy)` to choose `Stop { Confident }`.
    pub min_p_easy: f32,
    /// Minimum `p(Saturated)` to choose `Stop { SaturatedNoBenefit }`.
    pub min_p_saturated: f32,
    /// Minimum `p(Sparse)` to choose `Abstain`. Higher than the others
    /// because abstain is the only action that's externally visible to a
    /// downstream LLM.
    pub min_p_sparse: f32,
    /// Minimum `p(DistractorHeavy)` to choose `EscalateReranker`.
    pub min_p_distractor: f32,
    /// Minimum `p(Ambiguous)` to choose `ExpandTopK`.
    pub min_p_ambiguous: f32,
    /// Additive top-k step. The orchestrator caps the new top-k at
    /// [`Budget::max_top_k`][bud].
    ///
    /// [bud]: neorag_core::Budget::max_top_k
    pub top_k_step: usize,
    /// Below this measured `actual_gain` from the previous action,
    /// terminate with `Stop { NoImprovement }`. Defaults to `0.02` —
    /// anything smaller is noise.
    pub min_continuation_gain: f32,
}

impl Default for PolicyThresholds {
    fn default() -> Self {
        Self {
            min_p_easy: 0.40,
            min_p_saturated: 0.45,
            min_p_sparse: 0.50,
            min_p_distractor: 0.40,
            min_p_ambiguous: 0.40,
            top_k_step: 8,
            min_continuation_gain: 0.02,
        }
    }
}

/// The default conservative policy.
#[derive(Debug, Clone, Default)]
pub struct ConservativeRulePolicy {
    thresholds: PolicyThresholds,
}

impl ConservativeRulePolicy {
    /// Construct with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with caller-provided thresholds.
    pub fn with_thresholds(thresholds: PolicyThresholds) -> Self {
        Self { thresholds }
    }

    /// Borrow the active thresholds.
    pub fn thresholds(&self) -> &PolicyThresholds {
        &self.thresholds
    }
}

impl Policy for ConservativeRulePolicy {
    fn decide(&self, state: &RetrievalState) -> PolicyDecision {
        let t = &self.thresholds;

        // ---- Budget exhaustion is the first thing we check ----
        if state.budget.remaining_iterations == 0 {
            return decision(
                RetrievalAction::Stop {
                    reason: StopReason::BudgetExhausted,
                },
                0.0,
                "iteration budget exhausted".to_string(),
            );
        }

        // ---- No-improvement gate: if the previous action ran but didn't
        // help, stop. This is the conservative answer to "the controller
        // tried something and it didn't work". ----
        if let Some(last) = state.history.last() {
            if let Some(gain) = last.actual_gain {
                if gain < t.min_continuation_gain && !last.action.is_terminal() {
                    return decision(
                        RetrievalAction::Stop {
                            reason: StopReason::NoImprovement,
                        },
                        0.0,
                        format!(
                            "previous {} produced actual_gain={:.3} < min_continuation_gain={:.3}",
                            last.action.code(),
                            gain,
                            t.min_continuation_gain
                        ),
                    );
                }
            }
        }

        // ---- Regime-driven decisions ----
        let Some(regime) = state.regime.as_ref() else {
            // No classifier configured. The policy must do *nothing*.
            return decision(
                RetrievalAction::Stop {
                    reason: StopReason::NoSignal,
                },
                0.0,
                "no regime classification available".to_string(),
            );
        };

        // 1. Sparse → Abstain (highest priority — guards against feeding
        //    the LLM hallucination fuel).
        let p_sparse = regime.p(RetrievalRegime::Sparse);
        if p_sparse >= t.min_p_sparse {
            return decision(
                RetrievalAction::Abstain {
                    reason: AbstainReason::Sparse,
                },
                0.0,
                format!(
                    "p(Sparse)={:.2}≥{:.2}: corpus likely does not contain the answer",
                    p_sparse, t.min_p_sparse
                ),
            );
        }

        // 2. Easy → Stop(Confident). One of the two most common outcomes;
        //    this is the "do nothing" path for well-grounded queries.
        let p_easy = regime.p(RetrievalRegime::Easy);
        if p_easy >= t.min_p_easy {
            return decision(
                RetrievalAction::Stop {
                    reason: StopReason::Confident,
                },
                0.0,
                format!(
                    "p(Easy)={:.2}≥{:.2}: retrieval looks well-grounded",
                    p_easy, t.min_p_easy
                ),
            );
        }

        // 3. Saturated → Stop(SaturatedNoBenefit). More retrieval will not
        //    help.
        let p_saturated = regime.p(RetrievalRegime::Saturated);
        if p_saturated >= t.min_p_saturated {
            return decision(
                RetrievalAction::Stop {
                    reason: StopReason::SaturatedNoBenefit,
                },
                0.0,
                format!(
                    "p(Saturated)={:.2}≥{:.2}: top-k tail repeats head",
                    p_saturated, t.min_p_saturated
                ),
            );
        }

        // 4. DistractorHeavy → EscalateReranker, if we can.
        let p_distractor = regime.p(RetrievalRegime::DistractorHeavy);
        if p_distractor >= t.min_p_distractor {
            if state.budget.remaining_rerank_calls == 0 {
                return decision(
                    RetrievalAction::Stop {
                        reason: StopReason::SaturatedNoBenefit,
                    },
                    0.0,
                    format!(
                        "p(DistractorHeavy)={:.2}≥{:.2} but reranker budget exhausted",
                        p_distractor, t.min_p_distractor
                    ),
                );
            }
            let Some(next_level) = state.reranker_level.escalate() else {
                return decision(
                    RetrievalAction::Stop {
                        reason: StopReason::SaturatedNoBenefit,
                    },
                    0.0,
                    "reranker already at top tier; nothing to escalate".to_string(),
                );
            };
            // One-escalation rule: if we've escalated already in this
            // session, do not do so again. Repeated escalation does not
            // earn its complexity on our traces.
            let already_escalated = state.history.iter().any(|t| {
                matches!(t.action, RetrievalAction::EscalateReranker { .. })
            });
            if already_escalated {
                return decision(
                    RetrievalAction::Stop {
                        reason: StopReason::NoImprovement,
                    },
                    0.0,
                    "already escalated reranker once; one-escalation policy".to_string(),
                );
            }
            return decision(
                RetrievalAction::EscalateReranker {
                    from: state.reranker_level,
                    to: next_level,
                },
                expected_gain_escalate(p_distractor),
                format!(
                    "p(DistractorHeavy)={:.2}≥{:.2}: escalate {} → {}",
                    p_distractor, t.min_p_distractor, state.reranker_level, next_level
                ),
            );
        }

        // 5. Ambiguous → ExpandTopK, if we can.
        let p_ambiguous = regime.p(RetrievalRegime::Ambiguous);
        if p_ambiguous >= t.min_p_ambiguous {
            let proposed = state.current_top_k + t.top_k_step;
            let new_k = proposed.min(state.budget.max_top_k);
            if new_k <= state.current_top_k {
                return decision(
                    RetrievalAction::Stop {
                        reason: StopReason::SaturatedNoBenefit,
                    },
                    0.0,
                    format!(
                        "p(Ambiguous)={:.2}≥{:.2} but top_k {} already at max {}",
                        p_ambiguous,
                        t.min_p_ambiguous,
                        state.current_top_k,
                        state.budget.max_top_k
                    ),
                );
            }
            // One-expansion rule (parallel to one-escalation rule).
            let already_expanded = state.history.iter().any(|t| {
                matches!(t.action, RetrievalAction::ExpandTopK { .. })
            });
            if already_expanded {
                return decision(
                    RetrievalAction::Stop {
                        reason: StopReason::NoImprovement,
                    },
                    0.0,
                    "already expanded top-k once; one-expansion policy".to_string(),
                );
            }
            return decision(
                RetrievalAction::ExpandTopK {
                    from: state.current_top_k,
                    to: new_k,
                },
                expected_gain_expand(p_ambiguous),
                format!(
                    "p(Ambiguous)={:.2}≥{:.2}: expand top_k {} → {}",
                    p_ambiguous, t.min_p_ambiguous, state.current_top_k, new_k
                ),
            );
        }

        // 6. Nothing fired with enough confidence — do nothing.
        decision(
            RetrievalAction::Stop {
                reason: StopReason::NoSignal,
            },
            0.0,
            format!(
                "no regime cleared its threshold: argmax={} with p={:.2}",
                regime.argmax,
                regime.p(regime.argmax)
            ),
        )
    }

    fn name(&self) -> &'static str {
        "conservative_rule"
    }
}

fn decision(action: RetrievalAction, expected_gain: f32, rationale: String) -> PolicyDecision {
    PolicyDecision {
        action,
        expected_gain,
        rationale,
    }
}

/// Calibrated expected gain for an `EscalateReranker` action.
///
/// On our internal traces, escalating from lexical to semantic reranker
/// in the DistractorHeavy regime improves the composite evidence-quality
/// metric by ≈0.10 on average. We scale by `p(DistractorHeavy)` so a
/// barely-firing rule predicts less gain than a confidently-firing one.
fn expected_gain_escalate(p_distractor: f32) -> f32 {
    (0.10 * p_distractor).clamp(0.0, 1.0)
}

/// Calibrated expected gain for an `ExpandTopK` action.
///
/// Smaller than escalation because expansion typically just *finds* better
/// evidence to feed the reranker, rather than directly improving the
/// candidate set.
fn expected_gain_expand(p_ambiguous: f32) -> f32 {
    (0.05 * p_ambiguous).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neorag_core::{
        Budget, ConfidenceProfile, DiagnosticsReport, Query, RegimeDistribution, RerankerLevel,
        RetrievalRegime,
    };
    use std::collections::BTreeMap;

    fn state_with_regime(probs: &[(RetrievalRegime, f32)]) -> RetrievalState {
        let mut probabilities = BTreeMap::new();
        for &(r, p) in probs {
            probabilities.insert(r, p);
        }
        // Fill in any missing regimes with 0.
        for r in RetrievalRegime::all() {
            probabilities.entry(*r).or_insert(0.0);
        }
        let argmax = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(r, _)| *r)
            .unwrap();
        let regime = RegimeDistribution {
            probabilities,
            argmax,
            trace: Default::default(),
        };
        let mut s = RetrievalState::new(
            Query::new("q"),
            Vec::new(),
            DiagnosticsReport::default(),
            ConfidenceProfile::default(),
        );
        s.regime = Some(regime);
        s.current_top_k = 10;
        s.budget = Budget::default();
        s
    }

    #[test]
    fn easy_yields_stop_confident() {
        let s = state_with_regime(&[(RetrievalRegime::Easy, 0.7)]);
        let d = ConservativeRulePolicy::new().decide(&s);
        assert!(matches!(
            d.action,
            RetrievalAction::Stop {
                reason: StopReason::Confident
            }
        ));
        assert_eq!(d.expected_gain, 0.0);
    }

    #[test]
    fn sparse_yields_abstain_above_threshold() {
        let s = state_with_regime(&[(RetrievalRegime::Sparse, 0.6)]);
        let d = ConservativeRulePolicy::new().decide(&s);
        assert!(matches!(
            d.action,
            RetrievalAction::Abstain {
                reason: AbstainReason::Sparse
            }
        ));
    }

    #[test]
    fn saturated_yields_stop_no_benefit() {
        let s = state_with_regime(&[(RetrievalRegime::Saturated, 0.6)]);
        let d = ConservativeRulePolicy::new().decide(&s);
        assert!(matches!(
            d.action,
            RetrievalAction::Stop {
                reason: StopReason::SaturatedNoBenefit
            }
        ));
    }

    #[test]
    fn distractor_heavy_escalates_reranker() {
        let s = state_with_regime(&[(RetrievalRegime::DistractorHeavy, 0.6)]);
        let d = ConservativeRulePolicy::new().decide(&s);
        match d.action {
            RetrievalAction::EscalateReranker { from, to } => {
                assert_eq!(from, RerankerLevel::None);
                assert_eq!(to, RerankerLevel::Lexical);
            }
            other => panic!("expected EscalateReranker, got {other:?}"),
        }
        assert!(d.expected_gain > 0.0);
    }

    #[test]
    fn distractor_heavy_does_not_escalate_twice() {
        let mut s = state_with_regime(&[(RetrievalRegime::DistractorHeavy, 0.6)]);
        s.history.push(neorag_core::TakenAction {
            action: RetrievalAction::EscalateReranker {
                from: RerankerLevel::None,
                to: RerankerLevel::Lexical,
            },
            iteration: 0,
            expected_gain: 0.06,
            actual_gain: Some(0.05),
            pre_diagnostics: DiagnosticsReport::default(),
            post_diagnostics: Some(DiagnosticsReport::default()),
            latency_ms: 1,
            cost: Default::default(),
            rationale: "n/a".into(),
        });
        s.reranker_level = RerankerLevel::Lexical;
        let d = ConservativeRulePolicy::new().decide(&s);
        // Either we hit the one-escalation rule (NoImprovement) or the
        // no-improvement gate fires first; either is conservative.
        assert!(matches!(d.action, RetrievalAction::Stop { .. }));
    }

    #[test]
    fn ambiguous_expands_top_k() {
        let s = state_with_regime(&[(RetrievalRegime::Ambiguous, 0.6)]);
        let d = ConservativeRulePolicy::new().decide(&s);
        match d.action {
            RetrievalAction::ExpandTopK { from, to } => {
                assert_eq!(from, 10);
                assert!(to > from);
            }
            other => panic!("expected ExpandTopK, got {other:?}"),
        }
    }

    #[test]
    fn no_signal_yields_stop_no_signal() {
        // All regimes below their thresholds.
        let s = state_with_regime(&[
            (RetrievalRegime::Easy, 0.2),
            (RetrievalRegime::Saturated, 0.2),
            (RetrievalRegime::DistractorHeavy, 0.2),
            (RetrievalRegime::Ambiguous, 0.2),
            (RetrievalRegime::Sparse, 0.2),
        ]);
        let d = ConservativeRulePolicy::new().decide(&s);
        assert!(matches!(
            d.action,
            RetrievalAction::Stop {
                reason: StopReason::NoSignal
            }
        ));
    }

    #[test]
    fn no_classifier_yields_stop_no_signal() {
        let s = RetrievalState::new(
            Query::new("q"),
            Vec::new(),
            DiagnosticsReport::default(),
            ConfidenceProfile::default(),
        );
        let d = ConservativeRulePolicy::new().decide(&s);
        assert!(matches!(
            d.action,
            RetrievalAction::Stop {
                reason: StopReason::NoSignal
            }
        ));
    }

    #[test]
    fn budget_exhaustion_terminates_first() {
        let mut s = state_with_regime(&[(RetrievalRegime::DistractorHeavy, 0.9)]);
        s.budget.remaining_iterations = 0;
        let d = ConservativeRulePolicy::new().decide(&s);
        assert!(matches!(
            d.action,
            RetrievalAction::Stop {
                reason: StopReason::BudgetExhausted
            }
        ));
    }

    #[test]
    fn previous_action_with_no_gain_stops_loop() {
        let mut s = state_with_regime(&[(RetrievalRegime::Ambiguous, 0.7)]);
        s.history.push(neorag_core::TakenAction {
            action: RetrievalAction::ExpandTopK { from: 10, to: 18 },
            iteration: 0,
            expected_gain: 0.04,
            actual_gain: Some(0.001), // below threshold
            pre_diagnostics: DiagnosticsReport::default(),
            post_diagnostics: Some(DiagnosticsReport::default()),
            latency_ms: 1,
            cost: Default::default(),
            rationale: "n/a".into(),
        });
        let d = ConservativeRulePolicy::new().decide(&s);
        assert!(matches!(
            d.action,
            RetrievalAction::Stop {
                reason: StopReason::NoImprovement
            }
        ));
    }
}
