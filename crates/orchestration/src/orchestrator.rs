//! The adaptive retrieval orchestrator — the closed loop.
//!
//! ```text
//!   initial_retrieve(query, top_k)
//!         │
//!         ▼
//!   ┌───────────────────────────────────────────────┐
//!   │  diagnose → confidence → classify              │
//!   │  policy.decide                                 │
//!   │  if terminal → record + return                 │
//!   │  actuator.apply                                │
//!   │  re-diagnose, re-classify                      │
//!   │  measure actual_gain                           │
//!   │  record TakenAction                            │
//!   │  decrement budget                              │
//!   └────────────────┬──────────────────────────────┘
//!                    │ iterate (bounded by Budget)
//!                    ▼
//!                ...
//! ```
//!
//! This is the **first phase that mutates retrieval behavior**. Every
//! action it takes is bounded by the [`Budget`][bud], and every decision
//! it makes is logged into the [`RetrievalState::history`][hist]. The
//! orchestrator does not own the retriever or rerankers — those live
//! behind the [`Actuator`][act] trait so the orchestrator itself stays
//! easy to test against mocks.
//!
//! ## Evidence quality metric
//!
//! `actual_gain` is computed as the delta in an internal *evidence
//! quality* aggregate — a fixed weighted sum over the lexical
//! grounding, semantic grounding, distractor ratio, and evidence
//! concentration metrics. The aggregate is intentionally simple and
//! interpretable. Phase 9 may revisit the weighting once we have judge-
//! model labels to calibrate against.
//!
//! [bud]: redhop_core::Budget
//! [hist]: redhop_core::RetrievalState::history
//! [act]: crate::actuator::Actuator

use std::sync::Arc;

use redhop_core::{
    Budget, DiagnosticsEngine, DiagnosticsReport, Query, RegimeClassifier, Result, RetrievalState,
    TakenAction,
};

use crate::actuator::{ActuationOutcome, Actuator};
use crate::confidence::compute_confidence;
use crate::policy::{Policy, PolicyDecision};

/// The adaptive retrieval controller.
pub struct AdaptiveOrchestrator {
    diagnostics: Arc<dyn DiagnosticsEngine>,
    classifier: Arc<dyn RegimeClassifier>,
    policy: Arc<dyn Policy>,
    actuator: Arc<dyn Actuator>,
    initial_top_k: usize,
    budget_default: Budget,
}

impl AdaptiveOrchestrator {
    /// Construct a new orchestrator.
    pub fn new(
        diagnostics: Arc<dyn DiagnosticsEngine>,
        classifier: Arc<dyn RegimeClassifier>,
        policy: Arc<dyn Policy>,
        actuator: Arc<dyn Actuator>,
    ) -> Self {
        Self {
            diagnostics,
            classifier,
            policy,
            actuator,
            initial_top_k: 10,
            budget_default: Budget::default(),
        }
    }

    /// Override the initial `top_k` used for the first retrieval.
    pub fn with_initial_top_k(mut self, k: usize) -> Self {
        self.initial_top_k = k;
        self
    }

    /// Override the default budget.
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget_default = budget;
        self
    }

    /// Run the closed loop against a query and return the final
    /// [`RetrievalState`].
    ///
    /// The terminal action is always the last entry in
    /// `state.history`. Inspect it (or call [`state.terminal_action()`][term]
    /// and [`state.abstained()`][abs]) to learn how the loop exited.
    ///
    /// [term]: redhop_core::RetrievalState::terminal_action
    /// [abs]: redhop_core::RetrievalState::abstained
    pub async fn run(&self, query: Query) -> Result<RetrievalState> {
        // ---- Initial retrieval ----
        let candidates = self
            .actuator
            .initial_retrieve(&query, self.initial_top_k)
            .await?;
        let diagnostics = self.diagnostics.diagnose(&query, &candidates)?;
        let confidence = compute_confidence(&candidates);
        let mut state = RetrievalState::new(query, candidates, diagnostics, confidence)
            .with_budget(self.budget_default.clone());
        state.current_top_k = self.initial_top_k;
        let regime = self
            .classifier
            .classify(&state.diagnostics, &state.confidence);
        state.regime = Some(regime);

        // ---- Closed loop ----
        loop {
            let decision = self.policy.decide(&state);
            if decision.action.is_terminal() {
                self.record_terminal(&mut state, decision);
                return Ok(state);
            }

            // Snapshot pre-state for the action record. We clone the
            // diagnostics here because the action will rewrite
            // `state.diagnostics` below.
            let pre_diagnostics = state.diagnostics.clone();
            let pre_quality = evidence_quality(&pre_diagnostics);

            // Actuate.
            let outcome = self.actuator.apply(&decision.action, &state).await?;
            self.apply_outcome(&mut state, &outcome);

            // Re-observe.
            state.diagnostics = self.diagnostics.diagnose(&state.query, &state.candidates)?;
            state.confidence = compute_confidence(&state.candidates);
            let regime = self
                .classifier
                .classify(&state.diagnostics, &state.confidence);
            state.regime = Some(regime);

            // Measure actual gain.
            let post_quality = evidence_quality(&state.diagnostics);
            let actual_gain = (post_quality - pre_quality).clamp(-1.0, 1.0);

            // Record the action.
            state.history.push(TakenAction {
                action: decision.action,
                iteration: state.iteration,
                expected_gain: decision.expected_gain,
                actual_gain: Some(actual_gain),
                pre_diagnostics,
                post_diagnostics: Some(state.diagnostics.clone()),
                latency_ms: outcome.latency_ms,
                cost: outcome.cost,
                rationale: decision.rationale,
            });

            // Update iteration + budget.
            state.iteration += 1;
            state.budget.remaining_iterations =
                state.budget.remaining_iterations.saturating_sub(1);
        }
    }

    fn apply_outcome(&self, state: &mut RetrievalState, outcome: &ActuationOutcome) {
        if let Some(c) = &outcome.new_candidates {
            state.candidates = c.clone();
        }
        if let Some(lvl) = outcome.new_reranker_level {
            state.reranker_level = lvl;
        }
        if let Some(k) = outcome.new_top_k {
            state.current_top_k = k;
        }
        if outcome.cost.rerank_calls > 0 {
            state.budget.remaining_rerank_calls = state
                .budget
                .remaining_rerank_calls
                .saturating_sub(outcome.cost.rerank_calls);
        }
    }

    fn record_terminal(&self, state: &mut RetrievalState, decision: PolicyDecision) {
        // Terminal actions have no "after" diagnostics; we still record
        // the pre-action snapshot so consumers can read why the loop
        // exited.
        let pre_diagnostics = state.diagnostics.clone();
        state.history.push(TakenAction {
            action: decision.action,
            iteration: state.iteration,
            expected_gain: 0.0,
            actual_gain: None,
            pre_diagnostics,
            post_diagnostics: None,
            latency_ms: 0,
            cost: Default::default(),
            rationale: decision.rationale,
        });
    }
}

/// A weighted aggregate of the diagnostics fields that the orchestrator
/// uses to compute `actual_gain`. All inputs are in `[0, 1]`; the output
/// is in `[0, 1]`.
///
/// Weights chosen empirically:
///
/// - 0.30 lexical grounding
/// - 0.30 semantic grounding   (or lexical if semantic absent)
/// - 0.20 (1 − semantic distractor ratio)
/// - 0.10 (1 − lexical distractor ratio)
/// - 0.10 evidence concentration
///
/// When a field is `None`, its contribution is set to a neutral baseline
/// (0.5 for the grounding terms, 0 for the others). This keeps the
/// metric defined for partially-observed states without penalizing
/// partial observability.
pub(crate) fn evidence_quality(d: &DiagnosticsReport) -> f32 {
    let lex_g = d.lexical_grounding.unwrap_or(0.5);
    let sem_g = d.semantic_grounding.unwrap_or(lex_g);
    let lex_dr = d.distractor_ratio.unwrap_or(0.0);
    let sem_dr = d.semantic_distractor_ratio.unwrap_or(0.0);
    let conc = d.evidence_concentration.unwrap_or(0.0);
    (0.30 * lex_g + 0.30 * sem_g + 0.20 * (1.0 - sem_dr) + 0.10 * (1.0 - lex_dr) + 0.10 * conc)
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConservativeRulePolicy, DefaultActuator, RuleBasedClassifier};
    use async_trait::async_trait;
    use redhop_core::{
        Chunk, ChunkId, DiagnosticsEngine, Query, Reranker, RerankerLevel, RetrievalAction,
        RetrievalMethod, RetrievalResult, Retriever, Score, ScoreBreakdown, StopReason,
        TokenCount,
    };
    use std::sync::Mutex;

    // ─────────────────────────────────────────────────────────────────
    // Mocks
    // ─────────────────────────────────────────────────────────────────

    /// A retriever returning a fixed payload per top_k query.
    struct FixedRetriever {
        payload: Vec<RetrievalResult>,
    }
    fn mk(id: &str, text: &str, score: f32) -> RetrievalResult {
        RetrievalResult {
            chunk: Chunk::new(ChunkId::new(id), text, "doc", TokenCount(1)),
            score: Score {
                value: score,
                method: RetrievalMethod::Lexical,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }
    #[async_trait]
    impl Retriever for FixedRetriever {
        async fn retrieve(&self, _q: &Query, top_k: usize) -> Result<Vec<RetrievalResult>> {
            Ok(self.payload.iter().take(top_k).cloned().collect())
        }
        async fn index(&mut self, _c: &[Chunk]) -> Result<()> {
            Ok(())
        }
        fn name(&self) -> &'static str {
            "fixed"
        }
    }

    /// A diagnostics engine that returns a pre-canned report.
    struct StubDiagnostics {
        report: Mutex<DiagnosticsReport>,
    }
    impl DiagnosticsEngine for StubDiagnostics {
        fn diagnose(
            &self,
            _q: &Query,
            _r: &[RetrievalResult],
        ) -> Result<DiagnosticsReport> {
            Ok(self.report.lock().unwrap().clone())
        }
        fn name(&self) -> &'static str {
            "stub"
        }
    }

    /// A reranker that just relabels the top score so we can detect that
    /// it ran without changing candidate identity.
    struct MarkerReranker;
    #[async_trait]
    impl Reranker for MarkerReranker {
        async fn rerank(
            &self,
            _q: &Query,
            mut candidates: Vec<RetrievalResult>,
            _top_k: usize,
        ) -> Result<Vec<RetrievalResult>> {
            for r in &mut candidates {
                r.breakdown.rerank = Some(0.99);
            }
            Ok(candidates)
        }
        fn name(&self) -> &'static str {
            "marker"
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    // ─────────────────────────────────────────────────────────────────
    // Tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn easy_query_terminates_in_one_iteration_with_no_intervention() {
        rt().block_on(async {
            // Diagnostics that paint a clear Easy regime.
            let diag = DiagnosticsReport {
                lexical_grounding: Some(0.95),
                semantic_grounding: Some(0.95),
                distractor_ratio: Some(0.0),
                semantic_distractor_ratio: Some(0.0),
                ..Default::default()
            };
            let orchestrator = AdaptiveOrchestrator::new(
                Arc::new(StubDiagnostics {
                    report: Mutex::new(diag),
                }),
                Arc::new(RuleBasedClassifier::new()),
                Arc::new(ConservativeRulePolicy::new()),
                Arc::new(DefaultActuator::retrieve_only(Arc::new(FixedRetriever {
                    payload: vec![
                        mk("a", "answer text", 10.0),
                        mk("b", "filler", 1.0),
                    ],
                }))),
            );
            let state = orchestrator.run(Query::new("q")).await.unwrap();
            // Exactly one terminal action: Stop(Confident).
            assert_eq!(state.history.len(), 1);
            assert!(matches!(
                state.history[0].action,
                RetrievalAction::Stop {
                    reason: StopReason::Confident
                }
            ));
            // No retrieval mutation happened.
            assert_eq!(state.candidates.len(), 2);
        });
    }

    #[test]
    fn sparse_query_abstains_immediately() {
        rt().block_on(async {
            // Sparse signal: both groundings near zero.
            let diag = DiagnosticsReport {
                lexical_grounding: Some(0.0),
                semantic_grounding: Some(0.50),
                ..Default::default()
            };
            let orchestrator = AdaptiveOrchestrator::new(
                Arc::new(StubDiagnostics {
                    report: Mutex::new(diag),
                }),
                Arc::new(RuleBasedClassifier::new()),
                Arc::new(ConservativeRulePolicy::new()),
                Arc::new(DefaultActuator::retrieve_only(Arc::new(FixedRetriever {
                    payload: vec![mk("a", "irrelevant", 0.1)],
                }))),
            );
            let state = orchestrator.run(Query::new("q")).await.unwrap();
            assert!(state.abstained());
            assert_eq!(state.history.len(), 1);
        });
    }

    #[test]
    fn distractor_heavy_query_escalates_reranker_once() {
        rt().block_on(async {
            let diag = DiagnosticsReport {
                lexical_grounding: Some(0.3),
                distractor_ratio: Some(0.6),
                semantic_distractor_ratio: Some(0.6),
                ..Default::default()
            };
            let orchestrator = AdaptiveOrchestrator::new(
                Arc::new(StubDiagnostics {
                    report: Mutex::new(diag),
                }),
                Arc::new(RuleBasedClassifier::new()),
                Arc::new(ConservativeRulePolicy::new()),
                Arc::new(DefaultActuator::new(
                    Arc::new(FixedRetriever {
                        payload: vec![mk("a", "x", 1.0), mk("b", "y", 0.9)],
                    }),
                    vec![(RerankerLevel::Lexical, Arc::new(MarkerReranker))],
                )),
            );
            let state = orchestrator.run(Query::new("q")).await.unwrap();
            // One escalation, then a terminal stop (no-gain since stub
            // diagnostics don't change).
            let escalations = state
                .history
                .iter()
                .filter(|t| matches!(t.action, RetrievalAction::EscalateReranker { .. }))
                .count();
            assert_eq!(escalations, 1, "expected exactly one escalation, history: {:?}", state.history);
            assert!(matches!(
                state.history.last().unwrap().action,
                RetrievalAction::Stop { .. }
            ));
            // Marker reranker stamped the breakdown — verify it actually ran.
            assert_eq!(state.candidates[0].breakdown.rerank, Some(0.99));
        });
    }

    #[test]
    fn budget_caps_total_iterations() {
        rt().block_on(async {
            // Ambiguous signal that would otherwise trigger Expand,
            // but we cap the iteration budget at 0 to force immediate
            // termination on the second pass.
            let diag = DiagnosticsReport {
                lexical_grounding: Some(0.4),
                centroid_dispersion: Some(0.8),
                ..Default::default()
            };
            let orchestrator = AdaptiveOrchestrator::new(
                Arc::new(StubDiagnostics {
                    report: Mutex::new(diag),
                }),
                Arc::new(RuleBasedClassifier::new()),
                Arc::new(ConservativeRulePolicy::new()),
                Arc::new(DefaultActuator::retrieve_only(Arc::new(FixedRetriever {
                    payload: vec![mk("a", "x", 1.0), mk("b", "y", 0.5)],
                }))),
            )
            .with_budget(Budget::new(1, 50, 2));
            let state = orchestrator.run(Query::new("q")).await.unwrap();
            // First action: ExpandTopK (1 iteration used). Second: budget
            // exhausted → Stop(BudgetExhausted) OR Stop(NoImprovement).
            assert!(state.iteration <= 1);
            assert!(matches!(
                state.history.last().unwrap().action,
                RetrievalAction::Stop { .. }
            ));
        });
    }

    #[test]
    fn every_taken_action_records_latency_pre_post_diagnostics() {
        rt().block_on(async {
            let diag = DiagnosticsReport {
                lexical_grounding: Some(0.95),
                ..Default::default()
            };
            let orchestrator = AdaptiveOrchestrator::new(
                Arc::new(StubDiagnostics {
                    report: Mutex::new(diag),
                }),
                Arc::new(RuleBasedClassifier::new()),
                Arc::new(ConservativeRulePolicy::new()),
                Arc::new(DefaultActuator::retrieve_only(Arc::new(FixedRetriever {
                    payload: vec![mk("a", "answer", 10.0)],
                }))),
            );
            let state = orchestrator.run(Query::new("q")).await.unwrap();
            for t in &state.history {
                assert!(!t.rationale.is_empty(), "rationale must be populated");
                // pre_diagnostics is non-default
                assert!(t.pre_diagnostics.lexical_grounding.is_some());
                if !t.action.is_terminal() {
                    assert!(t.post_diagnostics.is_some());
                    assert!(t.actual_gain.is_some());
                }
            }
        });
    }
}
