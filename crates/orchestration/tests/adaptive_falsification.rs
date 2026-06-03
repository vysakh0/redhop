//! Adaptive-controller falsification suite.
//!
//! The empirical bar the adaptive controller has to clear:
//!
//!   1. Adaptive must IMPROVE hard regimes (DistractorHeavy / Ambiguous /
//!      Sparse) compared to a static pipeline on the same retrieval, OR
//!      mark them as such so a downstream LLM is not misled.
//!   2. Adaptive must NOT HARM easy regimes. On `Easy`-classified queries
//!      the adaptive controller must take zero retrieval-mutating actions,
//!      and the final candidate list must equal what static retrieval
//!      would return.
//!   3. Adaptive must AVOID OVER-ACTUATION. Action counts must be tightly
//!      bounded by the policy (≤1 escalation, ≤1 expansion per session).
//!
//! Each test constructs synthetic states or a mock retriever whose
//! diagnostics paint a target regime. The tests assert behaviour, not
//! incidental properties — they should remain stable across reasonable
//! refactors of the policy weights, the actuator implementation, and the
//! classifier thresholds.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use redhop::core::{
    Chunk, ChunkId, DiagnosticsEngine, DiagnosticsReport, Query, Reranker, RerankerLevel,
    Result as CoreResult, RetrievalAction, RetrievalMethod, RetrievalResult, Retriever, Score,
    ScoreBreakdown, StopReason, TokenCount,
};
use redhop_orchestration::{
    AdaptiveOrchestrator, ConservativeRulePolicy, DefaultActuator, RuleBasedClassifier,
};

// ─── Test fixtures ───────────────────────────────────────────────────

struct FixedRetriever {
    payload: Vec<RetrievalResult>,
    calls: Mutex<u32>,
}
fn mk(id: &str, score: f32) -> RetrievalResult {
    RetrievalResult {
        chunk: Chunk::new(ChunkId::new(id), id, "doc", TokenCount(1)),
        score: Score {
            value: score,
            method: RetrievalMethod::Lexical,
        },
        breakdown: ScoreBreakdown::default(),
    }
}
#[async_trait]
impl Retriever for FixedRetriever {
    async fn retrieve(&self, _q: &Query, top_k: usize) -> CoreResult<Vec<RetrievalResult>> {
        *self.calls.lock().unwrap() += 1;
        Ok(self.payload.iter().take(top_k).cloned().collect())
    }
    async fn index(&mut self, _c: &[Chunk]) -> CoreResult<()> {
        Ok(())
    }
    fn name(&self) -> &'static str {
        "fixed"
    }
}

/// A stub diagnostics engine that returns one report on the first call
/// and a different one on subsequent calls — useful for simulating "the
/// reranker fixed things" mid-loop.
struct ScriptedDiagnostics {
    reports: Mutex<Vec<DiagnosticsReport>>,
}
impl ScriptedDiagnostics {
    fn new(reports: Vec<DiagnosticsReport>) -> Self {
        Self {
            reports: Mutex::new(reports),
        }
    }
}
impl DiagnosticsEngine for ScriptedDiagnostics {
    fn diagnose(&self, _q: &Query, _r: &[RetrievalResult]) -> CoreResult<DiagnosticsReport> {
        let mut g = self.reports.lock().unwrap();
        if g.is_empty() {
            return Ok(DiagnosticsReport::default());
        }
        if g.len() == 1 {
            return Ok(g[0].clone());
        }
        Ok(g.remove(0))
    }
    fn name(&self) -> &'static str {
        "scripted"
    }
}

/// Reranker that "cleans up" distractor signal by trimming the back half
/// of the candidate list — the actuator-side analog of "semantic
/// reranker pushed off-topic candidates down."
struct CleanupReranker;
#[async_trait]
impl Reranker for CleanupReranker {
    async fn rerank(
        &self,
        _q: &Query,
        mut candidates: Vec<RetrievalResult>,
        _top_k: usize,
    ) -> CoreResult<Vec<RetrievalResult>> {
        let half = candidates.len().div_ceil(2);
        candidates.truncate(half);
        for r in &mut candidates {
            r.breakdown.rerank = Some(0.99);
        }
        Ok(candidates)
    }
    fn name(&self) -> &'static str {
        "cleanup"
    }
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

// ─── Falsification claim 1: neutral on easy ──────────────────────────

#[test]
fn easy_query_takes_exactly_one_terminal_stop_action() {
    rt().block_on(async {
        let diag = DiagnosticsReport {
            lexical_grounding: Some(0.92),
            semantic_grounding: Some(0.95),
            distractor_ratio: Some(0.0),
            semantic_distractor_ratio: Some(0.0),
            ..Default::default()
        };
        let retriever = Arc::new(FixedRetriever {
            payload: vec![mk("a", 10.0), mk("b", 8.0), mk("c", 6.0)],
            calls: Mutex::new(0),
        });
        let orchestrator = AdaptiveOrchestrator::new(
            Arc::new(ScriptedDiagnostics::new(vec![diag])),
            Arc::new(RuleBasedClassifier::new()),
            Arc::new(ConservativeRulePolicy::new()),
            Arc::new(DefaultActuator::new(
                retriever.clone(),
                vec![(RerankerLevel::Lexical, Arc::new(CleanupReranker))],
            )),
        );
        let state = orchestrator.run(Query::new("q")).await.unwrap();

        // Exactly one action, terminal, Stop(Confident).
        assert_eq!(state.history.len(), 1, "history: {:?}", state.history);
        assert!(
            matches!(
                state.history[0].action,
                RetrievalAction::Stop {
                    reason: StopReason::Confident
                }
            ),
            "got action={:?}, rationale={}",
            state.history[0].action,
            state.history[0].rationale
        );
        // Exactly one retriever call (the initial retrieve). No mutation.
        assert_eq!(*retriever.calls.lock().unwrap(), 1);
        // Candidates unchanged from initial retrieval.
        assert_eq!(state.candidates.len(), 3);
        assert_eq!(state.candidates[0].chunk.id.as_str(), "a");
    });
}

#[test]
fn easy_query_output_identical_to_static_pipeline() {
    // The strongest "neutral on easy" check: the candidate set after
    // adaptive must equal the candidate set after static retrieve.
    rt().block_on(async {
        let easy_diag = DiagnosticsReport {
            lexical_grounding: Some(0.92),
            semantic_grounding: Some(0.95),
            ..Default::default()
        };
        let static_payload = vec![mk("alpha", 9.0), mk("bravo", 5.0)];
        let retriever_adaptive = Arc::new(FixedRetriever {
            payload: static_payload.clone(),
            calls: Mutex::new(0),
        });
        let orchestrator = AdaptiveOrchestrator::new(
            Arc::new(ScriptedDiagnostics::new(vec![easy_diag])),
            Arc::new(RuleBasedClassifier::new()),
            Arc::new(ConservativeRulePolicy::new()),
            Arc::new(DefaultActuator::retrieve_only(retriever_adaptive)),
        );
        let state = orchestrator.run(Query::new("q")).await.unwrap();
        let adaptive_ids: Vec<_> = state
            .candidates
            .iter()
            .map(|r| r.chunk.id.as_str().to_string())
            .collect();
        let static_ids: Vec<_> = static_payload
            .iter()
            .map(|r| r.chunk.id.as_str().to_string())
            .collect();
        assert_eq!(adaptive_ids, static_ids);
    });
}

// ─── Falsification claim 2: improves hard regimes ────────────────────

#[test]
fn distractor_heavy_escalates_and_records_measurable_actual_gain() {
    rt().block_on(async {
        // Pre-action: distractor-heavy. Post-action (after CleanupReranker
        // trims the back half): semantically clean.
        let pre = DiagnosticsReport {
            lexical_grounding: Some(0.3),
            distractor_ratio: Some(0.6),
            semantic_distractor_ratio: Some(0.6),
            ..Default::default()
        };
        let post = DiagnosticsReport {
            lexical_grounding: Some(0.65),
            semantic_grounding: Some(0.85),
            distractor_ratio: Some(0.0),
            semantic_distractor_ratio: Some(0.0),
            evidence_concentration: Some(0.8),
            ..Default::default()
        };
        let retriever = Arc::new(FixedRetriever {
            payload: vec![mk("a", 10.0), mk("b", 5.0), mk("c", 4.0), mk("d", 3.0)],
            calls: Mutex::new(0),
        });
        let orchestrator = AdaptiveOrchestrator::new(
            Arc::new(ScriptedDiagnostics::new(vec![pre, post])),
            Arc::new(RuleBasedClassifier::new()),
            Arc::new(ConservativeRulePolicy::new()),
            Arc::new(DefaultActuator::new(
                retriever.clone(),
                vec![(RerankerLevel::Lexical, Arc::new(CleanupReranker))],
            )),
        );
        let state = orchestrator.run(Query::new("q")).await.unwrap();

        // First action must be EscalateReranker.
        match &state.history[0].action {
            RetrievalAction::EscalateReranker { from, to } => {
                assert_eq!(*from, RerankerLevel::None);
                assert_eq!(*to, RerankerLevel::Lexical);
            }
            other => panic!("expected EscalateReranker first, got {other:?}"),
        }
        // Actual gain MUST be positive — that's the empirical claim.
        let gain = state.history[0]
            .actual_gain
            .expect("non-terminal action must have actual_gain");
        assert!(gain > 0.05, "expected meaningful positive gain, got {gain}");
        // And expected_gain must be positive too — the policy predicted
        // some improvement.
        assert!(state.history[0].expected_gain > 0.0);
        // CleanupReranker actually ran.
        assert_eq!(state.candidates[0].breakdown.rerank, Some(0.99));
    });
}

#[test]
fn sparse_query_abstains_with_zero_retrieval_mutation() {
    rt().block_on(async {
        let diag = DiagnosticsReport {
            lexical_grounding: Some(0.02),
            semantic_grounding: Some(0.50),
            ..Default::default()
        };
        let retriever = Arc::new(FixedRetriever {
            payload: vec![mk("noise", 0.1)],
            calls: Mutex::new(0),
        });
        let orchestrator = AdaptiveOrchestrator::new(
            Arc::new(ScriptedDiagnostics::new(vec![diag])),
            Arc::new(RuleBasedClassifier::new()),
            Arc::new(ConservativeRulePolicy::new()),
            Arc::new(DefaultActuator::retrieve_only(retriever.clone())),
        );
        let state = orchestrator.run(Query::new("q")).await.unwrap();
        assert!(state.abstained(), "sparse query should abstain");
        // Exactly one retrieval call (initial); no mutation.
        assert_eq!(*retriever.calls.lock().unwrap(), 1);
        assert_eq!(state.history.len(), 1);
    });
}

#[test]
fn saturated_query_stops_with_zero_retrieval_mutation() {
    rt().block_on(async {
        let diag = DiagnosticsReport {
            lexical_grounding: Some(0.5),
            retrieval_saturation: Some(0.95),
            semantic_redundancy: Some(0.95),
            ..Default::default()
        };
        let retriever = Arc::new(FixedRetriever {
            payload: vec![mk("a", 1.0), mk("b", 0.9), mk("c", 0.85)],
            calls: Mutex::new(0),
        });
        let orchestrator = AdaptiveOrchestrator::new(
            Arc::new(ScriptedDiagnostics::new(vec![diag])),
            Arc::new(RuleBasedClassifier::new()),
            Arc::new(ConservativeRulePolicy::new()),
            Arc::new(DefaultActuator::retrieve_only(retriever.clone())),
        );
        let state = orchestrator.run(Query::new("q")).await.unwrap();
        assert!(matches!(
            state.history.last().unwrap().action,
            RetrievalAction::Stop {
                reason: StopReason::SaturatedNoBenefit
            }
        ));
        // No retrieval mutation.
        assert_eq!(*retriever.calls.lock().unwrap(), 1);
    });
}

// ─── Falsification claim 3: avoids over-actuation ────────────────────

#[test]
fn no_more_than_one_escalation_in_any_session() {
    rt().block_on(async {
        // Every diagnostic call returns distractor-heavy state. A naive
        // controller might escalate forever; ours must escalate at most
        // once.
        let dh = DiagnosticsReport {
            lexical_grounding: Some(0.3),
            distractor_ratio: Some(0.6),
            semantic_distractor_ratio: Some(0.6),
            ..Default::default()
        };
        let scripted =
            ScriptedDiagnostics::new(vec![dh.clone(), dh.clone(), dh.clone(), dh.clone()]);
        let retriever = Arc::new(FixedRetriever {
            payload: vec![mk("a", 5.0), mk("b", 4.0), mk("c", 3.0), mk("d", 2.0)],
            calls: Mutex::new(0),
        });
        let orchestrator = AdaptiveOrchestrator::new(
            Arc::new(scripted),
            Arc::new(RuleBasedClassifier::new()),
            Arc::new(ConservativeRulePolicy::new()),
            Arc::new(DefaultActuator::new(
                retriever.clone(),
                vec![(RerankerLevel::Lexical, Arc::new(CleanupReranker))],
            )),
        );
        let state = orchestrator.run(Query::new("q")).await.unwrap();
        let escalations = state
            .history
            .iter()
            .filter(|t| matches!(t.action, RetrievalAction::EscalateReranker { .. }))
            .count();
        assert!(
            escalations <= 1,
            "expected at most one escalation, got {escalations}: {:?}",
            state
                .history
                .iter()
                .map(|t| t.action.code())
                .collect::<Vec<_>>()
        );
        // And the final action is terminal.
        assert!(state.history.last().unwrap().action.is_terminal());
    });
}

#[test]
fn ambiguous_then_no_progress_stops_after_one_expansion() {
    rt().block_on(async {
        // Always report Ambiguous. The expansion will not change
        // quality (same diagnostic returned). The controller must stop
        // after one expansion via the "no improvement" gate.
        let amb = DiagnosticsReport {
            lexical_grounding: Some(0.4),
            centroid_dispersion: Some(0.85),
            ..Default::default()
        };
        let retriever = Arc::new(FixedRetriever {
            payload: vec![mk("a", 1.0); 50],
            calls: Mutex::new(0),
        });
        let orchestrator = AdaptiveOrchestrator::new(
            Arc::new(ScriptedDiagnostics::new(vec![amb])),
            Arc::new(RuleBasedClassifier::new()),
            Arc::new(ConservativeRulePolicy::new()),
            Arc::new(DefaultActuator::retrieve_only(retriever.clone())),
        );
        let state = orchestrator.run(Query::new("q")).await.unwrap();
        let expansions = state
            .history
            .iter()
            .filter(|t| matches!(t.action, RetrievalAction::ExpandTopK { .. }))
            .count();
        assert!(
            expansions <= 1,
            "expected at most one expansion, got {expansions}"
        );
        assert!(state.history.last().unwrap().action.is_terminal());
    });
}
