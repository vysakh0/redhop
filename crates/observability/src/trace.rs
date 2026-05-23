//! The [`RetrievalTrace`] — a serializable journey of one query.

use std::collections::BTreeMap;

use redhop_core::RetrievalState;
use serde::{Deserialize, Serialize};

/// A complete trace of one query's path through the adaptive controller.
///
/// Built from a finished [`RetrievalState`] via
/// [`RetrievalTrace::from_state`]. Everything here is derived from data
/// already present on the state — no extra recording happens on the hot
/// path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalTrace {
    /// The query text.
    pub query: String,
    /// Final argmax regime code, if a classifier ran.
    pub final_regime: Option<String>,
    /// Final regime probability distribution (code → mass).
    pub regime_probabilities: BTreeMap<String, f32>,
    /// One entry per action the controller took.
    pub iterations: Vec<TraceIteration>,
    /// Number of candidates in the final evidence set.
    pub final_candidate_count: usize,
    /// Final top-k in effect.
    pub final_top_k: usize,
    /// Final reranker tier reached.
    pub final_reranker_level: String,
    /// Code of the terminal action (`stop` / `abstain`).
    pub terminal_action: Option<String>,
    /// True iff the controller abstained.
    pub abstained: bool,
    /// Did the controller take any non-terminal (mutating) action?
    pub intervened: bool,
    /// Sum of per-action latency.
    pub total_latency_ms: u64,
    /// Total retrieval calls (initial + actions).
    pub total_retrieval_calls: u32,
    /// Total reranker calls.
    pub total_rerank_calls: u32,
}

/// One iteration of the control loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceIteration {
    /// Zero-indexed iteration.
    pub iteration: u32,
    /// Action code.
    pub action: String,
    /// The policy's human-readable rationale for the action.
    pub rationale: String,
    /// Policy-predicted gain.
    pub expected_gain: f32,
    /// Measured gain (None for terminal actions).
    pub actual_gain: Option<f32>,
    /// Action latency.
    pub latency_ms: u64,
    /// Retrieval calls this action triggered.
    pub retrieval_calls: u32,
    /// Reranker calls this action triggered.
    pub rerank_calls: u32,
    /// Net candidate-count delta.
    pub chunks_delta: i32,
    // Selected diagnostics observed *before* the action ran, for display.
    /// Lexical grounding at this iteration.
    pub lexical_grounding: Option<f32>,
    /// Semantic grounding at this iteration.
    pub semantic_grounding: Option<f32>,
    /// Lexical distractor ratio at this iteration.
    pub distractor_ratio: Option<f32>,
    /// Semantic redundancy at this iteration.
    pub semantic_redundancy: Option<f32>,
}

impl RetrievalTrace {
    /// Build a trace from a finished retrieval state.
    pub fn from_state(state: &RetrievalState) -> Self {
        let regime_probabilities = state
            .regime
            .as_ref()
            .map(|r| {
                r.probabilities
                    .iter()
                    .map(|(reg, p)| (reg.code().to_string(), *p))
                    .collect()
            })
            .unwrap_or_default();
        let final_regime = state.regime.as_ref().map(|r| r.argmax.code().to_string());

        let iterations: Vec<TraceIteration> = state
            .history
            .iter()
            .map(|t| TraceIteration {
                iteration: t.iteration,
                action: t.action.code().to_string(),
                rationale: t.rationale.clone(),
                expected_gain: t.expected_gain,
                actual_gain: t.actual_gain,
                latency_ms: t.latency_ms,
                retrieval_calls: t.cost.retrieval_calls,
                rerank_calls: t.cost.rerank_calls,
                chunks_delta: t.cost.chunks_delta,
                lexical_grounding: t.pre_diagnostics.lexical_grounding,
                semantic_grounding: t.pre_diagnostics.semantic_grounding,
                distractor_ratio: t.pre_diagnostics.distractor_ratio,
                semantic_redundancy: t.pre_diagnostics.semantic_redundancy,
            })
            .collect();

        let total_latency_ms = iterations.iter().map(|i| i.latency_ms).sum();
        let total_retrieval_calls = 1 + iterations.iter().map(|i| i.retrieval_calls).sum::<u32>();
        let total_rerank_calls = iterations.iter().map(|i| i.rerank_calls).sum();
        let intervened = state.history.iter().any(|t| !t.action.is_terminal());
        let terminal_action = state.history.last().and_then(|t| {
            if t.action.is_terminal() {
                Some(t.action.code().to_string())
            } else {
                None
            }
        });

        RetrievalTrace {
            query: state.query.text.clone(),
            final_regime,
            regime_probabilities,
            iterations,
            final_candidate_count: state.candidates.len(),
            final_top_k: state.current_top_k,
            final_reranker_level: state.reranker_level.code().to_string(),
            terminal_action,
            abstained: state.abstained(),
            intervened,
            total_latency_ms,
            total_retrieval_calls,
            total_rerank_calls,
        }
    }

    /// Serialize to a single-line JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Convenience: did this query's controller fire a reranker
    /// escalation?
    pub fn escalated(&self) -> bool {
        self.iterations
            .iter()
            .any(|i| i.action == "escalate_reranker")
    }

    /// Convenience: did this query's controller expand top-k?
    pub fn expanded(&self) -> bool {
        self.iterations.iter().any(|i| i.action == "expand_top_k")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop_core::{
        Budget, Chunk, ConfidenceProfile, DiagnosticsReport, Query, RegimeDistribution,
        RerankerLevel, RetrievalAction, RetrievalRegime, Score, StopReason, TakenAction,
        TokenCount,
    };
    use std::collections::BTreeMap;

    fn easy_state() -> RetrievalState {
        let mut probs = BTreeMap::new();
        for r in RetrievalRegime::all() {
            probs.insert(*r, 0.1);
        }
        probs.insert(RetrievalRegime::Easy, 0.6);
        let regime = RegimeDistribution {
            probabilities: probs,
            argmax: RetrievalRegime::Easy,
            trace: Default::default(),
        };
        let chunk = Chunk::new("c0", "evidence text", "doc", TokenCount(2));
        let mut state = RetrievalState::new(
            Query::new("what is the answer?"),
            vec![redhop_core::RetrievalResult::new(
                chunk,
                Score {
                    value: 9.0,
                    method: redhop_core::RetrievalMethod::Lexical,
                },
            )],
            DiagnosticsReport {
                lexical_grounding: Some(0.9),
                ..Default::default()
            },
            ConfidenceProfile::default(),
        );
        state.regime = Some(regime);
        state.budget = Budget::default();
        state.history.push(TakenAction {
            action: RetrievalAction::Stop {
                reason: StopReason::Confident,
            },
            iteration: 0,
            expected_gain: 0.0,
            actual_gain: None,
            pre_diagnostics: DiagnosticsReport {
                lexical_grounding: Some(0.9),
                ..Default::default()
            },
            post_diagnostics: None,
            latency_ms: 0,
            cost: Default::default(),
            rationale: "p(Easy)=0.60 ≥ 0.40: retrieval looks well-grounded".into(),
        });
        let _ = RerankerLevel::None;
        state
    }

    #[test]
    fn trace_captures_easy_query_journey() {
        let state = easy_state();
        let trace = RetrievalTrace::from_state(&state);
        assert_eq!(trace.query, "what is the answer?");
        assert_eq!(trace.final_regime.as_deref(), Some("easy"));
        assert_eq!(trace.iterations.len(), 1);
        assert_eq!(trace.iterations[0].action, "stop");
        assert!(!trace.intervened);
        assert_eq!(trace.terminal_action.as_deref(), Some("stop"));
        assert!((trace.regime_probabilities["easy"] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn trace_roundtrips_through_json() {
        let state = easy_state();
        let trace = RetrievalTrace::from_state(&state);
        let json = trace.to_json();
        let parsed: RetrievalTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.query, trace.query);
        assert_eq!(parsed.iterations.len(), trace.iterations.len());
    }
}
