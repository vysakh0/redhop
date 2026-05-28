//! Per-query runner — runs static and adaptive paths side-by-side and
//! returns a [`QueryOutcome`] with the metrics the sweep aggregates.
//!
//! ## What the runner measures
//!
//! For each [`LabeledQuery`]:
//!
//! - **Gold-chunk recall**, both static and adaptive. Recall is computed
//!   against `query.gold_chunk_ids`: `|retrieved ∩ gold| / |gold|`.
//! - **Recall lift** = `recall_adaptive − recall_static`. The single
//!   most important number in the entire harness. Positive means
//!   intervention helped; negative means intervention hurt.
//! - **Intervention** information: which actions ran, what they predicted,
//!   what they actually achieved.
//! - **Latency / cost** overhead introduced by adaptive over static.
//! - **Regime prediction** (argmax + full distribution) so reliability
//!   diagrams can be built.
//!
//! The runner does not aggregate; it just produces one structured record
//! per query that downstream sweep / reliability code consumes.

use std::sync::Arc;

use redhop::core::{
    ChunkId, DiagnosticsEngine, Query, RegimeClassifier, Reranker, RerankerLevel, Result,
    RetrievalAction, RetrievalRegime, RetrievalResult, Retriever, TakenAction,
};
use redhop_orchestration::{AdaptiveOrchestrator, ConservativeRulePolicy, DefaultActuator, Policy};
use serde::{Deserialize, Serialize};

use crate::dataset::LabeledQuery;

/// All per-query metrics from one calibration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOutcome {
    /// Query identifier from the labeled corpus.
    pub query_id: String,
    /// Ground-truth regime label.
    pub true_regime: RetrievalRegime,
    /// Argmax regime predicted by the classifier (after retrieval).
    pub predicted_regime: Option<RetrievalRegime>,
    /// Probability mass on the predicted regime.
    pub predicted_regime_p: Option<f32>,
    /// Probability mass on the *true* regime.
    pub true_regime_p: Option<f32>,
    /// Gold-chunk recall under the static pipeline.
    pub gold_recall_static: f32,
    /// Gold-chunk recall under the adaptive pipeline.
    pub gold_recall_adaptive: f32,
    /// `gold_recall_adaptive − gold_recall_static`. Headline utility.
    pub recall_lift: f32,
    /// True iff at least one non-terminal action ran.
    pub intervened: bool,
    /// True iff the adaptive run terminated via Abstain.
    pub abstained: bool,
    /// Number of escalations performed (≤ 1 under the conservative policy).
    pub escalations: u32,
    /// Number of expansions performed (≤ 1 under the conservative policy).
    pub expansions: u32,
    /// Sum of latency_ms across all actions.
    pub latency_ms_adaptive: u64,
    /// Total retrieval calls during the adaptive run (initial + actions).
    pub retrieval_calls_adaptive: u32,
    /// Total reranker calls during the adaptive run.
    pub rerank_calls_adaptive: u32,
    /// Sum of `actual_gain` over non-terminal actions.
    pub sum_actual_gain: f32,
    /// Final reranker tier reached.
    pub final_reranker_level: RerankerLevel,
    /// Full sequence of `(action_code, expected_gain, actual_gain)` for
    /// the action trace. Kept terse so a `SweepReport` can serialize
    /// many of these.
    pub action_trace: Vec<ActionTraceEntry>,
}

/// One row of an action trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionTraceEntry {
    /// Stable action code (`stop`, `expand_top_k`, …).
    pub action: String,
    /// What the policy predicted this action would buy.
    pub expected_gain: f32,
    /// What it actually bought. `None` for terminal actions.
    pub actual_gain: Option<f32>,
}

/// Configuration for one calibration run. Components are shared across
/// queries; only the policy (which carries thresholds) typically varies
/// during a sweep.
pub struct RunnerConfig {
    /// The retriever (after indexing).
    pub retriever: Arc<dyn Retriever>,
    /// The diagnostics engine.
    pub diagnostics: Arc<dyn DiagnosticsEngine>,
    /// The regime classifier.
    pub classifier: Arc<dyn RegimeClassifier>,
    /// The policy. Different policies = different threshold settings =
    /// different sweep rows.
    pub policy: Arc<dyn Policy>,
    /// Reranker cascade for the adaptive actuator.
    pub rerankers: Vec<(RerankerLevel, Arc<dyn Reranker>)>,
    /// Top-k used for both static and adaptive initial retrieval.
    pub top_k: usize,
}

/// Run one labeled query, returning a [`QueryOutcome`].
///
/// This function deliberately does *both* a static pass and an adaptive
/// pass. We need both to compute `recall_lift`; running adaptive alone
/// would lose the static baseline.
pub async fn run_query(query: &LabeledQuery, cfg: &RunnerConfig) -> Result<QueryOutcome> {
    let mut q = Query::new(&query.text);
    if let Some(e) = &query.embedding {
        q.embedding = Some(e.clone());
    }

    // ---- Static path ----
    let static_candidates = cfg.retriever.retrieve(&q, cfg.top_k).await?;
    let gold_recall_static = recall(&static_candidates, &query.gold_chunk_ids);

    // ---- Adaptive path ----
    let actuator = Arc::new(DefaultActuator::new(
        cfg.retriever.clone(),
        cfg.rerankers.clone(),
    ));
    let orchestrator = AdaptiveOrchestrator::new(
        cfg.diagnostics.clone(),
        cfg.classifier.clone(),
        cfg.policy.clone(),
        actuator,
    )
    .with_initial_top_k(cfg.top_k);
    let state = orchestrator.run(q).await?;
    let gold_recall_adaptive = recall(&state.candidates, &query.gold_chunk_ids);

    // ---- Metrics ----
    let (predicted_regime, predicted_regime_p, true_regime_p) = match state.regime.as_ref() {
        Some(r) => (
            Some(r.argmax),
            Some(r.p(r.argmax)),
            Some(r.p(query.true_regime)),
        ),
        None => (None, None, None),
    };

    let escalations = state
        .history
        .iter()
        .filter(|t| matches!(t.action, RetrievalAction::EscalateReranker { .. }))
        .count() as u32;
    let expansions = state
        .history
        .iter()
        .filter(|t| matches!(t.action, RetrievalAction::ExpandTopK { .. }))
        .count() as u32;
    let intervened = state.history.iter().any(|t| !t.action.is_terminal());
    let latency_ms_adaptive = state.history.iter().map(|t| t.latency_ms).sum();
    let retrieval_calls_adaptive = 1 // initial
        + state
            .history
            .iter()
            .map(|t| t.cost.retrieval_calls)
            .sum::<u32>();
    let rerank_calls_adaptive = state
        .history
        .iter()
        .map(|t| t.cost.rerank_calls)
        .sum::<u32>();
    let sum_actual_gain = state
        .history
        .iter()
        .filter_map(|t| t.actual_gain)
        .sum::<f32>();

    let action_trace = state
        .history
        .iter()
        .map(|t: &TakenAction| ActionTraceEntry {
            action: t.action.code().to_string(),
            expected_gain: t.expected_gain,
            actual_gain: t.actual_gain,
        })
        .collect();

    Ok(QueryOutcome {
        query_id: query.id.clone(),
        true_regime: query.true_regime,
        predicted_regime,
        predicted_regime_p,
        true_regime_p,
        gold_recall_static,
        gold_recall_adaptive,
        recall_lift: gold_recall_adaptive - gold_recall_static,
        intervened,
        abstained: state.abstained(),
        escalations,
        expansions,
        latency_ms_adaptive,
        retrieval_calls_adaptive,
        rerank_calls_adaptive,
        sum_actual_gain,
        final_reranker_level: state.reranker_level,
        action_trace,
    })
}

fn recall(candidates: &[RetrievalResult], gold: &[ChunkId]) -> f32 {
    if gold.is_empty() {
        return 1.0;
    }
    let mut hits = 0;
    for g in gold {
        if candidates.iter().any(|r| r.chunk.id == *g) {
            hits += 1;
        }
    }
    hits as f32 / gold.len() as f32
}

/// Convenience: construct a [`RunnerConfig`] with the default
/// [`ConservativeRulePolicy`].
pub fn default_runner_config(
    retriever: Arc<dyn Retriever>,
    diagnostics: Arc<dyn DiagnosticsEngine>,
    classifier: Arc<dyn RegimeClassifier>,
    rerankers: Vec<(RerankerLevel, Arc<dyn Reranker>)>,
    top_k: usize,
) -> RunnerConfig {
    RunnerConfig {
        retriever,
        diagnostics,
        classifier,
        policy: Arc::new(ConservativeRulePolicy::new()),
        rerankers,
        top_k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop::core::{Chunk, RetrievalMethod, Score, ScoreBreakdown, TokenCount};

    fn rr(id: &str) -> RetrievalResult {
        RetrievalResult {
            chunk: Chunk::new(id, id, "doc", TokenCount(1)),
            score: Score {
                value: 1.0,
                method: RetrievalMethod::Lexical,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }

    #[test]
    fn recall_is_intersection_over_gold() {
        let cand = vec![rr("a"), rr("b"), rr("c")];
        let gold = vec![ChunkId::new("a"), ChunkId::new("c"), ChunkId::new("z")];
        // a ∈ cand, c ∈ cand, z ∉ cand → 2/3.
        let r = recall(&cand, &gold);
        assert!((r - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn recall_is_one_when_gold_empty() {
        let cand = vec![rr("a")];
        assert_eq!(recall(&cand, &[]), 1.0);
    }
}
