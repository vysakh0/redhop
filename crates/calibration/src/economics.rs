//! Cost / quality economics of adaptive retrieval.
//!
//! The moat is *not spending compute you don't need to*. This module
//! turns that into numbers a deployment can act on.
//!
//! A [`CostModel`] assigns a unit cost to each kind of work. From a set
//! of [`QueryOutcome`]s (produced by the sweep or by the NeoTrace
//! loader) we then derive:
//!
//! - **adaptive cost** — what the conservative controller actually
//!   spent (it reranks only the queries it chose to).
//! - **uniform-rerank cost** — what reranking *every* query would have
//!   cost (the naive baseline).
//! - **compute reduction** — `1 − adaptive_reranks / uniform_reranks`.
//! - **cost per unit lift** — `adaptive_cost / mean_recall_lift`.
//! - **selective-escalation ROI** — `(lift / cost)_adaptive ÷
//!   (lift / cost)_uniform`, the headline efficiency multiple. Requires
//!   the uniform-rerank lift as an input (it's measured separately by
//!   the method-pair analysis; we do not fabricate it).
//!
//! Cost units are abstract (call them "compute units"); plug real
//! dollar/latency figures into [`CostModel`] for a deployment-specific
//! readout.

use serde::{Deserialize, Serialize};

use crate::runner::QueryOutcome;

/// Per-action unit costs. Units are abstract; set them to whatever your
/// deployment measures (ms, dollars, FLOPs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Cost of embedding one query online.
    pub cost_per_query_embed: f32,
    /// Cost of one retrieval (BM25 / ANN) call.
    pub cost_per_retrieval: f32,
    /// Cost of scoring one (query, candidate) pair through the
    /// cross-encoder. The dominant term in any reranking deployment.
    pub cost_per_rerank_candidate: f32,
    /// Candidates considered per rerank call (for converting rerank
    /// *calls* into rerank *candidate-scorings*).
    pub candidates_per_rerank: f32,
}

impl Default for CostModel {
    fn default() -> Self {
        // Defaults reflect the *relative* cost structure of a typical
        // CPU deployment: a cross-encoder pair-scoring is ~50× a BM25
        // lookup and ~10× a cached query embed. Override with real
        // measurements for a deployment-specific readout.
        Self {
            cost_per_query_embed: 1.0,
            cost_per_retrieval: 0.2,
            cost_per_rerank_candidate: 10.0,
            candidates_per_rerank: 4.0,
        }
    }
}

impl CostModel {
    /// Cost of a single query outcome under this model.
    pub fn outcome_cost(&self, o: &QueryOutcome) -> f32 {
        self.cost_per_query_embed
            + o.retrieval_calls_adaptive as f32 * self.cost_per_retrieval
            + o.rerank_calls_adaptive as f32
                * self.candidates_per_rerank
                * self.cost_per_rerank_candidate
    }

    /// Cost of the same query if it were *always* reranked once (the
    /// uniform baseline). One query embed, one retrieval, one rerank.
    pub fn uniform_cost(&self) -> f32 {
        self.cost_per_query_embed
            + self.cost_per_retrieval
            + self.candidates_per_rerank * self.cost_per_rerank_candidate
    }
}

/// Economics summary over a set of outcomes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EconomicsReport {
    /// Number of outcomes considered.
    pub n: usize,
    /// Mean adaptive cost per query.
    pub mean_adaptive_cost: f32,
    /// Cost of uniformly reranking every query.
    pub uniform_cost: f32,
    /// `mean_adaptive_cost / uniform_cost` — fraction of uniform spend.
    pub cost_fraction_vs_uniform: f32,
    /// `1 − adaptive_reranks / uniform_reranks`. Fraction of rerank
    /// compute *avoided* by selective escalation.
    pub rerank_compute_reduction: f32,
    /// Mean recall lift the adaptive controller achieved.
    pub mean_recall_lift: f32,
    /// Adaptive cost spent per unit of recall lift. Lower is better.
    /// `None` when mean lift is non-positive (cost-per-lift undefined).
    pub cost_per_unit_lift: Option<f32>,
    /// Mean rerank calls per query (intervention intensity).
    pub mean_rerank_calls: f32,
}

/// Compute the economics summary for a set of outcomes.
pub fn economics(outcomes: &[QueryOutcome], cost: &CostModel) -> EconomicsReport {
    if outcomes.is_empty() {
        return EconomicsReport::default();
    }
    let n = outcomes.len();
    let nf = n as f32;
    let total_cost: f32 = outcomes.iter().map(|o| cost.outcome_cost(o)).sum();
    let mean_adaptive_cost = total_cost / nf;
    let uniform_cost = cost.uniform_cost();
    let mean_recall_lift = outcomes.iter().map(|o| o.recall_lift).sum::<f32>() / nf;
    let total_reranks: u32 = outcomes.iter().map(|o| o.rerank_calls_adaptive).sum();
    let mean_rerank_calls = total_reranks as f32 / nf;
    // Uniform reranks = one per query.
    let rerank_compute_reduction = 1.0 - (mean_rerank_calls / 1.0).min(1.0);
    let cost_per_unit_lift = if mean_recall_lift > 1e-6 {
        Some(mean_adaptive_cost / mean_recall_lift)
    } else {
        None
    };
    EconomicsReport {
        n,
        mean_adaptive_cost,
        uniform_cost,
        cost_fraction_vs_uniform: mean_adaptive_cost / uniform_cost.max(1e-9),
        rerank_compute_reduction,
        mean_recall_lift,
        cost_per_unit_lift,
        mean_rerank_calls,
    }
}

/// The headline selective-escalation ROI multiple.
///
/// `(adaptive_lift / adaptive_cost) ÷ (uniform_lift / uniform_cost)`.
///
/// `uniform_lift` must be supplied — it's the recall lift you'd get by
/// reranking *every* query with the same reranker, which the method-pair
/// analysis measures directly (e.g. +0.046 for HotpotQA cross_encoder).
/// We require it as a parameter rather than estimating it, because
/// estimating it from adaptive-only data would be a fabrication.
///
/// Returns `None` when either efficiency is undefined (non-positive
/// lift or zero cost).
pub fn selective_escalation_roi(
    report: &EconomicsReport,
    uniform_lift: f32,
    cost: &CostModel,
) -> Option<f32> {
    if report.mean_recall_lift <= 0.0 || uniform_lift <= 0.0 {
        return None;
    }
    let adaptive_eff = report.mean_recall_lift / report.mean_adaptive_cost.max(1e-9);
    let uniform_eff = uniform_lift / cost.uniform_cost().max(1e-9);
    if uniform_eff <= 0.0 {
        return None;
    }
    Some(adaptive_eff / uniform_eff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neorag_core::{RerankerLevel, RetrievalRegime};

    fn outcome(rerank_calls: u32, lift: f32) -> QueryOutcome {
        QueryOutcome {
            query_id: "q".into(),
            true_regime: RetrievalRegime::Easy,
            predicted_regime: None,
            predicted_regime_p: None,
            true_regime_p: None,
            gold_recall_static: 0.5,
            gold_recall_adaptive: 0.5 + lift,
            recall_lift: lift,
            intervened: rerank_calls > 0,
            abstained: false,
            escalations: rerank_calls,
            expansions: 0,
            latency_ms_adaptive: 0,
            retrieval_calls_adaptive: 1,
            rerank_calls_adaptive: rerank_calls,
            sum_actual_gain: 0.0,
            final_reranker_level: RerankerLevel::None,
            action_trace: vec![],
        }
    }

    #[test]
    fn empty_outcomes_yields_default() {
        let r = economics(&[], &CostModel::default());
        assert_eq!(r.n, 0);
    }

    #[test]
    fn selective_costs_less_than_uniform() {
        // 10 queries, only 4 reranked → 40% rerank rate.
        let mut outs = Vec::new();
        for i in 0..10 {
            outs.push(outcome(if i < 4 { 1 } else { 0 }, if i < 4 { 0.3 } else { 0.0 }));
        }
        let cost = CostModel::default();
        let r = economics(&outs, &cost);
        assert_eq!(r.n, 10);
        // Adaptive cost should be well under uniform.
        assert!(r.cost_fraction_vs_uniform < 1.0);
        // 60% of rerank compute avoided.
        assert!((r.rerank_compute_reduction - 0.6).abs() < 1e-5);
        assert!((r.mean_rerank_calls - 0.4).abs() < 1e-5);
    }

    #[test]
    fn roi_exceeds_one_when_selective_beats_uniform() {
        // Adaptive: 4/10 reranked, mean lift 0.12.
        let mut outs = Vec::new();
        for i in 0..10 {
            outs.push(outcome(if i < 4 { 1 } else { 0 }, if i < 4 { 0.30 } else { 0.0 }));
        }
        let cost = CostModel::default();
        let r = economics(&outs, &cost);
        // Uniform rerank measured at +0.046 (from method-pair analysis).
        let roi = selective_escalation_roi(&r, 0.046, &cost).unwrap();
        // Adaptive should be dramatically more efficient.
        assert!(roi > 2.0, "expected ROI > 2, got {roi}");
    }

    #[test]
    fn cost_per_unit_lift_undefined_when_no_lift() {
        let outs = vec![outcome(1, 0.0), outcome(1, 0.0)];
        let r = economics(&outs, &CostModel::default());
        assert!(r.cost_per_unit_lift.is_none());
    }
}
