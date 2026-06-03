//! Threshold sweep — runs an entire labeled corpus at multiple policy
//! settings and aggregates per-setting metrics.
//!
//! The output ([`SweepReport`]) is the central artifact of this crate:
//! one row per threshold setting, each row carrying the headline
//! aggregates (intervention rate, mean recall lift, latency overhead,
//! cost, regime accuracy). It is what a production operator inspects to
//! answer:
//!
//! > For *my* workload, at which threshold does the adaptive controller
//! > Pareto-dominate static retrieval?
//!
//! ## Sweep dimensions
//!
//! The adaptive policy has two thresholds with strong leverage:
//!
//! - `min_p_distractor` — controls when EscalateReranker fires.
//! - `min_p_ambiguous` — controls when ExpandTopK fires.
//!
//! We sweep both. The other thresholds (`min_p_easy`, `min_p_saturated`,
//! `min_p_sparse`) control terminal actions only and have less impact on
//! the cost/quality tradeoff. They are held at their defaults; the
//! [`ThresholdSweep::with_static_thresholds`] hook lets callers override.

use std::sync::Arc;

use redhop::core::{
    DiagnosticsEngine, RegimeClassifier, Reranker, RerankerLevel, Result, Retriever,
};
use redhop_orchestration::{ConservativeRulePolicy, Policy, PolicyThresholds};
use serde::{Deserialize, Serialize};

use crate::dataset::LabeledCorpus;
use crate::runner::{run_query, QueryOutcome, RunnerConfig};

/// One row of a [`SweepReport`] — aggregated metrics at one threshold
/// setting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepRow {
    /// The `min_p_distractor` setting for this row.
    pub min_p_distractor: f32,
    /// The `min_p_ambiguous` setting for this row.
    pub min_p_ambiguous: f32,
    /// Number of queries evaluated.
    pub n: usize,
    /// Fraction of queries where adaptive took at least one non-terminal
    /// action.
    pub intervention_rate: f32,
    /// Mean `recall_lift` across all queries. The headline utility number.
    pub mean_recall_lift: f32,
    /// Mean `recall_lift` restricted to queries where adaptive
    /// intervened. Answers "was the intervention itself useful?"
    pub mean_recall_lift_when_intervened: f32,
    /// Fraction of interventions where `recall_lift > 0`.
    pub fraction_useful_interventions: f32,
    /// Fraction of interventions where `recall_lift < 0` (harmful).
    pub fraction_harmful_interventions: f32,
    /// Mean total latency (ms) across queries. Adaptive includes the
    /// static call latency plus any action latency.
    pub mean_latency_ms: f32,
    /// Mean rerank calls per query. Cost proxy.
    pub mean_rerank_calls: f32,
    /// Mean retrieval calls per query.
    pub mean_retrieval_calls: f32,
    /// Mean `sum_actual_gain` — what the orchestrator believed it bought,
    /// independent of gold-recall lift.
    pub mean_internal_actual_gain: f32,
    /// Fraction of queries where the classifier's argmax matched the
    /// true regime label.
    pub regime_argmax_accuracy: f32,
}

/// Per-row + per-query detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepReport {
    /// One row per threshold setting.
    pub rows: Vec<SweepRow>,
    /// All per-query outcomes, indexed `[setting_idx][query_idx]`.
    pub outcomes: Vec<Vec<QueryOutcome>>,
    /// The labels of the threshold settings tried, in the same order as
    /// `rows`. Useful for plotting.
    pub setting_labels: Vec<String>,
}

impl SweepReport {
    /// Locate the row that maximizes `mean_recall_lift`. The simplest
    /// "best threshold" criterion; richer Pareto analysis is the job of
    /// [`crate::report`].
    pub fn argmax_lift(&self) -> Option<&SweepRow> {
        self.rows.iter().max_by(|a, b| {
            a.mean_recall_lift
                .partial_cmp(&b.mean_recall_lift)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// The "no intervention" baseline row, if it's in the sweep — the
    /// row whose thresholds are high enough that the controller never
    /// intervenes. Useful for relative comparisons.
    pub fn no_intervention_baseline(&self) -> Option<&SweepRow> {
        self.rows
            .iter()
            .filter(|r| r.intervention_rate == 0.0)
            .min_by(|a, b| {
                a.mean_latency_ms
                    .partial_cmp(&b.mean_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Sweep configuration.
pub struct ThresholdSweep {
    /// Distractor probability grid (e.g. `[0.25, 0.30, 0.35, 0.40, 0.45]`).
    pub min_p_distractor_grid: Vec<f32>,
    /// Ambiguous probability grid.
    pub min_p_ambiguous_grid: Vec<f32>,
    /// Top-k used for both static and adaptive retrieval.
    pub top_k: usize,
    /// Other policy thresholds held constant across the sweep.
    pub static_thresholds: PolicyThresholds,
}

impl ThresholdSweep {
    /// Construct with sensible default grids covering the conservative-
    /// to-aggressive range.
    pub fn default_grid(top_k: usize) -> Self {
        Self {
            min_p_distractor_grid: vec![0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            min_p_ambiguous_grid: vec![0.30, 0.40, 0.50],
            top_k,
            static_thresholds: PolicyThresholds::default(),
        }
    }

    /// Pin the non-swept thresholds at custom values.
    pub fn with_static_thresholds(mut self, t: PolicyThresholds) -> Self {
        self.static_thresholds = t;
        self
    }

    /// Run the sweep against a labeled corpus.
    ///
    /// `retriever` should already be indexed against `corpus.docs`; the
    /// sweep does *not* re-index, since indexing is typically the most
    /// expensive part of any setup.
    pub async fn run(
        &self,
        corpus: &LabeledCorpus,
        retriever: Arc<dyn Retriever>,
        diagnostics: Arc<dyn DiagnosticsEngine>,
        classifier: Arc<dyn RegimeClassifier>,
        rerankers: Vec<(RerankerLevel, Arc<dyn Reranker>)>,
    ) -> Result<SweepReport> {
        let mut rows = Vec::new();
        let mut outcomes_per_setting = Vec::new();
        let mut setting_labels = Vec::new();

        for &p_d in &self.min_p_distractor_grid {
            for &p_a in &self.min_p_ambiguous_grid {
                let mut t = self.static_thresholds.clone();
                t.min_p_distractor = p_d;
                t.min_p_ambiguous = p_a;
                let policy: Arc<dyn Policy> = Arc::new(ConservativeRulePolicy::with_thresholds(t));
                let cfg = RunnerConfig {
                    retriever: retriever.clone(),
                    diagnostics: diagnostics.clone(),
                    classifier: classifier.clone(),
                    policy,
                    rerankers: rerankers.clone(),
                    top_k: self.top_k,
                };
                let mut outcomes = Vec::with_capacity(corpus.queries.len());
                for q in &corpus.queries {
                    outcomes.push(run_query(q, &cfg).await?);
                }
                let row = aggregate(&outcomes, p_d, p_a);
                setting_labels.push(format!("d={p_d:.2}, a={p_a:.2}"));
                rows.push(row);
                outcomes_per_setting.push(outcomes);
            }
        }
        Ok(SweepReport {
            rows,
            outcomes: outcomes_per_setting,
            setting_labels,
        })
    }
}

fn aggregate(outcomes: &[QueryOutcome], min_p_distractor: f32, min_p_ambiguous: f32) -> SweepRow {
    let n = outcomes.len();
    if n == 0 {
        return SweepRow {
            min_p_distractor,
            min_p_ambiguous,
            n: 0,
            intervention_rate: 0.0,
            mean_recall_lift: 0.0,
            mean_recall_lift_when_intervened: 0.0,
            fraction_useful_interventions: 0.0,
            fraction_harmful_interventions: 0.0,
            mean_latency_ms: 0.0,
            mean_rerank_calls: 0.0,
            mean_retrieval_calls: 0.0,
            mean_internal_actual_gain: 0.0,
            regime_argmax_accuracy: 0.0,
        };
    }
    let nf = n as f32;
    let intervened: Vec<&QueryOutcome> = outcomes.iter().filter(|o| o.intervened).collect();
    let n_intervened = intervened.len();
    let mean_recall_lift = outcomes.iter().map(|o| o.recall_lift).sum::<f32>() / nf;
    let mean_recall_lift_when_intervened = if n_intervened > 0 {
        intervened.iter().map(|o| o.recall_lift).sum::<f32>() / n_intervened as f32
    } else {
        0.0
    };
    let useful = intervened.iter().filter(|o| o.recall_lift > 0.0).count();
    let harmful = intervened.iter().filter(|o| o.recall_lift < 0.0).count();
    let fraction_useful_interventions = if n_intervened > 0 {
        useful as f32 / n_intervened as f32
    } else {
        0.0
    };
    let fraction_harmful_interventions = if n_intervened > 0 {
        harmful as f32 / n_intervened as f32
    } else {
        0.0
    };
    let mean_latency_ms = outcomes
        .iter()
        .map(|o| o.latency_ms_adaptive as f32)
        .sum::<f32>()
        / nf;
    let mean_rerank_calls = outcomes
        .iter()
        .map(|o| o.rerank_calls_adaptive as f32)
        .sum::<f32>()
        / nf;
    let mean_retrieval_calls = outcomes
        .iter()
        .map(|o| o.retrieval_calls_adaptive as f32)
        .sum::<f32>()
        / nf;
    let mean_internal_actual_gain = outcomes.iter().map(|o| o.sum_actual_gain).sum::<f32>() / nf;
    let argmax_correct = outcomes
        .iter()
        .filter(|o| o.predicted_regime == Some(o.true_regime))
        .count();
    let regime_argmax_accuracy = argmax_correct as f32 / nf;

    SweepRow {
        min_p_distractor,
        min_p_ambiguous,
        n,
        intervention_rate: n_intervened as f32 / nf,
        mean_recall_lift,
        mean_recall_lift_when_intervened,
        fraction_useful_interventions,
        fraction_harmful_interventions,
        mean_latency_ms,
        mean_rerank_calls,
        mean_retrieval_calls,
        mean_internal_actual_gain,
        regime_argmax_accuracy,
    }
}
