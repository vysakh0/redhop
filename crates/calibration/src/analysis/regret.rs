//! Intervention regret analysis.
//!
//! Two views on whether the controller's interventions paid off:
//!
//! 1. **Calibration of `expected_gain`.** For every non-terminal action
//!    the policy emitted an `expected_gain` prediction. After the
//!    action ran the orchestrator measured `actual_gain` from the
//!    pre/post diagnostics. *Calibration regret* is the mean signed
//!    error of `actual − expected`: positive means the policy
//!    *underestimated* what its interventions would buy; negative
//!    means it *overpromised*.
//! 2. **Operational regret on the workload.** For each query we observe
//!    `recall_lift = gold_recall_adaptive − gold_recall_static`.
//!    Aggregated:
//!    - `mean_useful_lift` — average lift among queries where
//!      `recall_lift > 0`. The upside if the controller fired *only*
//!      on the useful cases.
//!    - `mean_harmful_lift` — average lift among queries where
//!      `recall_lift < 0` (negative number — bigger magnitude = worse).
//!    - `unused_useful_opportunity` — sum of `recall_lift` we'd have
//!      gotten on queries where the controller chose NOT to
//!      intervene but adaptive happened to produce lift anyway. This
//!      is the *false-negative* regret: how much we left on the table
//!      by being too conservative.
//!
//! The two views answer different questions. (1) tells you whether the
//! `expected_gain` heuristic in the policy needs recalibrating. (2)
//! tells you whether the intervention *thresholds* are at the right
//! place on this workload.

use serde::{Deserialize, Serialize};

use crate::runner::QueryOutcome;

/// Bundle of regret metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterventionRegret {
    /// Number of queries where adaptive took at least one non-terminal
    /// action.
    pub n_interventions: usize,
    /// `mean(actual_gain − expected_gain)` across all action records.
    /// Positive means policy underestimates. Negative means
    /// overpromise.
    pub mean_expected_actual_error: f32,
    /// `mean(|actual_gain − expected_gain|)`. Pure miscalibration
    /// magnitude.
    pub mean_abs_expected_actual_error: f32,
    /// Mean `recall_lift` among interventions with `lift > 0`. The
    /// upside-when-it-works number.
    pub mean_useful_lift: f32,
    /// Mean `recall_lift` among interventions with `lift < 0`
    /// (negative). The damage-when-it-fails number.
    pub mean_harmful_lift: f32,
    /// Number of queries where adaptive did NOT intervene but
    /// `recall_adaptive > recall_static` (typically zero unless the
    /// reranker cascade ordering surprises us — we still record it).
    pub n_unused_useful_opportunities: usize,
    /// Number of queries where adaptive intervened but
    /// `recall_lift == 0` — wasted compute that neither helped nor
    /// hurt.
    pub n_wasted_interventions: usize,
}

/// Compute a regret summary from a flat list of [`QueryOutcome`]s.
pub fn regret_summary(outcomes: &[QueryOutcome]) -> InterventionRegret {
    if outcomes.is_empty() {
        return InterventionRegret::default();
    }

    let mut n_interventions = 0;
    let mut sum_err = 0.0f32;
    let mut sum_abs_err = 0.0f32;
    let mut n_err = 0usize;

    let mut useful_lifts = Vec::new();
    let mut harmful_lifts = Vec::new();
    let mut wasted = 0usize;
    let mut unused = 0usize;

    for o in outcomes {
        if o.intervened {
            n_interventions += 1;
            for entry in &o.action_trace {
                if let Some(actual) = entry.actual_gain {
                    let err = actual - entry.expected_gain;
                    sum_err += err;
                    sum_abs_err += err.abs();
                    n_err += 1;
                }
            }
            if o.recall_lift > 0.0 {
                useful_lifts.push(o.recall_lift);
            } else if o.recall_lift < 0.0 {
                harmful_lifts.push(o.recall_lift);
            } else {
                wasted += 1;
            }
        } else if o.recall_lift > 0.0 {
            // Adaptive didn't take any non-terminal action, yet final
            // recall differs from static? This shouldn't happen
            // normally — the only way is if the actuator's static path
            // and adaptive's initial retrieval landed on different
            // ordering due to top_k effects. We surface this as the
            // "missed-opportunity" count.
            unused += 1;
        }
    }

    InterventionRegret {
        n_interventions,
        mean_expected_actual_error: if n_err > 0 {
            sum_err / n_err as f32
        } else {
            0.0
        },
        mean_abs_expected_actual_error: if n_err > 0 {
            sum_abs_err / n_err as f32
        } else {
            0.0
        },
        mean_useful_lift: if !useful_lifts.is_empty() {
            useful_lifts.iter().sum::<f32>() / useful_lifts.len() as f32
        } else {
            0.0
        },
        mean_harmful_lift: if !harmful_lifts.is_empty() {
            harmful_lifts.iter().sum::<f32>() / harmful_lifts.len() as f32
        } else {
            0.0
        },
        n_wasted_interventions: wasted,
        n_unused_useful_opportunities: unused,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::ActionTraceEntry;
    use neorag_core::{RerankerLevel, RetrievalRegime};

    fn outcome(intervened: bool, lift: f32, expected: f32, actual: Option<f32>) -> QueryOutcome {
        QueryOutcome {
            query_id: "q".into(),
            true_regime: RetrievalRegime::Easy,
            predicted_regime: None,
            predicted_regime_p: None,
            true_regime_p: None,
            gold_recall_static: 0.5,
            gold_recall_adaptive: 0.5 + lift,
            recall_lift: lift,
            intervened,
            abstained: false,
            escalations: 0,
            expansions: 0,
            latency_ms_adaptive: 0,
            retrieval_calls_adaptive: 1,
            rerank_calls_adaptive: 0,
            sum_actual_gain: actual.unwrap_or(0.0),
            final_reranker_level: RerankerLevel::None,
            action_trace: if intervened {
                vec![ActionTraceEntry {
                    action: "escalate_reranker".into(),
                    expected_gain: expected,
                    actual_gain: actual,
                }]
            } else {
                vec![]
            },
        }
    }

    #[test]
    fn empty_input_yields_default() {
        let r = regret_summary(&[]);
        assert_eq!(r.n_interventions, 0);
    }

    #[test]
    fn captures_useful_and_harmful_lifts_separately() {
        let outs = vec![
            outcome(true, 0.5, 0.10, Some(0.30)),  // useful, underestimated
            outcome(true, 0.3, 0.10, Some(0.20)),  // useful, underestimated
            outcome(true, -0.2, 0.10, Some(-0.10)), // harmful
            outcome(true, 0.0, 0.05, Some(0.0)),   // wasted
            outcome(false, 0.0, 0.0, None),        // no intervention, no lift
        ];
        let r = regret_summary(&outs);
        assert_eq!(r.n_interventions, 4);
        assert!((r.mean_useful_lift - 0.4).abs() < 1e-5);
        assert!((r.mean_harmful_lift - (-0.2)).abs() < 1e-5);
        assert_eq!(r.n_wasted_interventions, 1);
        // Expected_actual error: (0.30 − 0.10), (0.20 − 0.10), (−0.10 −
        // 0.10), (0 − 0.05) = 0.20, 0.10, −0.20, −0.05 → mean 0.0125.
        assert!((r.mean_expected_actual_error - 0.0125).abs() < 1e-5);
        // Mean absolute error: (0.20 + 0.10 + 0.20 + 0.05) / 4 = 0.1375.
        assert!((r.mean_abs_expected_actual_error - 0.1375).abs() < 1e-5);
    }
}
