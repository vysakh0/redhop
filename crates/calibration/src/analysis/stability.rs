//! Threshold stability via bootstrap resampling.
//!
//! A single sweep tells you which threshold setting maximized
//! `mean_recall_lift` on *this* labeled corpus. It does NOT tell you
//! whether that winner is stable under sample noise — and on a real
//! workload of a few hundred queries the difference between two top
//! settings can easily be smaller than the bootstrap variance.
//!
//! This module computes:
//!
//! - **Per-setting lift stddev** — for each threshold setting, the
//!   stddev of `mean_recall_lift` across `B` bootstrap resamples of
//!   the original query outcomes. Small stddev → the lift number is
//!   stable; large stddev → don't trust the headline.
//! - **Argmax frequency** — fraction of bootstrap resamples where a
//!   given setting was the argmax. Settings with argmax frequency
//!   `>= 0.8` are stable winners; `< 0.5` suggests the sweep grid has
//!   multiple ~tied settings.
//!
//! Bootstrap is deterministic given a seed — the default seed is
//! `0xC0FFEE` so two runs of the same harness produce identical
//! stability numbers.

use serde::{Deserialize, Serialize};

use crate::sweep::SweepReport;

/// Result of a bootstrap stability analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BootstrapStability {
    /// Number of bootstrap resamples performed.
    pub n_bootstrap: usize,
    /// For each sweep row: stddev of `mean_recall_lift` across
    /// resamples. Same ordering as `SweepReport::rows`.
    pub lift_stddev: Vec<f32>,
    /// For each sweep row: fraction of resamples where this row was the
    /// argmax of `mean_recall_lift`. Same ordering as
    /// `SweepReport::rows`.
    pub argmax_frequency: Vec<f32>,
    /// For each sweep row: 90% confidence interval on `mean_recall_lift`
    /// (low, high), via the 5th and 95th percentiles of the bootstrap
    /// distribution.
    pub ci90: Vec<(f32, f32)>,
}

/// Run a bootstrap stability analysis.
///
/// Resamples *with replacement* from the per-query outcomes of each
/// sweep row `B` times and re-aggregates the mean lift. Note that this
/// is a stability analysis on the *measured* outcomes — it does not
/// re-run retrieval. Re-running retrieval on resampled queries would
/// also be valid but is far more expensive; this faster variant tells
/// you the same thing about per-query variance.
pub fn bootstrap_stability(report: &SweepReport, b: usize, seed: u64) -> BootstrapStability {
    let n_settings = report.rows.len();
    if n_settings == 0 || b == 0 {
        return BootstrapStability {
            n_bootstrap: 0,
            lift_stddev: Vec::new(),
            argmax_frequency: Vec::new(),
            ci90: Vec::new(),
        };
    }

    let mut lifts: Vec<Vec<f32>> = vec![Vec::with_capacity(b); n_settings];
    // Float-valued counts to accept fractional credit on ties.
    let mut argmax_counts_f: Vec<f32> = vec![0.0; n_settings];

    let mut rng = Lcg::new(seed);
    for _ in 0..b {
        // One resample = same indices into every setting's outcomes.
        // This preserves the within-query coupling: setting A and
        // setting B see the same resampled query order, so paired
        // comparisons stay meaningful.
        let n_queries = report.outcomes.first().map(|v| v.len()).unwrap_or(0);
        if n_queries == 0 {
            break;
        }
        let resample_indices: Vec<usize> = (0..n_queries)
            .map(|_| (rng.next_u64() as usize) % n_queries)
            .collect();

        let mut means = Vec::with_capacity(n_settings);
        for (s_idx, outs) in report.outcomes.iter().enumerate() {
            let mean = resample_indices
                .iter()
                .map(|&i| outs[i].recall_lift)
                .sum::<f32>()
                / resample_indices.len() as f32;
            lifts[s_idx].push(mean);
            means.push(mean);
        }
        // Tie-aware argmax credit: every setting whose mean equals the
        // maximum gets `1 / tied_count`. This is important because under
        // pure ties (a common failure mode on small or saturated
        // workloads), a naive first-wins argmax falsely concentrates
        // "stability" on the first row.
        let max_v = means.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let tied: Vec<usize> = means
            .iter()
            .enumerate()
            .filter(|(_, &m)| (m - max_v).abs() <= 1e-9)
            .map(|(i, _)| i)
            .collect();
        let credit = 1.0 / tied.len() as f32;
        for &i in &tied {
            argmax_counts_f[i] += credit;
        }
    }

    let lift_stddev: Vec<f32> = lifts.iter().map(|v| stddev(v)).collect();
    let argmax_frequency: Vec<f32> = argmax_counts_f.iter().map(|&c| c / b as f32).collect();
    let ci90: Vec<(f32, f32)> = lifts
        .iter()
        .map(|v| percentile_band(v, 0.05, 0.95))
        .collect();

    BootstrapStability {
        n_bootstrap: b,
        lift_stddev,
        argmax_frequency,
        ci90,
    }
}

fn stddev(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (n - 1.0);
    var.sqrt()
}

fn percentile_band(values: &[f32], lo_q: f32, hi_q: f32) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let lo_idx = ((n as f32 - 1.0) * lo_q).round() as usize;
    let hi_idx = ((n as f32 - 1.0) * hi_q).round() as usize;
    (sorted[lo_idx], sorted[hi_idx.min(n - 1)])
}

/// Tiny linear-congruential RNG. Bootstrap doesn't need cryptographic
/// quality and we want zero-dependency determinism.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{ActionTraceEntry, QueryOutcome};
    use crate::sweep::SweepRow;
    use redhop_core::{RerankerLevel, RetrievalRegime};

    fn mk_row(min_p: f32) -> SweepRow {
        SweepRow {
            min_p_distractor: min_p,
            min_p_ambiguous: 0.30,
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
        }
    }

    fn outcome(lift: f32) -> QueryOutcome {
        QueryOutcome {
            query_id: "q".into(),
            true_regime: RetrievalRegime::Easy,
            predicted_regime: None,
            predicted_regime_p: None,
            true_regime_p: None,
            gold_recall_static: 0.0,
            gold_recall_adaptive: lift,
            recall_lift: lift,
            intervened: false,
            abstained: false,
            escalations: 0,
            expansions: 0,
            latency_ms_adaptive: 0,
            retrieval_calls_adaptive: 1,
            rerank_calls_adaptive: 0,
            sum_actual_gain: 0.0,
            final_reranker_level: RerankerLevel::None,
            action_trace: Vec::<ActionTraceEntry>::new(),
        }
    }

    #[test]
    fn empty_report_yields_default() {
        let r = SweepReport {
            rows: vec![],
            outcomes: vec![],
            setting_labels: vec![],
        };
        let s = bootstrap_stability(&r, 100, 1);
        assert_eq!(s.n_bootstrap, 0);
    }

    #[test]
    fn dominant_setting_wins_argmax_frequency() {
        // Setting 0 has uniformly positive lift; setting 1 has 0 lift.
        let outs0: Vec<QueryOutcome> = (0..20).map(|_| outcome(0.5)).collect();
        let outs1: Vec<QueryOutcome> = (0..20).map(|_| outcome(0.0)).collect();
        let report = SweepReport {
            rows: vec![mk_row(0.30), mk_row(0.40)],
            outcomes: vec![outs0, outs1],
            setting_labels: vec!["s0".into(), "s1".into()],
        };
        let s = bootstrap_stability(&report, 100, 0xC0FFEE);
        // Setting 0 must dominate.
        assert!(s.argmax_frequency[0] > 0.95);
        assert!(s.argmax_frequency[1] < 0.05);
        // CI on setting 0 around 0.5.
        let (lo, hi) = s.ci90[0];
        assert!(lo > 0.4 && hi < 0.6);
    }

    #[test]
    fn variance_zero_when_all_outcomes_identical() {
        let outs: Vec<QueryOutcome> = (0..10).map(|_| outcome(0.25)).collect();
        let report = SweepReport {
            rows: vec![mk_row(0.30)],
            outcomes: vec![outs],
            setting_labels: vec!["s0".into()],
        };
        let s = bootstrap_stability(&report, 50, 1);
        // All outcomes identical → every resample has the same mean →
        // stddev is 0.
        assert!(s.lift_stddev[0].abs() < 1e-6);
    }
}
