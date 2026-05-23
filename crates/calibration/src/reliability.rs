//! Reliability diagram for the regime classifier.
//!
//! The headline calibration question — *is `p ≈ 0.35` for DistractorHeavy
//! genuinely weak signal, or is the classifier underconfident?* — is what
//! this module answers.
//!
//! For each predicted regime `R`, we bin queries by `p(R)`. Within each
//! bin we report the fraction of queries where `R` was actually the true
//! label. A well-calibrated classifier produces bins whose midpoint
//! approximately equals their empirical-true fraction.
//!
//! Concretely:
//!
//! ```text
//!   p ∈ [0.0, 0.1):  10 queries, 1 correct → empirical 0.10  ✓
//!   p ∈ [0.1, 0.2):  20 queries, 3 correct → empirical 0.15  ✓
//!   p ∈ [0.3, 0.4):  15 queries, 9 correct → empirical 0.60  ← underconfident
//!   p ∈ [0.6, 0.7):  8 queries, 4 correct  → empirical 0.50  ← overconfident
//! ```
//!
//! If the third bin's empirical fraction is much higher than the bin
//! midpoint, the classifier is *under*confident — lowering policy
//! thresholds in that range will trade a small fairness loss for a real
//! utility gain. If it's much lower, the classifier is *over*confident —
//! raising thresholds protects against false positives.
//!
//! ## What this diagram does NOT do
//!
//! It does not by itself recalibrate the classifier. Recalibration
//! (Platt scaling, isotonic regression) is a future possibility that
//! consumes the reliability data this module produces.

use redhop_core::RetrievalRegime;
use serde::{Deserialize, Serialize};

use crate::runner::QueryOutcome;

/// One bin in a reliability diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityBin {
    /// Lower edge of the probability bin (inclusive).
    pub lo: f32,
    /// Upper edge of the probability bin (exclusive, except the last bin
    /// where it's inclusive).
    pub hi: f32,
    /// Number of queries that fell into this bin.
    pub count: usize,
    /// Fraction of those queries where the predicted regime equaled the
    /// true regime. `0.0` when `count == 0`.
    pub empirical_correct: f32,
    /// Mean predicted probability among queries in this bin. Useful for
    /// fine-grained calibration: a well-calibrated classifier has
    /// `mean_predicted_p ≈ empirical_correct` per bin.
    pub mean_predicted_p: f32,
}

/// A complete reliability diagram for one regime, or aggregated across
/// all regimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityDiagram {
    /// The regime this diagram describes, or `None` for the aggregated
    /// "predicted-vs-true" diagram across all regimes.
    pub regime: Option<RetrievalRegime>,
    /// The bins, in ascending probability order.
    pub bins: Vec<ReliabilityBin>,
    /// Expected Calibration Error: `Σ (count / total) · |mean_p −
    /// empirical_correct|`. Lower is better; `0` means perfectly
    /// calibrated.
    pub ece: f32,
}

/// Build a reliability diagram for one regime.
///
/// Queries are bucketed by `p(regime)` from their `predicted_regime` /
/// `true_regime_p` fields. Queries where the classifier did not emit a
/// distribution for that regime are skipped.
pub fn reliability_diagram_for(
    outcomes: &[QueryOutcome],
    regime: RetrievalRegime,
    n_bins: usize,
) -> ReliabilityDiagram {
    assert!(n_bins >= 1, "n_bins must be ≥ 1");
    let mut bins: Vec<(usize, f32, f32)> = vec![(0, 0.0, 0.0); n_bins];

    // For each query, compute p(regime). We use `true_regime_p` when the
    // regime in question is the *true* regime — that is the natural
    // diagram for "how confident is the classifier in the truth?". For
    // other regimes, we'd need the full distribution. Phase 8 doesn't
    // expose that on QueryOutcome by default; we approximate with
    // `predicted_regime == regime` ? `predicted_regime_p` : 0.0.
    for o in outcomes {
        let p = if o.predicted_regime == Some(regime) {
            o.predicted_regime_p.unwrap_or(0.0)
        } else if regime == o.true_regime {
            o.true_regime_p.unwrap_or(0.0)
        } else {
            continue;
        };
        let correct = o.true_regime == regime;
        let bin_idx = bucket(p, n_bins);
        let b = &mut bins[bin_idx];
        b.0 += 1;
        if correct {
            b.1 += 1.0;
        }
        b.2 += p;
    }

    let mut total = 0usize;
    let bin_rows: Vec<ReliabilityBin> = (0..n_bins)
        .map(|i| {
            let (count, sum_correct, sum_p) = bins[i];
            total += count;
            let lo = i as f32 / n_bins as f32;
            let hi = (i + 1) as f32 / n_bins as f32;
            ReliabilityBin {
                lo,
                hi,
                count,
                empirical_correct: if count > 0 {
                    sum_correct / count as f32
                } else {
                    0.0
                },
                mean_predicted_p: if count > 0 {
                    sum_p / count as f32
                } else {
                    (lo + hi) / 2.0
                },
            }
        })
        .collect();

    let ece = if total > 0 {
        bin_rows
            .iter()
            .map(|b| {
                let w = b.count as f32 / total as f32;
                w * (b.mean_predicted_p - b.empirical_correct).abs()
            })
            .sum()
    } else {
        0.0
    };

    ReliabilityDiagram {
        regime: Some(regime),
        bins: bin_rows,
        ece,
    }
}

/// Convenience: reliability diagram for the *predicted* regime —
/// "whatever the classifier guessed, how often was it right at this
/// confidence level?"
pub fn reliability_diagram(outcomes: &[QueryOutcome], n_bins: usize) -> ReliabilityDiagram {
    assert!(n_bins >= 1, "n_bins must be ≥ 1");
    let mut bins: Vec<(usize, f32, f32)> = vec![(0, 0.0, 0.0); n_bins];

    for o in outcomes {
        let Some(p) = o.predicted_regime_p else {
            continue;
        };
        let correct = o.predicted_regime == Some(o.true_regime);
        let bin_idx = bucket(p, n_bins);
        let b = &mut bins[bin_idx];
        b.0 += 1;
        if correct {
            b.1 += 1.0;
        }
        b.2 += p;
    }

    let mut total = 0usize;
    let bin_rows: Vec<ReliabilityBin> = (0..n_bins)
        .map(|i| {
            let (count, sum_correct, sum_p) = bins[i];
            total += count;
            let lo = i as f32 / n_bins as f32;
            let hi = (i + 1) as f32 / n_bins as f32;
            ReliabilityBin {
                lo,
                hi,
                count,
                empirical_correct: if count > 0 {
                    sum_correct / count as f32
                } else {
                    0.0
                },
                mean_predicted_p: if count > 0 {
                    sum_p / count as f32
                } else {
                    (lo + hi) / 2.0
                },
            }
        })
        .collect();

    let ece = if total > 0 {
        bin_rows
            .iter()
            .map(|b| {
                let w = b.count as f32 / total as f32;
                w * (b.mean_predicted_p - b.empirical_correct).abs()
            })
            .sum()
    } else {
        0.0
    };

    ReliabilityDiagram {
        regime: None,
        bins: bin_rows,
        ece,
    }
}

fn bucket(p: f32, n_bins: usize) -> usize {
    let idx = (p * n_bins as f32).floor() as usize;
    idx.min(n_bins - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(
        predicted: RetrievalRegime,
        predicted_p: f32,
        true_r: RetrievalRegime,
    ) -> QueryOutcome {
        QueryOutcome {
            query_id: "q".into(),
            true_regime: true_r,
            predicted_regime: Some(predicted),
            predicted_regime_p: Some(predicted_p),
            true_regime_p: None,
            gold_recall_static: 0.0,
            gold_recall_adaptive: 0.0,
            recall_lift: 0.0,
            intervened: false,
            abstained: false,
            escalations: 0,
            expansions: 0,
            latency_ms_adaptive: 0,
            retrieval_calls_adaptive: 1,
            rerank_calls_adaptive: 0,
            sum_actual_gain: 0.0,
            final_reranker_level: redhop_core::RerankerLevel::None,
            action_trace: vec![],
        }
    }

    #[test]
    fn perfectly_calibrated_classifier_has_zero_ece() {
        // 10 queries at p=0.95 for Easy, 9 correct.
        // 10 queries at p=0.05 for Easy, 1 correct.
        let mut outs = Vec::new();
        for _ in 0..9 {
            outs.push(outcome(RetrievalRegime::Easy, 0.95, RetrievalRegime::Easy));
        }
        outs.push(outcome(
            RetrievalRegime::Easy,
            0.95,
            RetrievalRegime::Sparse,
        ));
        for _ in 0..1 {
            outs.push(outcome(RetrievalRegime::Easy, 0.05, RetrievalRegime::Easy));
        }
        for _ in 0..9 {
            outs.push(outcome(
                RetrievalRegime::Easy,
                0.05,
                RetrievalRegime::Sparse,
            ));
        }
        let d = reliability_diagram(&outs, 10);
        // ECE should be very small for this scenario.
        assert!(d.ece < 0.05, "got ECE {}", d.ece);
    }

    #[test]
    fn underconfident_classifier_has_high_ece() {
        // 10 queries at p=0.30 for Easy, 9 actually correct — underconfident.
        let outs: Vec<QueryOutcome> = (0..10)
            .map(|i| {
                outcome(
                    RetrievalRegime::Easy,
                    0.30,
                    if i < 9 {
                        RetrievalRegime::Easy
                    } else {
                        RetrievalRegime::Sparse
                    },
                )
            })
            .collect();
        let d = reliability_diagram(&outs, 10);
        // empirical = 0.9, mean_p ≈ 0.3 → bin contributes |0.3 − 0.9| = 0.6.
        // With only one populated bin (weight = 1), ECE ≈ 0.6.
        assert!(d.ece > 0.5, "got ECE {}", d.ece);
    }
}
