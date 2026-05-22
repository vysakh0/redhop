//! Regime confusion matrix and per-regime precision / recall / F1.
//!
//! The confusion matrix is the canonical multi-class evaluation:
//! rows = true regime, columns = predicted regime, cells = counts.
//! From it we derive:
//!
//! - **Precision per regime** — when the classifier says `R`, what
//!   fraction of those calls were correct? Low precision on a regime
//!   that drives intervention (DistractorHeavy, Ambiguous) means
//!   wasted compute.
//! - **Recall per regime** — when the truth is `R`, how often does
//!   the classifier catch it? Low recall on regimes that *should*
//!   trigger intervention means missed lift.
//! - **F1 per regime** — harmonic mean of the two, the single-number
//!   summary.
//! - **Accuracy** — total correct / total.
//!
//! In the conservative-policy world, recall on `DistractorHeavy` and
//! `Ambiguous` is what limits adaptive's ceiling; precision on those
//! regimes is what governs intervention cost. Both numbers should be
//! reported.

use std::collections::BTreeMap;

use redhop_core::RetrievalRegime;
use serde::{Deserialize, Serialize};

use crate::runner::QueryOutcome;

/// Per-regime precision / recall / F1 / support.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegimeMetrics {
    /// `TP / (TP + FP)`. `0.0` if the classifier never predicted this
    /// regime.
    pub precision: f32,
    /// `TP / (TP + FN)`. `0.0` if the regime never appears in truth.
    pub recall: f32,
    /// `2·P·R / (P + R)`. `0.0` if either P or R is 0.
    pub f1: f32,
    /// Number of true examples of this regime in the input.
    pub support: usize,
}

/// Complete confusion matrix + derived metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegimeConfusionMatrix {
    /// `matrix[true][predicted] = count`. Predicted regime is `None`
    /// when the classifier did not emit one.
    pub matrix: BTreeMap<RetrievalRegime, BTreeMap<RetrievalRegime, usize>>,
    /// Per-regime metrics.
    pub per_regime: BTreeMap<RetrievalRegime, RegimeMetrics>,
    /// Overall classification accuracy.
    pub accuracy: f32,
    /// Total number of queries with a predicted regime.
    pub n_predicted: usize,
    /// Number of queries the classifier did NOT emit a prediction for.
    pub n_unpredicted: usize,
}

/// Build a confusion matrix from a list of [`QueryOutcome`]s.
pub fn confusion_matrix(outcomes: &[QueryOutcome]) -> RegimeConfusionMatrix {
    let regimes = RetrievalRegime::all();
    let mut matrix: BTreeMap<RetrievalRegime, BTreeMap<RetrievalRegime, usize>> = BTreeMap::new();
    for r in regimes {
        let mut row = BTreeMap::new();
        for p in regimes {
            row.insert(*p, 0usize);
        }
        matrix.insert(*r, row);
    }

    let mut n_predicted = 0usize;
    let mut n_unpredicted = 0usize;
    for o in outcomes {
        match o.predicted_regime {
            Some(p) => {
                n_predicted += 1;
                if let Some(row) = matrix.get_mut(&o.true_regime) {
                    *row.entry(p).or_insert(0) += 1;
                }
            }
            None => n_unpredicted += 1,
        }
    }

    let total_predicted = n_predicted as f32;
    let mut correct = 0usize;
    let mut per_regime: BTreeMap<RetrievalRegime, RegimeMetrics> = BTreeMap::new();
    for &r in regimes {
        // Support = sum over predicted of matrix[r][_]
        let row = matrix.get(&r).cloned().unwrap_or_default();
        let support: usize = row.values().sum();
        // True positives: matrix[r][r]
        let tp = *row.get(&r).unwrap_or(&0);
        correct += tp;
        // False positives: sum over (r' != r) of matrix[r'][r]
        let mut fp = 0usize;
        for &r_other in regimes {
            if r_other == r {
                continue;
            }
            if let Some(other_row) = matrix.get(&r_other) {
                fp += other_row.get(&r).copied().unwrap_or(0);
            }
        }
        // False negatives: support - tp (predictions of other regimes
        // for true r).
        let fn_ = support.saturating_sub(tp);

        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };
        let recall = if support > 0 { tp as f32 / support as f32 } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let _ = fn_;
        per_regime.insert(
            r,
            RegimeMetrics {
                precision,
                recall,
                f1,
                support,
            },
        );
    }

    let accuracy = if total_predicted > 0.0 {
        correct as f32 / total_predicted
    } else {
        0.0
    };

    RegimeConfusionMatrix {
        matrix,
        per_regime,
        accuracy,
        n_predicted,
        n_unpredicted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop_core::RerankerLevel;

    fn outcome(true_r: RetrievalRegime, predicted: Option<RetrievalRegime>) -> QueryOutcome {
        QueryOutcome {
            query_id: "q".into(),
            true_regime: true_r,
            predicted_regime: predicted,
            predicted_regime_p: predicted.map(|_| 0.5),
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
            final_reranker_level: RerankerLevel::None,
            action_trace: vec![],
        }
    }

    #[test]
    fn perfect_classifier_has_accuracy_one() {
        let outs = vec![
            outcome(RetrievalRegime::Easy, Some(RetrievalRegime::Easy)),
            outcome(RetrievalRegime::Sparse, Some(RetrievalRegime::Sparse)),
            outcome(RetrievalRegime::DistractorHeavy, Some(RetrievalRegime::DistractorHeavy)),
        ];
        let m = confusion_matrix(&outs);
        assert_eq!(m.accuracy, 1.0);
        assert_eq!(m.per_regime[&RetrievalRegime::Easy].precision, 1.0);
        assert_eq!(m.per_regime[&RetrievalRegime::Easy].recall, 1.0);
        assert_eq!(m.per_regime[&RetrievalRegime::Easy].f1, 1.0);
    }

    #[test]
    fn confuses_distractor_with_easy_correctly_measured() {
        // Truth: 4 DistractorHeavy. Classifier: predicts Easy on 3 of
        // them, DistractorHeavy on 1.
        let outs = vec![
            outcome(RetrievalRegime::DistractorHeavy, Some(RetrievalRegime::Easy)),
            outcome(RetrievalRegime::DistractorHeavy, Some(RetrievalRegime::Easy)),
            outcome(RetrievalRegime::DistractorHeavy, Some(RetrievalRegime::Easy)),
            outcome(RetrievalRegime::DistractorHeavy, Some(RetrievalRegime::DistractorHeavy)),
        ];
        let m = confusion_matrix(&outs);
        // Recall on DistractorHeavy = 1/4 = 0.25.
        assert!((m.per_regime[&RetrievalRegime::DistractorHeavy].recall - 0.25).abs() < 1e-5);
        // Precision on DistractorHeavy = 1/1 = 1.0 (only 1 prediction,
        // and it was correct).
        assert!((m.per_regime[&RetrievalRegime::DistractorHeavy].precision - 1.0).abs() < 1e-5);
        // Precision on Easy = 0/3 = 0 (3 Easy predictions, all wrong).
        assert!(m.per_regime[&RetrievalRegime::Easy].precision < 1e-5);
    }

    #[test]
    fn missing_predictions_counted_separately() {
        let outs = vec![
            outcome(RetrievalRegime::Easy, Some(RetrievalRegime::Easy)),
            outcome(RetrievalRegime::Easy, None),
        ];
        let m = confusion_matrix(&outs);
        assert_eq!(m.n_predicted, 1);
        assert_eq!(m.n_unpredicted, 1);
        assert_eq!(m.accuracy, 1.0); // 1 correct out of 1 predicted
    }
}
