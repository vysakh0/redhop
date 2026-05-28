//! Derive a [`ConfidenceProfile`] from a list of retrieval results.
//!
//! The profile combines three orthogonal views on top-k score peakedness:
//!
//! - `decision_margin` — *is the top-1 clearly the top-1?*
//! - `score_entropy`   — *is the whole distribution peaked or flat?*
//! - `posterior_concentration` — *how much mass does the top-1 carry?*
//!
//! Each captures a different failure mode. A high `decision_margin` with
//! flat tail tells you the top-1 is safe but the tail is ambiguous. Low
//! `decision_margin` with low `score_entropy` means the top-1 and top-2
//! are tied but the rest is far behind — a different problem. The policy
//! layer in Phase 8 will read all three.
//!
//! Implementation notes:
//!
//! - Scores live on whatever scale the source retriever uses (BM25 is
//!   unbounded, cosine is `[-1, 1]`, RRF is small positive). We normalize
//!   via softmax with a *fitted* temperature: `τ = std(scores) + ε`. This
//!   makes the entropy comparable across retrievers without destroying
//!   the relative ordering. The choice of `std` as temperature is
//!   empirical: it produces sensible entropies on BM25, cosine, and RRF
//!   distributions without per-retriever tuning.
//! - All fields are `Option` so consumers can distinguish "we measured
//!   zero" from "we couldn't measure". With fewer than two results,
//!   `decision_margin` and `score_entropy` are honestly undefined.

use redhop::core::{ConfidenceProfile, RetrievalResult};

const EPS: f32 = 1e-6;

/// Compute a [`ConfidenceProfile`] over a result list.
///
/// `results` is expected in descending score order; if it isn't, the
/// function works but `decision_margin` will be uninformative.
pub fn compute_confidence(results: &[RetrievalResult]) -> ConfidenceProfile {
    let mut p = ConfidenceProfile::default();
    if results.is_empty() {
        return p;
    }

    let scores: Vec<f32> = results.iter().map(|r| r.score.value).collect();

    // Posterior via temperature-scaled softmax.
    let posterior = softmax_with_fitted_temperature(&scores);
    p.posterior_concentration = posterior.first().copied();

    if scores.len() >= 2 {
        // Decision margin on the *raw* scores so it stays interpretable.
        // `max(EPS)` (rather than `+ EPS`) keeps the denominator equal to
        // `|s0|` whenever the score is comfortably above zero; the EPS
        // only kicks in for near-zero degenerate cases. Without this, a
        // 10 vs 8 split was returning a margin of 0.19999998 instead of
        // 0.2 — close enough to defeat any threshold set at exactly 0.2.
        let s0 = scores[0];
        let s1 = scores[1];
        let denom = s0.abs().max(EPS);
        let margin = ((s0 - s1) / denom).clamp(0.0, 1.0);
        p.decision_margin = Some(margin);

        // Normalized Shannon entropy: H / ln(k). Result in [0, 1] where
        // 1 means a flat distribution.
        let h: f32 = posterior
            .iter()
            .filter(|&&pi| pi > 0.0)
            .map(|&pi| -pi * pi.ln())
            .sum();
        let max_h = (scores.len() as f32).ln().max(EPS);
        p.score_entropy = Some((h / max_h).clamp(0.0, 1.0));
    }

    p.aggregate = match (p.posterior_concentration, p.score_entropy) {
        (Some(c), Some(e)) => Some((c * (1.0 - e)).clamp(0.0, 1.0)),
        (Some(c), None) => Some(c),
        _ => None,
    };

    p
}

fn softmax_with_fitted_temperature(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    // Temperature = standard deviation of scores. Falls back to `1.0` for
    // degenerate (all-equal) distributions to avoid division by zero.
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let var = scores.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
    let temperature = var.sqrt().max(EPS);

    let max_scaled = scores
        .iter()
        .map(|&s| s / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores
        .iter()
        .map(|&s| (s / temperature - max_scaled).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        // All-equal degenerate case.
        let n = scores.len() as f32;
        return vec![1.0 / n; scores.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop::core::{Chunk, RetrievalMethod, Score, ScoreBreakdown, TokenCount};

    fn r(score: f32) -> RetrievalResult {
        RetrievalResult {
            chunk: Chunk::new("c", "c", "doc", TokenCount(1)),
            score: Score {
                value: score,
                method: RetrievalMethod::Lexical,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }

    #[test]
    fn empty_returns_default() {
        let p = compute_confidence(&[]);
        assert!(p.decision_margin.is_none());
        assert!(p.score_entropy.is_none());
        assert!(p.posterior_concentration.is_none());
        assert!(p.aggregate.is_none());
    }

    #[test]
    fn single_result_no_margin_no_entropy() {
        let p = compute_confidence(&[r(5.0)]);
        assert!(p.decision_margin.is_none());
        assert!(p.score_entropy.is_none());
        // Posterior on a single item is 1.0 by definition.
        assert!(p.posterior_concentration.is_some());
        assert!((p.posterior_concentration.unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn peaked_distribution_low_entropy_high_margin() {
        let p = compute_confidence(&[r(10.0), r(1.0), r(0.5), r(0.1)]);
        // The std-fitted temperature deliberately does NOT collapse the
        // tail to zero — keeping some posterior mass on the tail is the
        // honest answer when the top is large but a tail exists. So a
        // moderately peaked distribution like this lands around 0.55..0.60
        // on the normalized entropy scale, not near zero.
        assert!(
            p.score_entropy.unwrap() < 0.6,
            "entropy {} should be < 0.6",
            p.score_entropy.unwrap()
        );
        assert!(p.decision_margin.unwrap() > 0.5);
        assert!(p.posterior_concentration.unwrap() > 0.7);
        // Aggregate = posterior × (1 - entropy); with entropy ≈ 0.57 and
        // posterior ≈ 0.77 we expect ≈ 0.33.
        assert!(p.aggregate.unwrap() > 0.25);
    }

    #[test]
    fn flat_distribution_high_entropy_low_margin() {
        let p = compute_confidence(&[r(1.0), r(1.0), r(1.0), r(1.0)]);
        // All-equal scores → max entropy.
        assert!((p.score_entropy.unwrap() - 1.0).abs() < 1e-3);
        assert!(p.decision_margin.unwrap() < 1e-3);
        // Posterior concentration ≈ 1/k.
        assert!((p.posterior_concentration.unwrap() - 0.25).abs() < 1e-3);
    }

    #[test]
    fn aggregate_higher_for_peaked_than_flat() {
        let peaked = compute_confidence(&[r(10.0), r(1.0), r(0.5)])
            .aggregate
            .unwrap();
        let flat = compute_confidence(&[r(1.0), r(1.0), r(1.0)])
            .aggregate
            .unwrap();
        assert!(peaked > flat);
    }
}
