//! Rule-based regime classifier.
//!
//! This is the **interpretable** classifier the user should reach for by
//! default. Every classification produces a complete
//! [`ClassificationTrace`] — every feature inspected, every threshold
//! applied, every rule that fired, and the pre-softmax regime scores.
//! Given a [`RegimeDistribution`][rd] you can fully reconstruct *why*
//! the classifier reached its verdict without rerunning anything.
//!
//! ## Anatomy of a classification
//!
//! ```text
//!   diagnostics + confidence
//!         │
//!         ▼
//!   ┌─────────────────────────────────────────┐
//!   │  Rule 1: low_lexical_grounding → Sparse │ (weight w₁)
//!   │  Rule 2: redundant_semantics  → Saturated│ (weight w₂)
//!   │  Rule 3: high_dispersion      → Ambiguous│ (weight w₃)
//!   │  …                                       │
//!   └────────────────┬────────────────────────┘
//!                    │
//!     accumulate per-regime score
//!                    │
//!                    ▼
//!     softmax(regime scores, τ) → RegimeDistribution
//! ```
//!
//! Each rule is a small named function returning `Option<RuleFire>`. If
//! the rule's condition isn't met it returns `None` and contributes
//! nothing. If it fires it returns a [`RuleFire`] including a
//! human-readable justification with the actual numbers it saw.
//!
//! ## Why softmax over linear normalization
//!
//! We tried linear normalization first. The problem: when only one regime
//! has any fired rules (e.g. a textbook `Easy` case), linear
//! normalization makes its probability `1.0` and every other regime
//! exactly `0.0`. That's overconfident — it hides the (small but real)
//! probability that the diagnostics are noisy. Softmax preserves a thin
//! mass on the other regimes, which is what we want when later phases
//! make hedged decisions.
//!
//! [rd]: redhop_core::RegimeDistribution

use std::collections::BTreeMap;

use redhop_core::{
    ClassificationTrace, ConfidenceProfile, DiagnosticsReport, RegimeClassifier,
    RegimeDistribution, RetrievalRegime, RuleFire,
};

/// Visible, tweakable threshold values used by [`RuleBasedClassifier`].
///
/// Every threshold appears in the [`ClassificationTrace`] of every
/// classification this classifier produces.
#[derive(Debug, Clone)]
pub struct ClassifierThresholds {
    // Easy regime
    /// Below this lexical grounding the "easy" rule will not fire.
    pub easy_min_lexical_grounding: f32,
    /// Below this semantic grounding the "easy" rule will not fire.
    pub easy_min_semantic_grounding: f32,
    /// Above this distractor ratio the "easy" rule is suppressed.
    pub easy_max_distractor_ratio: f32,
    /// Below this decision margin the "easy" rule is suppressed.
    pub easy_min_decision_margin: f32,

    // Saturated regime
    /// Above this lexical retrieval saturation, classify as saturated.
    pub saturated_min_lexical: f32,
    /// Above this semantic redundancy, classify as saturated.
    pub saturated_min_semantic: f32,

    // Distractor-heavy regime
    /// Above this lexical distractor ratio, classify as distractor-heavy.
    pub distractor_min_lexical: f32,
    /// Above this semantic distractor ratio, classify as distractor-heavy.
    pub distractor_min_semantic: f32,

    // Ambiguous regime
    /// Above this score entropy (`[0, 1]`), classify as ambiguous.
    pub ambiguous_min_entropy: f32,
    /// Above this centroid dispersion, classify as ambiguous.
    pub ambiguous_min_dispersion: f32,
    /// Below this decision margin, classify as ambiguous.
    pub ambiguous_max_margin: f32,

    // Sparse regime
    /// Below this lexical grounding the "sparse" rule fires.
    pub sparse_max_lexical_grounding: f32,
    /// Below this semantic grounding the "sparse" rule fires.
    /// The semantic baseline for "no signal" is around `0.5` (shifted
    /// cosine of `0`); a value just above that still counts as sparse.
    pub sparse_max_semantic_grounding: f32,

    /// Softmax temperature applied to the per-regime scores. Lower values
    /// produce more peaked distributions. The default `0.5` is calibrated to
    /// produce a clearly dominant regime when one rule fires strongly while
    /// still preserving non-trivial mass on the other regimes (≈10% each
    /// for a single-rule classification, rising as more rules fire).
    pub softmax_temperature: f32,
}

impl Default for ClassifierThresholds {
    fn default() -> Self {
        // Defaults tuned on our internal HotpotQA + judge-model traces.
        // They are explicit constants rather than magic numbers — callers
        // wanting different behavior should construct a fresh instance.
        Self {
            easy_min_lexical_grounding: 0.40,
            easy_min_semantic_grounding: 0.75,
            easy_max_distractor_ratio: 0.25,
            easy_min_decision_margin: 0.20,

            saturated_min_lexical: 0.75,
            saturated_min_semantic: 0.80,

            distractor_min_lexical: 0.40,
            distractor_min_semantic: 0.40,

            ambiguous_min_entropy: 0.75,
            ambiguous_min_dispersion: 0.50,
            ambiguous_max_margin: 0.10,

            sparse_max_lexical_grounding: 0.10,
            sparse_max_semantic_grounding: 0.55,

            softmax_temperature: 0.5,
        }
    }
}

/// Interpretable, threshold-driven regime classifier.
#[derive(Debug, Clone, Default)]
pub struct RuleBasedClassifier {
    thresholds: ClassifierThresholds,
}

impl RuleBasedClassifier {
    /// Construct with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with explicit thresholds.
    pub fn with_thresholds(thresholds: ClassifierThresholds) -> Self {
        Self { thresholds }
    }

    /// Borrow the in-effect thresholds. Useful for callers that want to
    /// log the configuration alongside the classification.
    pub fn thresholds(&self) -> &ClassifierThresholds {
        &self.thresholds
    }
}

impl RegimeClassifier for RuleBasedClassifier {
    fn classify(&self, d: &DiagnosticsReport, c: &ConfidenceProfile) -> RegimeDistribution {
        let t = &self.thresholds;
        let mut trace = ClassificationTrace::default();

        // ---- Record features ----
        macro_rules! feat {
            ($name:expr, $expr:expr) => {
                if let Some(v) = $expr {
                    trace.features.insert($name.to_string(), v);
                }
            };
        }
        feat!("lexical_grounding", d.lexical_grounding);
        feat!("semantic_grounding", d.semantic_grounding);
        feat!("answer_density", d.answer_density);
        feat!("distractor_ratio", d.distractor_ratio);
        feat!("semantic_distractor_ratio", d.semantic_distractor_ratio);
        feat!("retrieval_saturation", d.retrieval_saturation);
        feat!("semantic_redundancy", d.semantic_redundancy);
        feat!("evidence_concentration", d.evidence_concentration);
        feat!("centroid_dispersion", d.centroid_dispersion);
        feat!("chunk_purity", d.chunk_purity);
        feat!("decision_margin", c.decision_margin);
        feat!("score_entropy", c.score_entropy);
        feat!("posterior_concentration", c.posterior_concentration);

        // ---- Record thresholds ----
        macro_rules! thr {
            ($name:expr, $val:expr) => {
                trace.thresholds.insert($name.to_string(), $val);
            };
        }
        thr!("easy_min_lexical_grounding", t.easy_min_lexical_grounding);
        thr!("easy_min_semantic_grounding", t.easy_min_semantic_grounding);
        thr!("easy_max_distractor_ratio", t.easy_max_distractor_ratio);
        thr!("easy_min_decision_margin", t.easy_min_decision_margin);
        thr!("saturated_min_lexical", t.saturated_min_lexical);
        thr!("saturated_min_semantic", t.saturated_min_semantic);
        thr!("distractor_min_lexical", t.distractor_min_lexical);
        thr!("distractor_min_semantic", t.distractor_min_semantic);
        thr!("ambiguous_min_entropy", t.ambiguous_min_entropy);
        thr!("ambiguous_min_dispersion", t.ambiguous_min_dispersion);
        thr!("ambiguous_max_margin", t.ambiguous_max_margin);
        thr!(
            "sparse_max_lexical_grounding",
            t.sparse_max_lexical_grounding
        );
        thr!(
            "sparse_max_semantic_grounding",
            t.sparse_max_semantic_grounding
        );
        thr!("softmax_temperature", t.softmax_temperature);

        // ---- Evaluate rules ----
        let mut fires: Vec<RuleFire> = Vec::new();
        for rule in RULES {
            if let Some(fire) = (rule.eval)(d, c, t) {
                fires.push(fire);
            }
        }

        // Accumulate per-regime score.
        let mut raw_scores: BTreeMap<RetrievalRegime, f32> = BTreeMap::new();
        for r in RetrievalRegime::all() {
            raw_scores.insert(*r, 0.0);
        }
        for fire in &fires {
            *raw_scores.entry(fire.regime).or_insert(0.0) += fire.weight;
        }

        // Edge case: no rules fired at all. This happens when every signal
        // is `None` (e.g. retrieval returned zero results). Emit a uniform
        // distribution and record a meta-rule so the trace is honest about
        // it.
        if fires.is_empty() {
            let r = RuleFire {
                rule: "no_signals_available".to_string(),
                regime: RetrievalRegime::Sparse,
                weight: 0.0,
                justification:
                    "no diagnostic signals available; defaulting to uniform regime distribution"
                        .to_string(),
            };
            fires.push(r);
        }

        trace.rules_fired = fires;
        trace.raw_scores = raw_scores.clone();

        let probabilities = softmax_regimes(&raw_scores, t.softmax_temperature);
        let argmax = argmax_regime(&probabilities);

        RegimeDistribution {
            probabilities,
            argmax,
            trace,
        }
    }

    fn name(&self) -> &'static str {
        "rule_based"
    }
}

// ─────────────────────────────────────────────────────────────────────
// Rule table
// ─────────────────────────────────────────────────────────────────────

/// A single classification rule.
struct Rule {
    eval: fn(&DiagnosticsReport, &ConfidenceProfile, &ClassifierThresholds) -> Option<RuleFire>,
}

const RULES: &[Rule] = &[
    Rule {
        eval: rule_easy_lexically_grounded,
    },
    Rule {
        eval: rule_easy_semantically_grounded,
    },
    Rule {
        eval: rule_saturated_lexical,
    },
    Rule {
        eval: rule_saturated_semantic,
    },
    Rule {
        eval: rule_distractor_lexical,
    },
    Rule {
        eval: rule_distractor_semantic,
    },
    Rule {
        eval: rule_ambiguous_high_entropy,
    },
    Rule {
        eval: rule_ambiguous_high_dispersion,
    },
    Rule {
        eval: rule_ambiguous_low_margin,
    },
    Rule {
        eval: rule_sparse_low_grounding,
    },
];

fn fire(rule: &'static str, regime: RetrievalRegime, weight: f32, msg: String) -> Option<RuleFire> {
    Some(RuleFire {
        rule: rule.to_string(),
        regime,
        weight,
        justification: msg,
    })
}

fn rule_easy_lexically_grounded(
    d: &DiagnosticsReport,
    c: &ConfidenceProfile,
    t: &ClassifierThresholds,
) -> Option<RuleFire> {
    let g = d.lexical_grounding?;
    let dr = d.distractor_ratio.unwrap_or(0.0);
    // decision_margin = None means fewer than two candidates were
    // returned; that is *not* a strike against Easy — a single-candidate
    // retrieval has trivially infinite margin. Only enforce the margin
    // threshold when we actually measured one.
    let (margin_ok, m_display) = match c.decision_margin {
        Some(m) => (m >= t.easy_min_decision_margin, m),
        None => (true, f32::NAN),
    };
    if g >= t.easy_min_lexical_grounding && margin_ok && dr <= t.easy_max_distractor_ratio {
        // Weight grows with the *evidence* of easiness; a clean retrieval
        // gets a larger vote than a borderline one.
        let base = (g - t.easy_min_lexical_grounding).max(0.0);
        let margin_bonus = c
            .decision_margin
            .map(|m| (m - t.easy_min_decision_margin).max(0.0))
            .unwrap_or(0.0);
        let weight = (base + margin_bonus).max(0.3);
        fire(
            "easy_lexically_grounded",
            RetrievalRegime::Easy,
            weight,
            if m_display.is_nan() {
                format!(
                    "lexical_grounding={:.2}≥{:.2} ∧ decision_margin unmeasured (single candidate) ∧ distractor_ratio={:.2}≤{:.2}",
                    g, t.easy_min_lexical_grounding,
                    dr, t.easy_max_distractor_ratio
                )
            } else {
                format!(
                    "lexical_grounding={:.2}≥{:.2} ∧ decision_margin={:.2}≥{:.2} ∧ distractor_ratio={:.2}≤{:.2}",
                    g, t.easy_min_lexical_grounding,
                    m_display, t.easy_min_decision_margin,
                    dr, t.easy_max_distractor_ratio
                )
            },
        )
    } else {
        None
    }
}

fn rule_easy_semantically_grounded(
    d: &DiagnosticsReport,
    c: &ConfidenceProfile,
    t: &ClassifierThresholds,
) -> Option<RuleFire> {
    let g = d.semantic_grounding?;
    let dr = d.semantic_distractor_ratio.unwrap_or(0.0);
    // Same rationale as in rule_easy_lexically_grounded: a missing
    // decision_margin means "could not measure" not "low confidence".
    let (margin_ok, m_display) = match c.decision_margin {
        Some(m) => (m >= t.easy_min_decision_margin, m),
        None => (true, f32::NAN),
    };
    if g >= t.easy_min_semantic_grounding && margin_ok && dr <= t.easy_max_distractor_ratio {
        let base = (g - t.easy_min_semantic_grounding).max(0.0);
        let margin_bonus = c
            .decision_margin
            .map(|m| (m - t.easy_min_decision_margin).max(0.0))
            .unwrap_or(0.0);
        let weight = (base + margin_bonus).max(0.3);
        fire(
            "easy_semantically_grounded",
            RetrievalRegime::Easy,
            weight,
            if m_display.is_nan() {
                format!(
                    "semantic_grounding={:.2}≥{:.2} ∧ decision_margin unmeasured (single candidate) ∧ semantic_distractor_ratio={:.2}≤{:.2}",
                    g, t.easy_min_semantic_grounding,
                    dr, t.easy_max_distractor_ratio
                )
            } else {
                format!(
                    "semantic_grounding={:.2}≥{:.2} ∧ decision_margin={:.2}≥{:.2} ∧ semantic_distractor_ratio={:.2}≤{:.2}",
                    g, t.easy_min_semantic_grounding,
                    m_display, t.easy_min_decision_margin,
                    dr, t.easy_max_distractor_ratio
                )
            },
        )
    } else {
        None
    }
}

fn rule_saturated_lexical(
    d: &DiagnosticsReport,
    _c: &ConfidenceProfile,
    t: &ClassifierThresholds,
) -> Option<RuleFire> {
    let s = d.retrieval_saturation?;
    if s >= t.saturated_min_lexical {
        fire(
            "saturated_lexical",
            RetrievalRegime::Saturated,
            (s - t.saturated_min_lexical) + 0.5,
            format!(
                "retrieval_saturation={:.2}≥{:.2}: top-k tail reuses head vocabulary",
                s, t.saturated_min_lexical
            ),
        )
    } else {
        None
    }
}

fn rule_saturated_semantic(
    d: &DiagnosticsReport,
    _c: &ConfidenceProfile,
    t: &ClassifierThresholds,
) -> Option<RuleFire> {
    let r = d.semantic_redundancy?;
    if r >= t.saturated_min_semantic {
        fire(
            "saturated_semantic",
            RetrievalRegime::Saturated,
            (r - t.saturated_min_semantic) + 0.5,
            format!(
                "semantic_redundancy={:.2}≥{:.2}: retrieved chunks cluster tightly in embedding space",
                r, t.saturated_min_semantic
            ),
        )
    } else {
        None
    }
}

fn rule_distractor_lexical(
    d: &DiagnosticsReport,
    _c: &ConfidenceProfile,
    t: &ClassifierThresholds,
) -> Option<RuleFire> {
    let r = d.distractor_ratio?;
    if r >= t.distractor_min_lexical {
        fire(
            "distractor_lexical",
            RetrievalRegime::DistractorHeavy,
            (r - t.distractor_min_lexical) + 0.5,
            format!(
                "distractor_ratio={:.2}≥{:.2}: many chunks fall below per-chunk grounding cutoff",
                r, t.distractor_min_lexical
            ),
        )
    } else {
        None
    }
}

fn rule_distractor_semantic(
    d: &DiagnosticsReport,
    _c: &ConfidenceProfile,
    t: &ClassifierThresholds,
) -> Option<RuleFire> {
    let r = d.semantic_distractor_ratio?;
    if r >= t.distractor_min_semantic {
        fire(
            "distractor_semantic",
            RetrievalRegime::DistractorHeavy,
            (r - t.distractor_min_semantic) + 0.5,
            format!(
                "semantic_distractor_ratio={:.2}≥{:.2}: many chunks fall below query-cosine cutoff",
                r, t.distractor_min_semantic
            ),
        )
    } else {
        None
    }
}

fn rule_ambiguous_high_entropy(
    _d: &DiagnosticsReport,
    c: &ConfidenceProfile,
    t: &ClassifierThresholds,
) -> Option<RuleFire> {
    let e = c.score_entropy?;
    if e >= t.ambiguous_min_entropy {
        fire(
            "ambiguous_high_entropy",
            RetrievalRegime::Ambiguous,
            (e - t.ambiguous_min_entropy) + 0.4,
            format!(
                "score_entropy={:.2}≥{:.2}: top-k score distribution is flat",
                e, t.ambiguous_min_entropy
            ),
        )
    } else {
        None
    }
}

fn rule_ambiguous_high_dispersion(
    d: &DiagnosticsReport,
    _c: &ConfidenceProfile,
    t: &ClassifierThresholds,
) -> Option<RuleFire> {
    let d_ = d.centroid_dispersion?;
    if d_ >= t.ambiguous_min_dispersion {
        fire(
            "ambiguous_high_dispersion",
            RetrievalRegime::Ambiguous,
            (d_ - t.ambiguous_min_dispersion) + 0.4,
            format!(
                "centroid_dispersion={:.2}≥{:.2}: retrieved chunks scatter in embedding space",
                d_, t.ambiguous_min_dispersion
            ),
        )
    } else {
        None
    }
}

fn rule_ambiguous_low_margin(
    _d: &DiagnosticsReport,
    c: &ConfidenceProfile,
    t: &ClassifierThresholds,
) -> Option<RuleFire> {
    let m = c.decision_margin?;
    if m <= t.ambiguous_max_margin {
        fire(
            "ambiguous_low_margin",
            RetrievalRegime::Ambiguous,
            (t.ambiguous_max_margin - m) + 0.3,
            format!(
                "decision_margin={:.2}≤{:.2}: top-1 and top-2 nearly tied",
                m, t.ambiguous_max_margin
            ),
        )
    } else {
        None
    }
}

fn rule_sparse_low_grounding(
    d: &DiagnosticsReport,
    _c: &ConfidenceProfile,
    t: &ClassifierThresholds,
) -> Option<RuleFire> {
    // Sparse needs *both* signals to be low to fire — a low lexical
    // grounding alone could just be the paraphrase regime.
    let l = d.lexical_grounding;
    let s = d.semantic_grounding;
    let lex_low = l
        .map(|x| x <= t.sparse_max_lexical_grounding)
        .unwrap_or(false);
    let sem_low = s
        .map(|x| x <= t.sparse_max_semantic_grounding)
        .unwrap_or(false);
    // If we only have one tier and it's low, fire with reduced weight —
    // we have less evidence than with both.
    match (l, s) {
        (Some(l), Some(s)) if lex_low && sem_low => fire(
            "sparse_both_tiers_low",
            RetrievalRegime::Sparse,
            1.0,
            format!(
                "lexical_grounding={:.2}≤{:.2} ∧ semantic_grounding={:.2}≤{:.2}",
                l, t.sparse_max_lexical_grounding, s, t.sparse_max_semantic_grounding
            ),
        ),
        (Some(l), None) if lex_low => fire(
            "sparse_lexical_only",
            RetrievalRegime::Sparse,
            0.4,
            format!(
                "lexical_grounding={:.2}≤{:.2} and no semantic signal available",
                l, t.sparse_max_lexical_grounding
            ),
        ),
        (None, Some(s)) if sem_low => fire(
            "sparse_semantic_only",
            RetrievalRegime::Sparse,
            0.4,
            format!(
                "semantic_grounding={:.2}≤{:.2} and no lexical signal available",
                s, t.sparse_max_semantic_grounding
            ),
        ),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Softmax + argmax helpers
// ─────────────────────────────────────────────────────────────────────

fn softmax_regimes(
    raw: &BTreeMap<RetrievalRegime, f32>,
    temperature: f32,
) -> BTreeMap<RetrievalRegime, f32> {
    let t = temperature.max(1e-6);
    // Numerical stability: subtract max before exp.
    let max = raw.values().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
    let exps: Vec<(RetrievalRegime, f32)> = raw
        .iter()
        .map(|(&r, &v)| (r, ((v - max) / t).exp()))
        .collect();
    let sum: f32 = exps.iter().map(|(_, e)| e).sum();
    let mut out = BTreeMap::new();
    if sum <= 0.0 {
        let n = exps.len() as f32;
        for (r, _) in exps {
            out.insert(r, 1.0 / n);
        }
    } else {
        for (r, e) in exps {
            out.insert(r, e / sum);
        }
    }
    out
}

fn argmax_regime(probs: &BTreeMap<RetrievalRegime, f32>) -> RetrievalRegime {
    probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(r, _)| *r)
        // Should never trigger — the map always has the five regimes
        // populated to zero. Defensive fallback.
        .unwrap_or(RetrievalRegime::Sparse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop_core::ConfidenceProfile;

    fn cls() -> RuleBasedClassifier {
        RuleBasedClassifier::new()
    }

    fn diag() -> DiagnosticsReport {
        DiagnosticsReport::default()
    }

    fn conf() -> ConfidenceProfile {
        ConfidenceProfile::default()
    }

    #[test]
    fn no_signals_yields_uniform_distribution_with_trace() {
        let r = cls().classify(&diag(), &conf());
        // All five regimes have probability ~0.2 since raw scores are all
        // zero → softmax of zeros → uniform.
        for &reg in RetrievalRegime::all() {
            assert!(
                (r.p(reg) - 0.2).abs() < 1e-4,
                "regime {:?} got {}",
                reg,
                r.p(reg)
            );
        }
        assert_eq!(r.trace.rules_fired.len(), 1);
        assert_eq!(r.trace.rules_fired[0].rule, "no_signals_available");
    }

    #[test]
    fn clean_lexical_retrieval_classifies_as_easy() {
        let d = DiagnosticsReport {
            lexical_grounding: Some(0.9),
            distractor_ratio: Some(0.0),
            ..Default::default()
        };
        let c = ConfidenceProfile {
            decision_margin: Some(0.6),
            score_entropy: Some(0.2),
            ..Default::default()
        };
        let r = cls().classify(&d, &c);
        assert_eq!(r.argmax, RetrievalRegime::Easy);
        assert!(r.p(RetrievalRegime::Easy) > 0.4);
        assert!(r
            .trace
            .rules_fired
            .iter()
            .any(|f| f.rule == "easy_lexically_grounded"));
    }

    #[test]
    fn paraphrase_retrieval_classifies_as_easy_via_semantic() {
        // Lexical grounding low — paraphrase signature.
        // Semantic grounding high, distractor low.
        let d = DiagnosticsReport {
            lexical_grounding: Some(0.05),
            semantic_grounding: Some(0.95),
            semantic_distractor_ratio: Some(0.0),
            ..Default::default()
        };
        let c = ConfidenceProfile {
            decision_margin: Some(0.5),
            score_entropy: Some(0.3),
            ..Default::default()
        };
        let r = cls().classify(&d, &c);
        assert_eq!(r.argmax, RetrievalRegime::Easy);
        assert!(r
            .trace
            .rules_fired
            .iter()
            .any(|f| f.rule == "easy_semantically_grounded"));
    }

    #[test]
    fn redundant_top_k_classifies_as_saturated() {
        let d = DiagnosticsReport {
            retrieval_saturation: Some(0.9),
            semantic_redundancy: Some(0.95),
            lexical_grounding: Some(0.4),
            ..Default::default()
        };
        let r = cls().classify(&d, &conf());
        assert_eq!(r.argmax, RetrievalRegime::Saturated);
        assert!(r
            .trace
            .rules_fired
            .iter()
            .any(|f| f.rule == "saturated_semantic"));
    }

    #[test]
    fn off_topic_results_classify_as_distractor_heavy() {
        let d = DiagnosticsReport {
            lexical_grounding: Some(0.3),
            distractor_ratio: Some(0.7),
            semantic_distractor_ratio: Some(0.6),
            ..Default::default()
        };
        let r = cls().classify(&d, &conf());
        assert_eq!(r.argmax, RetrievalRegime::DistractorHeavy);
    }

    #[test]
    fn flat_scores_with_dispersion_classify_as_ambiguous() {
        let d = DiagnosticsReport {
            lexical_grounding: Some(0.3),
            centroid_dispersion: Some(0.7),
            ..Default::default()
        };
        let c = ConfidenceProfile {
            decision_margin: Some(0.02),
            score_entropy: Some(0.95),
            ..Default::default()
        };
        let r = cls().classify(&d, &c);
        assert_eq!(r.argmax, RetrievalRegime::Ambiguous);
    }

    #[test]
    fn no_matches_classifies_as_sparse() {
        let d = DiagnosticsReport {
            lexical_grounding: Some(0.0),
            semantic_grounding: Some(0.50),
            ..Default::default()
        };
        let r = cls().classify(&d, &conf());
        assert_eq!(r.argmax, RetrievalRegime::Sparse);
        assert!(r
            .trace
            .rules_fired
            .iter()
            .any(|f| f.rule == "sparse_both_tiers_low"));
    }

    #[test]
    fn trace_records_features_and_thresholds() {
        let d = DiagnosticsReport {
            lexical_grounding: Some(0.42),
            ..Default::default()
        };
        let r = cls().classify(&d, &conf());
        assert!(r.trace.features.contains_key("lexical_grounding"));
        assert!(r
            .trace
            .thresholds
            .contains_key("easy_min_lexical_grounding"));
    }

    #[test]
    fn probabilities_sum_to_one() {
        let d = DiagnosticsReport {
            lexical_grounding: Some(0.5),
            ..Default::default()
        };
        let r = cls().classify(&d, &conf());
        let sum: f32 = r.probabilities.values().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
    }

    #[test]
    fn distribution_entropy_higher_when_signals_split_across_regimes() {
        // Clean easy case — one rule fires for one regime, peaked posterior.
        let clean = cls().classify(
            &DiagnosticsReport {
                lexical_grounding: Some(0.9),
                distractor_ratio: Some(0.0),
                ..Default::default()
            },
            &ConfidenceProfile {
                decision_margin: Some(0.6),
                ..Default::default()
            },
        );
        // Split case — rules fire toward *different* regimes (Saturated +
        // DistractorHeavy), so posterior mass is divided. This is what
        // "conflicting signals" should mean and what should raise the
        // distribution's entropy.
        let split = cls().classify(
            &DiagnosticsReport {
                lexical_grounding: Some(0.3),
                semantic_redundancy: Some(0.90), // → Saturated
                distractor_ratio: Some(0.60),    // → DistractorHeavy
                ..Default::default()
            },
            &ConfidenceProfile::default(),
        );
        assert!(
            split.entropy() > clean.entropy(),
            "split-regime entropy {} should exceed clean entropy {}",
            split.entropy(),
            clean.entropy()
        );
    }
}
