//! Default diagnostics engine.
//!
//! Computes every metric in [`crate::metrics`] and assembles them into a
//! single [`DiagnosticsReport`]. Configurable thresholds drive warning
//! emission; the actual numeric values are always reported.

use redhop_core::{DiagnosticsEngine, DiagnosticsReport, Query, Result, RetrievalResult};

use crate::metrics;

/// Configurable thresholds for warning emission.
#[derive(Debug, Clone)]
pub struct DiagnosticsThresholds {
    /// Below this lexical-grounding value, emit a `low_lexical_grounding`
    /// warning.
    pub min_lexical_grounding: f32,
    /// Per-chunk grounding cutoff used to classify a result as a distractor.
    pub distractor_min_grounding: f32,
    /// Above this distractor ratio, emit a `high_distractor_ratio` warning.
    pub max_distractor_ratio: f32,
    /// Above this saturation value, emit a `retrieval_saturated` warning.
    pub max_retrieval_saturation: f32,
}

impl Default for DiagnosticsThresholds {
    fn default() -> Self {
        Self {
            min_lexical_grounding: 0.15,
            distractor_min_grounding: 0.10,
            max_distractor_ratio: 0.50,
            max_retrieval_saturation: 0.85,
        }
    }
}

/// Default diagnostics engine.
#[derive(Debug, Clone, Default)]
pub struct DefaultDiagnosticsEngine {
    thresholds: DiagnosticsThresholds,
}

impl DefaultDiagnosticsEngine {
    /// Construct with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with caller-provided thresholds.
    pub fn with_thresholds(thresholds: DiagnosticsThresholds) -> Self {
        Self { thresholds }
    }
}

impl DiagnosticsEngine for DefaultDiagnosticsEngine {
    fn diagnose(&self, query: &Query, results: &[RetrievalResult]) -> Result<DiagnosticsReport> {
        let grounding = metrics::lexical_grounding(query, results);
        let purity = metrics::chunk_purity(results);
        let density = metrics::answer_density(query, results);
        let distractor =
            metrics::distractor_ratio(query, results, self.thresholds.distractor_min_grounding);
        let saturation = metrics::retrieval_saturation(results);
        let concentration = metrics::evidence_concentration(results);
        let confidence = metrics::retrieval_confidence(grounding, concentration);

        let mut report = DiagnosticsReport {
            answer_density: density,
            distractor_ratio: distractor,
            retrieval_confidence: confidence,
            retrieval_saturation: saturation,
            evidence_concentration: concentration,
            lexical_grounding: grounding,
            chunk_purity: purity,
            // Semantic-tier fields are owned by SemanticDiagnosticsEngine;
            // this engine leaves them None and lets a LayeredDiagnosticsEngine
            // fill them in.
            ..Default::default()
        };

        if let Some(g) = grounding {
            if g < self.thresholds.min_lexical_grounding {
                report = report.with_warning(
                    "low_lexical_grounding",
                    format!(
                        "average query/chunk lexical overlap {:.3} is below threshold {:.3}; the reader model may struggle to anchor evidence",
                        g, self.thresholds.min_lexical_grounding
                    ),
                );
            }
        }
        if let Some(d) = distractor {
            if d > self.thresholds.max_distractor_ratio {
                report = report.with_warning(
                    "high_distractor_ratio",
                    format!(
                        "{:.0}% of retrieved chunks fall below the per-chunk grounding cutoff",
                        d * 100.0
                    ),
                );
            }
        }
        if let Some(s) = saturation {
            if s > self.thresholds.max_retrieval_saturation {
                report = report.with_warning(
                    "retrieval_saturated",
                    format!(
                        "retrieval has saturated ({:.0}% tail/head term overlap); increasing top_k is unlikely to add new evidence",
                        s * 100.0
                    ),
                );
            }
        }
        Ok(report)
    }

    fn name(&self) -> &'static str {
        "default"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop_core::{Chunk, RetrievalMethod, Score, ScoreBreakdown, TokenCount};

    fn r(text: &str, score: f32) -> RetrievalResult {
        RetrievalResult {
            chunk: Chunk::new(
                text,
                text,
                "doc",
                TokenCount(text.split_whitespace().count()),
            ),
            score: Score {
                value: score,
                method: RetrievalMethod::Lexical,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }

    #[test]
    fn emits_warnings_on_bad_retrieval() {
        let e = DefaultDiagnosticsEngine::new();
        let q = Query::new("rust async runtime");
        // All chunks unrelated → low grounding, high distractor.
        let results = vec![
            r("cats nap a lot in the sun", 1.0),
            r("the weather is sunny today", 0.9),
            r("breakfast was tasty this morning", 0.8),
        ];
        let report = e.diagnose(&q, &results).unwrap();
        let codes: Vec<&str> = report.warnings.iter().map(|w| w.code.as_str()).collect();
        assert!(codes.contains(&"low_lexical_grounding"));
        assert!(codes.contains(&"high_distractor_ratio"));
    }

    #[test]
    fn clean_retrieval_no_warnings() {
        let e = DefaultDiagnosticsEngine::new();
        let q = Query::new("rust async runtime");
        let results = vec![
            r("rust async runtime tokio executor", 1.0),
            r("rust futures async await runtime", 0.9),
        ];
        let report = e.diagnose(&q, &results).unwrap();
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert!(report.retrieval_confidence.unwrap() > 0.3);
    }
}
