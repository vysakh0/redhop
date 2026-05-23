//! Composition of multiple diagnostics engines into a single report.
//!
//! The motivating use case is layering [`DefaultDiagnosticsEngine`] (lexical
//! tier) and [`SemanticDiagnosticsEngine`] (semantic tier) so callers see a
//! single unified [`DiagnosticsReport`] with both lexical and semantic
//! fields populated.
//!
//! The layering is *order-preserving*: the first engine's report establishes
//! the base, and each subsequent engine fills in only the fields the
//! previous layers left as `None`. Warnings accumulate. This matters because
//! a layered configuration like `[default, semantic, custom]` should not
//! have the custom engine silently overwrite an explicit `lexical_grounding`
//! from `default`.
//!
//! [`DefaultDiagnosticsEngine`]: crate::engine::DefaultDiagnosticsEngine
//! [`SemanticDiagnosticsEngine`]: crate::semantic::SemanticDiagnosticsEngine

use std::sync::Arc;

use redhop_core::{DiagnosticsEngine, DiagnosticsReport, Query, Result, RetrievalResult};

/// A diagnostics engine that runs a list of underlying engines and merges
/// their reports.
pub struct LayeredDiagnosticsEngine {
    layers: Vec<Arc<dyn DiagnosticsEngine>>,
}

impl LayeredDiagnosticsEngine {
    /// Construct from a non-empty vector of engines.
    ///
    /// Empty `layers` is legal but useless; the resulting engine returns an
    /// empty `DiagnosticsReport` on every call.
    pub fn new(layers: Vec<Arc<dyn DiagnosticsEngine>>) -> Self {
        Self { layers }
    }

    /// Convenience constructor for the canonical two-tier setup.
    pub fn lexical_and_semantic(
        lexical: Arc<dyn DiagnosticsEngine>,
        semantic: Arc<dyn DiagnosticsEngine>,
    ) -> Self {
        Self::new(vec![lexical, semantic])
    }
}

impl DiagnosticsEngine for LayeredDiagnosticsEngine {
    fn diagnose(&self, query: &Query, results: &[RetrievalResult]) -> Result<DiagnosticsReport> {
        let mut combined = DiagnosticsReport::default();
        for layer in &self.layers {
            let partial = layer.diagnose(query, results)?;
            combined = combined.merge(partial);
        }
        Ok(combined)
    }

    fn name(&self) -> &'static str {
        "layered"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DefaultDiagnosticsEngine, SemanticDiagnosticsEngine};
    use redhop_core::{Chunk, Embedding, RetrievalMethod, Score, ScoreBreakdown, TokenCount};

    fn r(text: &str, emb: Option<Vec<f32>>) -> RetrievalResult {
        let mut c = Chunk::new(
            text,
            text,
            "doc",
            TokenCount(text.split_whitespace().count()),
        );
        if let Some(e) = emb {
            c = c.with_embedding(Embedding::from(e));
        }
        RetrievalResult {
            chunk: c,
            score: Score {
                value: 1.0,
                method: RetrievalMethod::Dense,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }

    #[test]
    fn merges_lexical_and_semantic_fields() {
        let lexical = Arc::new(DefaultDiagnosticsEngine::new()) as Arc<dyn DiagnosticsEngine>;
        let semantic = Arc::new(SemanticDiagnosticsEngine::new()) as Arc<dyn DiagnosticsEngine>;
        let engine = LayeredDiagnosticsEngine::lexical_and_semantic(lexical, semantic);

        let query =
            Query::new("rust async runtime").with_embedding(Embedding::from(vec![1.0, 0.0, 0.0]));
        let results = vec![
            r("rust async runtime tokio", Some(vec![0.9, 0.1, 0.0])),
            r("rust futures executor", Some(vec![0.8, 0.2, 0.0])),
        ];
        let report = engine.diagnose(&query, &results).unwrap();
        assert!(report.lexical_grounding.is_some(), "lexical tier populated");
        assert!(
            report.semantic_grounding.is_some(),
            "semantic tier populated"
        );
    }

    #[test]
    fn semantic_silently_skipped_when_chunks_lack_embeddings() {
        let lexical = Arc::new(DefaultDiagnosticsEngine::new()) as Arc<dyn DiagnosticsEngine>;
        let semantic = Arc::new(SemanticDiagnosticsEngine::new()) as Arc<dyn DiagnosticsEngine>;
        let engine = LayeredDiagnosticsEngine::lexical_and_semantic(lexical, semantic);

        let query = Query::new("rust async runtime");
        let results = vec![r("rust async runtime tokio", None)];
        let report = engine.diagnose(&query, &results).unwrap();
        assert!(report.lexical_grounding.is_some());
        assert!(report.semantic_grounding.is_none());
    }

    #[test]
    fn first_layer_wins_on_overlap() {
        // Two semantic engines layered: the first one's value should be
        // preserved even if the second would compute a different value.
        let q = Query::new("ignored").with_embedding(Embedding::from(vec![1.0, 0.0, 0.0]));
        let results = vec![r("a", Some(vec![1.0, 0.0, 0.0]))];

        let strict = Arc::new(SemanticDiagnosticsEngine::with_config(
            crate::semantic::SemanticDiagnosticsConfig {
                distractor_min_cosine: 0.95,
                ..Default::default()
            },
        )) as Arc<dyn DiagnosticsEngine>;
        let lenient = Arc::new(SemanticDiagnosticsEngine::new()) as Arc<dyn DiagnosticsEngine>;

        let engine = LayeredDiagnosticsEngine::new(vec![strict.clone(), lenient]);
        let report = engine.diagnose(&q, &results).unwrap();
        // strict layer's distractor classification (cosine 1.0 >= 0.95) wins.
        assert_eq!(report.semantic_distractor_ratio.unwrap(), 0.0);
    }
}
