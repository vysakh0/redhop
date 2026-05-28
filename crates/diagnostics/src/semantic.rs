//! Semantic-tier diagnostics.
//!
//! Where the lexical tier ([`crate::engine::DefaultDiagnosticsEngine`])
//! computes everything from raw text, the semantic tier reads the
//! embeddings already present on the [`Query`] and on each retrieved
//! [`Chunk`]. It closes the *paraphrase blind spot* of the lexical tier:
//!
//! > A chunk like "Tim Cook earned $99M in fiscal 2023" has zero lexical
//! > grounding against the query "What is the CEO's salary?" — but high
//! > semantic grounding. The lexical engine treats it as a distractor; the
//! > semantic engine recognizes it as evidence.
//!
//! Crucially, *no embedding model is invoked here*. The query embedding is
//! whatever the caller already produced for dense retrieval; the chunk
//! embeddings are whatever the caller persisted at ingest time. If either
//! is missing the engine leaves the relevant fields as `None` and returns
//! cleanly. This keeps RedHop retrieval-centric: we observe what other
//! parts of the pipeline already paid for, we do not pay again.
//!
//! ## Metrics
//!
//! Four numbers, all in `[0, 1]`:
//!
//! - **`semantic_grounding`** — mean cosine between query and each chunk,
//!   shifted from `[-1, 1]` into `[0, 1]`. *Higher is better.*
//! - **`semantic_redundancy`** — mean pairwise cosine across the retrieved
//!   chunks. Semantic counterpart to `retrieval_saturation`. *Higher means
//!   more redundant.* Used as the early-stop signal in adaptive reranking.
//! - **`centroid_dispersion`** — `1 − mean cosine(chunk_i, centroid)`,
//!   measuring how widely the chunks scatter in embedding space.
//!   *Higher means more ambiguous.* The semantic-side mirror of
//!   `evidence_concentration`.
//! - **`semantic_distractor_ratio`** — fraction of chunks with
//!   `cosine(query, chunk) < threshold`. *Lower is better.*

use redhop::core::{
    DiagnosticsEngine, DiagnosticsReport, Embedding, Query, Result, RetrievalResult,
};

/// Configurable thresholds for the semantic engine.
#[derive(Debug, Clone)]
pub struct SemanticDiagnosticsConfig {
    /// Per-chunk cosine cutoff used to classify a result as a semantic
    /// distractor.
    pub distractor_min_cosine: f32,
    /// Below this `semantic_grounding`, emit a `low_semantic_grounding`
    /// warning.
    pub min_semantic_grounding: f32,
    /// Above this `semantic_redundancy`, emit a `semantic_redundancy_high`
    /// warning — the policy layer will read this to trigger
    /// `EarlyStopReranker`.
    pub max_semantic_redundancy: f32,
    /// Above this `semantic_distractor_ratio`, emit a
    /// `high_semantic_distractor_ratio` warning.
    pub max_semantic_distractor_ratio: f32,
}

impl Default for SemanticDiagnosticsConfig {
    fn default() -> Self {
        // Defaults are calibrated for cosine on unit-normalized vectors
        // from reasonable text embedders. The distractor cutoff of 0.20
        // is intentionally generous; semantic relevance can be real even
        // at modest cosines and a tighter cutoff produced too many
        // false-positive "distractor" classifications on our internal
        // traces.
        Self {
            distractor_min_cosine: 0.20,
            min_semantic_grounding: 0.50,
            max_semantic_redundancy: 0.85,
            max_semantic_distractor_ratio: 0.50,
        }
    }
}

/// Diagnostics engine that reads embeddings off the query and chunks.
///
/// Implements the same [`DiagnosticsEngine`] trait as the lexical engine
/// so the two compose freely through
/// [`crate::layered::LayeredDiagnosticsEngine`].
#[derive(Debug, Clone, Default)]
pub struct SemanticDiagnosticsEngine {
    config: SemanticDiagnosticsConfig,
}

impl SemanticDiagnosticsEngine {
    /// Construct with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with caller-provided thresholds.
    pub fn with_config(config: SemanticDiagnosticsConfig) -> Self {
        Self { config }
    }
}

impl DiagnosticsEngine for SemanticDiagnosticsEngine {
    fn diagnose(&self, query: &Query, results: &[RetrievalResult]) -> Result<DiagnosticsReport> {
        let mut report = DiagnosticsReport::default();
        if results.is_empty() {
            return Ok(report);
        }
        let Some(q_emb) = query.embedding.as_ref() else {
            // No query embedding → semantic tier silently degrades. The
            // lexical tier will still produce a report; the layered engine
            // just won't pick up semantic fields.
            return Ok(report);
        };
        // Gather chunk embeddings, normalize once, and bail if any chunk
        // is missing — partial computation here would give misleading
        // dispersion/redundancy numbers.
        let mut chunk_vecs: Vec<Vec<f32>> = Vec::with_capacity(results.len());
        for r in results {
            let Some(e) = r.chunk.embedding.as_ref() else {
                return Ok(report);
            };
            if e.dim() != q_emb.dim() {
                // Dimension skew: refuse rather than report nonsense.
                return Ok(report.with_warning(
                    "semantic_dim_mismatch",
                    format!(
                        "chunk {} embedding dim {} does not match query dim {}",
                        r.chunk.id,
                        e.dim(),
                        q_emb.dim()
                    ),
                ));
            }
            chunk_vecs.push(normalize(e));
        }
        let qn = normalize(q_emb);

        // semantic_grounding: mean cosine(query, chunk_i)
        let grounding_cosines: Vec<f32> = chunk_vecs.iter().map(|c| dot(&qn, c)).collect();
        let mean_grounding = grounding_cosines.iter().sum::<f32>() / grounding_cosines.len() as f32;
        let semantic_grounding = unit_clamp((mean_grounding + 1.0) * 0.5);
        report.semantic_grounding = Some(semantic_grounding);

        // semantic_distractor_ratio: fraction below the raw cosine cutoff.
        // We deliberately apply the cutoff to the *raw* cosine, not the
        // shifted [0,1] score, so the config value reads naturally.
        let distractors = grounding_cosines
            .iter()
            .filter(|c| **c < self.config.distractor_min_cosine)
            .count();
        report.semantic_distractor_ratio =
            Some(distractors as f32 / grounding_cosines.len() as f32);

        // semantic_redundancy: mean pairwise cosine across chunks.
        // Skipped when there is only one chunk (the pair count would be 0
        // and the metric would be undefined).
        if chunk_vecs.len() >= 2 {
            let mut acc = 0.0f32;
            let mut n = 0usize;
            for i in 0..chunk_vecs.len() {
                for j in (i + 1)..chunk_vecs.len() {
                    acc += dot(&chunk_vecs[i], &chunk_vecs[j]);
                    n += 1;
                }
            }
            let mean_pair = acc / n as f32;
            report.semantic_redundancy = Some(unit_clamp((mean_pair + 1.0) * 0.5));
        }

        // centroid_dispersion: 1 - mean cosine(chunk_i, centroid).
        // Note we compute the centroid from the *normalized* vectors and
        // then renormalize it; this is the standard spherical centroid and
        // keeps cosine the right notion of distance.
        if chunk_vecs.len() >= 2 {
            let dim = chunk_vecs[0].len();
            let mut centroid = vec![0f32; dim];
            for v in &chunk_vecs {
                for k in 0..dim {
                    centroid[k] += v[k];
                }
            }
            let n = chunk_vecs.len() as f32;
            for k in 0..dim {
                centroid[k] /= n;
            }
            let centroid = normalize(&Embedding(centroid));
            let mean_to_centroid = chunk_vecs.iter().map(|v| dot(v, &centroid)).sum::<f32>() / n;
            // Dispersion goes up as cosine to centroid goes down.
            let dispersion = unit_clamp(1.0 - (mean_to_centroid + 1.0) * 0.5);
            report.centroid_dispersion = Some(dispersion);
        }

        // Warnings — codes are stable; messages are advisory.
        if semantic_grounding < self.config.min_semantic_grounding {
            report = report.with_warning(
                "low_semantic_grounding",
                format!(
                    "mean query/chunk cosine {:.3} (scaled) is below threshold {:.3}; even paraphrase-friendly retrieval is failing",
                    semantic_grounding, self.config.min_semantic_grounding
                ),
            );
        }
        if let Some(r) = report.semantic_redundancy {
            if r > self.config.max_semantic_redundancy {
                report = report.with_warning(
                    "semantic_redundancy_high",
                    format!(
                        "mean pairwise chunk cosine {:.3} (scaled) exceeds threshold {:.3}; the top-k is rehashing the same evidence",
                        r, self.config.max_semantic_redundancy
                    ),
                );
            }
        }
        if let Some(d) = report.semantic_distractor_ratio {
            if d > self.config.max_semantic_distractor_ratio {
                report = report.with_warning(
                    "high_semantic_distractor_ratio",
                    format!(
                        "{:.0}% of retrieved chunks fall below the query-cosine cutoff",
                        d * 100.0
                    ),
                );
            }
        }

        Ok(report)
    }

    fn name(&self) -> &'static str {
        "semantic"
    }
}

fn normalize(e: &Embedding) -> Vec<f32> {
    let mut v = e.as_slice().to_vec();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn unit_clamp(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop::core::{Chunk, Embedding, RetrievalMethod, Score, ScoreBreakdown, TokenCount};

    fn r(text: &str, emb: Vec<f32>) -> RetrievalResult {
        RetrievalResult {
            chunk: Chunk::new(text, text, "doc", TokenCount(1))
                .with_embedding(Embedding::from(emb)),
            score: Score {
                value: 1.0,
                method: RetrievalMethod::Dense,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }

    fn q(emb: Vec<f32>) -> Query {
        Query::new("ignored").with_embedding(Embedding::from(emb))
    }

    #[test]
    fn grounding_is_high_when_query_and_chunks_align() {
        let engine = SemanticDiagnosticsEngine::new();
        let query = q(vec![1.0, 0.0, 0.0]);
        let results = vec![r("a", vec![0.95, 0.05, 0.0]), r("b", vec![0.90, 0.10, 0.0])];
        let report = engine.diagnose(&query, &results).unwrap();
        let g = report.semantic_grounding.unwrap();
        assert!(g > 0.95, "expected near-1, got {g}");
        assert_eq!(report.semantic_distractor_ratio.unwrap(), 0.0);
    }

    #[test]
    fn distractors_classified_below_cosine_cutoff() {
        let engine = SemanticDiagnosticsEngine::with_config(SemanticDiagnosticsConfig {
            distractor_min_cosine: 0.5,
            ..Default::default()
        });
        let query = q(vec![1.0, 0.0, 0.0]);
        // Two on-topic, two off-topic.
        let results = vec![
            r("a", vec![1.0, 0.0, 0.0]),
            r("b", vec![0.9, 0.1, 0.0]),
            r("c", vec![0.0, 1.0, 0.0]),
            r("d", vec![0.0, 0.0, 1.0]),
        ];
        let report = engine.diagnose(&query, &results).unwrap();
        assert!((report.semantic_distractor_ratio.unwrap() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn redundancy_high_for_near_duplicates() {
        let engine = SemanticDiagnosticsEngine::new();
        let query = q(vec![1.0, 0.0, 0.0]);
        let results = vec![
            r("a", vec![1.0, 0.0, 0.0]),
            r("b", vec![0.99, 0.01, 0.0]),
            r("c", vec![0.98, 0.02, 0.0]),
        ];
        let report = engine.diagnose(&query, &results).unwrap();
        let red = report.semantic_redundancy.unwrap();
        assert!(red > 0.95, "expected near-1 redundancy, got {red}");
    }

    #[test]
    fn dispersion_low_for_clustered_orthogonal_for_spread() {
        let engine = SemanticDiagnosticsEngine::new();
        let query = q(vec![1.0, 0.0, 0.0]);

        // Clustered.
        let r_clustered = vec![r("a", vec![1.0, 0.0, 0.0]), r("b", vec![0.99, 0.01, 0.0])];
        let d_cluster = engine
            .diagnose(&query, &r_clustered)
            .unwrap()
            .centroid_dispersion
            .unwrap();

        // Spread across orthogonal axes.
        let r_spread = vec![
            r("a", vec![1.0, 0.0, 0.0]),
            r("b", vec![0.0, 1.0, 0.0]),
            r("c", vec![0.0, 0.0, 1.0]),
        ];
        let d_spread = engine
            .diagnose(&query, &r_spread)
            .unwrap()
            .centroid_dispersion
            .unwrap();

        assert!(
            d_spread > d_cluster,
            "spread dispersion {d_spread} should exceed clustered {d_cluster}"
        );
    }

    #[test]
    fn empty_results_yields_empty_report() {
        let engine = SemanticDiagnosticsEngine::new();
        let query = q(vec![1.0, 0.0, 0.0]);
        let report = engine.diagnose(&query, &[]).unwrap();
        assert!(report.semantic_grounding.is_none());
    }

    #[test]
    fn missing_query_embedding_degrades_silently() {
        let engine = SemanticDiagnosticsEngine::new();
        let query = Query::new("no embedding");
        let results = vec![r("a", vec![1.0, 0.0, 0.0])];
        let report = engine.diagnose(&query, &results).unwrap();
        assert!(report.semantic_grounding.is_none());
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn missing_chunk_embedding_degrades_silently() {
        let engine = SemanticDiagnosticsEngine::new();
        let query = q(vec![1.0, 0.0, 0.0]);
        let results = vec![RetrievalResult {
            chunk: Chunk::new("a", "a", "doc", TokenCount(1)),
            score: Score {
                value: 1.0,
                method: RetrievalMethod::Dense,
            },
            breakdown: ScoreBreakdown::default(),
        }];
        let report = engine.diagnose(&query, &results).unwrap();
        assert!(report.semantic_grounding.is_none());
    }

    #[test]
    fn dimension_mismatch_emits_warning_and_skips() {
        let engine = SemanticDiagnosticsEngine::new();
        let query = q(vec![1.0, 0.0, 0.0]);
        let results = vec![r("a", vec![1.0, 0.0])]; // dim 2 vs 3
        let report = engine.diagnose(&query, &results).unwrap();
        assert!(report.semantic_grounding.is_none());
        assert!(report
            .warnings
            .iter()
            .any(|w| w.code == "semantic_dim_mismatch"));
    }

    #[test]
    fn warning_fires_below_grounding_threshold() {
        let engine = SemanticDiagnosticsEngine::with_config(SemanticDiagnosticsConfig {
            min_semantic_grounding: 0.99,
            ..Default::default()
        });
        let query = q(vec![1.0, 0.0, 0.0]);
        // Cosine 0 → grounding scaled to 0.5, below the 0.99 threshold.
        let results = vec![r("a", vec![0.0, 1.0, 0.0])];
        let report = engine.diagnose(&query, &results).unwrap();
        assert!(report
            .warnings
            .iter()
            .any(|w| w.code == "low_semantic_grounding"));
    }
}
