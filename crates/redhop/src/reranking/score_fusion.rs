//! Reranker that recombines per-stage scores already present in
//! [`ScoreBreakdown`].
//!
//! This is the cheapest possible reranker: it does not look at chunk text at
//! all. It is useful as a *second* fusion pass after a hybrid retriever — for
//! example, to apply a different weight ratio than the retriever's RRF would
//! have produced, without re-running the candidate retrieval.

use crate::core::{Query, Reranker, Result, RetrievalMethod, RetrievalResult, Score};
use async_trait::async_trait;

/// Weighted recombination of per-stage scores.
#[derive(Debug, Clone)]
pub struct ScoreFusionReranker {
    /// Weight applied to `breakdown.lexical`.
    pub lexical_weight: f32,
    /// Weight applied to `breakdown.dense`.
    pub dense_weight: f32,
    /// Weight applied to `breakdown.rerank`, if present.
    pub rerank_weight: f32,
}

impl Default for ScoreFusionReranker {
    fn default() -> Self {
        Self {
            lexical_weight: 1.0,
            dense_weight: 1.0,
            rerank_weight: 1.0,
        }
    }
}

#[async_trait]
impl Reranker for ScoreFusionReranker {
    async fn rerank(
        &self,
        _query: &Query,
        mut candidates: Vec<RetrievalResult>,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        for r in candidates.iter_mut() {
            let mut score = 0.0f32;
            if let Some(v) = r.breakdown.lexical {
                score += self.lexical_weight * v;
            }
            if let Some(v) = r.breakdown.dense {
                score += self.dense_weight * v;
            }
            if let Some(v) = r.breakdown.rerank {
                score += self.rerank_weight * v;
            }
            r.score = Score {
                value: score,
                method: RetrievalMethod::Rerank,
            };
            r.breakdown.rerank = Some(score);
        }
        candidates.sort_by(|a, b| {
            b.score
                .value
                .partial_cmp(&a.score.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(top_k);
        Ok(candidates)
    }

    fn name(&self) -> &'static str {
        "score_fusion"
    }
}
