//! Evidence-density reranker.
//!
//! Where [`LexicalGroundingReranker`] rewards *coverage* of distinct query
//! terms, this reranker rewards *density*: how many of the chunk's tokens
//! are query-relevant. Same retrieved chunk but half the length →
//! higher density → ranked higher. Useful when context budget is the
//! bottleneck, which it usually is.
//!
//! [`LexicalGroundingReranker`]: crate::LexicalGroundingReranker

use std::collections::HashSet;

use crate::core::{Query, Reranker, Result, RetrievalMethod, RetrievalResult, Score};
use async_trait::async_trait;
use unicode_segmentation::UnicodeSegmentation;

/// Reranker that rewards per-token evidence density.
#[derive(Debug, Clone)]
pub struct EvidenceDensityReranker {
    /// Weight applied to the existing retrieval score.
    pub base_weight: f32,
    /// Weight applied to the density signal.
    pub density_weight: f32,
}

impl Default for EvidenceDensityReranker {
    fn default() -> Self {
        Self {
            base_weight: 1.0,
            density_weight: 1.0,
        }
    }
}

fn query_term_set(text: &str) -> HashSet<String> {
    text.unicode_words()
        .map(|w| w.to_lowercase())
        .filter(|w| w.chars().count() > 1)
        .collect()
}

#[async_trait]
impl Reranker for EvidenceDensityReranker {
    async fn rerank(
        &self,
        query: &Query,
        mut candidates: Vec<RetrievalResult>,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        let q = query_term_set(&query.text);
        if q.is_empty() {
            candidates.truncate(top_k);
            return Ok(candidates);
        }

        for r in candidates.iter_mut() {
            let mut total = 0usize;
            let mut hits = 0usize;
            for w in r.chunk.text.unicode_words() {
                total += 1;
                if q.contains(&w.to_lowercase()) {
                    hits += 1;
                }
            }
            let density = if total == 0 {
                0.0
            } else {
                hits as f32 / total as f32
            };
            // Raw base score plus density bonus; see the equivalent comment
            // in `LexicalGroundingReranker` for the rationale on not
            // min-max normalizing here.
            let new = self.base_weight * r.score.value + self.density_weight * density;
            r.breakdown.rerank = Some(new);
            r.score = Score {
                value: new,
                method: RetrievalMethod::Rerank,
            };
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
        "evidence_density"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Chunk, ScoreBreakdown, TokenCount};

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
                method: RetrievalMethod::Dense,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn density_prefers_shorter_focused_chunks() {
        rt().block_on(async {
            let q = Query::new("rust async");
            // Same query terms present in both, but second one is much
            // longer with low per-token density.
            let cand = vec![
                r("rust async", 0.8),
                r(
                    "rust async ... lots and lots and lots and lots of filler text here",
                    0.8,
                ),
            ];
            let rr = EvidenceDensityReranker::default();
            let out = rr.rerank(&q, cand, 2).await.unwrap();
            assert_eq!(out[0].chunk.text, "rust async");
        });
    }
}
