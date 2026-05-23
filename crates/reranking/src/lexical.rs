//! Lexical-grounding reranker.
//!
//! Boosts candidates whose chunks share more *distinct* query terms. This is
//! a simple but surprisingly effective signal in QA workloads where the
//! reader model needs anchor terms to localize evidence inside a long
//! context. It runs in microseconds per candidate and meaningfully reduces
//! the rate at which paraphrase-only dense matches dominate the top of the
//! list at the expense of more directly grounded chunks.

use std::collections::HashSet;

use async_trait::async_trait;
use redhop_core::{Query, Reranker, Result, RetrievalMethod, RetrievalResult, Score};
use unicode_segmentation::UnicodeSegmentation;

/// Reranker that boosts candidates with high query-term overlap.
#[derive(Debug, Clone)]
pub struct LexicalGroundingReranker {
    /// Weight on the existing retrieval score (lexical/dense/fused).
    pub base_weight: f32,
    /// Weight on the lexical-grounding signal.
    pub grounding_weight: f32,
}

impl Default for LexicalGroundingReranker {
    fn default() -> Self {
        Self {
            base_weight: 1.0,
            grounding_weight: 1.0,
        }
    }
}

fn terms(text: &str) -> HashSet<String> {
    text.unicode_words()
        .map(|w| w.to_lowercase())
        .filter(|w| w.chars().count() > 1)
        .collect()
}

#[async_trait]
impl Reranker for LexicalGroundingReranker {
    async fn rerank(
        &self,
        query: &Query,
        mut candidates: Vec<RetrievalResult>,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        let q = terms(&query.text);
        if q.is_empty() {
            // No query terms to ground against; pass through.
            candidates.truncate(top_k);
            return Ok(candidates);
        }
        let qn = q.len() as f32;

        // Use raw retrieval scores so a tight cluster of candidates (e.g.
        // dense scores all in 0.79..0.82) doesn't have its noise amplified
        // by min-max normalization. The grounding signal ∈ [0, 1] is large
        // enough on its own to break ties and reorder the head.
        for r in candidates.iter_mut() {
            let c = terms(&r.chunk.text);
            let grounding = q.intersection(&c).count() as f32 / qn;
            let new = self.base_weight * r.score.value + self.grounding_weight * grounding;
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
        "lexical_grounding"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop_core::{Chunk, ScoreBreakdown, TokenCount};

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
            breakdown: ScoreBreakdown {
                dense: Some(score),
                ..Default::default()
            },
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn grounding_promotes_overlapping_chunk() {
        rt().block_on(async {
            let q = Query::new("rust async runtime");
            // Dense scores both nearly identical, but only the second chunk
            // actually shares query terms.
            let cand = vec![
                r("memory safety guarantees with ownership", 0.81),
                r("rust async runtime executor model", 0.80),
            ];
            let rr = LexicalGroundingReranker::default();
            let out = rr.rerank(&q, cand, 2).await.unwrap();
            assert!(out[0].chunk.text.contains("async runtime"));
        });
    }
}
