//! Score- and rank-level fusion of retrieval result lists.
//!
//! Fusion is the cheapest and most reliable way to combine signals from
//! heterogenous retrievers (e.g. BM25 + dense). RedHop provides two
//! strategies:
//!
//! - **Reciprocal Rank Fusion (RRF)** — rank-based, scale-free, robust to
//!   wildly different score distributions. The default.
//! - **Weighted sum** — linear combination of per-retriever scores. Useful
//!   when scores are already comparable (e.g. both cosines), or as a
//!   diagnostic baseline.
//!
//! Both functions take per-retriever result lists already sorted in
//! descending order of score and return a fused, deduplicated list also in
//! descending order.

use std::collections::HashMap;

use crate::core::{Chunk, ChunkId, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown};

/// Which fusion strategy to apply.
#[derive(Debug, Clone, Copy)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion with the standard `k=60` smoothing.
    ReciprocalRank,
    /// Weighted sum with per-list weights. Length must match the number of
    /// input lists at the call site.
    WeightedSum,
}

/// Reciprocal Rank Fusion (Cormack, Clarke & Buettcher 2009).
///
/// `k` is the rank smoothing constant; `60` is the canonical default and
/// almost never needs tuning. Each input list is expected to be in
/// descending score order.
///
/// Returns the fused list sorted by fused score, capped at `top_k`.
pub fn reciprocal_rank_fusion(
    lists: &[Vec<RetrievalResult>],
    k: f32,
    top_k: usize,
) -> Vec<RetrievalResult> {
    let mut by_id: HashMap<ChunkId, (Chunk, f32, ScoreBreakdown)> = HashMap::new();

    for list in lists {
        for (rank, result) in list.iter().enumerate() {
            let contribution = 1.0 / (k + (rank as f32) + 1.0);
            let entry = by_id
                .entry(result.chunk.id.clone())
                .or_insert_with(|| (result.chunk.clone(), 0.0, result.breakdown.clone()));
            entry.1 += contribution;
            merge_breakdown(&mut entry.2, &result.breakdown, result.score);
        }
    }

    let mut fused: Vec<RetrievalResult> = by_id
        .into_iter()
        .map(|(_, (chunk, score, mut bd))| {
            bd.fused = Some(score);
            RetrievalResult {
                chunk,
                score: Score {
                    value: score,
                    method: RetrievalMethod::Hybrid,
                },
                breakdown: bd,
            }
        })
        .collect();
    fused.sort_by(|a, b| {
        b.score
            .value
            .partial_cmp(&a.score.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused.truncate(top_k);
    fused
}

/// Weighted-sum fusion with min-max normalization per list.
///
/// Per-list min-max normalization brings each retriever's scores into
/// `[0, 1]` before the weighted sum. This makes the strategy reasonably
/// robust to BM25 (`~0..40`) vs cosine (`~-1..1`) mixing, but it remains
/// rank-fusion's weaker cousin in the presence of long-tailed distributions.
pub fn weighted_sum_fusion(
    lists: &[Vec<RetrievalResult>],
    weights: &[f32],
    top_k: usize,
) -> Vec<RetrievalResult> {
    assert_eq!(lists.len(), weights.len(), "weights/lists length mismatch");
    let mut by_id: HashMap<ChunkId, (Chunk, f32, ScoreBreakdown)> = HashMap::new();

    for (list, &w) in lists.iter().zip(weights.iter()) {
        if list.is_empty() {
            continue;
        }
        let (min, max) = list
            .iter()
            .map(|r| r.score.value)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            });
        // When min == max (single-item lists, or all-tied), treat every item
        // as a "1" — it's the top of the only ranking we have.
        let degenerate = (max - min).abs() <= f32::EPSILON;
        let range = (max - min).max(f32::EPSILON);
        for result in list {
            let normalized = if degenerate {
                1.0
            } else {
                (result.score.value - min) / range
            };
            let entry = by_id
                .entry(result.chunk.id.clone())
                .or_insert_with(|| (result.chunk.clone(), 0.0, result.breakdown.clone()));
            entry.1 += w * normalized;
            merge_breakdown(&mut entry.2, &result.breakdown, result.score);
        }
    }

    let mut fused: Vec<RetrievalResult> = by_id
        .into_iter()
        .map(|(_, (chunk, score, mut bd))| {
            bd.fused = Some(score);
            RetrievalResult {
                chunk,
                score: Score {
                    value: score,
                    method: RetrievalMethod::Hybrid,
                },
                breakdown: bd,
            }
        })
        .collect();
    fused.sort_by(|a, b| {
        b.score
            .value
            .partial_cmp(&a.score.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused.truncate(top_k);
    fused
}

fn merge_breakdown(dst: &mut ScoreBreakdown, src: &ScoreBreakdown, src_score: Score) {
    match src_score.method {
        RetrievalMethod::Lexical => dst.lexical = Some(src_score.value),
        RetrievalMethod::Dense => dst.dense = Some(src_score.value),
        RetrievalMethod::Rerank => dst.rerank = Some(src_score.value),
        _ => {}
    }
    if dst.lexical.is_none() {
        dst.lexical = src.lexical;
    }
    if dst.dense.is_none() {
        dst.dense = src.dense;
    }
    if dst.rerank.is_none() {
        dst.rerank = src.rerank;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Chunk, TokenCount};

    fn r(id: &str, method: RetrievalMethod, score: f32) -> RetrievalResult {
        RetrievalResult::new(
            Chunk::new(id, id, "doc", TokenCount(1)),
            Score {
                value: score,
                method,
            },
        )
    }

    #[test]
    fn rrf_prefers_consistently_ranked_items() {
        let l1 = vec![
            r("a", RetrievalMethod::Lexical, 3.0),
            r("b", RetrievalMethod::Lexical, 2.0),
            r("c", RetrievalMethod::Lexical, 1.0),
        ];
        let l2 = vec![
            r("b", RetrievalMethod::Dense, 0.9),
            r("a", RetrievalMethod::Dense, 0.7),
            r("d", RetrievalMethod::Dense, 0.5),
        ];
        let fused = reciprocal_rank_fusion(&[l1, l2], 60.0, 10);
        // `a` appears at ranks 0 and 1, `b` at ranks 1 and 0 — both should
        // outrank items that appear in only one list.
        assert!(fused[0].chunk.id.as_str() == "a" || fused[0].chunk.id.as_str() == "b");
        let tail: Vec<_> = fused.iter().map(|r| r.chunk.id.as_str()).collect();
        assert!(tail.contains(&"c"));
        assert!(tail.contains(&"d"));
    }

    #[test]
    fn weighted_sum_respects_weights() {
        let l1 = vec![r("a", RetrievalMethod::Lexical, 10.0)];
        let l2 = vec![r("b", RetrievalMethod::Dense, 10.0)];
        let f1 = weighted_sum_fusion(&[l1.clone(), l2.clone()], &[1.0, 0.0], 2);
        assert_eq!(f1[0].chunk.id.as_str(), "a");
        let f2 = weighted_sum_fusion(&[l1, l2], &[0.0, 1.0], 2);
        assert_eq!(f2[0].chunk.id.as_str(), "b");
    }
}
