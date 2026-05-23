//! Individual diagnostic metrics.
//!
//! All metrics are stateless and operate on already-retrieved results. They
//! deliberately avoid any model dependence: every metric here can be computed
//! from text alone, which keeps the diagnostics path cheap, deterministic,
//! and binding-friendly.

use std::collections::HashSet;

use redhop_core::{Query, RetrievalResult};
use unicode_segmentation::UnicodeSegmentation;

/// Lowercased Unicode-word terms of a string, with single-character noise
/// dropped. Used by every metric below as a uniform "bag of terms" model.
pub fn terms(text: &str) -> HashSet<String> {
    text.unicode_words()
        .map(|w| w.to_lowercase())
        .filter(|w| w.chars().count() > 1)
        .collect()
}

/// Average lexical grounding: mean over results of
/// `|query_terms ∩ chunk_terms| / |query_terms|`.
///
/// Returns `None` if the query contains no usable terms.
pub fn lexical_grounding(query: &Query, results: &[RetrievalResult]) -> Option<f32> {
    let q = terms(&query.text);
    if q.is_empty() || results.is_empty() {
        return None;
    }
    let qn = q.len() as f32;
    let total: f32 = results
        .iter()
        .map(|r| {
            let c = terms(&r.chunk.text);
            q.intersection(&c).count() as f32 / qn
        })
        .sum();
    Some(total / results.len() as f32)
}

/// Per-chunk topical purity, averaged across results.
///
/// For each chunk we sentence-segment with a coarse regex (any of `. ! ?`
/// followed by whitespace) and compute the average Jaccard similarity
/// between adjacent sentence term-sets. High pairwise similarity → coherent
/// chunk; low → topical drift inside the chunk, which the chunker should
/// probably have split.
pub fn chunk_purity(results: &[RetrievalResult]) -> Option<f32> {
    if results.is_empty() {
        return None;
    }
    let mut total = 0.0f32;
    let mut counted = 0usize;
    for r in results {
        if let Some(p) = single_chunk_purity(&r.chunk.text) {
            total += p;
            counted += 1;
        }
    }
    if counted == 0 {
        None
    } else {
        Some(total / counted as f32)
    }
}

fn single_chunk_purity(text: &str) -> Option<f32> {
    let sentences: Vec<HashSet<String>> = text
        .split(['.', '!', '?'])
        .map(terms)
        .filter(|s| !s.is_empty())
        .collect();
    if sentences.len() < 2 {
        // Single-sentence chunks are trivially "pure".
        return Some(1.0);
    }
    let mut acc = 0.0f32;
    let mut n = 0usize;
    for i in 0..sentences.len() - 1 {
        let a = &sentences[i];
        let b = &sentences[i + 1];
        let inter = a.intersection(b).count() as f32;
        let union = a.union(b).count() as f32;
        if union > 0.0 {
            acc += inter / union;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some(acc / n as f32)
    }
}

/// Answer-bearing evidence density.
///
/// Computed as the fraction of *retrieved tokens* that match query terms.
/// This is a coarse proxy — a real answer-density metric would require an
/// answer span — but it tracks the same quantity a reader model uses to
/// localize evidence within a long context, which is what matters in
/// practice.
pub fn answer_density(query: &Query, results: &[RetrievalResult]) -> Option<f32> {
    let q = terms(&query.text);
    if q.is_empty() || results.is_empty() {
        return None;
    }
    let mut total_tokens = 0usize;
    let mut relevant_tokens = 0usize;
    for r in results {
        for w in r.chunk.text.unicode_words() {
            total_tokens += 1;
            if q.contains(&w.to_lowercase()) {
                relevant_tokens += 1;
            }
        }
    }
    if total_tokens == 0 {
        return None;
    }
    Some(relevant_tokens as f32 / total_tokens as f32)
}

/// Fraction of retrieved chunks whose per-chunk lexical grounding is below
/// `min_grounding`. *Lower is better*.
///
/// Returns `None` if the query has no usable terms.
pub fn distractor_ratio(
    query: &Query,
    results: &[RetrievalResult],
    min_grounding: f32,
) -> Option<f32> {
    let q = terms(&query.text);
    if q.is_empty() || results.is_empty() {
        return None;
    }
    let qn = q.len() as f32;
    let distractors = results
        .iter()
        .filter(|r| {
            let c = terms(&r.chunk.text);
            let g = q.intersection(&c).count() as f32 / qn;
            g < min_grounding
        })
        .count();
    Some(distractors as f32 / results.len() as f32)
}

/// Retrieval saturation: does the tail of results contribute new vocabulary?
///
/// We measure the term-set overlap between the *head* (top half of results)
/// and the *tail* (bottom half). High overlap → the bottom is just repeating
/// the top → saturated. Returned in `[0, 1]` where `1.0` is fully saturated
/// (no new information).
///
/// Returns `None` if fewer than two results are available.
pub fn retrieval_saturation(results: &[RetrievalResult]) -> Option<f32> {
    if results.len() < 2 {
        return None;
    }
    let split = results.len() / 2;
    let head: HashSet<String> = results[..split]
        .iter()
        .flat_map(|r| terms(&r.chunk.text))
        .collect();
    let tail: HashSet<String> = results[split..]
        .iter()
        .flat_map(|r| terms(&r.chunk.text))
        .collect();
    if tail.is_empty() {
        return None;
    }
    let inter = head.intersection(&tail).count() as f32;
    Some(inter / tail.len() as f32)
}

/// Evidence concentration: how peaked the top scores are.
///
/// We use the gap between the top score and the median of the rest,
/// normalized into `[0, 1]`. A single dominant result yields a value near
/// `1.0`; a flat plateau yields `0.0`.
///
/// Returns `None` if fewer than two results are available or scores are
/// degenerate.
pub fn evidence_concentration(results: &[RetrievalResult]) -> Option<f32> {
    if results.len() < 2 {
        return None;
    }
    let mut scores: Vec<f32> = results.iter().map(|r| r.score.value).collect();
    let top = scores[0];
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = *scores.first().unwrap();
    let max = *scores.last().unwrap();
    let range = max - min;
    if range.abs() <= f32::EPSILON {
        return Some(0.0);
    }
    let rest_median = scores[scores.len() / 2];
    let gap = (top - rest_median).max(0.0);
    Some((gap / range).clamp(0.0, 1.0))
}

/// Retrieval confidence — a weighted blend of grounding and concentration.
///
/// Diagnostics engines may use this as a single scalar summary; it is *not*
/// a substitute for inspecting the per-metric report.
pub fn retrieval_confidence(grounding: Option<f32>, concentration: Option<f32>) -> Option<f32> {
    match (grounding, concentration) {
        (Some(g), Some(c)) => Some((0.7 * g + 0.3 * c).clamp(0.0, 1.0)),
        (Some(g), None) => Some(g),
        (None, Some(c)) => Some(c),
        (None, None) => None,
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
    fn lexical_grounding_basic() {
        let q = Query::new("rust async runtime");
        let results = vec![
            r("rust async runtime tokio", 1.0),
            r("python sync libraries", 0.5),
        ];
        let g = lexical_grounding(&q, &results).unwrap();
        // First chunk hits all 3 terms (1.0); second hits 0 (0.0). Mean = 0.5.
        assert!((g - 0.5).abs() < 1e-5);
    }

    #[test]
    fn distractor_ratio_classifies_low_grounding() {
        let q = Query::new("rust async runtime");
        let results = vec![
            r("rust async runtime tokio", 1.0),
            r("totally unrelated content here", 0.5),
        ];
        let d = distractor_ratio(&q, &results, 0.5).unwrap();
        assert!((d - 0.5).abs() < 1e-5);
    }

    #[test]
    fn saturation_reports_repetition() {
        let results = vec![
            r("alpha bravo charlie delta", 1.0),
            r("alpha bravo charlie delta", 0.9),
            r("alpha bravo charlie delta", 0.8),
            r("alpha bravo charlie delta", 0.7),
        ];
        let s = retrieval_saturation(&results).unwrap();
        assert!(s > 0.9, "expected near-1, got {s}");
    }

    #[test]
    fn concentration_high_for_single_peak() {
        let results = vec![r("a", 100.0), r("b", 1.0), r("c", 1.0), r("d", 1.0)];
        let c = evidence_concentration(&results).unwrap();
        assert!(c > 0.9, "expected near-1, got {c}");
    }

    #[test]
    fn answer_density_reasonable() {
        let q = Query::new("rust");
        let results = vec![r("rust rust rust filler filler filler", 1.0)];
        let d = answer_density(&q, &results).unwrap();
        assert!((d - 0.5).abs() < 1e-5);
    }
}
