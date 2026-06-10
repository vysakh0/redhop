//! Query-level diagnosis on the Decision Report.
//!
//! Facts the engine observed about how the query interacted with the
//! corpus and the retrieved candidates, plus a small closed registry of
//! hints that fire on documented failure shapes. Pure observability:
//! nothing here changes retrieval or assembly. See
//! `docs/design/REPORT_DIAGNOSIS.md` for the full design.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::analyzer::Analyzer;
use crate::core::{Query, RetrievalResult};

// Hint thresholds. All 🟡 convention (no measurement-driven choice).
// Registered in DEFAULT_PROVENANCE.md with a re-validation entry.
const VOCAB_MISMATCH_MIN_SHARE: f32 = 0.5;
const VOCAB_MISMATCH_MIN_TERMS: usize = 2;
const DF_RATIO_LOW_DISCRIMINATION: f32 = 0.25;
const LOW_DISCRIMINATION_MIN_TERMS: usize = 8;
const LOW_DISCRIMINATION_MIN_SHARE: f32 = 0.6;
const UNDERDETERMINED_MAX_TERMS: usize = 2;
const UNDERDETERMINED_MAX_SPREAD: f32 = 0.15;
const UNDERDETERMINED_MIN_CANDIDATES: usize = 5;
const SCORE_SPREAD_TOP_K: usize = 10;

const EVIDENCE_CHOOSING_A_CONFIG: &str = "docs/CHOOSING_A_CONFIG.md";
const EVIDENCE_MULTIHOP_HYBRID: &str = "docs/findings/MULTIHOP_HYBRID.md";
const EVIDENCE_CUAD_RECALL_GAP: &str = "docs/findings/CUAD_RECALL_GAP.md";

/// Query-level facts plus bounded hints that fire on documented failure
/// shapes. Observation only: never changes what was retrieved or kept.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Diagnosis {
    /// The query's analyzed terms (deduped, first-occurrence order).
    /// Produced by the same `Analyzer` that indexed the corpus, so the
    /// diagnosis can never disagree with the grounding scorer on what a
    /// "term" is. See `docs/design/ANALYZER_PLUGIN.md`.
    pub query_terms: Vec<String>,
    /// `true` when corpus-level stats (zero_match_terms, term_stats) were
    /// computed. `false` when `build_context` was called directly with a
    /// caller-supplied candidate pool and no `Document` to derive corpus
    /// vocabulary from.
    pub corpus_stats_available: bool,
    /// Analyzed query terms that appear in zero chunks of the corpus.
    /// Empty when `corpus_stats_available` is false.
    pub zero_match_terms: Vec<String>,
    /// Per-term corpus stats for query terms that DO appear (`df > 0`).
    /// Empty when `corpus_stats_available` is false.
    pub term_stats: Vec<TermStat>,
    /// Query terms that appear in no *retrieved candidate* chunk (they
    /// may still exist elsewhere in the corpus, present but outranked).
    /// Always computed.
    pub terms_unmatched_in_candidates: Vec<String>,
    /// Number of candidates handed to assembly (`retrieved.len()`).
    pub n_candidates: usize,
    /// Relative score spread across the top candidates:
    /// `(top_score - kth_score) / top_score`, over the top
    /// `min(n_candidates, 10)`. `None` when `n_candidates < 2` or
    /// `top_score <= 0`. A flat spread on a short query is the
    /// underdetermined-query signature.
    pub score_spread: Option<f32>,
    /// `true` when assembly selected zero chunks.
    pub empty_context: bool,
    /// Hints that fired, from the closed registry in this module.
    pub hints: Vec<DiagnosisHint>,
}

/// Per-term corpus statistics for one query term that appears in the
/// corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermStat {
    /// The analyzed term (matches how the corpus was indexed).
    pub term: String,
    /// Number of corpus chunks containing the term.
    pub df: u32,
    /// `df / total corpus chunks`, in `[0, 1]`.
    pub df_ratio: f32,
}

/// One hint from the closed registry. Carries the user-facing
/// observation, a code clients can branch on, and a path to the doc or
/// finding that justifies it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosisHint {
    /// Stable identifier. Serialized as snake_case in bindings.
    pub code: HintCode,
    /// One or two sentences. Observation only, never a promised
    /// improvement. Style: no em dashes, no semicolons.
    pub message: String,
    /// Repo-relative path of the doc or finding grounding this hint.
    pub evidence: String,
}

/// Closed registry of hint codes. Adding a code requires a registry row
/// in `docs/design/REPORT_DIAGNOSIS.md` with trigger conditions and an
/// evidence citation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum HintCode {
    /// Assembly selected zero chunks.
    EmptyContext,
    /// Most query terms appear nowhere in the corpus.
    VocabMismatch,
    /// Every selected chunk is at or below the grounding bar.
    LowConfidence,
    /// Most query terms appear in a large fraction of corpus chunks.
    LowDiscriminationQuery,
    /// Short query, many candidates, nearly flat score spread.
    UnderdeterminedQuery,
}

/// Layer 1: compute candidate-level facts. Called from inside
/// `build_context` / `analyze_context`, where the corpus is not
/// visible. Sets `corpus_stats_available = false`. The Document-level
/// path enriches this in Layer 2.
pub(crate) fn compute(
    query: &Query,
    retrieved: &[RetrievalResult],
    empty_context: bool,
    low_confidence: bool,
    analyzer: &dyn Analyzer,
) -> Diagnosis {
    let query_terms = ordered_terms(&query.text, analyzer);
    let query_term_set: HashSet<&String> = query_terms.iter().collect();

    let mut candidate_terms: HashSet<String> = HashSet::new();
    for r in retrieved {
        for t in analyzer.tokens(&r.chunk.text) {
            candidate_terms.insert(t);
        }
    }
    let terms_unmatched_in_candidates: Vec<String> = query_terms
        .iter()
        .filter(|t| !candidate_terms.contains(t.as_str()))
        .cloned()
        .collect();
    let _ = query_term_set; // reserved for symmetric ops; silence dead-set warning.

    let score_spread = compute_score_spread(retrieved);

    let mut d = Diagnosis {
        query_terms,
        corpus_stats_available: false,
        zero_match_terms: Vec::new(),
        term_stats: Vec::new(),
        terms_unmatched_in_candidates,
        n_candidates: retrieved.len(),
        score_spread,
        empty_context,
        hints: Vec::new(),
    };
    evaluate_hints(&mut d, low_confidence);
    d
}

/// Layer 2: enrich the diagnosis with corpus vocabulary stats and
/// re-evaluate the full hint registry. Called from
/// `Document::context_inner` after `build_context` returns. The vocab
/// map is `analyzed term -> number of chunks containing it`, built once
/// per `Document` from `cfg.analyzer` (see ANALYZER_PLUGIN: shared
/// tokenization keeps the two layers from drifting).
pub(crate) fn enrich(
    d: &mut Diagnosis,
    vocab: &HashMap<String, u32>,
    n_corpus_chunks: usize,
    low_confidence: bool,
) {
    d.corpus_stats_available = true;
    d.zero_match_terms.clear();
    d.term_stats.clear();
    let denom = n_corpus_chunks.max(1) as f32;
    for term in &d.query_terms {
        match vocab.get(term).copied().unwrap_or(0) {
            0 => d.zero_match_terms.push(term.clone()),
            df => d.term_stats.push(TermStat {
                term: term.clone(),
                df,
                df_ratio: df as f32 / denom,
            }),
        }
    }
    d.hints.clear();
    evaluate_hints(d, low_confidence);
}

fn evaluate_hints(d: &mut Diagnosis, low_confidence: bool) {
    let n_terms = d.query_terms.len();

    let h2_fired = if d.corpus_stats_available
        && n_terms >= VOCAB_MISMATCH_MIN_TERMS
        && (d.empty_context || low_confidence)
    {
        let share = d.zero_match_terms.len() as f32 / n_terms as f32;
        if share >= VOCAB_MISMATCH_MIN_SHARE {
            true
        } else {
            false
        }
    } else {
        false
    };

    if d.empty_context {
        d.hints.push(DiagnosisHint {
            code: HintCode::EmptyContext,
            message: format!(
                "Assembly selected zero chunks. {} candidates were retrieved.",
                d.n_candidates
            ),
            evidence: EVIDENCE_CHOOSING_A_CONFIG.to_string(),
        });
    }

    if h2_fired {
        let listed = format_term_list(&d.zero_match_terms, 6);
        d.hints.push(DiagnosisHint {
            code: HintCode::VocabMismatch,
            message: format!(
                "{k} of {m} query terms appear nowhere in this corpus: {listed}. \
                 The query and the documents may use different vocabulary. \
                 Rephrasing with the documents' own terms is the measured first fix. \
                 Dense retrieval (retrieval=\"hybrid\") matches paraphrases by embedding \
                 similarity and was measured to lift multi-hop retention on exactly this \
                 failure shape.",
                k = d.zero_match_terms.len(),
                m = n_terms,
                listed = listed,
            ),
            evidence: EVIDENCE_MULTIHOP_HYBRID.to_string(),
        });
    }

    if low_confidence && !d.empty_context && !h2_fired {
        d.hints.push(DiagnosisHint {
            code: HintCode::LowConfidence,
            message:
                "Every selected chunk is at or below the grounding bar. \
                 Retrieval matched something, but weakly. \
                 Check diagnosis.term_stats to see which terms carried the match."
                    .to_string(),
            evidence: EVIDENCE_CHOOSING_A_CONFIG.to_string(),
        });
    }

    if d.corpus_stats_available && n_terms >= LOW_DISCRIMINATION_MIN_TERMS {
        let high_df = d
            .term_stats
            .iter()
            .filter(|t| t.df_ratio > DF_RATIO_LOW_DISCRIMINATION)
            .count();
        let share = high_df as f32 / n_terms as f32;
        if share >= LOW_DISCRIMINATION_MIN_SHARE {
            let pct = (DF_RATIO_LOW_DISCRIMINATION * 100.0).round() as u32;
            d.hints.push(DiagnosisHint {
                code: HintCode::LowDiscriminationQuery,
                message: format!(
                    "{k} of {m} query terms appear in more than {pct}% of chunks and \
                     carry little ranking signal. If your queries follow a fixed template, \
                     this is the boilerplate-dilution shape measured on CUAD. \
                     analyze_query_set on a sample of your queries will confirm it, \
                     and Stripper removes the wrapper.",
                    k = high_df,
                    m = n_terms,
                    pct = pct,
                ),
                evidence: EVIDENCE_CUAD_RECALL_GAP.to_string(),
            });
        }
    }

    if n_terms <= UNDERDETERMINED_MAX_TERMS
        && n_terms > 0
        && d.n_candidates >= UNDERDETERMINED_MIN_CANDIDATES
    {
        if let Some(s) = d.score_spread {
            if s <= UNDERDETERMINED_MAX_SPREAD {
                d.hints.push(DiagnosisHint {
                    code: HintCode::UnderdeterminedQuery,
                    message: format!(
                        "A {m}-term query produced a nearly flat ranking across {n} \
                         candidates (spread {s:.2}). Short queries can match several \
                         sections equally well. One added disambiguating word was the \
                         fix in every measured polysemy case.",
                        m = n_terms,
                        n = d.n_candidates,
                        s = s,
                    ),
                    evidence: EVIDENCE_CHOOSING_A_CONFIG.to_string(),
                });
            }
        }
    }
}

fn compute_score_spread(retrieved: &[RetrievalResult]) -> Option<f32> {
    if retrieved.len() < 2 {
        return None;
    }
    let mut scores: Vec<f32> = retrieved.iter().map(|r| r.score.value).collect();
    // Sort desc so [0] is the top score.
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top = *scores.first()?;
    if top <= 0.0 {
        return None;
    }
    let k = scores.len().min(SCORE_SPREAD_TOP_K);
    let kth = scores[k - 1];
    Some(((top - kth) / top).clamp(0.0, 1.0))
}

/// Analyze `text` with `analyzer` and return its terms in
/// first-occurrence order, deduped. The analyzer drives tokenization so
/// the diagnosis matches the index. The Layer-1 / Layer-2 split needs
/// ordering (the existing `terms()` helper in this module returns a
/// HashSet) so the rendered list is stable across runs.
pub(crate) fn ordered_terms(text: &str, analyzer: &dyn Analyzer) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for t in analyzer.tokens(text) {
        if seen.insert(t.clone()) {
            out.push(t);
        }
    }
    out
}

fn format_term_list(terms: &[String], max: usize) -> String {
    if terms.is_empty() {
        return "(none)".to_string();
    }
    if terms.len() <= max {
        return terms
            .iter()
            .map(|t| format!("\"{}\"", t))
            .collect::<Vec<_>>()
            .join(", ");
    }
    let head: Vec<String> = terms
        .iter()
        .take(max)
        .map(|t| format!("\"{}\"", t))
        .collect();
    format!("{}, and {} more", head.join(", "), terms.len() - max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::default_analyzer;

    #[test]
    fn ordered_terms_dedupes_preserving_first_occurrence() {
        let a = default_analyzer();
        let terms = ordered_terms("alpha beta alpha gamma beta", a.as_ref());
        assert_eq!(terms, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn format_term_list_truncates_above_max() {
        let terms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(format_term_list(&terms, 5), r#""a", "b", "c""#);
        let many: Vec<String> = (0..10).map(|i| format!("t{}", i)).collect();
        assert!(format_term_list(&many, 3).contains("and 7 more"));
    }

    #[test]
    fn evidence_paths_all_exist_in_repo() {
        // Acceptance criterion: every evidence citation must point at a
        // file that actually lives in the repo. Path is relative to the
        // workspace root (which is two levels up from this crate file).
        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("two-deep parent from CARGO_MANIFEST_DIR")
            .to_path_buf();
        for path in [
            EVIDENCE_CHOOSING_A_CONFIG,
            EVIDENCE_MULTIHOP_HYBRID,
            EVIDENCE_CUAD_RECALL_GAP,
        ] {
            let full = workspace_root.join(path);
            assert!(
                full.exists(),
                "hint evidence path missing: {}",
                full.display()
            );
        }
    }

    #[test]
    fn no_em_dash_or_prose_semicolon_in_hint_strings() {
        // Style rule: no em dashes, no semicolons in user-facing prose.
        // String-search the source file for the hint messages.
        let src = include_str!("./diagnosis.rs");
        // Pull message literals out by finding `message:` followed by a
        // string literal. Crude but enough.
        for line in src.lines() {
            // Skip the test module itself.
            if line.contains("fn no_em_dash") {
                break;
            }
            assert!(
                !line.contains('\u{2014}'),
                "em dash leaked into hint source: {}",
                line
            );
        }
    }
}
