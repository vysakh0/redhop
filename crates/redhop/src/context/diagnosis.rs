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
///
/// `candidate_terms` is the union of the candidates' analyzed term
/// sets. The caller passes the `c_terms` that `characterize()` already
/// computed for grounding, so the candidates are tokenized exactly once
/// per call (and the two layers cannot disagree on tokenization).
pub(crate) fn compute(
    query: &Query,
    retrieved: &[RetrievalResult],
    candidate_terms: &HashSet<String>,
    empty_context: bool,
    low_confidence: bool,
    analyzer: &dyn Analyzer,
) -> Diagnosis {
    let query_terms = ordered_terms(&query.text, analyzer);

    let terms_unmatched_in_candidates: Vec<String> = query_terms
        .iter()
        .filter(|t| !candidate_terms.contains(t.as_str()))
        .cloned()
        .collect();

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
        let listed = format_term_list(&display_order(&d.zero_match_terms), 6);
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

/// Order terms for *display*: content words first, stopwords after,
/// stable within each group. The structured fields keep first-occurrence
/// order untouched (facts are facts); this only decides what a human
/// reads first. Without it, the raw analyzer's kept stopwords ("how",
/// "do", "i") bury the informative terms ("cancel", "money") in the
/// hint message.
pub(crate) fn display_order(terms: &[String]) -> Vec<String> {
    let is_stop = |t: &String| super::STOPWORDS.contains(&t.as_str());
    let mut out: Vec<String> = terms.iter().filter(|t| !is_stop(t)).cloned().collect();
    out.extend(terms.iter().filter(|t| is_stop(t)).cloned());
    out
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

// ── Workload audit ─────────────────────────────────────────────────────────
//
// Aggregates per-query `Diagnosis` data across N `ContextReport`s into a
// single workload summary with at most ONE focus recommendation. See
// `docs/design/WORKLOAD_AUDIT.md` for the full design.

// Focus-resolution thresholds. All 🟡 convention (no measurement-driven
// choice), registered in DEFAULT_PROVENANCE.md.
const SUMMARY_MIN_QUERIES: usize = 20;
const DOMINANT_HINT_SHARE: f32 = 0.20;
const WEAK_RETRIEVAL_MIN_RATE: f32 = 0.30;
const TOP_TERMS_CAP: usize = 20;

/// Workload-level aggregation of per-query diagnoses. Observation only:
/// reports the shape of the workload's failures and at most one
/// findings-cited focus recommendation. See `summarize_diagnoses`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagnosisSummary {
    /// Number of reports aggregated.
    pub n: usize,
    /// Count + share per hint code. All five codes always present
    /// (count 0 included) so consumers can chart without key checks.
    pub hint_counts: Vec<HintCount>,
    /// Fraction of reports where assembly selected zero chunks.
    pub empty_context_rate: f32,
    /// Fraction of reports with `low_confidence_retrieval == true`.
    pub low_confidence_rate: f32,
    /// Fraction of reports that carried corpus stats (Layer 2). Below
    /// 1.0 means part of the workload ran through direct
    /// `build_context` / `analyze_context` and got candidate-level
    /// facts only.
    pub corpus_stats_coverage: f32,
    /// Terms that zero-matched the corpus, ranked by how many queries
    /// listed them. Capped at `TOP_TERMS_CAP`. Directly actionable as a
    /// `Vocabulary` dict or doc-glossary fix.
    pub top_zero_match_terms: Vec<TermCount>,
    /// Mean `score_spread` over reports where it was `Some(_)`.
    /// `None` when no report carried one (mirrors `EvalSummary`'s
    /// "None if zero present" convention).
    pub mean_score_spread: Option<f32>,
    /// Number of reports that carried a `score_spread`.
    pub n_with_score_spread: usize,
    /// The single focus recommendation (or `Healthy` / `SampleTooSmall`).
    pub focus: WorkloadFocus,
}

/// One entry in the workload's hint histogram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HintCount {
    /// The hint code.
    pub code: HintCode,
    /// Number of reports that fired this hint at least once.
    pub count: usize,
    /// `count / n`, in `[0, 1]`. `0.0` when `n == 0`.
    pub share: f32,
}

/// One entry in the top-zero-match-terms ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermCount {
    /// The analyzed term (matches how the corpus was indexed).
    pub term: String,
    /// Number of queries whose diagnosis listed this term as
    /// zero-match.
    pub count: usize,
}

/// The single workload focus recommendation: what the data points at
/// (if anything) and the finding that justifies the recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadFocus {
    /// Stable identifier. Serialized as snake_case in bindings.
    pub code: FocusCode,
    /// One or two sentences. Observation only, never a promised
    /// improvement. Style: no em dashes, no semicolons.
    pub message: String,
    /// Repo-relative path of the doc or finding grounding this
    /// recommendation. Empty for `Healthy` / `SampleTooSmall` (nothing
    /// to cite).
    pub evidence: String,
}

impl Default for WorkloadFocus {
    fn default() -> Self {
        Self {
            code: FocusCode::SampleTooSmall,
            message: String::new(),
            evidence: String::new(),
        }
    }
}

/// Closed registry of workload-focus codes. Adding a code requires a
/// row in the focus-resolution table in
/// `docs/design/WORKLOAD_AUDIT.md` with priority order and evidence.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FocusCode {
    /// Fewer than `SUMMARY_MIN_QUERIES` reports; no recommendation.
    SampleTooSmall,
    /// No failure shape reached `DOMINANT_HINT_SHARE`.
    Healthy,
    /// `vocab_mismatch` dominates.
    VocabMismatch,
    /// `low_discrimination_query` dominates.
    TemplatedQueries,
    /// `underdetermined_query` dominates.
    UnderdeterminedQueries,
    /// Empty / low-confidence rates are high but no specific hint
    /// dominates. The corpus may simply not cover the questions.
    WeakRetrieval,
}

/// Aggregate a list of [`crate::context::ContextReport`]s into a
/// workload-level summary. Single pass. Empty input returns a zeroed
/// summary with [`FocusCode::SampleTooSmall`]; below
/// [`SUMMARY_MIN_QUERIES`] the same code applies and no recommendation
/// is made. Mirrors [`crate::context::eval::summarize`]'s shape.
pub fn summarize_diagnoses(reports: &[crate::context::ContextReport]) -> DiagnosisSummary {
    let n = reports.len();
    let mut summary = DiagnosisSummary {
        n,
        hint_counts: zeroed_hint_counts(),
        ..Default::default()
    };
    if n == 0 {
        summary.focus = WorkloadFocus {
            code: FocusCode::SampleTooSmall,
            message: format!(
                "Only 0 queries aggregated. {min} or more are needed before the failure-shape shares are meaningful.",
                min = SUMMARY_MIN_QUERIES,
            ),
            evidence: String::new(),
        };
        return summary;
    }

    // Single pass: tally hints, rates, score_spread, vocab gaps.
    let mut empty_count = 0usize;
    let mut low_conf_count = 0usize;
    let mut corpus_stats_count = 0usize;
    let mut score_spread_sum = 0.0f32;
    let mut score_spread_n = 0usize;
    // term -> reports listing it as zero-match. Counts the term once
    // per query; the per-query list is already deduped.
    let mut term_freq: HashMap<String, usize> = HashMap::new();
    // Per-code hit counts; index aligned with HINT_CODE_ORDER.
    let mut hint_hits = [0usize; HINT_CODE_ORDER.len()];

    for report in reports {
        if report.low_confidence_retrieval {
            low_conf_count += 1;
        }
        let d = &report.diagnosis;
        if d.empty_context {
            empty_count += 1;
        }
        if d.corpus_stats_available {
            corpus_stats_count += 1;
        }
        if let Some(s) = d.score_spread {
            score_spread_sum += s;
            score_spread_n += 1;
        }
        // A hint code counts once per report even if it fires twice
        // (no current registry does, but be defensive).
        let mut seen: [bool; HINT_CODE_ORDER.len()] = [false; HINT_CODE_ORDER.len()];
        for h in &d.hints {
            if let Some(i) = hint_code_index(h.code) {
                if !seen[i] {
                    seen[i] = true;
                    hint_hits[i] += 1;
                }
            }
        }
        for t in &d.zero_match_terms {
            *term_freq.entry(t.clone()).or_insert(0) += 1;
        }
    }

    let nf = n as f32;
    summary.empty_context_rate = empty_count as f32 / nf;
    summary.low_confidence_rate = low_conf_count as f32 / nf;
    summary.corpus_stats_coverage = corpus_stats_count as f32 / nf;
    summary.n_with_score_spread = score_spread_n;
    summary.mean_score_spread = if score_spread_n == 0 {
        None
    } else {
        Some(score_spread_sum / score_spread_n as f32)
    };
    for (i, code) in HINT_CODE_ORDER.iter().enumerate() {
        summary.hint_counts[i] = HintCount {
            code: *code,
            count: hint_hits[i],
            share: hint_hits[i] as f32 / nf,
        };
    }

    // Rank top zero-match terms: count desc, then term asc.
    let mut terms: Vec<(String, usize)> = term_freq.into_iter().collect();
    terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    terms.truncate(TOP_TERMS_CAP);
    summary.top_zero_match_terms = terms
        .into_iter()
        .map(|(term, count)| TermCount { term, count })
        .collect();

    summary.focus = resolve_focus(&summary);
    summary
}

const HINT_CODE_ORDER: [HintCode; 5] = [
    HintCode::EmptyContext,
    HintCode::VocabMismatch,
    HintCode::LowConfidence,
    HintCode::LowDiscriminationQuery,
    HintCode::UnderdeterminedQuery,
];

fn hint_code_index(c: HintCode) -> Option<usize> {
    HINT_CODE_ORDER.iter().position(|x| *x == c)
}

fn zeroed_hint_counts() -> Vec<HintCount> {
    HINT_CODE_ORDER
        .iter()
        .map(|c| HintCount {
            code: *c,
            count: 0,
            share: 0.0,
        })
        .collect()
}

fn share_of(summary: &DiagnosisSummary, code: HintCode) -> f32 {
    summary
        .hint_counts
        .iter()
        .find(|h| h.code == code)
        .map(|h| h.share)
        .unwrap_or(0.0)
}

fn resolve_focus(summary: &DiagnosisSummary) -> WorkloadFocus {
    let n = summary.n;
    if n < SUMMARY_MIN_QUERIES {
        return WorkloadFocus {
            code: FocusCode::SampleTooSmall,
            message: format!(
                "Only {n} queries aggregated. {min} or more are needed before the failure-shape shares are meaningful.",
                n = n,
                min = SUMMARY_MIN_QUERIES,
            ),
            evidence: String::new(),
        };
    }

    let top_terms_listed = format_top_terms_for_message(&summary.top_zero_match_terms, 6);

    let vocab_share = share_of(summary, HintCode::VocabMismatch);
    if vocab_share >= DOMINANT_HINT_SHARE {
        return WorkloadFocus {
            code: FocusCode::VocabMismatch,
            message: format!(
                "{pct}% of queries had most terms missing from the corpus. \
                 Top gap terms: {terms}. \
                 Rephrasing toward the documents' vocabulary is the measured first fix, \
                 and dense retrieval (retrieval=\"hybrid\") was measured to lift retention \
                 on exactly this shape.",
                pct = pct(vocab_share),
                terms = top_terms_listed,
            ),
            evidence: EVIDENCE_MULTIHOP_HYBRID.to_string(),
        };
    }

    let templated_share = share_of(summary, HintCode::LowDiscriminationQuery);
    if templated_share >= DOMINANT_HINT_SHARE {
        return WorkloadFocus {
            code: FocusCode::TemplatedQueries,
            message: format!(
                "{pct}% of queries are boilerplate-shaped. \
                 Run analyze_query_set on a sample to extract the template, then compile a Stripper. \
                 Template stripping was measured to lift retention on exactly this shape \
                 (CUAD three-arm run).",
                pct = pct(templated_share),
            ),
            evidence: EVIDENCE_CUAD_CLAUSE_EXPANSION.to_string(),
        };
    }

    let underdet_share = share_of(summary, HintCode::UnderdeterminedQuery);
    if underdet_share >= DOMINANT_HINT_SHARE {
        return WorkloadFocus {
            code: FocusCode::UnderdeterminedQueries,
            message: format!(
                "{pct}% of queries were too short to discriminate between candidates. \
                 One added disambiguating word was the fix in every measured polysemy case. \
                 If queries come from a UI, consider prompting for one more keyword.",
                pct = pct(underdet_share),
            ),
            evidence: EVIDENCE_CHOOSING_A_CONFIG.to_string(),
        };
    }

    let weak_rate = summary.empty_context_rate.max(summary.low_confidence_rate);
    if weak_rate >= WEAK_RETRIEVAL_MIN_RATE {
        return WorkloadFocus {
            code: FocusCode::WeakRetrieval,
            message: format!(
                "{pct}% of queries retrieved nothing usable but no single failure shape dominates. \
                 The corpus may not cover these questions. \
                 Inspect top_zero_match_terms for what users ask about that the documents never mention.",
                pct = pct(weak_rate),
            ),
            evidence: EVIDENCE_CHOOSING_A_CONFIG.to_string(),
        };
    }

    WorkloadFocus {
        code: FocusCode::Healthy,
        message: format!(
            "No failure shape exceeded {pct}% of queries. No intervention indicated.",
            pct = pct(DOMINANT_HINT_SHARE),
        ),
        evidence: String::new(),
    }
}

fn pct(share: f32) -> u32 {
    (share * 100.0).round() as u32
}

fn format_top_terms_for_message(terms: &[TermCount], max: usize) -> String {
    if terms.is_empty() {
        return "(none recorded; corpus stats may be off)".to_string();
    }
    let words: Vec<String> = terms.iter().map(|t| t.term.clone()).collect();
    // Stopwords last (display rule, same as per-query hints).
    format_term_list(&display_order(&words), max)
}

const EVIDENCE_CUAD_CLAUSE_EXPANSION: &str = "docs/findings/CUAD_CLAUSE_EXPANSION.md";

impl DiagnosisSummary {
    /// Render the summary as a human-readable string. Follows the
    /// rendered-report conventions in
    /// [`crate::context::ContextReport::render`]. Unstable string,
    /// like the report's; parse structured fields for programmatic
    /// use.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str("RedHop Workload Audit\n");
        s.push_str("═════════════════════\n");
        s.push_str(&format!("\n  Reports aggregated: {}\n", self.n));

        s.push_str("\nHint histogram\n──────────────\n");
        for hc in &self.hint_counts {
            if hc.count == 0 {
                continue;
            }
            s.push_str(&format!(
                "  - {:<25} {:>4}  ({:>3}%)\n",
                hint_code_label(hc.code),
                hc.count,
                pct(hc.share),
            ));
        }
        // If every count was 0, still emit a marker so the section
        // isn't silently empty.
        if self.hint_counts.iter().all(|h| h.count == 0) {
            s.push_str("  - no hints fired across the workload\n");
        }

        s.push_str("\nRates\n─────\n");
        s.push_str(&format!(
            "  Empty-context rate:    {}%\n",
            pct(self.empty_context_rate)
        ));
        s.push_str(&format!(
            "  Low-confidence rate:   {}%\n",
            pct(self.low_confidence_rate)
        ));
        s.push_str(&format!(
            "  Corpus-stats coverage: {}%\n",
            pct(self.corpus_stats_coverage)
        ));
        if let Some(spread) = self.mean_score_spread {
            s.push_str(&format!(
                "  Mean score spread:     {:.2}  (over {} reports)\n",
                spread, self.n_with_score_spread,
            ));
        }

        if !self.top_zero_match_terms.is_empty() {
            s.push_str("\nTop zero-match terms\n────────────────────\n");
            let words: Vec<String> = self
                .top_zero_match_terms
                .iter()
                .map(|t| t.term.clone())
                .collect();
            let ordered = display_order(&words);
            let listed: Vec<String> = ordered
                .iter()
                .take(10)
                .map(|t| format!("\"{}\"", t))
                .collect();
            s.push_str(&format!("  {}\n", listed.join(", ")));
        }

        s.push_str("\nFocus\n─────\n");
        s.push_str(&format!("  Code: {}\n", focus_code_label(self.focus.code)));
        s.push_str(&format!("  {}\n", self.focus.message));
        if !self.focus.evidence.is_empty() {
            s.push_str(&format!("      evidence: {}\n", self.focus.evidence));
        }
        s
    }
}

fn hint_code_label(c: HintCode) -> &'static str {
    match c {
        HintCode::EmptyContext => "empty_context",
        HintCode::VocabMismatch => "vocab_mismatch",
        HintCode::LowConfidence => "low_confidence",
        HintCode::LowDiscriminationQuery => "low_discrimination_query",
        HintCode::UnderdeterminedQuery => "underdetermined_query",
        _ => "unknown",
    }
}

fn focus_code_label(c: FocusCode) -> &'static str {
    match c {
        FocusCode::SampleTooSmall => "sample_too_small",
        FocusCode::Healthy => "healthy",
        FocusCode::VocabMismatch => "vocab_mismatch",
        FocusCode::TemplatedQueries => "templated_queries",
        FocusCode::UnderdeterminedQueries => "underdetermined_queries",
        FocusCode::WeakRetrieval => "weak_retrieval",
        _ => "unknown",
    }
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
    fn display_order_puts_content_words_before_stopwords() {
        let terms: Vec<String> = ["how", "cancel", "do", "money"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let ordered = display_order(&terms);
        assert_eq!(ordered, vec!["cancel", "money", "how", "do"]);
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
            EVIDENCE_CUAD_CLAUSE_EXPANSION,
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

    // ── Workload-audit tests (docs/design/WORKLOAD_AUDIT.md §7) ─────────

    fn make_report(d: Diagnosis, low_confidence: bool) -> crate::context::ContextReport {
        // ContextReport doesn't derive Default (strategy enums lack one);
        // build a minimal report with just the fields summarize_diagnoses
        // actually reads. All others default-zero.
        crate::context::ContextReport {
            strategy: crate::context::ContextStrategy::RawTopK,
            requested_strategy: crate::context::ContextStrategy::Auto,
            token_budget: 1000,
            input_tokens: 0,
            auto_gate_tokens: 1500,
            total_tokens: 0,
            token_utilization: 0.0,
            n_input_chunks: 0,
            n_selected: if d.empty_context { 0 } else { 1 },
            input_distractor_ratio: 0.0,
            retained_evidence_ratio: 1.0,
            second_hop_rescue_count: 0,
            reasoning_preservation_delta: 0,
            removed: Default::default(),
            n_expanded: 0,
            low_confidence_retrieval: low_confidence,
            low_confidence_threshold: 0.10,
            economics: Default::default(),
            query_rewrites: vec![],
            diagnosis: d,
        }
    }

    fn fire(code: HintCode) -> DiagnosisHint {
        DiagnosisHint {
            code,
            message: "test hint".to_string(),
            evidence: "docs/CHOOSING_A_CONFIG.md".to_string(),
        }
    }

    fn diagnosis_with(hints: Vec<HintCode>, zero_match: Vec<&str>) -> Diagnosis {
        Diagnosis {
            query_terms: vec![],
            corpus_stats_available: true,
            zero_match_terms: zero_match.into_iter().map(String::from).collect(),
            term_stats: vec![],
            terms_unmatched_in_candidates: vec![],
            n_candidates: 5,
            score_spread: Some(0.4),
            empty_context: false,
            hints: hints.into_iter().map(fire).collect(),
        }
    }

    #[test]
    fn summarize_empty_input_is_sample_too_small() {
        let summary = summarize_diagnoses(&[]);
        assert_eq!(summary.n, 0);
        assert_eq!(summary.focus.code, FocusCode::SampleTooSmall);
        assert!(summary.focus.evidence.is_empty());
        assert!(summary.mean_score_spread.is_none());
        // All 5 codes present with count 0 so consumers can chart.
        assert_eq!(summary.hint_counts.len(), 5);
        for h in &summary.hint_counts {
            assert_eq!(h.count, 0);
            assert_eq!(h.share, 0.0);
        }
    }

    #[test]
    fn summarize_below_min_queries_makes_no_recommendation() {
        let reports: Vec<_> = (0..5)
            .map(|_| {
                make_report(
                    diagnosis_with(vec![HintCode::VocabMismatch], vec!["cancel"]),
                    true,
                )
            })
            .collect();
        let summary = summarize_diagnoses(&reports);
        assert_eq!(summary.n, 5);
        assert_eq!(summary.focus.code, FocusCode::SampleTooSmall);
        assert!(
            summary.focus.evidence.is_empty(),
            "below min: no evidence cited"
        );
    }

    #[test]
    fn summarize_vocab_dominant_workload() {
        // n=25; 10 carry VocabMismatch (40% share, well above 20%).
        let mut reports: Vec<_> = (0..10)
            .map(|_| {
                make_report(
                    diagnosis_with(
                        vec![HintCode::VocabMismatch],
                        vec!["cancel", "money", "refund"],
                    ),
                    true,
                )
            })
            .collect();
        reports.extend(
            (0..15)
                .map(|_| make_report(diagnosis_with(vec![], vec!["zanzibar"]), false)),
        );

        let summary = summarize_diagnoses(&reports);
        assert_eq!(summary.n, 25);
        assert_eq!(summary.focus.code, FocusCode::VocabMismatch);
        assert!(summary.focus.evidence.ends_with("MULTIHOP_HYBRID.md"));
        // Ranking: "cancel"/"money"/"refund" each in 10 queries,
        // "zanzibar" in 15 -> zanzibar first by count, then ties by
        // term asc.
        assert_eq!(summary.top_zero_match_terms[0].term, "zanzibar");
        assert_eq!(summary.top_zero_match_terms[0].count, 15);
        let tied: Vec<&str> = summary
            .top_zero_match_terms
            .iter()
            .skip(1)
            .take(3)
            .map(|t| t.term.as_str())
            .collect();
        assert_eq!(tied, vec!["cancel", "money", "refund"]);
    }

    #[test]
    fn summarize_vocab_outranks_templated() {
        // Both shapes fire at 40%. VocabMismatch wins by priority.
        let mut reports = Vec::new();
        for _ in 0..10 {
            reports.push(make_report(
                diagnosis_with(vec![HintCode::VocabMismatch], vec!["cancel"]),
                false,
            ));
        }
        for _ in 0..10 {
            reports.push(make_report(
                diagnosis_with(vec![HintCode::LowDiscriminationQuery], vec![]),
                false,
            ));
        }
        for _ in 0..5 {
            reports.push(make_report(diagnosis_with(vec![], vec![]), false));
        }
        let summary = summarize_diagnoses(&reports);
        assert_eq!(summary.focus.code, FocusCode::VocabMismatch);
    }

    #[test]
    fn summarize_weak_retrieval_without_dominant_hint() {
        // 12 of 25 reports (48%) carry low_confidence but no hint code
        // crosses 20%. Should resolve to WeakRetrieval.
        let mut reports = Vec::new();
        // 4 vocab, 4 templated, 4 underdetermined (each 16% share).
        for code in [
            HintCode::VocabMismatch,
            HintCode::LowDiscriminationQuery,
            HintCode::UnderdeterminedQuery,
        ] {
            for _ in 0..4 {
                reports.push(make_report(diagnosis_with(vec![code], vec![]), true));
            }
        }
        // 13 unhinted low-confidence reports to push the rate over 30%.
        for _ in 0..13 {
            reports.push(make_report(diagnosis_with(vec![], vec![]), false));
        }
        assert_eq!(reports.len(), 25);
        let summary = summarize_diagnoses(&reports);
        assert_eq!(
            summary.focus.code,
            FocusCode::WeakRetrieval,
            "expected WeakRetrieval; hint shares: {:?}",
            summary
                .hint_counts
                .iter()
                .map(|h| (h.code, h.share))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn summarize_healthy_workload_recommends_nothing() {
        // 30 hint-free reports.
        let reports: Vec<_> = (0..30)
            .map(|_| make_report(diagnosis_with(vec![], vec![]), false))
            .collect();
        let summary = summarize_diagnoses(&reports);
        assert_eq!(summary.focus.code, FocusCode::Healthy);
        assert!(summary.focus.evidence.is_empty(), "healthy: no evidence");
    }

    #[test]
    fn summary_render_sections_and_focus() {
        let reports: Vec<_> = (0..25)
            .map(|_| {
                make_report(
                    diagnosis_with(vec![HintCode::VocabMismatch], vec!["cancel", "money"]),
                    true,
                )
            })
            .collect();
        let rendered = summarize_diagnoses(&reports).render();
        assert!(
            rendered.contains("Hint histogram"),
            "missing histogram section: {}",
            rendered
        );
        assert!(rendered.contains("Top zero-match terms"));
        assert!(rendered.contains("Focus"));
        assert!(rendered.contains("vocab_mismatch"));
        assert!(rendered.contains("MULTIHOP_HYBRID.md"));
    }

    #[test]
    fn summarize_corpus_stats_coverage_split() {
        // 10 Layer-2 (corpus_stats_available=true), 15 Layer-1 (false).
        let mut reports = Vec::new();
        for _ in 0..10 {
            let mut d = diagnosis_with(vec![], vec![]);
            d.corpus_stats_available = true;
            reports.push(make_report(d, false));
        }
        for _ in 0..15 {
            let mut d = diagnosis_with(vec![], vec![]);
            d.corpus_stats_available = false;
            reports.push(make_report(d, false));
        }
        let summary = summarize_diagnoses(&reports);
        assert!(
            (summary.corpus_stats_coverage - 0.4).abs() < 1e-4,
            "coverage should be 10/25 = 0.4, got {}",
            summary.corpus_stats_coverage
        );
    }

    #[test]
    fn summarize_score_spread_none_when_absent() {
        let mut d = diagnosis_with(vec![], vec![]);
        d.score_spread = None;
        let reports: Vec<_> = (0..25).map(|_| make_report(d.clone(), false)).collect();
        let summary = summarize_diagnoses(&reports);
        assert!(summary.mean_score_spread.is_none());
        assert_eq!(summary.n_with_score_spread, 0);
    }
}
