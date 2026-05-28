//! Ingestion diagnostics — corpus-level retrieval-reliability checks.
//!
//! Real-world RAG silently fails on messy corpora: OCR'd scans,
//! repeated headers/footers, duplicated sections, chunks broken
//! mid-sentence, flattened tables. None of these are visible to a
//! retriever — it happily returns garbage chunks with high scores. This
//! module surfaces the corruption *before* retrieval, so a deployment
//! can react (re-chunk, re-OCR, dedup) instead of serving noise.
//!
//! Unlike the per-query [`DiagnosticsEngine`][de] tier, ingestion
//! diagnostics run **once over the whole chunk corpus** at index time.
//! They are deliberately *text-only* — RedHop does not parse PDFs (that
//! stays in the Python ingestion layer); these metrics work on whatever
//! text the chunker produced regardless of source.
//!
//! Five cheap, interpretable signals:
//!
//! - [`IngestionReport::ocr_noise_score`] — vowelless tokens + isolated
//!   single characters. OCR garbage like `"rn th wrd"` lights this up.
//! - [`IngestionReport::duplicate_ratio`] — fraction of chunks that are
//!   near-duplicates of another (shingle-Jaccard). Catches repeated
//!   sections and copy-paste.
//! - [`IngestionReport::boilerplate_ratio`] — fraction of lines that
//!   repeat across many chunks (headers, footers, page numbers).
//! - [`IngestionReport::fragmentation_score`] — fraction of chunks that
//!   start mid-sentence or end without terminal punctuation.
//! - [`IngestionReport::table_noise_score`] — fraction of chunks that
//!   are digit/delimiter-heavy with little prose structure (flattened
//!   tables).
//!
//! Each metric is in `[0, 1]`; for all five, **lower is better**. The
//! report emits a [`DiagnosticsWarning`] per metric that crosses its
//! threshold, reusing the same warning channel as the per-query tier.
//!
//! These are *diagnostics*, not a controller: they surface problems,
//! they do not silently rewrite the corpus.
//!
//! [de]: crate::engine::DefaultDiagnosticsEngine

use std::collections::HashMap;

use redhop::core::{Chunk, ChunkId, DiagnosticsWarning};
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

/// Tunable thresholds for ingestion warnings.
#[derive(Debug, Clone)]
pub struct IngestionThresholds {
    /// Above this `ocr_noise_score`, warn.
    pub max_ocr_noise: f32,
    /// Above this `duplicate_ratio`, warn.
    pub max_duplicate_ratio: f32,
    /// Above this `boilerplate_ratio`, warn.
    pub max_boilerplate_ratio: f32,
    /// Above this `fragmentation_score`, warn.
    pub max_fragmentation: f32,
    /// Above this `table_noise_score`, warn.
    pub max_table_noise: f32,
    /// Shingle width (in words) for near-duplicate detection.
    pub shingle_k: usize,
    /// Jaccard similarity at/above which two chunks are near-duplicates.
    pub near_dup_jaccard: f32,
    /// A normalized line counts as boilerplate if it appears in at least
    /// this fraction of chunks.
    pub boilerplate_min_chunk_fraction: f32,
    /// How many worst-offender chunks to retain per metric for
    /// drill-down.
    pub flagged_per_metric: usize,
}

impl Default for IngestionThresholds {
    fn default() -> Self {
        Self {
            max_ocr_noise: 0.10,
            max_duplicate_ratio: 0.20,
            max_boilerplate_ratio: 0.15,
            max_fragmentation: 0.40,
            max_table_noise: 0.25,
            shingle_k: 5,
            near_dup_jaccard: 0.80,
            boilerplate_min_chunk_fraction: 0.50,
            flagged_per_metric: 5,
        }
    }
}

/// A chunk flagged by a specific metric, for drill-down.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlaggedChunk {
    /// The offending chunk.
    pub chunk_id: ChunkId,
    /// Which metric flagged it.
    pub metric: String,
    /// The chunk's per-chunk score for that metric.
    pub score: f32,
}

/// Result of an ingestion diagnosis over a chunk corpus.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestionReport {
    /// Number of chunks analyzed.
    pub n_chunks: usize,
    /// OCR-noise score in `[0, 1]`. Lower is better.
    pub ocr_noise_score: f32,
    /// Near-duplicate fraction in `[0, 1]`. Lower is better.
    pub duplicate_ratio: f32,
    /// Boilerplate-line fraction in `[0, 1]`. Lower is better.
    pub boilerplate_ratio: f32,
    /// Mid-sentence-break fraction in `[0, 1]`. Lower is better.
    pub fragmentation_score: f32,
    /// Flattened-table fraction in `[0, 1]`. Lower is better.
    pub table_noise_score: f32,
    /// Worst-offender chunks per metric, for drill-down.
    pub flagged_chunks: Vec<FlaggedChunk>,
    /// Advisory warnings, one per metric that crossed its threshold.
    pub warnings: Vec<DiagnosticsWarning>,
}

/// Run ingestion diagnostics over a chunk corpus.
pub fn diagnose_ingestion(chunks: &[Chunk], cfg: &IngestionThresholds) -> IngestionReport {
    let mut report = IngestionReport {
        n_chunks: chunks.len(),
        ..Default::default()
    };
    if chunks.is_empty() {
        return report;
    }

    // ── Per-chunk scores (used both for aggregates and flagging) ──
    let mut ocr_scores: Vec<(usize, f32)> = Vec::with_capacity(chunks.len());
    let mut frag_flags: Vec<(usize, bool)> = Vec::with_capacity(chunks.len());
    let mut table_scores: Vec<(usize, f32)> = Vec::with_capacity(chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        ocr_scores.push((i, ocr_noise_chunk(&c.text)));
        frag_flags.push((i, is_fragmented(&c.text)));
        table_scores.push((i, table_noise_chunk(&c.text)));
    }

    report.ocr_noise_score = ocr_scores.iter().map(|(_, s)| *s).sum::<f32>() / chunks.len() as f32;
    report.table_noise_score =
        table_scores.iter().map(|(_, s)| *s).sum::<f32>() / chunks.len() as f32;
    report.fragmentation_score =
        frag_flags.iter().filter(|(_, f)| *f).count() as f32 / chunks.len() as f32;

    // ── Duplicate ratio (shingle-Jaccard via inverted index) ──────
    let (dup_ratio, dup_flagged) = duplicate_ratio(chunks, cfg);
    report.duplicate_ratio = dup_ratio;

    // ── Boilerplate ratio (cross-chunk line frequency) ────────────
    report.boilerplate_ratio = boilerplate_ratio(chunks, cfg);

    // ── Flag worst offenders ──────────────────────────────────────
    flag_top(
        &mut report.flagged_chunks,
        chunks,
        &ocr_scores,
        "ocr_noise",
        cfg.flagged_per_metric,
        cfg.max_ocr_noise,
    );
    flag_top(
        &mut report.flagged_chunks,
        chunks,
        &table_scores,
        "table_noise",
        cfg.flagged_per_metric,
        cfg.max_table_noise,
    );
    for idx in dup_flagged.into_iter().take(cfg.flagged_per_metric) {
        report.flagged_chunks.push(FlaggedChunk {
            chunk_id: chunks[idx].id.clone(),
            metric: "duplicate".to_string(),
            score: 1.0,
        });
    }

    // ── Warnings ──────────────────────────────────────────────────
    push_warning(
        &mut report.warnings,
        report.ocr_noise_score,
        cfg.max_ocr_noise,
        "ocr_noise",
        "corpus shows OCR-corruption signatures (vowelless tokens / isolated characters); consider re-OCR or a more robust extractor",
    );
    push_warning(
        &mut report.warnings,
        report.duplicate_ratio,
        cfg.max_duplicate_ratio,
        "duplicate_content",
        "many chunks are near-duplicates; dedup before indexing or retrieval will surface redundant evidence",
    );
    push_warning(
        &mut report.warnings,
        report.boilerplate_ratio,
        cfg.max_boilerplate_ratio,
        "boilerplate",
        "repeated headers/footers/page-furniture detected across chunks; strip boilerplate during chunking",
    );
    push_warning(
        &mut report.warnings,
        report.fragmentation_score,
        cfg.max_fragmentation,
        "fragmentation",
        "many chunks start or end mid-sentence; consider sentence-aware chunking with overlap",
    );
    push_warning(
        &mut report.warnings,
        report.table_noise_score,
        cfg.max_table_noise,
        "table_noise",
        "many chunks are digit/delimiter-heavy with little prose (flattened tables); table-aware extraction recommended",
    );

    report
}

// ─────────────────────────────────────────────────────────────────────
// Metric implementations
// ─────────────────────────────────────────────────────────────────────

fn has_vowel(s: &str) -> bool {
    s.chars()
        .any(|c| matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u' | 'y'))
}

/// Per-chunk OCR-noise score: a blend of the vowelless-token ratio and
/// the isolated-single-character ratio. Clean English prose scores
/// near 0; OCR garbage scores high.
fn ocr_noise_chunk(text: &str) -> f32 {
    let words: Vec<&str> = text.unicode_words().collect();
    if words.is_empty() {
        return 0.0;
    }
    let mut alpha_long = 0usize; // alphabetic tokens length >= 3
    let mut vowelless = 0usize;
    let mut single_alpha = 0usize;
    let mut total_tokens = 0usize;
    for w in &words {
        total_tokens += 1;
        let is_alpha = w.chars().all(|c| c.is_alphabetic());
        let len = w.chars().count();
        if is_alpha && len == 1 {
            // "a" and "i" are legitimate single-letter words.
            let lower = w.to_lowercase();
            if lower != "a" && lower != "i" {
                single_alpha += 1;
            }
        }
        if is_alpha && len >= 3 {
            alpha_long += 1;
            if !has_vowel(w) {
                vowelless += 1;
            }
        }
    }
    let vowelless_ratio = if alpha_long > 0 {
        vowelless as f32 / alpha_long as f32
    } else {
        0.0
    };
    let isolated_ratio = single_alpha as f32 / total_tokens as f32;
    (0.7 * vowelless_ratio + 0.3 * isolated_ratio).clamp(0.0, 1.0)
}

/// Per-chunk fragmentation flag: starts lowercase OR doesn't end in
/// terminal punctuation.
fn is_fragmented(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }
    let first = t.chars().next().unwrap();
    let starts_mid = first.is_alphabetic() && first.is_lowercase();
    let last = t.chars().rev().find(|c| !c.is_whitespace()).unwrap_or(' ');
    let ends_clean = matches!(last, '.' | '!' | '?' | '"' | '\'' | ')' | ']' | '”' | '’');
    starts_mid || !ends_clean
}

/// Per-chunk table-noise score: high when the chunk is digit/delimiter
/// heavy and has little sentence structure.
fn table_noise_chunk(text: &str) -> f32 {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return 0.0;
    }
    let total = chars.len() as f32;
    let digits = chars.iter().filter(|c| c.is_ascii_digit()).count() as f32;
    let delimiters = chars
        .iter()
        .filter(|c| matches!(c, '|' | '\t' | ',' | ';' | ':' | '%' | '$'))
        .count() as f32;
    let digit_delim_density = (digits + delimiters) / total;

    // Prose structure proxy: words per sentence terminator. Tables have
    // very few sentence terminators relative to tokens.
    let words = text.unicode_words().count().max(1) as f32;
    let terminators = text
        .chars()
        .filter(|c| matches!(c, '.' | '!' | '?'))
        .count()
        .max(1) as f32;
    let words_per_sentence = words / terminators;
    // Low prose structure ⇒ words_per_sentence is very high (one long
    // run with no terminators) OR there simply aren't sentences.
    let low_prose = if words_per_sentence > 40.0 { 1.0 } else { 0.0 };

    // Combine: density is the primary signal; low-prose amplifies.
    let score = digit_delim_density * (0.6 + 0.4 * low_prose);
    // Scale: 0.3 raw density is already very table-like → map toward 1.
    (score / 0.30).clamp(0.0, 1.0)
}

/// Corpus duplicate ratio via shingle-Jaccard with an inverted index to
/// avoid the full O(n²) comparison.
fn duplicate_ratio(chunks: &[Chunk], cfg: &IngestionThresholds) -> (f32, Vec<usize>) {
    let k = cfg.shingle_k.max(1);
    // Build shingle sets.
    let shingle_sets: Vec<Vec<u64>> = chunks.iter().map(|c| shingles(&c.text, k)).collect();

    // Inverted index: shingle → chunk indices.
    let mut index: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, set) in shingle_sets.iter().enumerate() {
        for &sh in set {
            index.entry(sh).or_default().push(i);
        }
    }

    let mut is_dup = vec![false; chunks.len()];
    for (i, set_i) in shingle_sets.iter().enumerate() {
        if set_i.is_empty() || is_dup[i] {
            continue;
        }
        // Candidate partners: chunks sharing at least one shingle.
        let mut candidates: HashMap<usize, usize> = HashMap::new(); // j → shared count
        for &sh in set_i {
            if let Some(holders) = index.get(&sh) {
                for &j in holders {
                    if j != i {
                        *candidates.entry(j).or_insert(0) += 1;
                    }
                }
            }
        }
        for (j, _shared) in candidates {
            let set_j = &shingle_sets[j];
            if set_j.is_empty() {
                continue;
            }
            let jac = jaccard(set_i, set_j);
            if jac >= cfg.near_dup_jaccard {
                is_dup[i] = true;
                is_dup[j] = true;
            }
        }
    }
    let dup_indices: Vec<usize> = (0..chunks.len()).filter(|&i| is_dup[i]).collect();
    let ratio = dup_indices.len() as f32 / chunks.len() as f32;
    (ratio, dup_indices)
}

fn shingles(text: &str, k: usize) -> Vec<u64> {
    let words: Vec<String> = text.unicode_words().map(|w| w.to_lowercase()).collect();
    if words.len() < k {
        // Whole-chunk shingle when too short.
        if words.is_empty() {
            return Vec::new();
        }
        return vec![hash_words(&words)];
    }
    let mut set: Vec<u64> = words.windows(k).map(hash_words).collect();
    set.sort_unstable();
    set.dedup();
    set
}

fn hash_words(words: &[String]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for w in words {
        for b in w.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0x20; // word separator
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn jaccard(a: &[u64], b: &[u64]) -> f32 {
    // Both are sorted, deduped.
    let mut i = 0;
    let mut j = 0;
    let mut inter = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let union = a.len() + b.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
}

/// Corpus boilerplate ratio: fraction of (normalized) lines that recur
/// across at least `boilerplate_min_chunk_fraction` of chunks.
fn boilerplate_ratio(chunks: &[Chunk], cfg: &IngestionThresholds) -> f32 {
    // Count, per normalized line, how many *distinct chunks* contain it.
    let mut line_chunk_count: HashMap<String, usize> = HashMap::new();
    let mut total_lines = 0usize;
    for c in chunks {
        let mut seen_in_chunk: std::collections::HashSet<String> = std::collections::HashSet::new();
        for raw in c.text.lines() {
            let norm = normalize_line(raw);
            if norm.is_empty() {
                continue;
            }
            total_lines += 1;
            seen_in_chunk.insert(norm);
        }
        for line in seen_in_chunk {
            *line_chunk_count.entry(line).or_insert(0) += 1;
        }
    }
    if total_lines == 0 {
        return 0.0;
    }
    let min_chunks = (chunks.len() as f32 * cfg.boilerplate_min_chunk_fraction).ceil() as usize;
    let min_chunks = min_chunks.max(2); // a line in a single chunk isn't boilerplate
                                        // Count total line *occurrences* that belong to boilerplate lines.
    let mut boilerplate_occurrences = 0usize;
    for c in chunks {
        for raw in c.text.lines() {
            let norm = normalize_line(raw);
            if norm.is_empty() {
                continue;
            }
            if line_chunk_count.get(&norm).copied().unwrap_or(0) >= min_chunks {
                boilerplate_occurrences += 1;
            }
        }
    }
    (boilerplate_occurrences as f32 / total_lines as f32).clamp(0.0, 1.0)
}

/// Normalize a line for boilerplate matching: trim, lowercase, collapse
/// whitespace, and replace digit runs with `#` so "Page 3" and "Page 4"
/// collapse to the same boilerplate key.
fn normalize_line(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(trimmed.len());
    let mut prev_space = false;
    let mut prev_digit = false;
    for c in trimmed.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
            prev_digit = false;
        } else if c.is_ascii_digit() {
            if !prev_digit {
                out.push('#');
            }
            prev_digit = true;
            prev_space = false;
        } else {
            out.extend(c.to_lowercase());
            prev_space = false;
            prev_digit = false;
        }
    }
    out.trim().to_string()
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

fn flag_top(
    out: &mut Vec<FlaggedChunk>,
    chunks: &[Chunk],
    scores: &[(usize, f32)],
    metric: &str,
    n: usize,
    threshold: f32,
) {
    let mut sorted: Vec<&(usize, f32)> = scores.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (idx, score) in sorted.into_iter().take(n) {
        if *score <= threshold {
            break; // only flag chunks that actually cross the line
        }
        out.push(FlaggedChunk {
            chunk_id: chunks[*idx].id.clone(),
            metric: metric.to_string(),
            score: *score,
        });
    }
}

fn push_warning(
    out: &mut Vec<DiagnosticsWarning>,
    value: f32,
    threshold: f32,
    code: &str,
    message: &str,
) {
    if value > threshold {
        out.push(DiagnosticsWarning {
            code: code.to_string(),
            message: format!("{message} (score {value:.3} > threshold {threshold:.3})"),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop::core::TokenCount;

    fn chunk(id: &str, text: &str) -> Chunk {
        Chunk::new(id, text, "doc", TokenCount(text.split_whitespace().count()))
    }

    fn clean_corpus() -> Vec<Chunk> {
        vec![
            chunk("c1", "The transformer architecture introduced self-attention. It replaced recurrence with parallel attention over the sequence."),
            chunk("c2", "Retrieval augmented generation grounds language models in external evidence. The retriever selects relevant passages."),
            chunk("c3", "Dense retrieval encodes queries and documents into a shared vector space. Cosine similarity ranks the candidates."),
        ]
    }

    #[test]
    fn clean_corpus_scores_low_everywhere() {
        let r = diagnose_ingestion(&clean_corpus(), &IngestionThresholds::default());
        assert!(r.ocr_noise_score < 0.10, "ocr {}", r.ocr_noise_score);
        assert!(r.duplicate_ratio < 0.20, "dup {}", r.duplicate_ratio);
        assert!(r.boilerplate_ratio < 0.15, "boiler {}", r.boilerplate_ratio);
        assert!(
            r.fragmentation_score < 0.40,
            "frag {}",
            r.fragmentation_score
        );
        assert!(r.table_noise_score < 0.25, "table {}", r.table_noise_score);
        assert!(r.warnings.is_empty(), "warnings: {:?}", r.warnings);
    }

    #[test]
    fn ocr_garbage_detected() {
        let corpus = vec![
            chunk("c1", "rn th wrd xj qz mn bk th rn nw wht hppnd hr nd thr"),
            chunk("c2", "w h a t i s g o i n g o n h e r e t o d a y"),
        ];
        let r = diagnose_ingestion(&corpus, &IngestionThresholds::default());
        assert!(r.ocr_noise_score > 0.10, "ocr {}", r.ocr_noise_score);
        assert!(r.warnings.iter().any(|w| w.code == "ocr_noise"));
        assert!(r.flagged_chunks.iter().any(|f| f.metric == "ocr_noise"));
    }

    #[test]
    fn duplicates_detected() {
        let body = "Retrieval augmented generation grounds language models in external evidence and selects relevant passages from a corpus.";
        let corpus = vec![
            chunk("c1", body),
            chunk("c2", body), // exact dup
            chunk("c3", "A completely different chunk about transformer attention mechanisms and parallel sequence processing entirely."),
            chunk("c4", body), // another dup
        ];
        let r = diagnose_ingestion(&corpus, &IngestionThresholds::default());
        // 3 of 4 chunks are near-duplicates → ratio 0.75.
        assert!(r.duplicate_ratio >= 0.5, "dup {}", r.duplicate_ratio);
        assert!(r.warnings.iter().any(|w| w.code == "duplicate_content"));
    }

    #[test]
    fn boilerplate_detected() {
        // Same header line in every chunk + page numbers.
        let mk = |n: usize, body: &str| {
            chunk(
                &format!("c{n}"),
                &format!("ACME CORP — CONFIDENTIAL\nPage {n}\n{body}"),
            )
        };
        let corpus = vec![
            mk(
                1,
                "First section discusses the quarterly revenue figures and growth.",
            ),
            mk(
                2,
                "Second section covers the operating expenses and margins.",
            ),
            mk(3, "Third section reviews the cash flow and balance sheet."),
            mk(
                4,
                "Fourth section analyzes the competitive landscape and risks.",
            ),
        ];
        let r = diagnose_ingestion(&corpus, &IngestionThresholds::default());
        assert!(r.boilerplate_ratio > 0.15, "boiler {}", r.boilerplate_ratio);
        assert!(r.warnings.iter().any(|w| w.code == "boilerplate"));
    }

    #[test]
    fn fragmentation_detected() {
        let corpus = vec![
            chunk(
                "c1",
                "and then the process continues without a clear ending so the reader",
            ),
            chunk(
                "c2",
                "is left wondering what happened because the chunk boundary cut",
            ),
            chunk(
                "c3",
                "right through the middle of an important explanatory sentence",
            ),
        ];
        let r = diagnose_ingestion(&corpus, &IngestionThresholds::default());
        assert!(
            r.fragmentation_score > 0.40,
            "frag {}",
            r.fragmentation_score
        );
        assert!(r.warnings.iter().any(|w| w.code == "fragmentation"));
    }

    #[test]
    fn table_noise_detected() {
        let corpus = vec![
            chunk("c1", "2019 | 1,240 | 3.5% | $12,400 | 88 | 2020 | 1,560 | 4.1% | $15,600 | 92 | 2021 | 1,890 | 5.0% | $18,900 | 95"),
            chunk("c2", "Q1 | 45 | Q2 | 52 | Q3 | 61 | Q4 | 70 | total | 228 | avg | 57 | 2022 | 99 | 12 | 34 | 56 | 78"),
        ];
        let r = diagnose_ingestion(&corpus, &IngestionThresholds::default());
        assert!(r.table_noise_score > 0.25, "table {}", r.table_noise_score);
        assert!(r.warnings.iter().any(|w| w.code == "table_noise"));
    }

    #[test]
    fn empty_corpus_is_safe() {
        let r = diagnose_ingestion(&[], &IngestionThresholds::default());
        assert_eq!(r.n_chunks, 0);
        assert!(r.warnings.is_empty());
    }
}
