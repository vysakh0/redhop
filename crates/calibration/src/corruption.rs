//! Corpus corruption injection + the diagnostics-vs-degradation
//! correlation study.
//!
//! Phase C built ingestion diagnostics that *claim* to detect
//! retrieval-corrupting corpora. This module validates that claim
//! empirically: take a real corpus, corrupt it at increasing severity,
//! and measure at each level both (a) the ingestion diagnostic scores
//! and (b) actual retrieval recall. If the diagnostics are worth
//! anything, they rise as recall falls — i.e. they *predict*
//! degradation rather than just describing text.
//!
//! The corruption operators model real-world failure modes:
//!
//! - **OCR noise** — drop vowels / spread characters in a fraction of
//!   tokens (scanned-document garble).
//! - **Duplication** — replace a fraction of chunks with copies of
//!   other chunks (copy-paste / repeated sections).
//! - **Boilerplate** — prepend a shared header line to every chunk
//!   (page furniture).
//!
//! Corruption is deterministic given a seed, so the study is
//! reproducible.

use redhop::core::{Chunk, ChunkId, TokenCount};
use serde::{Deserialize, Serialize};

/// Which corruption to apply.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum CorruptionKind {
    /// Scanned/OCR garble: vowel-dropping + character spreading.
    OcrNoise,
    /// Replace chunks with duplicates of earlier chunks.
    Duplication,
    /// Prepend a shared boilerplate header to every chunk.
    Boilerplate,
}

impl CorruptionKind {
    /// Stable code.
    pub fn code(self) -> &'static str {
        match self {
            Self::OcrNoise => "ocr_noise",
            Self::Duplication => "duplication",
            Self::Boilerplate => "boilerplate",
        }
    }
}

/// A tiny deterministic RNG (LCG) so corruption is reproducible.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn frac(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Corrupt a corpus at a given severity in `[0, 1]`.
///
/// `severity` is the fraction of tokens (OCR), chunks (duplication), or
/// always-applied (boilerplate) affected. Returns a new corpus; the
/// input is untouched. Chunk ids are preserved so gold labels still
/// resolve.
pub fn corrupt(chunks: &[Chunk], kind: CorruptionKind, severity: f32, seed: u64) -> Vec<Chunk> {
    let sev = severity.clamp(0.0, 1.0);
    let mut rng = Lcg::new(seed);
    match kind {
        CorruptionKind::OcrNoise => chunks
            .iter()
            .map(|c| {
                let text = ocr_garble(&c.text, sev, &mut rng);
                rebuild(c, text)
            })
            .collect(),
        CorruptionKind::Boilerplate => {
            let header = "ACME CORP CONFIDENTIAL — DO NOT DISTRIBUTE — PAGE FOOTER";
            chunks
                .iter()
                .map(|c| rebuild(c, format!("{header}\n{}", c.text)))
                .collect()
        }
        CorruptionKind::Duplication => {
            if chunks.is_empty() {
                return Vec::new();
            }
            chunks
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    if i > 0 && rng.frac() < sev {
                        // Replace this chunk's *text* with an earlier
                        // chunk's text, but keep this chunk's id (so the
                        // pollution is real but gold ids stay valid).
                        let src = (rng.next() as usize) % i;
                        rebuild(c, chunks[src].text.clone())
                    } else {
                        c.clone()
                    }
                })
                .collect()
        }
    }
}

fn rebuild(orig: &Chunk, text: String) -> Chunk {
    let tokens = text.split_whitespace().count();
    let mut c = Chunk::new(orig.id.clone(), text, &orig.source, TokenCount(tokens));
    c.metadata = orig.metadata.clone();
    c.embedding = None; // corruption invalidates any precomputed embedding
    c
}

fn ocr_garble(text: &str, severity: f32, rng: &mut Lcg) -> String {
    let mut out = Vec::new();
    for word in text.split_whitespace() {
        if rng.frac() < severity {
            // Two garble modes, picked per-word.
            if rng.frac() < 0.5 {
                // Drop vowels.
                let garbled: String = word
                    .chars()
                    .filter(|c| !matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u'))
                    .collect();
                if !garbled.is_empty() {
                    out.push(garbled);
                } else {
                    out.push(word.to_string());
                }
            } else {
                // Spread characters into single-char tokens.
                for ch in word.chars() {
                    out.push(ch.to_string());
                }
            }
        } else {
            out.push(word.to_string());
        }
    }
    out.join(" ")
}

/// One severity level's measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationRow {
    /// Corruption severity in `[0, 1]`.
    pub severity: f32,
    /// The dominant ingestion-diagnostic score for the applied
    /// corruption kind (e.g. `ocr_noise_score` for OcrNoise).
    pub diagnostic_score: f32,
    /// Gold-chunk retrieval recall at this severity.
    pub retrieval_recall: f32,
    /// Number of ingestion warnings emitted.
    pub n_warnings: usize,
}

/// Result of a corruption sweep: rows + the headline correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationStudy {
    /// Corruption kind studied.
    pub kind: String,
    /// One row per severity level.
    pub rows: Vec<DegradationRow>,
    /// Pearson correlation between `diagnostic_score` and
    /// `(1 − retrieval_recall)`. Near `+1.0` means the diagnostic
    /// predicts degradation well.
    pub diagnostic_vs_degradation_correlation: f32,
}

/// Run a corruption sweep and measure diagnostics vs retrieval recall.
///
/// For each severity level: corrupt the corpus, re-embed + re-index with
/// the hashing embedder, run the gold queries to measure recall@`top_k`,
/// and run ingestion diagnostics to measure the corruption score. The
/// returned [`DegradationStudy`] carries the Pearson correlation between
/// the diagnostic score and `(1 − recall)` — the headline number that
/// answers "do the diagnostics predict real retrieval degradation?".
///
/// Queries are *not* corrupted (users send clean queries); only the
/// corpus degrades. As corpus text garbles, the gold chunk's embedding
/// drifts from the clean query embedding and recall falls.
pub fn run_degradation_study(
    clean_chunks: &[Chunk],
    queries: &[(String, Vec<ChunkId>)],
    kind: CorruptionKind,
    severities: &[f32],
    embedder: &crate::embedder::HashingEmbedder,
    top_k: usize,
    seed: u64,
) -> DegradationStudy {
    use redhop::core::VectorIndex;
    use redhop_diagnostics::{diagnose_ingestion, IngestionThresholds};
    use redhop::storage::FlatVectorIndex;

    let cfg = IngestionThresholds::default();
    let dim = embedder.dim;
    let mut rows = Vec::with_capacity(severities.len());

    for &sev in severities {
        let corpus = if sev == 0.0 {
            clean_chunks.to_vec()
        } else {
            corrupt(clean_chunks, kind, sev, seed)
        };

        // Build a dense index over the (possibly corrupted) corpus.
        let mut index = FlatVectorIndex::new(dim);
        for c in &corpus {
            let v = embedder.embed(&c.text);
            let _ = index.add(c.id.clone(), v);
        }

        // Measure recall over the gold queries (queries stay clean).
        let mut total_recall = 0f32;
        let mut counted = 0usize;
        for (qtext, gold) in queries {
            if gold.is_empty() {
                continue;
            }
            let qv = embedder.embed(qtext);
            let hits = index.search(&qv, top_k).unwrap_or_default();
            let retrieved: Vec<&ChunkId> = hits.iter().map(|(id, _)| id).collect();
            let found = gold.iter().filter(|g| retrieved.contains(g)).count();
            total_recall += found as f32 / gold.len() as f32;
            counted += 1;
        }
        let recall = if counted > 0 {
            total_recall / counted as f32
        } else {
            0.0
        };

        // Ingestion diagnostics on the corrupted corpus.
        let report = diagnose_ingestion(&corpus, &cfg);
        let diagnostic_score = match kind {
            CorruptionKind::OcrNoise => report.ocr_noise_score,
            CorruptionKind::Duplication => report.duplicate_ratio,
            CorruptionKind::Boilerplate => report.boilerplate_ratio,
        };

        rows.push(DegradationRow {
            severity: sev,
            diagnostic_score,
            retrieval_recall: recall,
            n_warnings: report.warnings.len(),
        });
    }

    let diag_scores: Vec<f32> = rows.iter().map(|r| r.diagnostic_score).collect();
    let degradation: Vec<f32> = rows.iter().map(|r| 1.0 - r.retrieval_recall).collect();
    let corr = pearson(&diag_scores, &degradation);

    DegradationStudy {
        kind: kind.code().to_string(),
        rows,
        diagnostic_vs_degradation_correlation: corr,
    }
}

/// Pearson correlation. Returns 0 for degenerate inputs.
pub fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return 0.0;
    }
    let nf = n as f32;
    let mx = xs.iter().sum::<f32>() / nf;
    let my = ys.iter().sum::<f32>() / nf;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom <= 1e-9 {
        0.0
    } else {
        cov / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, text: &str) -> Chunk {
        Chunk::new(id, text, "doc", TokenCount(text.split_whitespace().count()))
    }

    fn clean() -> Vec<Chunk> {
        vec![
            chunk(
                "c1",
                "The transformer architecture uses self attention over the sequence.",
            ),
            chunk(
                "c2",
                "Retrieval augmented generation grounds models in external evidence.",
            ),
            chunk(
                "c3",
                "Dense retrieval encodes text into a shared vector space for ranking.",
            ),
        ]
    }

    #[test]
    fn ocr_corruption_increases_with_severity() {
        use redhop_diagnostics::{diagnose_ingestion, IngestionThresholds};
        let cfg = IngestionThresholds::default();
        let base = clean();
        let low = corrupt(&base, CorruptionKind::OcrNoise, 0.1, 1);
        let high = corrupt(&base, CorruptionKind::OcrNoise, 0.9, 1);
        let s_low = diagnose_ingestion(&low, &cfg).ocr_noise_score;
        let s_high = diagnose_ingestion(&high, &cfg).ocr_noise_score;
        assert!(s_high > s_low, "high {s_high} should exceed low {s_low}");
    }

    #[test]
    fn corruption_preserves_chunk_ids() {
        let base = clean();
        let c = corrupt(&base, CorruptionKind::OcrNoise, 0.5, 7);
        assert_eq!(c.len(), base.len());
        for (a, b) in base.iter().zip(c.iter()) {
            assert_eq!(a.id, b.id);
        }
    }

    #[test]
    fn duplication_replaces_text_keeps_id() {
        let base = clean();
        let c = corrupt(&base, CorruptionKind::Duplication, 1.0, 3);
        // With severity 1.0, every chunk after the first is a dup.
        assert_eq!(c[0].id, base[0].id);
        // c[1] and c[2] should have text matching some earlier chunk.
        for i in 1..c.len() {
            let dup_of_earlier = (0..i).any(|j| c[i].text == base[j].text);
            assert!(dup_of_earlier, "chunk {i} should duplicate an earlier one");
        }
    }

    #[test]
    fn pearson_detects_perfect_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pearson_detects_anti_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [4.0, 3.0, 2.0, 1.0];
        assert!((pearson(&x, &y) + 1.0).abs() < 1e-5);
    }
}
