//! Signal-quality ablation harness (non-circular).
//!
//! RedHop's grounding signal is raw lexical query-term overlap. This harness
//! measures whether sharpening that signal (stopword removal, IDF weighting,
//! …) actually helps — using ground truth INDEPENDENT of the signal itself, so
//! we are not measuring the proxy agreeing with itself.
//!
//! Ground truth: dataset gold annotations (real supporting chunks) vs injected
//! off-document distractors (known junk). Primary metric:
//!
//!   AUC = P( grounding(gold) > grounding(injected distractor) )
//!
//! threshold-free and scale-invariant, so raw vs IDF-weighted grounding are
//! directly comparable. AUC 1.0 = perfect separation; 0.5 = random. A variant
//! "helps" only if its AUC 95% CI clears the baseline. We ablate variants HERE
//! and promote only winners into the core crate.
//!
//! Run:  cargo run -p redhop-examples --example signal_ablation --release

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use redhop_calibration::dataset::LabeledCorpus;
use redhop_calibration::loaders::hotpotqa::{default_regime as hp_regime, HotpotQADataset};
use redhop_calibration::loaders::musique::{default_regime as mq_regime, MuSiQueDataset};
use redhop_chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop_core::{Chunk, ChunkId, Chunker, TokenizerBackend};
use rust_stemmers::{Algorithm, Stemmer};
use unicode_segmentation::UnicodeSegmentation;

const HOTPOTQA_PATH: &str =
    "/Users/vysakh/projects/neorag/data/hotpotqa/hotpot_dev_distractor_v1.json";
const MUSIQUE_PATH: &str = "/Users/vysakh/projects/neorag/data/musique/dev.jsonl";
const SAMPLE_SIZE: usize = 1500;
const N_DISTRACTORS: usize = 8;

// A compact English stopword list (the high-frequency function words that
// inflate raw overlap; deliberately small and conventional).
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "as", "by", "with", "from", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "them", "his", "her", "their", "which", "who", "whom",
    "what", "when", "where", "how", "why", "into", "than", "then", "there", "here", "out", "up",
    "down", "over", "under", "do", "does", "did", "has", "have", "had", "not", "no", "can", "will",
    "would", "should", "could", "may", "might", "about", "between", "during", "such", "also",
];

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Baseline,        // raw lexical overlap
    Stopwords,       // drop stopwords
    Idf,             // IDF-weighted overlap (keeps stopwords; lets IDF down-weight them)
    StopwordsIdf,    // both
    StopwordsStem,   // stopwords + crude stemming
    StopwordsSnowball, // stopwords + Snowball (Porter2) stemming
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::Baseline => "baseline (raw overlap)",
            Variant::Stopwords => "+ stopwords",
            Variant::Idf => "+ idf",
            Variant::StopwordsIdf => "+ stopwords + idf",
            Variant::StopwordsStem => "+ stopwords + stem (crude)",
            Variant::StopwordsSnowball => "+ stopwords + stem (snowball)",
        }
    }
    fn drop_stop(self) -> bool {
        matches!(
            self,
            Variant::Stopwords | Variant::StopwordsIdf | Variant::StopwordsStem | Variant::StopwordsSnowball
        )
    }
    fn use_idf(self) -> bool {
        matches!(self, Variant::Idf | Variant::StopwordsIdf)
    }
    fn stem_mode(self) -> StemMode {
        match self {
            Variant::StopwordsStem => StemMode::Crude,
            Variant::StopwordsSnowball => StemMode::Snowball,
            _ => StemMode::Off,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum StemMode {
    Off,
    Crude,
    Snowball,
}

/// Crude suffix-stripping stemmer (a Porter stand-in to test whether stemming
/// helps the signal at all before taking a real stemmer dependency).
fn crude_stem(w: &str) -> String {
    let w = w.strip_suffix("'s").unwrap_or(w);
    for suf in ["ing", "edly", "ed", "ies", "es", "ly", "s"] {
        if let Some(base) = w.strip_suffix(suf) {
            if base.chars().count() >= 3 {
                return base.to_string();
            }
        }
    }
    w.to_string()
}

fn raw_terms(text: &str) -> Vec<String> {
    text.unicode_words()
        .map(|w| w.to_lowercase())
        .filter(|w| w.chars().count() > 1)
        .collect()
}

fn term_set(
    text: &str,
    drop_stop: bool,
    stem: StemMode,
    stop: &HashSet<&str>,
    stemmer: &Stemmer,
) -> HashSet<String> {
    raw_terms(text)
        .into_iter()
        .filter(|w| !drop_stop || !stop.contains(w.as_str()))
        .map(|w| match stem {
            StemMode::Off => w,
            StemMode::Crude => crude_stem(&w),
            StemMode::Snowball => stemmer.stem(&w).into_owned(),
        })
        .collect()
}

/// Grounding of a chunk wrt a query under a variant. Baseline = fraction of
/// query terms present; IDF = idf-weighted fraction.
fn grounding(
    q: &HashSet<String>,
    c: &HashSet<String>,
    use_idf: bool,
    idf: &HashMap<String, f32>,
) -> f32 {
    if q.is_empty() {
        return 0.0;
    }
    if !use_idf {
        return q.intersection(c).count() as f32 / q.len() as f32;
    }
    let denom: f32 = q.iter().map(|t| idf.get(t).copied().unwrap_or(0.0)).sum();
    if denom <= 0.0 {
        return 0.0;
    }
    let num: f32 = q.intersection(c).map(|t| idf.get(t).copied().unwrap_or(0.0)).sum();
    num / denom
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
}

/// Per query: grounding scores of the gold chunks and of injected distractors.
struct QCase {
    gold: Vec<f32>,
    distractor: Vec<f32>,
}

/// AUC = P(gold > distractor), pooled over all gold/distractor pairs across the
/// given cases. Ties count as 0.5.
fn auc(cases: &[&QCase]) -> f32 {
    let mut wins = 0.0f64;
    let mut total = 0.0f64;
    for c in cases {
        for &g in &c.gold {
            for &d in &c.distractor {
                total += 1.0;
                if g > d {
                    wins += 1.0;
                } else if (g - d).abs() < 1e-9 {
                    wins += 0.5;
                }
            }
        }
    }
    if total == 0.0 {
        0.5
    } else {
        (wins / total) as f32
    }
}

fn auc_ci(cases: &[QCase], rng: &mut Lcg) -> (f32, f32, f32) {
    let refs: Vec<&QCase> = cases.iter().collect();
    let point = auc(&refs);
    let mut samples = Vec::with_capacity(500);
    for _ in 0..500 {
        let boot: Vec<&QCase> = (0..cases.len())
            .map(|_| &cases[(rng.next() as usize) % cases.len()])
            .collect();
        samples.push(auc(&boot));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (point, samples[12], samples[487])
}

fn build_cases(corpus: &LabeledCorpus, chunks: &[Chunk], variant: Variant) -> Vec<QCase> {
    let stop: HashSet<&str> = STOPWORDS.iter().copied().collect();
    let stemmer = Stemmer::create(Algorithm::English);
    let by_id: HashMap<&ChunkId, &Chunk> = chunks.iter().map(|c| (&c.id, c)).collect();

    // IDF over the chunk corpus (document = chunk), using the variant's term set.
    let n_docs = chunks.len().max(1) as f32;
    let mut df: HashMap<String, u32> = HashMap::new();
    if variant.use_idf() {
        for c in chunks {
            for t in term_set(&c.text, variant.drop_stop(), variant.stem_mode(), &stop, &stemmer) {
                *df.entry(t).or_insert(0) += 1;
            }
        }
    }
    let idf: HashMap<String, f32> = df
        .into_iter()
        .map(|(t, d)| (t, (n_docs / d as f32).ln().max(0.0)))
        .collect();

    let mut rng = Lcg(0xC0FFEE);
    let mut cases = Vec::new();
    for lq in &corpus.queries {
        if lq.gold_chunk_ids.len() < 2 {
            continue;
        }
        let q = term_set(&lq.text, variant.drop_stop(), variant.stem_mode(), &stop, &stemmer);
        let gold_chunks: Vec<&Chunk> =
            lq.gold_chunk_ids.iter().filter_map(|id| by_id.get(id).copied()).collect();
        if gold_chunks.len() < 2 {
            continue;
        }
        let gold_docs: HashSet<&str> = gold_chunks.iter().map(|c| c.source.as_str()).collect();
        let gold: Vec<f32> = gold_chunks
            .iter()
            .map(|c| grounding(&q, &term_set(&c.text, variant.drop_stop(), variant.stem_mode(), &stop, &stemmer), variant.use_idf(), &idf))
            .collect();

        // Inject off-document distractors.
        let mut pool: Vec<&Chunk> =
            chunks.iter().filter(|c| !gold_docs.contains(c.source.as_str())).collect();
        for i in (1..pool.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            pool.swap(i, j);
        }
        let distractor: Vec<f32> = pool
            .iter()
            .take(N_DISTRACTORS)
            .map(|c| grounding(&q, &term_set(&c.text, variant.drop_stop(), variant.stem_mode(), &stop, &stemmer), variant.use_idf(), &idf))
            .collect();
        if gold.is_empty() || distractor.is_empty() {
            continue;
        }
        cases.push(QCase { gold, distractor });
    }
    cases
}

fn run(label: &str, corpus: &LabeledCorpus, chunks: &[Chunk]) {
    println!("\n=== {label} (gold-vs-distractor AUC; higher = sharper signal) ===");
    println!("  {:<24} {:>22}", "variant", "AUC [95% CI]");
    println!("  {}", "─".repeat(50));
    let mut rng = Lcg(0x5EED);
    for v in [Variant::Baseline, Variant::Stopwords, Variant::Idf, Variant::StopwordsIdf, Variant::StopwordsStem, Variant::StopwordsSnowball] {
        let cases = build_cases(corpus, chunks, v);
        let (m, lo, hi) = auc_ci(&cases, &mut rng);
        println!("  {:<24} {:>8.3} [{:.3}, {:.3}]  (n={})", v.label(), m, lo, hi, cases.len());
    }
}

fn main() -> anyhow::Result<()> {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;

    let mut hp = HotpotQADataset::from_path(HOTPOTQA_PATH)?;
    hp.examples.truncate(SAMPLE_SIZE);
    let hp_corpus = hp.to_labeled_corpus(&chunker, |_| None, hp_regime)?;
    let hp_chunks = chunker.chunk_batch(&hp_corpus.docs)?;
    run("HotpotQA", &hp_corpus, &hp_chunks);

    if let Ok(mut mq) = MuSiQueDataset::from_path(MUSIQUE_PATH) {
        mq.examples.truncate(SAMPLE_SIZE);
        let mq_corpus = mq.to_labeled_corpus(&chunker, |_| None, mq_regime)?;
        let mq_chunks = chunker.chunk_batch(&mq_corpus.docs)?;
        run("MuSiQue", &mq_corpus, &mq_chunks);
    }

    println!("\n  A variant is promoted to the core only if its AUC CI clears baseline.");
    Ok(())
}
