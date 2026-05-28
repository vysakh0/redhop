//! The second-hop tax, measured directly and at large n.
//!
//! Every prior experiment inferred the tax indirectly (gold retention,
//! recall deltas, an LLM sign-flip). This one measures it head-on, with
//! no LLM and no embeddings, so it is fast, deterministic, and runs at
//! n in the hundreds-to-thousands.
//!
//! For each multi-hop HotpotQA query we label the gold chunks:
//!   first hop  = the gold chunk(s) with the HIGHER query grounding
//!   second hop = the gold chunk with the LOWEST query grounding
//! The second hop is the reasoning-critical, low-query-relevance chunk —
//! linked to the first hop by a bridge entity, not by the query. It is
//! exactly what relevance-based selection taxes.
//!
//! We inject off-document distractors ("true junk", near-zero query
//! overlap), then build a context under each strategy and measure three
//! things per query, with bootstrap 95% CIs over queries:
//!   second_hop_retention : did the reasoning-critical hop survive?   (want HIGH)
//!   junk_suppression     : fraction of injected distractors removed   (want HIGH)
//!   first_hop_retention  : did the query-relevant hop survive?        (sanity)
//!
//! Two panels:
//!   A. filter aggressiveness sweep (generous budget) — isolates the
//!      FILTER tax: DistractorFiltered vs ReasoningPreserving as the
//!      grounding threshold rises. Same junk suppression; the question
//!      is whether the second hop is dropped with the junk.
//!   B. budget scarcity (tight budget, threshold fixed) — isolates the
//!      RANKING/BUDGET tax across all four strategies.
//!
//! Run:  cargo run -p redhop-examples --example second_hop_retention --release

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use redhop_calibration::dataset::LabeledCorpus;
use redhop_calibration::loaders::hotpotqa::{default_regime as hotpot_regime, HotpotQADataset};
use redhop_calibration::loaders::musique::{default_regime as musique_regime, MuSiQueDataset};
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::context::{build_context, ContextConfig, ContextStrategy};
use redhop::core::{
    Chunk, ChunkId, Chunker, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenizerBackend,
};
use unicode_segmentation::UnicodeSegmentation;

const SAMPLE_SIZE: usize = 1500;
const N_DISTRACTORS: usize = 8;
const GENEROUS_BUDGET: usize = 100_000;
const TIGHT_BUDGET: usize = 220; // ~half of (2 gold + 8 distractors) @ ~45 tok

// Same term/grounding primitive redhop-context uses internally (stopword
// removal + Snowball stemming), so our second-hop labels match the
// strategies' notion of query relevance.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "as", "by", "with", "from", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "them", "his", "her", "their", "which", "who", "whom",
    "what", "when", "where", "how", "why", "into", "than", "then", "there", "here", "out", "up",
    "down", "over", "under", "do", "does", "did", "has", "have", "had", "not", "no", "can", "will",
    "would", "should", "could", "may", "might", "about", "between", "during", "such", "also",
];
fn terms(text: &str) -> HashSet<String> {
    use rust_stemmers::{Algorithm, Stemmer};
    thread_local!(static ST: Stemmer = Stemmer::create(Algorithm::English));
    let stop: HashSet<&str> = STOPWORDS.iter().copied().collect();
    text.unicode_words()
        .map(|w| w.to_lowercase())
        .filter(|w| w.chars().count() > 1 && !stop.contains(w.as_str()))
        .map(|w| ST.with(|s| s.stem(&w).into_owned()))
        .collect()
}
fn grounding(q: &HashSet<String>, c: &HashSet<String>) -> f32 {
    if q.is_empty() {
        return 0.0;
    }
    q.intersection(c).count() as f32 / q.len() as f32
}

// Tiny deterministic RNG.
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(s.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn as_results(chunks: &[Chunk]) -> Vec<RetrievalResult> {
    chunks
        .iter()
        .map(|c| RetrievalResult {
            chunk: c.clone(),
            score: Score {
                value: 1.0,
                method: RetrievalMethod::Lexical,
            },
            breakdown: ScoreBreakdown::default(),
        })
        .collect()
}

/// One query's labeled evidence, ready to be assembled and scored.
struct Case {
    query: Query,
    first_hop_ids: Vec<ChunkId>,
    second_hop_id: ChunkId,
    junk_ids: HashSet<ChunkId>,
    retrieved: Vec<RetrievalResult>,
}

/// The three per-query measurements for one strategy.
#[derive(Clone, Copy)]
struct Outcome {
    second_hop_retained: f32, // 0/1
    first_hop_retention: f32, // fraction
    junk_suppression: f32,    // fraction of injected junk removed
}

fn score(case: &Case, cfg: &ContextConfig) -> Outcome {
    let ctx = build_context(&case.query, &case.retrieved, cfg);
    let kept: HashSet<&ChunkId> = ctx.chunks.iter().map(|c| &c.id).collect();

    let second = if kept.contains(&&case.second_hop_id) {
        1.0
    } else {
        0.0
    };
    let first = if case.first_hop_ids.is_empty() {
        1.0
    } else {
        case.first_hop_ids
            .iter()
            .filter(|id| kept.contains(id))
            .count() as f32
            / case.first_hop_ids.len() as f32
    };
    let junk_kept = case.junk_ids.iter().filter(|id| kept.contains(id)).count();
    let junk_supp = if case.junk_ids.is_empty() {
        1.0
    } else {
        (case.junk_ids.len() - junk_kept) as f32 / case.junk_ids.len() as f32
    };
    Outcome {
        second_hop_retained: second,
        first_hop_retention: first,
        junk_suppression: junk_supp,
    }
}

/// Mean + 95% bootstrap CI over a per-query vector.
fn mean_ci(xs: &[f32], rng: &mut Lcg) -> (f32, f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
    let mut means = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let mut s = 0.0;
        for _ in 0..xs.len() {
            s += xs[(rng.next() as usize) % xs.len()];
        }
        means.push(s / xs.len() as f32);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (mean, means[24], means[974])
}

fn cfg(strategy: ContextStrategy, tau: f32, budget: usize) -> ContextConfig {
    ContextConfig {
        token_budget: budget,
        strategy,
        distractor_min_grounding: tau,
        link_min_jaccard: 0.12,
        auto_passthrough_max_tokens: 8_000,
        redundancy_max_cosine: 1.0,
    }
}

fn run_panel(cases: &[Case], strategy: ContextStrategy, tau: f32, budget: usize, label: &str) {
    let c = cfg(strategy, tau, budget);
    let outs: Vec<Outcome> = cases.iter().map(|case| score(case, &c)).collect();
    let mut rng = Lcg::new(0x5EED);
    let second: Vec<f32> = outs.iter().map(|o| o.second_hop_retained).collect();
    let first: Vec<f32> = outs.iter().map(|o| o.first_hop_retention).collect();
    let junk: Vec<f32> = outs.iter().map(|o| o.junk_suppression).collect();
    let (s_m, s_lo, s_hi) = mean_ci(&second, &mut rng);
    let (f_m, _, _) = mean_ci(&first, &mut rng);
    let (j_m, _, _) = mean_ci(&junk, &mut rng);
    println!(
        "  {:<22} {:>6.3} [{:.3},{:.3}]   {:>6.3}      {:>6.3}",
        label, s_m, s_lo, s_hi, j_m, f_m
    );
}

/// Build labeled cases (one per gap-qualified multi-hop query) from a corpus.
/// Returns the cases and the mean first−second grounding gap.
fn build_cases(corpus: &LabeledCorpus, chunks: &[Chunk]) -> (Vec<Case>, f32) {
    let by_id: HashMap<ChunkId, Chunk> = chunks.iter().map(|c| (c.id.clone(), c.clone())).collect();
    let mut rng = Lcg::new(0xC0FFEE);
    let mut cases: Vec<Case> = Vec::new();
    let mut grounding_gap_sum = 0.0f32;

    for lq in &corpus.queries {
        if lq.gold_chunk_ids.len() < 2 {
            continue; // need a first and a second hop
        }
        let q_terms = terms(&lq.text);
        let mut gold: Vec<(ChunkId, Chunk, f32)> = lq
            .gold_chunk_ids
            .iter()
            .filter_map(|id| by_id.get(id).map(|c| (id.clone(), c.clone())))
            .map(|(id, c)| {
                let g = grounding(&q_terms, &terms(&c.text));
                (id, c, g)
            })
            .collect();
        if gold.len() < 2 {
            continue;
        }
        // Sort by grounding ascending: lowest = the second hop.
        gold.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let second = gold[0].clone();
        let firsts: Vec<(ChunkId, Chunk, f32)> = gold[1..].to_vec();
        // Only keep cases with a real relevance gap — where the second
        // hop is genuinely less query-relevant than the first hop. That
        // is the regime the tax lives in; ~equal-relevance pairs aren't
        // multi-hop in the adversarial sense.
        let max_first_g = firsts.iter().map(|x| x.2).fold(0.0f32, f32::max);
        if second.2 >= max_first_g {
            continue;
        }
        grounding_gap_sum += max_first_g - second.2;

        // Off-document distractor pool.
        let gold_docs: HashSet<String> = gold.iter().map(|(_, c, _)| c.source.clone()).collect();
        let mut pool: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| !gold_docs.contains(&c.source))
            .collect();
        for i in (1..pool.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            pool.swap(i, j);
        }
        let junk: Vec<Chunk> = pool
            .iter()
            .take(N_DISTRACTORS)
            .map(|c| (*c).clone())
            .collect();
        let junk_ids: HashSet<ChunkId> = junk.iter().map(|c| c.id.clone()).collect();

        // Retrieved set, presented relevance-ranked (descending grounding),
        // as a real retriever would return it — the second hop sits low.
        let mut all: Vec<Chunk> = gold
            .iter()
            .map(|(_, c, _)| c.clone())
            .chain(junk.into_iter())
            .collect();
        let g_of = |c: &Chunk| grounding(&q_terms, &terms(&c.text));
        all.sort_by(|a, b| g_of(b).partial_cmp(&g_of(a)).unwrap());

        cases.push(Case {
            query: Query::new(&lq.text),
            first_hop_ids: firsts.iter().map(|x| x.0.clone()).collect(),
            second_hop_id: second.0.clone(),
            junk_ids,
            retrieved: as_results(&all),
        });
    }
    let n = cases.len();
    (cases, grounding_gap_sum / n.max(1) as f32)
}

/// Print the two panels for one dataset.
fn report(label: &str, cases: &[Case], mean_gap: f32) {
    let n = cases.len();
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  The second-hop tax, measured directly — {label:<25}║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("\n  multi-hop queries with a query-relevance gap: n = {n}");
    println!("  mean grounding gap (first hop − second hop): {mean_gap:.3}");
    println!("  injected off-document distractors per query: {N_DISTRACTORS}");

    println!("\n──── Panel A: the FILTER tax (generous budget, nothing dropped for space) ────");
    println!("  As the grounding threshold rises, DistractorFiltered drops the");
    println!("  second hop along with the junk; ReasoningPreserving rescues it.\n");
    println!(
        "  {:<22} {:>20}   {:>9}   {:>9}",
        "strategy @ threshold", "second_hop_ret [95% CI]", "junk_supp", "first_ret"
    );
    println!("  {}", "─".repeat(70));
    for tau in [0.05f32, 0.10, 0.20, 0.30] {
        run_panel(
            cases,
            ContextStrategy::DistractorFiltered,
            tau,
            GENEROUS_BUDGET,
            &format!("distractor_filt @{tau:.2}"),
        );
        run_panel(
            cases,
            ContextStrategy::ReasoningPreserving,
            tau,
            GENEROUS_BUDGET,
            &format!("reasoning_pres @{tau:.2}"),
        );
        println!();
    }

    println!("──── Panel B: the RANKING/BUDGET tax (tight budget {TIGHT_BUDGET} tok, τ=0.10) ────");
    println!("  Under attention scarcity, relevance-ranked selection drops the");
    println!("  low-relevance second hop to fit the budget.\n");
    println!(
        "  {:<22} {:>20}   {:>9}   {:>9}",
        "strategy", "second_hop_ret [95% CI]", "junk_supp", "first_ret"
    );
    println!("  {}", "─".repeat(70));
    run_panel(
        cases,
        ContextStrategy::RawTopK,
        0.10,
        TIGHT_BUDGET,
        "raw_topk",
    );
    run_panel(
        cases,
        ContextStrategy::DistractorFiltered,
        0.10,
        TIGHT_BUDGET,
        "distractor_filt",
    );
    run_panel(
        cases,
        ContextStrategy::MaxDensity,
        0.10,
        TIGHT_BUDGET,
        "max_density",
    );
    run_panel(
        cases,
        ContextStrategy::ReasoningPreserving,
        0.10,
        TIGHT_BUDGET,
        "reasoning_pres",
    );

    println!("\n  (second_hop_ret = P(reasoning-critical hop survives); junk_supp =");
    println!("   fraction of injected distractors removed; both want HIGH.)");
}

fn main() -> anyhow::Result<()> {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;

    // HotpotQA
    let mut hotpot = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
    hotpot.examples.truncate(SAMPLE_SIZE);
    let hp_corpus = hotpot.to_labeled_corpus(&chunker, |_| None, hotpot_regime)?;
    let hp_chunks = chunker.chunk_batch(&hp_corpus.docs)?;
    let (hp_cases, hp_gap) = build_cases(&hp_corpus, &hp_chunks);
    report("HotpotQA", &hp_cases, hp_gap);

    // MuSiQue (a second multi-hop dataset, for cross-dataset replication)
    match MuSiQueDataset::from_path(redhop_examples::data_path("musique/dev.jsonl")) {
        Ok(mut musique) => {
            musique.examples.truncate(SAMPLE_SIZE);
            let mq_corpus = musique.to_labeled_corpus(&chunker, |_| None, musique_regime)?;
            let mq_chunks = chunker.chunk_batch(&mq_corpus.docs)?;
            let (mq_cases, mq_gap) = build_cases(&mq_corpus, &mq_chunks);
            report("MuSiQue", &mq_cases, mq_gap);
        }
        Err(e) => eprintln!("\n[MuSiQue skipped: {e}]"),
    }
    Ok(())
}
