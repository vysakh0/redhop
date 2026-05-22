//! Emit contexts for the large-n end-to-end test of ReasoningPreserving.
//!
//! The retention experiment (second_hop_retention.rs) showed, hermetically
//! and at n=1327 with CIs, that ReasoningPreserving rescues second hops an
//! aggressive relevance filter drops. That is *reachability*. This emitter
//! sets up the *reasoning-success* test: does keeping the second hop
//! actually produce better answers?
//!
//! For each multi-hop HotpotQA query WITH a query-relevance gap (the regime
//! the tax lives in), we build four contexts from the SAME polluted input
//! (gold + injected off-document distractors), at an AGGRESSIVE filter
//! threshold where the tax bites:
//!
//!   ctx_gold_only  : the supporting gold only (clean ceiling)
//!   ctx_polluted   : gold + N distractors (realistic noisy retrieval)
//!   ctx_filtered   : polluted → DistractorFiltered @ aggressive threshold
//!                    (suppresses junk but taxes the second hop)
//!   ctx_reasoning  : polluted → ReasoningPreserving @ same threshold
//!                    (suppresses junk, rescues the linked second hop)
//!
//! The decisive comparison is ctx_reasoning − ctx_filtered: same junk
//! suppression regime, the only systematic difference is the second hop.
//! We also emit per-query labels of whether each strategy KEPT the labeled
//! second hop, so the lab can correlate retention with answer correctness
//! directly — separating reachability from reasoning success.
//!
//! NeoRAG (Rust) builds contexts; the Python lab calls the LLM and scores.
//!
//! Run:  cargo run -p neorag-examples --example emit_reasoning_qa --release

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use neorag_calibration::loaders::hotpotqa::{default_regime, HotpotQADataset};
use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
use neorag_context::{build_context, ContextConfig, ContextStrategy};
use neorag_core::{
    Chunk, ChunkId, Chunker, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenizerBackend,
};
use unicode_segmentation::UnicodeSegmentation;

const HOTPOTQA_PATH: &str =
    "/Users/vysakh/projects/neorag/data/hotpotqa/hotpot_dev_distractor_v1.json";
const OUT: &str = "/Users/vysakh/projects/neorag/exports/reasoning_qa_contexts.jsonl";
const SAMPLE_SIZE: usize = 600; // yields ~500 gap-qualified multi-hop cases
const MAX_CASES: usize = 400; // cap emitted cases (lab runs >=300)
const N_DISTRACTORS: usize = 8;
// Aggressive enough that the tax bites (retention 0.75 vs 0.83 at n=1327)
// while the filter still removes most junk.
const AGGRESSIVE_THRESHOLD: f32 = 0.20;
const LINK_MIN_JACCARD: f32 = 0.12;

fn terms(text: &str) -> HashSet<String> {
    text.unicode_words()
        .map(|w| w.to_lowercase())
        .filter(|w| w.chars().count() > 1)
        .collect()
}
fn grounding(q: &HashSet<String>, c: &HashSet<String>) -> f32 {
    if q.is_empty() {
        return 0.0;
    }
    q.intersection(c).count() as f32 / q.len() as f32
}

struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(s.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
}

fn as_results(chunks: &[Chunk]) -> Vec<RetrievalResult> {
    chunks
        .iter()
        .map(|c| RetrievalResult {
            chunk: c.clone(),
            score: Score { value: 1.0, method: RetrievalMethod::Lexical },
            breakdown: ScoreBreakdown::default(),
        })
        .collect()
}

fn join_ctx(chunks: &[Chunk]) -> String {
    chunks
        .iter()
        .enumerate()
        .map(|(i, c)| format!("[{}] {}", i + 1, c.text))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn build(query: &Query, set: &[Chunk], strategy: ContextStrategy) -> Vec<Chunk> {
    build_context(
        query,
        &as_results(set),
        &ContextConfig {
            token_budget: 100_000, // generous: isolate the FILTER decision, not budget
            strategy,
            distractor_min_grounding: AGGRESSIVE_THRESHOLD,
            link_min_jaccard: LINK_MIN_JACCARD,
            redundancy_max_cosine: 1.0,
        },
    )
    .chunks
}

fn main() -> anyhow::Result<()> {
    let mut dataset = HotpotQADataset::from_path(HOTPOTQA_PATH)?;
    dataset.examples.truncate(SAMPLE_SIZE);
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;

    let gold_answer: HashMap<String, String> = dataset
        .examples
        .iter()
        .map(|e| (e.question.clone(), e.answer.clone()))
        .collect();

    let corpus = dataset.to_labeled_corpus(&chunker, |_| None, default_regime)?;
    let chunks = chunker.chunk_batch(&corpus.docs)?;
    let by_id: HashMap<ChunkId, Chunk> =
        chunks.iter().map(|c| (c.id.clone(), c.clone())).collect();

    let mut rng = Lcg::new(0xC0FFEE);
    let mut out = String::new();
    let mut n = 0;
    // Aggregate retention bookkeeping (the reachability side, for the header).
    let mut sh_kept_filtered = 0usize;
    let mut sh_kept_reasoning = 0usize;

    for lq in &corpus.queries {
        if n >= MAX_CASES {
            break;
        }
        let Some(answer) = gold_answer.get(&lq.text) else { continue };
        if lq.gold_chunk_ids.len() < 2 {
            continue;
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
        gold.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let second = gold[0].clone();
        let max_first_g = gold[1..].iter().map(|x| x.2).fold(0.0f32, f32::max);
        // Only the gap regime: the second hop is genuinely less query-relevant.
        if second.2 >= max_first_g {
            continue;
        }

        let gold_chunks: Vec<Chunk> = gold.iter().map(|(_, c, _)| c.clone()).collect();
        let gold_docs: HashSet<String> =
            gold_chunks.iter().map(|c| c.source.clone()).collect();

        // Off-document distractor pool, shuffled deterministically.
        let mut pool: Vec<&Chunk> =
            chunks.iter().filter(|c| !gold_docs.contains(&c.source)).collect();
        for i in (1..pool.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            pool.swap(i, j);
        }
        let mut polluted: Vec<Chunk> = gold_chunks.clone();
        polluted.extend(pool.iter().take(N_DISTRACTORS).map(|c| (*c).clone()));
        // Shuffle so order carries no signal.
        for i in (1..polluted.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            polluted.swap(i, j);
        }

        let query = Query::new(&lq.text);
        let filtered = build(&query, &polluted, ContextStrategy::DistractorFiltered);
        let reasoning = build(&query, &polluted, ContextStrategy::ReasoningPreserving);

        let sh_in = |set: &[Chunk]| set.iter().any(|c| c.id == second.0);
        let sh_filtered = sh_in(&filtered);
        let sh_reasoning = sh_in(&reasoning);
        sh_kept_filtered += sh_filtered as usize;
        sh_kept_reasoning += sh_reasoning as usize;

        // FULL gold retention per strategy (refinement over the single-chunk
        // second-hop proxy). reasoning's kept-gold is always a superset of
        // filtered's (both keep above-threshold seeds; only reasoning rescues
        // below-threshold linked gold), so:
        //   RESCUED  := gold_kept_reasoning > gold_kept_filtered
        //   CONTROL  := gold_kept_reasoning == gold_kept_filtered
        let gold_ids: HashSet<ChunkId> = gold.iter().map(|(id, _, _)| id.clone()).collect();
        let gold_kept = |set: &[Chunk]| set.iter().filter(|c| gold_ids.contains(&c.id)).count();
        let gold_kept_filtered = gold_kept(&filtered);
        let gold_kept_reasoning = gold_kept(&reasoning);

        let rec = serde_json::json!({
            "id": lq.id,
            "question": lq.text,
            "gold_answer": answer,
            "ctx_gold_only": join_ctx(&gold_chunks),
            "ctx_polluted": join_ctx(&polluted),
            "ctx_filtered": join_ctx(&filtered),
            "ctx_reasoning": join_ctx(&reasoning),
            // single-chunk proxy labels (kept for comparison):
            "second_hop_in_filtered": sh_filtered,
            "second_hop_in_reasoning": sh_reasoning,
            // full-gold-retention labels (the refined mechanism signal):
            "n_gold": gold_chunks.len(),
            "gold_kept_filtered": gold_kept_filtered,
            "gold_kept_reasoning": gold_kept_reasoning,
            "n_filtered": filtered.len(),
            "n_reasoning": reasoning.len(),
        });
        out.push_str(&serde_json::to_string(&rec)?);
        out.push('\n');
        n += 1;
    }

    std::fs::write(OUT, out)?;
    println!("wrote {n} gap-qualified multi-hop cases → {OUT}");
    println!("  filter threshold (aggressive): {AGGRESSIVE_THRESHOLD}");
    println!(
        "  second-hop retention (reachability): filtered {}/{} ({:.0}%), reasoning {}/{} ({:.0}%)",
        sh_kept_filtered, n, sh_kept_filtered as f32 / n.max(1) as f32 * 100.0,
        sh_kept_reasoning, n, sh_kept_reasoning as f32 / n.max(1) as f32 * 100.0,
    );
    println!("next: python ../neorag/scripts/score_reasoning_qa.py --n 300");
    Ok(())
}
