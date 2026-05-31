//! Emit controlled-distractor contexts for downstream QA scoring.
//!
//! The RedHop (Rust) half of the end-to-end distractor experiment. For
//! each HotpotQA query we build, from KNOWN gold chunks plus injected
//! off-topic distractors (chunks from unrelated documents):
//!
//!   gold_only   : just the supporting chunks (clean ceiling)
//!   polluted_4  : gold + 4 off-topic distractors (shuffled)
//!   polluted_8  : gold + 8 off-topic distractors (shuffled)
//!   filtered_8  : polluted_8 run through distractor filtering
//!
//! This isolates the distractor effect causally (controlled count) and
//! tests the SAFE filter regime: the injected distractors are clearly
//! off-topic (near-zero query overlap), so absolute-threshold filtering
//! removes them while keeping the gold. No embeddings needed.
//!
//! The Python lab (../redhop/scripts/score_context_qa.py) then calls an
//! LLM on each context and scores answer quality. RedHop builds
//! contexts; the lab judges answers.
//!
//! Run:  cargo run -p redhop-examples --example emit_qa_contexts --release

use std::collections::HashMap;
use std::sync::Arc;

use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::context::{build_context, ContextConfig, ContextStrategy};
use redhop::core::{
    Chunk, ChunkId, Chunker, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenizerBackend,
};
use redhop_calibration::loaders::hotpotqa::{default_regime, HotpotQADataset};

const HOTPOTQA_PATH: &str =
    "/Users/vysakh/projects/neorag/data/hotpotqa/hotpot_dev_distractor_v1.json";
const OUT: &str = "/Users/vysakh/projects/neorag/exports/qa_contexts.jsonl";
const SAMPLE_SIZE: usize = 30;
const DISTRACTOR_GROUNDING: f32 = 0.10;

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

fn join_ctx(chunks: &[Chunk]) -> String {
    chunks
        .iter()
        .enumerate()
        .map(|(i, c)| format!("[{}] {}", i + 1, c.text))
        .collect::<Vec<_>>()
        .join("\n\n")
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
    let by_id: HashMap<ChunkId, Chunk> = chunks.iter().map(|c| (c.id.clone(), c.clone())).collect();
    // Map chunk id -> source doc, to pick OFF-document distractors.
    let chunk_source: HashMap<ChunkId, String> = chunks
        .iter()
        .map(|c| (c.id.clone(), c.source.clone()))
        .collect();

    let mut rng = Lcg::new(0xC0FFEE);
    let mut out = String::new();
    let mut n = 0;
    let mut filtered_removed_total = 0usize;
    let mut filtered_kept_gold = 0usize;
    let mut gold_total = 0usize;

    for lq in &corpus.queries {
        let Some(gold) = gold_answer.get(&lq.text) else {
            continue;
        };
        if lq.gold_chunk_ids.is_empty() {
            continue;
        }
        // Gold chunks (the supporting evidence).
        let gold_chunks: Vec<Chunk> = lq
            .gold_chunk_ids
            .iter()
            .filter_map(|id| by_id.get(id).cloned())
            .collect();
        if gold_chunks.is_empty() {
            continue;
        }
        let gold_docs: std::collections::HashSet<String> =
            gold_chunks.iter().map(|c| c.source.clone()).collect();

        // Distractor pool: chunks from OTHER documents (off-topic).
        let mut pool: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| {
                chunk_source
                    .get(&c.id)
                    .map(|s| !gold_docs.contains(s))
                    .unwrap_or(false)
            })
            .collect();
        // Shuffle the pool deterministically.
        for i in (1..pool.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            pool.swap(i, j);
        }
        let take = |k: usize| -> Vec<Chunk> { pool.iter().take(k).map(|c| (*c).clone()).collect() };

        let mk_polluted = |k: usize, rng: &mut Lcg| -> Vec<Chunk> {
            let mut v = gold_chunks.clone();
            v.extend(take(k));
            for i in (1..v.len()).rev() {
                let j = (rng.next() as usize) % (i + 1);
                v.swap(i, j);
            }
            v
        };

        let gold_only = gold_chunks.clone();
        let polluted_4 = mk_polluted(4, &mut rng);
        let polluted_8 = mk_polluted(8, &mut rng);

        // Distractor-filter polluted_8 (lexical, safe absolute threshold).
        let query = Query::new(&lq.text);
        let filtered = build_context(
            &query,
            &as_results(&polluted_8),
            &ContextConfig {
                token_budget: 100_000,
                strategy: ContextStrategy::DistractorFiltered,
                distractor_min_grounding: DISTRACTOR_GROUNDING,
                link_min_jaccard: 0.12,
                auto_passthrough_max_tokens: 8_000,
                redundancy_max_cosine: 1.0,
                low_confidence_max_grounding: 0.10,
            },
        );
        // Measure filter safety: did it keep the gold chunks?
        let kept_gold = lq
            .gold_chunk_ids
            .iter()
            .filter(|g| filtered.chunks.iter().any(|c| &c.id == *g))
            .count();
        filtered_kept_gold += kept_gold;
        gold_total += lq.gold_chunk_ids.len();
        filtered_removed_total += polluted_8.len() - filtered.chunks.len();

        let rec = serde_json::json!({
            "id": lq.id,
            "question": lq.text,
            "gold_answer": gold,
            "ctx_gold_only": join_ctx(&gold_only),
            "ctx_polluted_4": join_ctx(&polluted_4),
            "ctx_polluted_8": join_ctx(&polluted_8),
            "ctx_filtered_8": join_ctx(&filtered.chunks),
            "n_gold": gold_chunks.len(),
            "n_filtered_kept": filtered.chunks.len(),
            "filtered_kept_gold": kept_gold,
        });
        out.push_str(&serde_json::to_string(&rec)?);
        out.push('\n');
        n += 1;
    }
    std::fs::write(OUT, out)?;
    println!("wrote {n} query context-sets → {OUT}");
    println!(
        "filter safety check: kept {}/{} gold chunks ({:.0}%), removed {} injected distractors total",
        filtered_kept_gold,
        gold_total,
        filtered_kept_gold as f32 / gold_total.max(1) as f32 * 100.0,
        filtered_removed_total
    );
    println!("next: python ../redhop/scripts/score_context_qa.py");
    Ok(())
}
