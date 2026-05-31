//! Emit contexts for the DILUTION test — the experiment that decides whether
//! build_context is an accuracy product at modern (large) context windows.
//!
//! The reasoning-QA test (emit_reasoning_qa) ran at a generous budget where
//! nothing had to be cut, and found pruning ≈ no-op on accuracy. But that
//! regime is rigged: when everything fits, any filter can only lose info.
//!
//! The real question for a 1M-window world is DILUTION: "fits" ≠ "used well".
//! A model with a huge window still degrades when you actually fill it with
//! mostly-junk (lost-in-the-middle). If pruning a bloated-but-fitting context
//! back to the load-bearing evidence RECOVERS accuracy, build_context has a
//! real accuracy home that has nothing to do with hard budget limits.
//!
//! Per gap-qualified multi-hop HotpotQA query we build, from the SAME large
//! polluted pool (gold + MANY off-document distractors):
//!
//!   ctx_gold_only : the supporting gold only (clean ceiling)
//!   ctx_polluted  : gold + N_DISTRACTORS distractors, ALL of it, unpruned
//!                   (the "stuff it all in the big window" baseline)
//!   ctx_pruned    : polluted → ReasoningPreserving, pruned to PRUNE_BUDGET
//!                   (bridge-aware: drop junk, keep seeds + linked second hop)
//!   ctx_topk      : polluted → MaxDensity, truncated to PRUNE_BUDGET
//!                   (naive relevance truncation — drops the low-relevance hop)
//!
//! Decisive comparisons:
//!   pruned − polluted : does pruning the bloated context RECOVER accuracy?
//!                       (> 0 ⇒ dilution is real ⇒ the optimizer earns its keep)
//!   pruned − topk     : does bridge-aware pruning beat naive truncation at
//!                       the same budget? (the second-hop-tax, under real cuts)
//!
//! Env knobs: REDHOP_N_DISTRACTORS (default 400 ≈ ~20k-token polluted),
//! REDHOP_PRUNE_BUDGET (default 2000 tokens), REDHOP_MAX_CASES (default 200).
//!
//! Run:  cargo run -p redhop-examples --example emit_dilution --release

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::context::{build_context, grounding_score, ContextConfig, ContextStrategy};
use redhop::core::{
    Chunk, ChunkId, Chunker, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenizerBackend,
};
use redhop_calibration::loaders::hotpotqa::{default_regime, HotpotQADataset};

const SAMPLE_SIZE: usize = 1000; // big enough for a large distractor pool
const LINK_MIN_JACCARD: f32 = 0.12;
// Distractor grounding bar for pruning (same as the library default regime).
const DISTRACTOR_TAU: f32 = 0.10;

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

fn tokens_of(chunks: &[Chunk]) -> usize {
    chunks.iter().map(|c| c.token_count.value().max(1)).sum()
}

fn build(query: &Query, set: &[Chunk], strategy: ContextStrategy, budget: usize) -> Vec<Chunk> {
    build_context(
        query,
        &as_results(set),
        &ContextConfig {
            token_budget: budget,
            strategy,
            distractor_min_grounding: DISTRACTOR_TAU,
            link_min_jaccard: LINK_MIN_JACCARD,
            auto_passthrough_max_tokens: 8_000,
            redundancy_max_cosine: 1.0,
            low_confidence_max_grounding: 0.10,
        },
    )
    .chunks
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> anyhow::Result<()> {
    let n_distractors = env_usize("REDHOP_N_DISTRACTORS", 400);
    let prune_budget = env_usize("REDHOP_PRUNE_BUDGET", 2000);
    let max_cases = env_usize("REDHOP_MAX_CASES", 200);

    let mut dataset = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
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

    let mut rng = Lcg::new(0xD1107E0F);
    let mut out = String::new();
    let mut n = 0;
    let mut sum_polluted_tok = 0usize;
    let mut sum_pruned_tok = 0usize;
    let mut sum_topk_tok = 0usize;
    let mut gold_kept_pruned_tot = 0usize;
    let mut gold_kept_topk_tot = 0usize;
    let mut gold_tot = 0usize;

    for lq in &corpus.queries {
        if n >= max_cases {
            break;
        }
        let Some(answer) = gold_answer.get(&lq.text) else {
            continue;
        };
        if lq.gold_chunk_ids.len() < 2 {
            continue;
        }
        let mut gold: Vec<(ChunkId, Chunk, f32)> = lq
            .gold_chunk_ids
            .iter()
            .filter_map(|id| by_id.get(id).map(|c| (id.clone(), c.clone())))
            .map(|(id, c)| {
                let g = grounding_score(&lq.text, &c.text);
                (id, c, g)
            })
            .collect();
        if gold.len() < 2 {
            continue;
        }
        gold.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let second = gold[0].clone();
        let max_first_g = gold[1..].iter().map(|x| x.2).fold(0.0f32, f32::max);
        // Gap regime only: the second hop is genuinely less query-relevant —
        // exactly the chunk naive truncation drops and the bridge rescues.
        if second.2 >= max_first_g {
            continue;
        }

        let gold_chunks: Vec<Chunk> = gold.iter().map(|(_, c, _)| c.clone()).collect();
        let gold_docs: HashSet<String> = gold_chunks.iter().map(|c| c.source.clone()).collect();

        // Large off-document distractor pool, shuffled deterministically.
        let mut pool: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| !gold_docs.contains(&c.source))
            .collect();
        for i in (1..pool.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            pool.swap(i, j);
        }
        if pool.len() < n_distractors {
            continue; // not enough distractors to build the diluted context
        }
        let mut polluted: Vec<Chunk> = gold_chunks.clone();
        polluted.extend(pool.iter().take(n_distractors).map(|c| (*c).clone()));
        // Shuffle so position carries no signal (gold scattered through the bloat).
        for i in (1..polluted.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            polluted.swap(i, j);
        }

        let query = Query::new(&lq.text);
        let pruned = build(
            &query,
            &polluted,
            ContextStrategy::ReasoningPreserving,
            prune_budget,
        );
        let topk = build(&query, &polluted, ContextStrategy::MaxDensity, prune_budget);

        let gold_ids: HashSet<ChunkId> = gold.iter().map(|(id, _, _)| id.clone()).collect();
        let gold_kept = |set: &[Chunk]| set.iter().filter(|c| gold_ids.contains(&c.id)).count();
        let gk_pruned = gold_kept(&pruned);
        let gk_topk = gold_kept(&topk);
        let sh_in = |set: &[Chunk]| set.iter().any(|c| c.id == second.0);

        sum_polluted_tok += tokens_of(&polluted);
        sum_pruned_tok += tokens_of(&pruned);
        sum_topk_tok += tokens_of(&topk);
        gold_kept_pruned_tot += gk_pruned;
        gold_kept_topk_tot += gk_topk;
        gold_tot += gold_chunks.len();

        let rec = serde_json::json!({
            "id": lq.id,
            "question": lq.text,
            "gold_answer": answer,
            "ctx_gold_only": join_ctx(&gold_chunks),
            "ctx_polluted": join_ctx(&polluted),
            "ctx_pruned": join_ctx(&pruned),
            "ctx_topk": join_ctx(&topk),
            "n_gold": gold_chunks.len(),
            "polluted_tokens": tokens_of(&polluted),
            "pruned_tokens": tokens_of(&pruned),
            "topk_tokens": tokens_of(&topk),
            "n_polluted": polluted.len(),
            "n_pruned": pruned.len(),
            "n_topk": topk.len(),
            "gold_kept_pruned": gk_pruned,
            "gold_kept_topk": gk_topk,
            "second_hop_in_pruned": sh_in(&pruned),
            "second_hop_in_topk": sh_in(&topk),
        });
        out.push_str(&serde_json::to_string(&rec)?);
        out.push('\n');
        n += 1;
    }

    let out_name = format!("dilution_contexts_d{n_distractors}.jsonl");
    let out_path = redhop_examples::exports_path(&out_name);
    if let Some(dir) = out_path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&out_path, out)?;
    let avg = |s: usize| s as f32 / n.max(1) as f32;
    println!(
        "wrote {n} gap-qualified multi-hop cases → {}",
        out_path.display()
    );
    println!("  config: n_distractors={n_distractors}, prune_budget={prune_budget} tok");
    println!(
        "  avg tokens/ctx:  polluted {:.0}  →  pruned {:.0}  /  topk {:.0}",
        avg(sum_polluted_tok),
        avg(sum_pruned_tok),
        avg(sum_topk_tok)
    );
    println!(
        "  gold retention:  pruned {}/{} ({:.0}%)  vs  topk {}/{} ({:.0}%)",
        gold_kept_pruned_tot,
        gold_tot,
        gold_kept_pruned_tot as f32 / gold_tot.max(1) as f32 * 100.0,
        gold_kept_topk_tot,
        gold_tot,
        gold_kept_topk_tot as f32 / gold_tot.max(1) as f32 * 100.0,
    );
    println!("next: python python/eval/score_dilution.py --n {n} --model <id>");
    Ok(())
}
