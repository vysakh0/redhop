//! First-class benchmark: context strategies head-to-head.
//!
//! Drives the four production strategies through the public
//! `build_context` API and scores them with labels (gold retention,
//! second-hop retention) + the `ContextReport` telemetry, swept across
//! token budgets and two populations:
//!
//!   multihop_gap  — multi-hop HotpotQA where the second hop is genuinely
//!                   less query-relevant than the first (the tax regime)
//!   shallow_nogap — multi-hop questions whose hops are similarly relevant
//!                   (a single-hop-like proxy within the same dataset)
//!
//! Hermetic: no LLM, no embeddings — deterministic and CI-friendly. Dense
//! retrieval and cross-generator axes are covered by the onnx/LLM harnesses
//! (see benchmarks/README.md); this one isolates the assembly strategy.
//!
//! Writes reproducible artifacts to benchmarks/context/:
//!   results.json  — metadata + per-(population,strategy,budget) metrics + CIs
//!   SUMMARY.md    — human-readable tables
//!
//! Run:  cargo run -p redhop-examples --example bench_context_strategies --release

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use redhop_calibration::loaders::hotpotqa::{default_regime, HotpotQADataset};
use redhop_chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop_context::{build_context, ContextConfig, ContextStrategy};
use redhop_core::{
    Chunk, ChunkId, Chunker, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenizerBackend,
};
use unicode_segmentation::UnicodeSegmentation;

const SAMPLE_SIZE: usize = 1500;
const N_DISTRACTORS: usize = 8;
const DISTRACTOR_MIN_GROUNDING: f32 = 0.20;
const LINK_MIN_JACCARD: f32 = 0.12;
const BUDGETS: [usize; 5] = [80, 150, 250, 400, 800];

const STRATEGIES: [(&str, ContextStrategy); 4] = [
    ("raw_topk", ContextStrategy::RawTopK),
    ("distractor_filtered", ContextStrategy::DistractorFiltered),
    ("max_density", ContextStrategy::MaxDensity),
    ("reasoning_preserving", ContextStrategy::ReasoningPreserving),
];

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

struct Case {
    query: Query,
    gold_ids: HashSet<ChunkId>,
    second_hop_id: ChunkId,
    retrieved: Vec<RetrievalResult>,
    is_gap: bool,
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}

/// 95% bootstrap CI of a mean.
fn mean_ci(xs: &[f32], rng: &mut Lcg) -> (f32, f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let m = mean(xs);
    let mut ms = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let mut s = 0.0;
        for _ in 0..xs.len() {
            s += xs[(rng.next() as usize) % xs.len()];
        }
        ms.push(s / xs.len() as f32);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (m, ms[24], ms[974])
}

#[derive(Default)]
struct Cell {
    gold_ret: Vec<f32>,
    second_ret: Vec<f32>,
    density: Vec<f32>,
    out_distractor: Vec<f32>,
    utilization: Vec<f32>,
    tokens: Vec<f32>,
    removed: Vec<f32>,
    rescue: Vec<f32>,
}

fn main() -> anyhow::Result<()> {
    let mut dataset = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
    dataset.examples.truncate(SAMPLE_SIZE);
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;
    let corpus = dataset.to_labeled_corpus(&chunker, |_| None, default_regime)?;
    let chunks = chunker.chunk_batch(&corpus.docs)?;
    let by_id: HashMap<ChunkId, Chunk> = chunks.iter().map(|c| (c.id.clone(), c.clone())).collect();

    let mut rng = Lcg::new(0xC0FFEE);
    let mut cases: Vec<Case> = Vec::new();
    for lq in &corpus.queries {
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
        let is_gap = second.2 < max_first_g;

        let gold_chunks: Vec<Chunk> = gold.iter().map(|(_, c, _)| c.clone()).collect();
        let gold_ids: HashSet<ChunkId> = gold.iter().map(|(id, _, _)| id.clone()).collect();
        let gold_docs: HashSet<String> = gold_chunks.iter().map(|c| c.source.clone()).collect();
        let mut pool: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| !gold_docs.contains(&c.source))
            .collect();
        for i in (1..pool.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            pool.swap(i, j);
        }
        let mut all: Vec<Chunk> = gold_chunks
            .iter()
            .cloned()
            .chain(pool.iter().take(N_DISTRACTORS).map(|c| (*c).clone()))
            .collect();
        let g_of = |c: &Chunk| grounding(&q_terms, &terms(&c.text));
        all.sort_by(|a, b| g_of(b).partial_cmp(&g_of(a)).unwrap());

        cases.push(Case {
            query: Query::new(&lq.text),
            gold_ids,
            second_hop_id: second.0.clone(),
            retrieved: as_results(&all),
            is_gap,
        });
    }

    let n_gap = cases.iter().filter(|c| c.is_gap).count();
    let n_nogap = cases.len() - n_gap;

    // population -> strategy -> budget -> Cell
    let mut cells: HashMap<(&str, &str, usize), Cell> = HashMap::new();
    for case in &cases {
        let pop = if case.is_gap {
            "multihop_gap"
        } else {
            "shallow_nogap"
        };
        for (sname, strat) in STRATEGIES {
            for budget in BUDGETS {
                let cfg = ContextConfig {
                    token_budget: budget,
                    strategy: strat,
                    distractor_min_grounding: DISTRACTOR_MIN_GROUNDING,
                    link_min_jaccard: LINK_MIN_JACCARD,
                    auto_passthrough_max_tokens: 8_000,
                    redundancy_max_cosine: 1.0,
                };
                let ctx = build_context(&case.query, &case.retrieved, &cfg);
                let kept: HashSet<&ChunkId> = ctx.chunks.iter().map(|c| &c.id).collect();
                let gold_in = case.gold_ids.iter().filter(|g| kept.contains(g)).count();
                let cell = cells.entry((pop, sname, budget)).or_default();
                cell.gold_ret
                    .push(gold_in as f32 / case.gold_ids.len() as f32);
                cell.second_ret
                    .push(if kept.contains(&&case.second_hop_id) {
                        1.0
                    } else {
                        0.0
                    });
                cell.density.push(ctx.report.economics.evidence_density);
                cell.out_distractor
                    .push(ctx.report.economics.distractor_ratio);
                cell.utilization.push(ctx.report.token_utilization);
                cell.tokens.push(ctx.report.total_tokens as f32);
                cell.removed.push(ctx.report.removed.total as f32);
                cell.rescue.push(ctx.report.second_hop_rescue_count as f32);
            }
        }
    }

    // ── emit JSON + markdown ──
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut rng = Lcg::new(0x5EED);
    let mut json = String::new();
    json.push_str("{\n");
    writeln!(json, "  \"benchmark\": \"context_strategies\",")?;
    writeln!(json, "  \"metadata\": {{")?;
    writeln!(json, "    \"generated_unix\": {ts},")?;
    writeln!(json, "    \"dataset\": \"hotpotqa_dev_distractor\",")?;
    writeln!(json, "    \"sample_size\": {SAMPLE_SIZE},")?;
    writeln!(json, "    \"n_multihop_gap\": {n_gap},")?;
    writeln!(json, "    \"n_shallow_nogap\": {n_nogap},")?;
    writeln!(json, "    \"n_distractors\": {N_DISTRACTORS},")?;
    writeln!(
        json,
        "    \"distractor_min_grounding\": {DISTRACTOR_MIN_GROUNDING},"
    )?;
    writeln!(json, "    \"link_min_jaccard\": {LINK_MIN_JACCARD},")?;
    writeln!(json, "    \"budgets\": {BUDGETS:?},")?;
    writeln!(json, "    \"hermetic\": true")?;
    writeln!(json, "  }},")?;
    writeln!(json, "  \"results\": [")?;

    let mut md = String::new();
    writeln!(md, "# Benchmark: context strategies\n")?;
    writeln!(
        md,
        "Hermetic (no LLM/embeddings), HotpotQA dev. Generated from"
    )?;
    writeln!(
        md,
        "`cargo run -p redhop-examples --example bench_context_strategies --release`.\n"
    )?;
    writeln!(
        md,
        "- multihop_gap: n={n_gap}  ·  shallow_nogap: n={n_nogap}"
    )?;
    writeln!(md, "- distractors/query={N_DISTRACTORS}, distractor_min_grounding={DISTRACTOR_MIN_GROUNDING}, link_min_jaccard={LINK_MIN_JACCARD}\n")?;
    writeln!(
        md,
        "second_hop_ret and gold_ret are means with 95% bootstrap CIs.\n"
    )?;

    let mut first = true;
    for pop in ["multihop_gap", "shallow_nogap"] {
        writeln!(md, "## {pop}\n")?;
        writeln!(md, "| strategy | budget | gold_ret [95% CI] | second_hop_ret [95% CI] | density | out_distr | util | tokens | rescue |")?;
        writeln!(md, "| -------- | ------ | ----------------- | ----------------------- | ------- | --------- | ---- | ------ | ------ |")?;
        for (sname, _) in STRATEGIES {
            for budget in BUDGETS {
                let Some(cell) = cells.get(&(pop, sname, budget)) else {
                    continue;
                };
                let (g, glo, ghi) = mean_ci(&cell.gold_ret, &mut rng);
                let (s, slo, shi) = mean_ci(&cell.second_ret, &mut rng);
                let dens = mean(&cell.density);
                let od = mean(&cell.out_distractor);
                let util = mean(&cell.utilization);
                let toks = mean(&cell.tokens);
                let resc = mean(&cell.rescue);
                writeln!(md, "| {sname} | {budget} | {g:.3} [{glo:.3},{ghi:.3}] | {s:.3} [{slo:.3},{shi:.3}] | {dens:.3} | {od:.3} | {util:.3} | {toks:.0} | {resc:.2} |")?;

                if !first {
                    json.push_str(",\n");
                }
                first = false;
                write!(json, "    {{\"population\":\"{pop}\",\"strategy\":\"{sname}\",\"budget\":{budget},\"n\":{},", cell.gold_ret.len())?;
                write!(
                    json,
                    "\"gold_ret\":{g:.4},\"gold_ret_ci\":[{glo:.4},{ghi:.4}],"
                )?;
                write!(
                    json,
                    "\"second_hop_ret\":{s:.4},\"second_hop_ret_ci\":[{slo:.4},{shi:.4}],"
                )?;
                write!(
                    json,
                    "\"evidence_density\":{dens:.4},\"output_distractor_ratio\":{od:.4},"
                )?;
                write!(json, "\"token_utilization\":{util:.4},\"mean_tokens\":{toks:.2},\"mean_second_hop_rescue\":{resc:.4}}}")?;
            }
        }
        writeln!(md)?;
    }
    json.push_str("\n  ]\n}\n");

    let out_dir = format!("{}/../../benchmarks/context", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(&out_dir)?;
    std::fs::write(format!("{out_dir}/results.json"), &json)?;
    std::fs::write(format!("{out_dir}/SUMMARY.md"), &md)?;
    println!("wrote {out_dir}/results.json and SUMMARY.md");
    println!("  multihop_gap n={n_gap}, shallow_nogap n={n_nogap}");
    Ok(())
}
