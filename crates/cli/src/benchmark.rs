//! `redhop benchmark` — reproducible strategy sweep over a labeled dataset.
//!
//! Input (`--input labeled.json`):
//! ```json
//! {
//!   "dataset": "my_eval",
//!   "queries": [
//!     {"query": "...", "chunks": [{"id","text",...}],
//!      "gold_ids": ["c3","c7"], "second_hop_id": "c7"}
//!   ]
//! }
//! ```
//! Emits results.json (+ SUMMARY.md) with per-(strategy,budget) gold/second-hop
//! retention (95% bootstrap CIs) and telemetry. No fabricated metrics — every
//! number is computed from the provided labels. For the canonical hermetic
//! HotpotQA run see benchmarks/context/ (the bench_context_strategies example).

use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use clap::Args as ClapArgs;
use redhop::context::{build_context, ContextConfig};
use redhop::core::Query;
use serde::Deserialize;
use serde_json::json;

use crate::io::{parse_strategy, ChunkInput};

#[derive(ClapArgs)]
pub struct Args {
    /// Labeled dataset JSON.
    #[arg(long, short)]
    input: String,
    /// Comma-separated strategies.
    #[arg(
        long,
        default_value = "raw_topk,distractor_filtered,max_density,reasoning_preserving"
    )]
    strategies: String,
    /// Comma-separated token budgets.
    #[arg(long, default_value = "250,800,12000")]
    budgets: String,
    #[arg(long, default_value_t = 0.10)]
    distractor_min_grounding: f32,
    #[arg(long, default_value_t = 0.12)]
    link_min_jaccard: f32,
    /// Directory for results.json + SUMMARY.md.
    #[arg(long, default_value = "redhop_bench_out")]
    out_dir: String,
}

#[derive(Deserialize)]
struct LabeledQuery {
    query: String,
    chunks: Vec<ChunkInput>,
    #[serde(default)]
    gold_ids: Vec<String>,
    #[serde(default)]
    second_hop_id: Option<String>,
}

#[derive(Deserialize)]
struct Dataset {
    #[serde(default = "default_name")]
    dataset: String,
    queries: Vec<LabeledQuery>,
}
fn default_name() -> String {
    "labeled".into()
}

#[derive(Default)]
struct Cell {
    gold: Vec<f32>,
    second: Vec<f32>,
    density: Vec<f32>,
    tokens: Vec<f32>,
    rescue: Vec<f32>,
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}
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

pub fn run(a: Args) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(&a.input).with_context(|| format!("reading {}", a.input))?;
    let ds: Dataset = serde_json::from_str(&raw).context("parsing labeled dataset JSON")?;
    let strategies: Vec<&str> = a.strategies.split(',').map(str::trim).collect();
    let budgets: Vec<usize> = a
        .budgets
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .context("budget must be an integer")
        })
        .collect::<anyhow::Result<_>>()?;

    let n = ds.queries.len();
    let n_labeled = ds.queries.iter().filter(|q| !q.gold_ids.is_empty()).count();

    // (strategy, budget) -> Cell
    let mut cells: std::collections::HashMap<(&str, usize), Cell> =
        std::collections::HashMap::new();
    for q in &ds.queries {
        let query = Query::new(&q.query);
        let results: Vec<_> = crate::io::RetrievalInput {
            query: None,
            chunks: q.chunks.clone(),
        }
        .to_results();
        let gold: HashSet<&str> = q.gold_ids.iter().map(String::as_str).collect();
        for sname in &strategies {
            let strat = parse_strategy(sname)?;
            for &budget in &budgets {
                let cfg = ContextConfig {
                    token_budget: budget,
                    strategy: strat,
                    distractor_min_grounding: a.distractor_min_grounding,
                    link_min_jaccard: a.link_min_jaccard,
                    auto_passthrough_max_tokens: 8_000,
                    redundancy_max_cosine: 0.92,
                    low_confidence_max_grounding: 0.10,
                    analyzer: redhop::analyzer::default_english(),
            preserve_order: false,
                };
                let ctx = build_context(&query, &results, &cfg);
                let kept: HashSet<&str> = ctx.chunks.iter().map(|c| c.id.as_str()).collect();
                let cell = cells.entry((sname, budget)).or_default();
                if !gold.is_empty() {
                    let g = gold.iter().filter(|id| kept.contains(*id)).count() as f32
                        / gold.len() as f32;
                    cell.gold.push(g);
                }
                if let Some(h) = &q.second_hop_id {
                    cell.second
                        .push(if kept.contains(h.as_str()) { 1.0 } else { 0.0 });
                }
                cell.density.push(ctx.report.economics.evidence_density);
                cell.tokens.push(ctx.report.total_tokens as f32);
                cell.rescue.push(ctx.report.second_hop_rescue_count as f32);
            }
        }
    }

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut rng = Lcg(0x5EED);
    let mut json_rows = Vec::new();
    let mut md = format!("# Benchmark: {}\n\n", ds.dataset);
    md.push_str(&format!(
        "- queries: {n} (labeled with gold: {n_labeled})\n"
    ));
    md.push_str(&format!(
        "- distractor_min_grounding: {}, link_min_jaccard: {}\n\n",
        a.distractor_min_grounding, a.link_min_jaccard
    ));
    md.push_str("| strategy | budget | gold_ret [95% CI] | second_hop_ret [95% CI] | density | tokens | rescue |\n");
    md.push_str("| -------- | ------ | ----------------- | ----------------------- | ------- | ------ | ------ |\n");

    for sname in &strategies {
        for &budget in &budgets {
            let Some(c) = cells.get(&(sname, budget)) else {
                continue;
            };
            let (g, glo, ghi) = mean_ci(&c.gold, &mut rng);
            let (s, slo, shi) = mean_ci(&c.second, &mut rng);
            let dens = mean(&c.density);
            let toks = mean(&c.tokens);
            let resc = mean(&c.rescue);
            let gold_cell = if c.gold.is_empty() {
                "-".to_string()
            } else {
                format!("{g:.3} [{glo:.3},{ghi:.3}]")
            };
            let sec_cell = if c.second.is_empty() {
                "-".to_string()
            } else {
                format!("{s:.3} [{slo:.3},{shi:.3}]")
            };
            md.push_str(&format!("| {sname} | {budget} | {gold_cell} | {sec_cell} | {dens:.3} | {toks:.0} | {resc:.2} |\n"));
            json_rows.push(json!({
                "strategy": sname, "budget": budget, "n": c.density.len(),
                "gold_ret": if c.gold.is_empty() { Value::Null } else { json!(g) },
                "gold_ret_ci": if c.gold.is_empty() { Value::Null } else { json!([glo, ghi]) },
                "second_hop_ret": if c.second.is_empty() { Value::Null } else { json!(s) },
                "second_hop_ret_ci": if c.second.is_empty() { Value::Null } else { json!([slo, shi]) },
                "evidence_density": dens, "mean_tokens": toks, "mean_second_hop_rescue": resc,
            }));
        }
    }

    let out = json!({
        "benchmark": ds.dataset,
        "metadata": {
            "generated_unix": ts, "n_queries": n, "n_labeled": n_labeled,
            "distractor_min_grounding": a.distractor_min_grounding,
            "link_min_jaccard": a.link_min_jaccard, "budgets": budgets, "hermetic": true,
        },
        "results": json_rows,
    });

    std::fs::create_dir_all(&a.out_dir).with_context(|| format!("creating {}", a.out_dir))?;
    std::fs::write(
        format!("{}/results.json", a.out_dir),
        serde_json::to_string_pretty(&out)?,
    )?;
    std::fs::write(format!("{}/SUMMARY.md", a.out_dir), &md)?;
    println!("wrote {}/results.json and SUMMARY.md", a.out_dir);
    println!(
        "  {n} queries ({n_labeled} gold-labeled), {} strategies × {} budgets",
        strategies.len(),
        budgets.len()
    );
    Ok(())
}

use serde_json::Value;
