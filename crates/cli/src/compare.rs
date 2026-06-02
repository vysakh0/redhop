//! `redhop compare` — side-by-side strategy comparison + context economics.
//! The strongest demo surface: shows reasoning-preserving optimization live.

use std::collections::HashSet;

use anyhow::Context as _;
use clap::Args as ClapArgs;
use redhop::context::{build_context, ContextConfig};
use redhop::core::Query;
use serde_json::json;

use crate::io::{parse_strategy, RetrievalInput};

#[derive(ClapArgs)]
pub struct Args {
    /// The user query (overrides any "query" in the input file).
    #[arg(long)]
    query: Option<String>,
    /// Retrieval JSON (`{query?, chunks:[...]}`), or `-` for stdin.
    #[arg(long, short)]
    input: String,
    /// Comma-separated strategies to compare.
    #[arg(
        long,
        default_value = "raw_topk,distractor_filtered,reasoning_preserving"
    )]
    strategies: String,
    /// Token budget.
    #[arg(long, default_value_t = 12000)]
    budget: usize,
    #[arg(long, default_value_t = 0.10)]
    distractor_min_grounding: f32,
    #[arg(long, default_value_t = 0.12)]
    link_min_jaccard: f32,
    /// Optional comma-separated gold chunk ids → enables retention columns.
    #[arg(long)]
    gold_ids: Option<String>,
    /// Optional id of the reasoning-critical second hop → enables a 2nd-hop column.
    #[arg(long)]
    second_hop_id: Option<String>,
    /// Characters of each assembled context to preview (0 = no preview).
    #[arg(long, default_value_t = 200)]
    preview_chars: usize,
    /// Write structured results to this JSON path.
    #[arg(long)]
    json: Option<String>,
}

pub fn run(a: Args) -> anyhow::Result<()> {
    let input = RetrievalInput::load(&a.input)?;
    let query_text = a
        .query
        .clone()
        .or_else(|| input.query.clone())
        .context("no query: pass --query or include \"query\" in the input file")?;
    let query = Query::new(&query_text);
    let retrieved = input.to_results();

    let strategies: Vec<&str> = a.strategies.split(',').map(|s| s.trim()).collect();
    let gold: Option<HashSet<String>> = a
        .gold_ids
        .as_ref()
        .map(|g| g.split(',').map(|s| s.trim().to_string()).collect());

    println!("Query: {query_text}");
    println!(
        "Retrieved: {} chunks · budget {}\n",
        retrieved.len(),
        a.budget
    );

    // Table header (retention columns only when gold is provided).
    let show_gold = gold.is_some();
    let show_hop = a.second_hop_id.is_some();
    let mut header = vec![
        "strategy", "chunks", "tokens", "removed", "rescued", "distr", "density",
    ];
    if show_gold {
        header.push("gold_ret");
    }
    if show_hop {
        header.push("2nd_hop");
    }
    let widths = [22usize, 9, 7, 7, 7, 6, 7, 8, 8];
    print_row(&header, &widths);
    println!(
        "{}",
        "─".repeat(header.iter().zip(widths).map(|(_, w)| w + 2).sum::<usize>())
    );

    let mut json_rows = Vec::new();
    let mut previews = Vec::new();
    for sname in &strategies {
        let strat = parse_strategy(sname)?;
        let cfg = ContextConfig {
            token_budget: a.budget,
            strategy: strat,
            distractor_min_grounding: a.distractor_min_grounding,
            link_min_jaccard: a.link_min_jaccard,
            auto_passthrough_max_tokens: 8_000,
            redundancy_max_cosine: 0.92,
            low_confidence_max_grounding: 0.10,
            analyzer: redhop::analyzer::default_english(),
        };
        let ctx = build_context(&query, &retrieved, &cfg);
        let r = &ctx.report;
        let kept: HashSet<&str> = ctx.chunks.iter().map(|c| c.id.as_str()).collect();

        let gold_ret = gold.as_ref().map(|g| {
            if g.is_empty() {
                return 1.0;
            }
            g.iter().filter(|id| kept.contains(id.as_str())).count() as f32 / g.len() as f32
        });
        let hop_kept = a.second_hop_id.as_ref().map(|h| kept.contains(h.as_str()));

        let mut row = vec![
            sname.to_string(),
            format!("{}→{}", r.n_input_chunks, r.n_selected),
            r.total_tokens.to_string(),
            r.removed.total.to_string(),
            r.second_hop_rescue_count.to_string(),
            format!("{:.2}", r.economics.distractor_ratio),
            format!("{:.2}", r.economics.evidence_density),
        ];
        if let Some(g) = gold_ret {
            row.push(format!("{g:.2}"));
        }
        if let Some(h) = hop_kept {
            row.push(if h { "✓".into() } else { "✗".into() });
        }
        print_row(&row.iter().map(|s| s.as_str()).collect::<Vec<_>>(), &widths);

        let mut obj = json!({
            "strategy": sname,
            "report": r,
        });
        if let Some(g) = gold_ret {
            obj["gold_retention"] = json!(g);
        }
        if let Some(h) = hop_kept {
            obj["second_hop_retained"] = json!(h);
        }
        json_rows.push(obj);

        if a.preview_chars > 0 {
            let text = ctx.text();
            let prev: String = text.chars().take(a.preview_chars).collect();
            let ell = if text.chars().count() > a.preview_chars {
                "…"
            } else {
                ""
            };
            previews.push((sname.to_string(), format!("{prev}{ell}")));
        }
    }

    if !previews.is_empty() {
        println!("\n── assembled context previews ──");
        for (s, p) in &previews {
            println!("\n[{s}]\n{p}");
        }
    }

    if let Some(path) = &a.json {
        let out = json!({
            "command": "compare",
            "query": query_text,
            "budget": a.budget,
            "n_input_chunks": retrieved.len(),
            "results": json_rows,
        });
        std::fs::write(path, serde_json::to_string_pretty(&out)?)
            .with_context(|| format!("writing {path}"))?;
        println!("\nwrote {path}");
    }
    Ok(())
}

fn print_row(cells: &[&str], widths: &[usize]) {
    let line: Vec<String> = cells
        .iter()
        .zip(widths)
        .map(|(c, w)| format!("{:<width$}", c, width = w))
        .collect();
    println!("{}", line.join("  "));
}
