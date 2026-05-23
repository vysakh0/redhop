//! `redhop analyze-context` — non-destructive observability for one context.
//! Reuses `ContextReport::render()`. For retrieval inspection / ops visibility.

use anyhow::Context as _;
use clap::Args as ClapArgs;
use redhop_context::{analyze_context, ContextConfig};
use redhop_core::Query;

use crate::io::RetrievalInput;

#[derive(ClapArgs)]
pub struct Args {
    /// Context JSON (`{query?, chunks:[...]}`), or `-` for stdin.
    input: String,
    /// Query (overrides any "query" in the file).
    #[arg(long)]
    query: Option<String>,
    #[arg(long, default_value_t = 0.10)]
    distractor_min_grounding: f32,
    #[arg(long, default_value_t = 0.12)]
    link_min_jaccard: f32,
    /// Emit the raw ContextReport JSON instead of the rendered report.
    #[arg(long)]
    json: bool,
}

pub fn run(a: Args) -> anyhow::Result<()> {
    let input = RetrievalInput::load(&a.input)?;
    let query_text = a
        .query
        .clone()
        .or_else(|| input.query.clone())
        .context("no query: pass --query or include \"query\" in the input file")?;
    let cfg = ContextConfig {
        token_budget: usize::MAX, // analysis is budget-free
        distractor_min_grounding: a.distractor_min_grounding,
        link_min_jaccard: a.link_min_jaccard,
        auto_passthrough_max_tokens: 8_000,
        ..Default::default()
    };
    let report = analyze_context(&Query::new(&query_text), &input.to_results(), &cfg);

    if a.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", report.render(None));
    }
    Ok(())
}
