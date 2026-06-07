//! 08 · Structural expansion — neighbors=N and include_heading=true.
//!
//! Real-world scenario:
//!     A SaaS company's internal handbook is heavily structured: each
//!     policy section has a heading + multiple paragraphs. The BM25 top
//!     hit lands on the specific paragraph that answers the question —
//!     but for an LLM to write a grounded answer it usually wants the
//!     section heading (so it knows the topic) and the adjacent
//!     paragraphs (in case the answer spans them).
//!
//! What this demonstrates:
//!     - `doc.context_expanded(query, budget, candidate_k, neighbors,
//!       include_heading)` — same retrieval selection, padded with
//!       adjacent context and the section heading within the token
//!       budget.
//!     - `ctx.report.n_expanded` — how many extra chunks expansion added.
//!     - Why we constrain `candidate_k=Some(2)`: on this small corpus
//!       the budget would otherwise swallow everything and there'd be
//!       nothing to expand.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 08_structural_expansion --release

use redhop::{text as load_text, LoadOptions};

const HANDBOOK: &str = "
# PTO (Paid Time Off)

Full-time employees accrue 1.5 days of PTO per month, totaling 18 days per year.

PTO carries over up to a maximum of 30 days at the end of the calendar year. Beyond that, unused PTO is forfeited.

To request PTO, submit a request through Workday at least two weeks in advance. Manager approval is required.

# Sick Leave

Sick leave is separate from PTO. Employees may take up to 10 paid sick days per year for personal illness or family caregiving.

Sick days do not carry over and do not count against your PTO balance.

# Parental Leave

New parents are eligible for 16 weeks of paid parental leave following the birth or adoption of a child.

Leave can be taken continuously or split into two blocks of at least four weeks each, within the first 12 months.
";

fn show_arm(
    label: &str,
    query: &str,
    neighbors: usize,
    include_heading: bool,
) -> anyhow::Result<()> {
    let opts = LoadOptions {
        source: Some("handbook.md".into()),
        chunk_size: Some(20),
        candidate_k: Some(2),
        ..LoadOptions::default()
    };
    let mut doc = load_text(HANDBOOK, &opts)?;
    let ctx = doc.context_expanded(query, None, None, neighbors, include_heading)?;
    println!("─── {} ─────────────────────────", label);
    println!("  n_selected     : {}", ctx.report.n_selected);
    println!("  n_expanded     : {}", ctx.report.n_expanded);
    println!("  total_tokens   : {}", ctx.report.total_tokens);
    println!("  assembled context:");
    for line in ctx.text().split('\n') {
        println!("    {}", line);
    }
    println!();
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let query = "how many PTO days do I get?";
    println!("Query: {:?}\n", query);

    show_arm("Arm A · plain context (no expansion)", query, 0, false)?;
    show_arm("Arm B · include_heading=true", query, 0, true)?;
    show_arm("Arm C · neighbors=1", query, 1, false)?;
    show_arm(
        "Arm D · neighbors=1 + include_heading=true (recommended for handbooks)",
        query,
        1,
        true,
    )?;

    println!("─── When to use each ─────────────────────────────");
    println!("- include_heading=true : structured docs (handbooks,");
    println!("  contracts, runbooks) where the topic label matters.");
    println!("- neighbors=1          : when the answer often spans");
    println!("  adjacent chunks (a fact stated, then qualified).");
    println!("- both                 : the safe default for structured");
    println!("  document QA.");
    println!("- neither              : code search, transcripts,");
    println!("  high-density technical content where the hit IS the answer.");
    Ok(())
}
