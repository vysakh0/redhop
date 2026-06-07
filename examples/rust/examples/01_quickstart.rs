//! 01 · Quickstart — load a document, ask a question, read the Decision Report.
//!
//! Real-world scenario:
//!     A contract analyst has a Master Services Agreement (MSA) and needs
//!     to answer "what's the governing law?" before handing the snippet
//!     to an LLM for summarization. They want a citation back to the
//!     clause and a reason RedHop chose those chunks.
//!
//! What this demonstrates:
//!     - The 3-call surface: `Document::from_text(source, text)`,
//!       `doc.context(query)`, `ctx.text() / ctx.chunks / ctx.report`.
//!     - The Decision Report explaining what RedHop did and why.
//!     - That for a small document, the runtime *deliberately* leaves the
//!       context untouched (the "Auto → passthrough" decision).
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 01_quickstart --release

use redhop::{citations, Document};

/// A short Master Services Agreement excerpt. In production this would
/// be `Document::read_file("msa.pdf")?` (requires the `files`
/// feature, on by default in this examples crate), but for a
/// self-contained demo we paste the text inline.
const MSA: &str = "
SECTION 8. CONFIDENTIALITY

Each party shall keep confidential all non-public information disclosed by
the other party in connection with this Agreement. The receiving party
shall not use such information for any purpose other than performance of
this Agreement.

SECTION 9. GOVERNING LAW AND JURISDICTION

This Agreement shall be governed by and construed in accordance with the
laws of the State of New York, without regard to its conflict-of-laws
principles. The parties consent to the exclusive jurisdiction of the
state and federal courts located in New York County, New York.

SECTION 10. ENTIRE AGREEMENT

This Agreement constitutes the entire agreement between the parties and
supersedes all prior negotiations, representations, and agreements,
whether written or oral, with respect to its subject matter.

SECTION 11. NOTICES

Any notice required under this Agreement shall be in writing and
delivered to the address set forth on the signature page.
";

fn main() -> anyhow::Result<()> {
    // 1. Load. `from_text` runs the default sentence chunker and indexes
    //    every chunk with BM25 — no model download, no vector DB.
    let mut doc = Document::from_text("acme_msa.txt", MSA)?;
    println!("Indexed {} chunks from acme_msa.txt\n", doc.chunks().len());

    // 2. Ask. RedHop retrieves, scores, and budgets the prompt all
    //    in-process. The BuiltContext carries the assembled text, the
    //    selected chunks (for citations), and the Decision Report.
    let ctx = doc.context("what's the governing law?")?;

    // 3. Hand the assembled text to whatever LLM you use — RedHop has
    //    no LLM lock-in. For the demo we just print it.
    println!("─── Prompt (ctx.text()) ────────────────────────────");
    println!("{}", ctx.text());
    println!();

    // Citations: where did each kept chunk come from? `redhop::citations`
    // walks ctx.chunks and surfaces the citation-shaped view
    // (source / page / heading / line).
    println!("─── Citations ──────────────────────────────────────");
    for c in citations(&ctx) {
        let snippet: String = c.text.chars().take(80).collect();
        println!("  source={:?} text={:?}…", c.source, snippet);
    }
    println!();

    // The Decision Report explains the runtime's choice. For a small
    // document like this, the size gate fires and the context is passed
    // through untouched — pruning small contexts is wash-to-harmful per
    // docs/findings/CONTEXT_DILUTION.md.
    println!("─── Decision Report ────────────────────────────────");
    println!("  strategy           : {:?}", ctx.report.strategy);
    println!("  auto decision      : {:?}", ctx.report.auto_decision());
    println!("  input chunks       : {}", ctx.report.n_input_chunks);
    println!("  selected chunks    : {}", ctx.report.n_selected);
    println!("  total tokens       : {}", ctx.report.total_tokens);
    println!(
        "  retained evidence  : {:.2}",
        ctx.report.retained_evidence_ratio
    );
    println!();
    println!("(For a human-readable version, call ctx.report.render(None))");
    Ok(())
}
