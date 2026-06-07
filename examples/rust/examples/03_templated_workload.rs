//! 03 · Templated workload — detect → strip → vocabulary → audit trail.
//!
//! Real-world scenario:
//!     A legal-ops team uses a fixed query template across hundreds of
//!     contracts. Each query is shaped like
//!         Highlight the parts (if any) of this contract related to "<X>"
//!         that should be reviewed by a lawyer. Details: <…>
//!     where only <X> varies. The boilerplate words dilute BM25's signal
//!     on the discriminating clause name, costing retention on the
//!     framework comparison (CUAD: 81% raw → 88% stripped → 90.7%
//!     stripped + clause-synonyms).
//!
//! What this demonstrates:
//!     - `redhop::analyze_query_set(...)` — flags whether a query set is
//!       templated and which words are doing the dilution.
//!     - `redhop::Stripper::new(boilerplate)` — compiled token-level
//!       boilerplate removal.
//!     - `redhop::Vocabulary::new(&[(key, &[syns])])` — compiled
//!       equivalence classes.
//!     - `doc.context_with_rewrites(query, &[&stripper, &vocab])` —
//!       runs the chain through retrieval; each stage's
//!       `RewriteRecord` lands on `ctx.report.query_rewrites`.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 03_templated_workload --release

use redhop::{analyze_query_set, Document, Stripper, Vocabulary};

const CONTRACT: &str = "
SECTION 7. CHANGE OF CONTROL

In the event of a Change of Control of either party, including any
merger, consolidation, or sale of substantially all assets, the
non-acquired party shall have the right to terminate this Agreement on
thirty days' written notice.

SECTION 8. NON-COMPETE

During the Term and for two years thereafter, the Distributor shall
not, directly or indirectly, engage in any business competitive with
the Company within the Territory.

SECTION 9. INDEMNIFICATION

Each party shall indemnify and hold harmless the other from any third-
party claims arising from the indemnifying party's gross negligence or
willful misconduct.

SECTION 10. CONFIDENTIALITY

Each party shall keep confidential all non-public information disclosed
by the other party in connection with this Agreement.
";

fn sample_queries() -> Vec<&'static str> {
    vec![
        "Highlight the parts (if any) of this contract related to \"Change of Control\" that should be reviewed by a lawyer.",
        "Highlight the parts (if any) of this contract related to \"Non-Compete\" that should be reviewed by a lawyer.",
        "Highlight the parts (if any) of this contract related to \"Indemnification\" that should be reviewed by a lawyer.",
        "Highlight the parts (if any) of this contract related to \"Confidentiality\" that should be reviewed by a lawyer.",
        "Highlight the parts (if any) of this contract related to \"Termination\" that should be reviewed by a lawyer.",
    ]
}

fn clause_synonyms() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            "change of control",
            &["merger", "consolidation", "acquisition", "successor"][..],
        ),
        ("non-compete", &["restraint", "compete", "competitive"]),
        (
            "indemnification",
            &["indemnify", "hold harmless", "third-party claims"],
        ),
        (
            "confidentiality",
            &["confidential", "non-disclosure", "non-public"],
        ),
        ("termination", &["terminate", "expire", "end"]),
    ]
}

fn main() -> anyhow::Result<()> {
    // ── Step 1: Detect ──────────────────────────────────────────────
    println!("─── Step 1 · Detect the template ────────────────");
    let queries: Vec<String> = sample_queries().iter().map(|s| s.to_string()).collect();
    let report = analyze_query_set(&queries);
    println!("  is_templated            : {}", report.is_templated);
    println!(
        "  template_word_share     : {:.2}",
        report.template_word_share
    );
    println!(
        "  estimated_dilution_cost : {:?}",
        report.estimated_dilution_cost
    );
    println!("  boilerplate_terms       : {:?}", report.boilerplate_terms);
    println!("  suggested_action        : {}", report.suggested_action);
    println!();
    if !report.is_templated {
        println!("(Template not detected — skip the rewrite chain.)");
        return Ok(());
    }

    // ── Step 2: Compile the rewrites ───────────────────────────────
    let stripper = Stripper::new(&report.boilerplate_terms);
    let vocabulary = Vocabulary::new(&clause_synonyms());
    println!("─── Step 2 · Compile the rewrites ───────────────");
    println!("  Stripper: {} boilerplate forms", stripper.len());
    println!("  Vocabulary: {} clause classes", vocabulary.len());
    println!();

    // ── Step 3: Run a query through the chain ──────────────────────
    println!("─── Step 3 · Run a query through the chain ──────");
    let mut doc = Document::from_text("msa.txt", CONTRACT)?;
    let query = sample_queries()[0];
    println!("  raw query: {:?}\n", query);

    let ctx = doc.context_with_rewrites(query, &[&stripper, &vocabulary])?;

    // The per-stage audit trail.
    println!("  query_rewrites audit trail:");
    for rec in &ctx.report.query_rewrites {
        println!("    [{}]", rec.stage);
        println!("      from   : {:?}", rec.from);
        println!("      to     : {:?}", rec.to);
        println!("      matched: {:?}", rec.matched);
        println!("      added  : {:?}", rec.added);
        println!("      removed: {:?}", rec.removed);
    }
    println!();

    let top = &ctx.chunks[0];
    let snippet: String = top.text.chars().take(80).collect();
    println!("  Top citation source : {}", top.source);
    println!("  Top citation text   : {}…", snippet.replace('\n', " "));
    println!();
    println!(
        "  Decision: {:?} / {:?}",
        ctx.report.auto_decision(),
        ctx.report.strategy
    );
    Ok(())
}
