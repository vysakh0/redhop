//! 05 · Deterministic A/B with `redhop::evaluate(...)` — no LLM judge.
//!
//! Real-world scenario:
//!     The legal-ops team from 03_templated_workload.rs wants to know
//!     whether adding clause-name synonyms (the Vocabulary step)
//!     actually lifts retrieval on *their* contracts, not on a
//!     published benchmark. They have a small gold set: for each query
//!     they labeled which chunk id(s) should appear in the assembled
//!     context. They compare two arms — baseline vs strip + vocab —
//!     and `evaluate` returns context_recall, context_precision, and
//!     a composite `overall` for each. Deterministic across runs, no
//!     LLM API key, no money spent.
//!
//! What this demonstrates:
//!     - `redhop::evaluate(&query, &ctx, gold)` with `EvalGold::Chunks`.
//!     - How to plug it into an A/B comparing two retrieval arms.
//!     - "Refraction not independent measurement" — same primitives the
//!       runtime uses (docs/findings/EVALUATE_API.md).
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 05_evaluate_ab --release

use std::collections::HashMap;

use redhop::core::{Chunk, ChunkId, Query, TokenCount};
use redhop::{evaluate, Document, EvalGold, Stripper, Vocabulary};
use serde_json::json;

struct Section {
    id: &'static str,
    heading: &'static str,
    text: &'static str,
}

fn sections() -> Vec<Section> {
    vec![
        Section { id: "sec-7", heading: "Change of Control",
            text: "SECTION 7. CHANGE OF CONTROL. In the event of a Change of Control of either party, including any merger, consolidation, or sale of substantially all assets, the non-acquired party shall have the right to terminate this Agreement on thirty days' written notice." },
        Section { id: "sec-8", heading: "Non-Compete",
            text: "SECTION 8. NON-COMPETE. During the Term and for two years thereafter, the Distributor shall not, directly or indirectly, engage in any business competitive with the Company within the Territory." },
        Section { id: "sec-9", heading: "Indemnification",
            text: "SECTION 9. INDEMNIFICATION. Each party shall indemnify and hold harmless the other from any third-party claims arising from the indemnifying party's gross negligence or willful misconduct." },
        Section { id: "sec-10", heading: "Confidentiality",
            text: "SECTION 10. CONFIDENTIALITY. Each party shall keep confidential all non-public information disclosed by the other party in connection with this Agreement." },
        Section { id: "sec-11", heading: "Termination",
            text: "SECTION 11. TERMINATION. Either party may terminate this Agreement upon thirty days' written notice in the event of a material breach by the other party." },
        Section { id: "sec-12", heading: "Notices",
            text: "SECTION 12. NOTICES. Any notice required under this Agreement shall be in writing and delivered to the address set forth on the signature page." },
    ]
}

fn gold_queries() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        ("Highlight the parts (if any) of this contract related to \"Change of Control\" that should be reviewed by a lawyer.", &["sec-7"][..]),
        ("Highlight the parts (if any) of this contract related to \"Non-Compete\" that should be reviewed by a lawyer.", &["sec-8"]),
        ("Highlight the parts (if any) of this contract related to \"Indemnification\" that should be reviewed by a lawyer.", &["sec-9"]),
        ("Highlight the parts (if any) of this contract related to \"Confidentiality\" that should be reviewed by a lawyer.", &["sec-10"]),
        ("Highlight the parts (if any) of this contract related to \"Termination\" that should be reviewed by a lawyer.", &["sec-11"]),
    ]
}

fn clause_synonyms() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            "change of control",
            &["merger", "consolidation", "acquisition"][..],
        ),
        ("non-compete", &["restraint", "compete", "competitive"]),
        ("indemnification", &["indemnify", "hold harmless"]),
        ("confidentiality", &["confidential", "non-disclosure"]),
        ("termination", &["terminate", "expire", "end"]),
    ]
}

fn build_doc() -> anyhow::Result<Document> {
    let chunks: Vec<Chunk> = sections()
        .iter()
        .map(|s| {
            let token_count = s.text.split_whitespace().count();
            let mut metadata = HashMap::new();
            metadata.insert("heading".to_string(), json!(s.heading));
            Chunk::new(
                ChunkId::new(s.id),
                s.text,
                "msa.txt",
                TokenCount(token_count),
            )
            .with_metadata(metadata)
        })
        .collect();
    Ok(Document::from_chunks(chunks)?)
}

fn evaluate_arm(label: &str, use_rewrites: bool) -> anyhow::Result<f32> {
    let mut doc = build_doc()?;
    let boilerplate: Vec<&str> = vec![
        "highlight",
        "the",
        "parts",
        "if",
        "any",
        "of",
        "this",
        "contract",
        "related",
        "to",
        "that",
        "should",
        "be",
        "reviewed",
        "by",
        "a",
        "lawyer",
    ];
    let stripper = Stripper::new(&boilerplate);
    let vocab = Vocabulary::new(&clause_synonyms());

    println!("─── arm {} ─────────────────────────────────", label);
    let mut total_recall = 0.0f32;
    let mut total_precision = 0.0f32;
    let mut total_overall = 0.0f32;
    let mut n = 0u32;
    for (query, gold_ids) in gold_queries() {
        let ctx = if use_rewrites {
            doc.context_with_rewrites(query, &[&stripper, &vocab])?
        } else {
            doc.context(query)?
        };
        let q = Query::new(query);
        let r = evaluate(&q, &ctx, EvalGold::Chunks(gold_ids));
        n += 1;
        total_recall += r.context_recall.unwrap_or(0.0);
        total_precision += r.context_precision.unwrap_or(0.0);
        total_overall += r.overall;
        if n == 1 {
            let preview: String = query.chars().take(60).collect();
            println!("  example query  : {}…", preview);
            println!(
                "  context_recall : {:.2}  context_precision : {:.2}  overall : {:.2}",
                r.context_recall.unwrap_or(0.0),
                r.context_precision.unwrap_or(0.0),
                r.overall,
            );
        }
    }
    let n_f = n as f32;
    let mean_overall = total_overall / n_f;
    println!(
        "  mean over {} queries: recall={:.2}  precision={:.2}  overall={:.2}",
        n,
        total_recall / n_f,
        total_precision / n_f,
        mean_overall,
    );
    println!();
    Ok(mean_overall)
}

fn main() -> anyhow::Result<()> {
    println!("Comparing two retrieval arms on the same gold set.\n");
    let a = evaluate_arm("A · baseline (no rewrites)", false)?;
    let b = evaluate_arm("B · stripped + clause-name vocabulary", true)?;
    let delta = b - a;
    println!("─── Verdict ──────────────────────────────────────");
    println!(
        "  ΔB−A on `overall`: {}{:.2}",
        if delta >= 0.0 { "+" } else { "" },
        delta
    );
    if b > a + 0.02 {
        println!("  ✓ The rewrite chain lifted retrieval on this gold set.");
    } else if b < a - 0.02 {
        println!("  ✗ The rewrite chain regressed retrieval. Inspect the audit");
        println!("    trail (ctx.report.query_rewrites) — likely the vocab is");
        println!("    appending workload-pervasive terms (the CUAD_PRF_NULL");
        println!("    failure mode).");
    } else {
        println!("  ~ Within sample noise — re-run with a larger gold set.");
    }
    Ok(())
}
