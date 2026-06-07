//! 04 · Chunk-side enrich — `vocab.enrich(chunk_text)` at ingest.
//!
//! Real-world scenario:
//!     A platform engineering team maintains a runbook keyed by short
//!     error codes (`ERR_4012`, `EVT_CHRGBCK`, `DB_5001`). When alerts
//!     fire, on-call engineers search the runbook in natural language
//!     ("payment declined", "checkout broken", "database timeout") —
//!     almost never by the code itself.
//!
//! ⚠ Honest framing (read before applying to your corpus):
//!     Enrich is shipped as a primitive on **mechanism reasoning with
//!     asymmetric measured evidence**:
//!       - Measured negative: CUAD prose chunks regressed −2.0pt
//!         (docs/findings/CUAD_ENRICH_DEFINITIONS_NULL.md).
//!       - Measured positive: none on RedHop's eval rigs yet.
//!     This example shows the mechanism on short opaque coded units
//!     (the regime where it's *predicted* to help) — but it is a
//!     synthetic demo with a hand-crafted dictionary, not a benchmark.
//!     **Always A/B with `redhop::evaluate(...)` against your gold
//!     set before adopting in production.** See 05_evaluate_ab.rs.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 04_chunk_enrich --release

use std::collections::HashMap;

use redhop::core::{Chunk, ChunkId, TokenCount};
use redhop::{citations, Document, Vocabulary};

struct RunbookEntry {
    code: &'static str,
    title: &'static str,
    body: &'static str,
}

fn runbook() -> Vec<RunbookEntry> {
    vec![
        RunbookEntry {
            code: "ERR_4012",
            title: "ERR_4012: PAYMENT_GATEWAY_DECLINED",
            body: "Stripe returned a 4012. Check the customer's card. Common causes: insufficient funds, expired card, blocked transaction. Retry strategy: exponential backoff with a max of 3 attempts.",
        },
        RunbookEntry {
            code: "ERR_5001",
            title: "ERR_5001: DB_CONNECTION_TIMEOUT",
            body: "The Postgres pool exhausted. Check `pg_stat_activity` for long-running queries. Restart the worker if connections aren't returning to the pool.",
        },
        RunbookEntry {
            code: "EVT_CHRGBCK",
            title: "EVT_CHRGBCK: chargeback notification",
            body: "Stripe sent a chargeback webhook. Flag the order, freeze the customer's account pending review. Respond to Stripe within 7 days with evidence.",
        },
        RunbookEntry {
            code: "ERR_6201",
            title: "ERR_6201: SHIPPING_LABEL_INVALID",
            body: "ShipStation rejected the label. Check the customer's address validity. Re-print the label after the address is corrected.",
        },
        RunbookEntry {
            code: "ERR_7301",
            title: "ERR_7301: EMAIL_DELIVERY_FAILED",
            body: "SendGrid bounced. Check the recipient's domain status. Most common cause: customer mistyped their email at signup.",
        },
    ]
}

/// Workload-specific decoder dictionary — the user supplies this.
/// Each key gets *term-specific* synonyms only. Generic words ("error",
/// "alert", "system") would re-create the CUAD_PRF_NULL low-IDF
/// dilution failure mode on the chunk side.
fn error_code_vocab() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            "ERR_4012",
            &["payment", "card", "charge", "stripe declined"][..],
        ),
        (
            "ERR_5001",
            &["database", "postgres", "timeout", "connection pool"],
        ),
        ("EVT_CHRGBCK", &["chargeback", "dispute", "refund request"]),
        ("ERR_6201", &["shipping", "label", "address", "delivery"]),
        ("ERR_7301", &["email", "bounce", "deliverability"]),
    ]
}

fn main() -> anyhow::Result<()> {
    let vocab = Vocabulary::new(&error_code_vocab());
    println!("Compiled vocabulary with {} classes\n", vocab.len());

    // ── Step 1: Enrich each chunk at ingest ─────────────────────────
    println!("─── Step 1 · Enrich chunks at ingest ────────────");
    let entries = runbook();
    let mut chunks: Vec<Chunk> = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        let chunk_text = format!("{}\n{}", entry.title, entry.body);
        // vocab.enrich(text) returns RewriteResult { query, record }.
        // The record carries the audit trail — what was matched, what
        // was added — so you can log it at ingest time.
        let result = vocab.enrich(&chunk_text);
        if !result.record.matched.is_empty() {
            println!(
                "  {:>14} ← matched={:?} added={:?}",
                entry.code, result.record.matched, result.record.added
            );
        }
        let token_count = result.query.split_whitespace().count();
        let mut metadata = HashMap::new();
        metadata.insert("heading".to_string(), serde_json::json!(entry.title));
        let _ = i;
        let chunk = Chunk::new(
            ChunkId::new(entry.code),
            result.query,
            format!("runbook/{}.md", entry.code),
            TokenCount(token_count),
        )
        .with_metadata(metadata);
        chunks.push(chunk);
    }
    println!();

    // ── Step 2: Build the document and run a natural-language query ─
    let mut doc = Document::from_chunks(chunks)?;
    let query = "customer's card got declined at checkout, what do we do?";
    println!("─── Step 2 · Query (natural language) ───────────");
    println!("  {:?}\n", query);

    let ctx = doc.context(query)?;
    let cites = citations(&ctx);
    let top = &cites[0];
    let excerpt: String = top.text.chars().take(100).collect();
    println!("─── Top hit ────────────────────────────────────");
    println!("  source : {}", top.source);
    println!("  heading: {:?}", top.heading);
    println!("  excerpt: {}…", excerpt.replace('\n', " "));
    println!();

    println!("Mechanism: the query has no overlap with the bare error code");
    println!("`ERR_4012` — the match landed via the appended `payment` / `card` /");
    println!("`charge` tokens that enrich attached at ingest. On your real");
    println!("runbook, A/B with `redhop::evaluate(...)` against a gold set (see");
    println!("05_evaluate_ab.rs) before committing to this in production.");
    Ok(())
}
