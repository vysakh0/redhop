//! 02 · Structured corpus — `redhop::core::Chunk` for content you
//!      already chunked elsewhere, with metadata that flows through to
//!      citations.
//!
//! Real-world scenario:
//!     A SaaS company has a customer-support knowledge base: each FAQ
//!     pair is one row in a database (question, answer, category,
//!     last_updated, article_url). Support agents query it in natural
//!     language. They need citations that point back to a specific
//!     article + metadata visible alongside (category, last_updated).
//!
//!     The 0.3.0 typed `Chunk` constructor + open metadata is what
//!     makes this clean.
//!
//! What this demonstrates:
//!     - `redhop::core::Chunk::new(id, text, source, token_count)`
//!       and `.with_metadata(...)` — the typed primitives the
//!       bindings wrap.
//!     - **source vs id**: `source` is *provenance*; `id` is *identity*.
//!     - **Open metadata flows to citations**: known keys (`page`,
//!       `heading`, `line`) appear on `redhop::citations(&ctx)`.
//!       Arbitrary keys (`category`, `last_updated`) are preserved on
//!       the chunk but not surfaced through citations — keep a parallel
//!       index on your side for display.
//!     - `Document::from_chunks(chunks)` — no chunker re-split.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 02_structured_corpus --release

use std::collections::HashMap;

use redhop::core::{Chunk, ChunkId, TokenCount};
use redhop::{citations, Document};
use serde_json::json;

#[derive(Clone)]
struct FaqRow {
    id: &'static str,
    category: &'static str,
    question: &'static str,
    answer: &'static str,
    url: &'static str,
    last_updated: &'static str,
}

fn faq_rows() -> Vec<FaqRow> {
    vec![
        FaqRow { id: "faq-001", category: "billing",
            question: "When is my credit card charged?",
            answer: "Your card is charged on the first day of each billing cycle. You can view upcoming charges under Settings → Billing.",
            url: "https://help.acme.com/billing/charge-date", last_updated: "2026-04-12" },
        FaqRow { id: "faq-002", category: "billing",
            question: "How do I request a refund?",
            answer: "Refunds are available within 30 days of charge. Email finance@acme.com with your invoice number and reason. We process refunds within 5 business days.",
            url: "https://help.acme.com/billing/refunds", last_updated: "2026-05-03" },
        FaqRow { id: "faq-003", category: "account",
            question: "How do I change my email address?",
            answer: "Settings → Account → Email. We send a confirmation link to the new address; click it within 24 hours to complete the change.",
            url: "https://help.acme.com/account/email", last_updated: "2026-03-21" },
        FaqRow { id: "faq-004", category: "account",
            question: "How do I delete my account?",
            answer: "Settings → Account → Delete Account. We retain billing records for 7 years for tax compliance but anonymize all profile data immediately.",
            url: "https://help.acme.com/account/delete", last_updated: "2026-02-18" },
        FaqRow { id: "faq-005", category: "shipping",
            question: "When will my order arrive?",
            answer: "Standard shipping is 3-5 business days. Express is 1-2 days. You'll get a tracking link by email once the package leaves our warehouse.",
            url: "https://help.acme.com/shipping/delivery-time", last_updated: "2026-05-30" },
        FaqRow { id: "faq-006", category: "shipping",
            question: "Can I change my shipping address after ordering?",
            answer: "Yes, if the order hasn't shipped yet. Go to Orders → Edit. After shipment we cannot reroute — you'll need to contact the carrier directly.",
            url: "https://help.acme.com/shipping/change-address", last_updated: "2026-04-05" },
        FaqRow { id: "faq-007", category: "returns",
            question: "What is your return policy?",
            answer: "Unworn items in original packaging may be returned within 30 days of delivery for a full refund. Print a prepaid label from Orders → Return.",
            url: "https://help.acme.com/returns/policy", last_updated: "2026-05-15" },
        FaqRow { id: "faq-008", category: "returns",
            question: "Do you cover return shipping?",
            answer: "Yes — return shipping is free in the US for unworn items. International returns are paid by the customer.",
            url: "https://help.acme.com/returns/shipping-costs", last_updated: "2026-04-22" },
    ]
}

fn build_chunks(rows: &[FaqRow]) -> Vec<Chunk> {
    rows.iter()
        .map(|r| {
            let text = format!("Q: {}\nA: {}", r.question, r.answer);
            let token_count = text.split_whitespace().count();
            let mut metadata = HashMap::new();
            metadata.insert("category".to_string(), json!(r.category));
            metadata.insert("last_updated".to_string(), json!(r.last_updated));
            metadata.insert("heading".to_string(), json!(r.question));
            Chunk::new(ChunkId::new(r.id), text, r.url, TokenCount(token_count))
                .with_metadata(metadata)
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let rows = faq_rows();
    let chunks = build_chunks(&rows);
    let mut doc = Document::from_chunks(chunks)?;
    println!("Indexed {} FAQ entries.\n", doc.chunks().len());

    let query = "what's the deadline for getting a refund?";
    println!("Query: {:?}\n", query);

    let ctx = doc.context(query)?;

    println!("─── Top hit ───────────────────────────────────────");
    let cites = citations(&ctx);
    let cite = &cites[0];
    println!("  source        : {}", cite.source);
    println!("  heading       : {:?}", cite.heading);

    // `category` and `last_updated` aren't first-class citation fields,
    // but we attached them to the chunk's metadata. Look them up from
    // your parallel index — here, the FAQ row vector.
    if let Some(row) = rows.iter().find(|r| r.url == cite.source) {
        println!("  category      : {}", row.category);
        println!("  last_updated  : {}", row.last_updated);
    }
    let snippet: String = cite.text.chars().take(80).collect();
    println!("  text (excerpt): {}…", snippet);
    println!();

    println!("─── Decision Report ───────────────────────────────");
    println!("  Final context tokens : {}", ctx.report.total_tokens);
    println!(
        "  Decision             : {:?} (strategy={:?})",
        ctx.report.auto_decision(),
        ctx.report.strategy,
    );
    println!(
        "  Chunks selected      : {} of {}",
        ctx.report.n_selected, ctx.report.n_input_chunks
    );
    Ok(())
}
