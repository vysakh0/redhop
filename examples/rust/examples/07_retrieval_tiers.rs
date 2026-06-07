//! 07 · Retrieval tiers — lexical / hybrid / semantic on the same query.
//!
//! Real-world scenario:
//!     A B2C support team's FAQ uses the company's polite phrasings
//!     ("refund", "return") but customers ask in colloquial English
//!     ("send back", "money back"). The same five-line FAQ corpus, hit
//!     with three different retrieval tiers, shows where each one fails
//!     and where each one succeeds — the trade-off documented in
//!     docs/findings/SEMANTIC_MISMATCH.md.
//!
//! What this demonstrates:
//!     - The three retrieval tiers via `redhop::text(text, &options)`:
//!       `"lexical"` (BM25, default, no model), `"hybrid"` (BM25
//!       candidate pool + dense rerank, ~80MB model), `"semantic"`
//!       (global exact-cosine dense, ~80MB model).
//!     - That for a synonym-mismatch query, lexical can miss; hybrid
//!       is pool-dependent; semantic catches it.
//!
//! First-run note:
//!     `hybrid` and `semantic` need an embedding model. The first call
//!     downloads `bge-small` (~80MB) to your local model cache;
//!     subsequent runs are fast.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 07_retrieval_tiers --release

use std::time::Instant;

use redhop::{citations, text as load_text, LoadOptions};

const SUPPORT_FAQ: &str = "
Q: When will my package arrive?
A: Standard shipping takes 3-5 business days from when your order leaves our warehouse.

Q: How do I get my money back if I'm not satisfied?
A: We offer a full refund within 30 days of delivery. Return the item using the prepaid label.

Q: What's the warranty?
A: Our products have a one-year manufacturer warranty against defects.

Q: Can I cancel a subscription?
A: You can cancel anytime from Settings, no fee.

Q: Do you ship internationally?
A: Yes, we ship to 50 countries. Express international is 5-7 days.
";

const QUERY: &str = "how do I send back something I do not want?";

fn try_tier(label: &str, retrieval: &str, with_model: bool) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let opts = LoadOptions {
        chunk_size: Some(30),
        retrieval: Some(retrieval.to_string()),
        model: if with_model {
            Some("bge-small".to_string())
        } else {
            None
        },
        ..LoadOptions::default()
    };
    let mut doc = load_text(SUPPORT_FAQ, &opts)?;
    let ctx = doc.context(QUERY)?;
    let elapsed = t0.elapsed().as_secs_f64();
    let cites = citations(&ctx);
    let top = if let Some(c) = cites.first() {
        let s: String = c.text.chars().take(80).collect();
        format!("{:?}", s)
    } else {
        "(none)".to_string()
    };
    println!("  {:<10} build+query: {:>5.2}s", label, elapsed);
    println!("               top hit  : {}", top);
    println!();
    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("Query: {:?}", QUERY);
    println!("Gold (the right answer): \"How do I get my money back …\" /");
    println!("                         \"We offer a full refund within 30 days …\"\n");

    println!("─── Arm A · retrieval=\"lexical\" (BM25, default, no model) ─");
    try_tier("lexical", "lexical", false)?;

    println!("─── Arm B · retrieval=\"hybrid\" (BM25 pool + dense rerank) ─");
    try_tier("hybrid", "hybrid", true)?;

    println!("─── Arm C · retrieval=\"semantic\" (global exact-cosine dense) ─");
    try_tier("semantic", "semantic", true)?;

    println!("─── How to read this ─────────────────────────────");
    println!("On this tiny 5-chunk corpus, BM25's candidate pool happens");
    println!("to fit all 5 chunks, so `hybrid` finds the right answer too.");
    println!("On a real synonym-heavy corpus (HR FAQs, support tickets");
    println!("translated from internal phrasing, multilingual content),");
    println!("BM25's top-K will often exclude the synonym-mismatch answer");
    println!("entirely — and then hybrid can't recover it because it only");
    println!("reranks within BM25's pool. That's the regime where");
    println!("`semantic` (global, no pruning) earns its keep, at the cost");
    println!("of embedding every chunk per query (only practical on small");
    println!("to medium corpora).");
    println!();
    println!("Don't read this as 'always use semantic.' For most document");
    println!("QA — code, runbooks, contracts, financial filings — the");
    println!("question and answer DO share surface words, and lexical");
    println!("wins on latency.");
    println!("Decision tree: docs/CHOOSING_A_CONFIG.md.");
    println!("Mechanism + measurement: docs/findings/SEMANTIC_MISMATCH.md.");
    Ok(())
}
