//! 14 · Catalog search — short, noisy queries over a near-duplicate catalog.
//!
//! Real-world scenario:
//!     A corner-store ordering assistant takes short, messy product
//!     requests ("liberty root beer", "summit cola", and plenty of
//!     typos like "1iberty"). The catalog is a near-duplicate lattice:
//!     one brand has the same product at several sizes and prices that
//!     differ by a token or two. Three things break here that don't
//!     break on prose QA, and each has a lever.
//!
//! What this demonstrates:
//!     - `language="char_ngram"` — the subword typo tier. A typo
//!       ("1iberty") still matches via shared character n-grams, no
//!       model. Word-token BM25 scores it at zero.
//!     - `bm25_field_weights = [text, source, heading]` — per-field BM25
//!       boosts, a domain lever (default equal weight is unchanged).
//!     - `EvalGold::AllOf` -> `set_coverage` — a catalog query maps to a
//!       SET (all sizes); recall@k hides a half-retrieved family,
//!       set_coverage catches it.
//!
//!     Honest framing (docs/findings/CATALOG_REGIME.md): char-ngram is a
//!     recall booster, not a drop-in. Field weights help only when the
//!     boosted field separates the answer from its near-duplicates.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 14_catalog_search --release

use std::collections::HashMap;

use redhop::core::{Chunk, ChunkId, Query, TokenCount};
use redhop::{chunks_typed, evaluate, Document, EvalConfig, EvalGold, LoadOptions};
use serde_json::json;

// (sku id, brand+product key, full product line). A small American
// convenience-store catalog with near-duplicate size/price variants.
const CATALOG: &[(&str, &str, &str)] = &[
    ("summit-cola-12", "Summit Cola", "Summit Cola 12 oz 1.49"),
    ("summit-cola-20", "Summit Cola", "Summit Cola 20 oz 1.99"),
    ("summit-cola-2l", "Summit Cola", "Summit Cola 2 liter 2.49"),
    ("summit-diet-12", "Summit Diet Cola", "Summit Diet Cola 12 oz 1.49"),
    ("summit-diet-20", "Summit Diet Cola", "Summit Diet Cola 20 oz 1.99"),
    ("liberty-rb-12", "Liberty Root Beer", "Liberty Root Beer 12 oz 1.49"),
    ("liberty-rb-20", "Liberty Root Beer", "Liberty Root Beer 20 oz 1.99"),
    ("eagle-bbq-2", "Eagle Potato Chips", "Eagle Potato Chips BBQ 2 oz 1.29"),
    ("eagle-bbq-8", "Eagle Potato Chips", "Eagle Potato Chips BBQ 8 oz 3.49"),
    ("eagle-salt-2", "Eagle Potato Chips", "Eagle Potato Chips Salted 2 oz 1.29"),
    ("pioneer-jerky-3", "Pioneer Beef Jerky", "Pioneer Beef Jerky Original 3 oz 5.99"),
    ("coastal-mix-6", "Coastal Trail Mix", "Coastal Trail Mix 6 oz 4.29"),
];

fn build(language: Option<&str>, field_weights: Option<Vec<f32>>) -> anyhow::Result<Document> {
    let chunks: Vec<Chunk> = CATALOG
        .iter()
        .map(|(sku, heading, text)| {
            let mut metadata = HashMap::new();
            metadata.insert("heading".to_string(), json!(heading));
            Chunk::new(
                ChunkId::new(*sku),
                *text,
                "catalog",
                TokenCount(text.split_whitespace().count()),
            )
            .with_metadata(metadata)
        })
        .collect();
    let opts = LoadOptions {
        language: language.map(str::to_string),
        bm25_field_weights: field_weights,
        ..Default::default()
    };
    Ok(chunks_typed(chunks, &opts)?)
}

/// Distinct brand+product labels in the assembled context, in order.
fn products(ctx: &redhop::BuiltContext) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for c in &ctx.chunks {
        if let Some(h) = c.metadata.get("heading").and_then(|v| v.as_str()) {
            if !seen.iter().any(|s| s == h) {
                seen.push(h.to_string());
            }
        }
    }
    seen
}

fn main() -> anyhow::Result<()> {
    // ── 1. Transcription typo: char-ngram recovers what word-BM25 drops ──
    // A realistic noisy order: the brand is typo'd ("1iberty") AND the
    // product is run together ("rootbeer"), so word-BM25 has no exact token
    // to match. char-ngram bridges both via shared character n-grams.
    println!("1) Typo recovery — query: '1iberty rootbeer'\n");
    let mut word = build(Some("raw"), None)?; // default word-token analyzer
    let mut ngram = build(Some("char_ngram"), None)?; // subword typo tier
    let q = "1iberty rootbeer";
    println!("   word-BM25  found : {:?}", products(&word.context(q)?));
    let ngram_found = products(&ngram.context(q)?);
    println!("   char-ngram found : {:?}", ngram_found);
    println!(
        "   -> char-ngram recovered Liberty Root Beer despite the typo: {}\n",
        ngram_found.iter().any(|p| p == "Liberty Root Beer")
    );

    // ── 2. Per-field weighting is a knob (default = equal weight) ─────────
    println!("2) Field weights — boost the brand/product 'heading' field 2x\n");
    let mut boosted = build(Some("char_ngram"), Some(vec![1.0, 1.0, 2.0]))?;
    println!("   'summit cola' -> {:?}", products(&boosted.context("summit cola")?));
    println!("   (a domain lever: sweep on your own gold set, it is not a");
    println!("    guaranteed lift; see docs/findings/CATALOG_REGIME.md)\n");

    // ── 3. set_coverage: did we retrieve the WHOLE variant family? ───────
    println!("3) Set coverage — 'summit cola' should return ALL its sizes\n");
    let ctx = ngram.context("summit cola")?;
    let family: &[&[&str]] = &[&["summit-cola-12", "summit-cola-20", "summit-cola-2l"]];
    let r = evaluate(
        &Query::new("summit cola"),
        &ctx,
        None,
        EvalGold::AllOf(family),
        None,
        EvalConfig::default(),
    );
    println!("   products offered : {:?}", products(&ctx));
    println!(
        "   set_coverage     : {:?}   (1.0 = whole family offerable)",
        r.set_coverage
    );
    println!("   context_recall   : {:?}", r.context_recall);
    println!("   recall@k can read fine while a family is half-retrieved;");
    println!("   set_coverage is the metric a disambiguation UX should gate on.");
    Ok(())
}
