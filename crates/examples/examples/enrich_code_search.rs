//! Code-symbol retrieval with `Vocabulary::enrich(...)` — chunks are
//! short and opaque (function signatures with cryptic identifiers), the
//! query is natural language, and a hand-curated symbol→meaning
//! dictionary lifts the answer chunk from "doesn't surface at all" to
//! "top hit."
//!
//! The use case: a legacy codebase where function names are abbreviated
//! (`usrSvc`, `calcAmt`, `chkInv`) and the developer's question is
//! ordinary English ("monthly billing run"). Lexical retrieval has
//! nothing to match — the query and the bare symbol `calcAmt` share
//! zero surface forms — so the right chunk never makes it into the
//! BM25 top-K. Enrichment fixes this by appending each symbol's
//! plain-language synonyms to its chunk at ingest time, raising the
//! chunk's matchable surface area without changing how it's authored.
//!
//! ## The mechanism (and why it's not the same as query-side expand)
//!
//! Query-side [`redhop::Vocabulary::apply`] patches gaps the *author of
//! the query rewriter* anticipated — you enumerate which queries get
//! which synonyms. Chunk-side `enrich` does the opposite: you describe
//! the *content* once, and any future query that uses a paraphrase of
//! that description finds it. Different jobs:
//!
//! - **expand:** known query reformulations. Surgical, cheap, narrow.
//! - **enrich:** raise the content's semantic floor for queries you
//!   can't predict. Broader, ingest-time cost, conditional on having a
//!   decoding dictionary.
//!
//! ## When this earns its keep
//!
//! `value ∝ shortness × opacity × (dictionary exists)`. Function
//! signatures are an extreme case. Other extreme cases: SQL schema
//! columns (`emp_compensation`, `ord_dt`), error codes (`ERR_4012`),
//! defined terms in contracts, clinical abbreviations (`MI`, `SOB`).
//! On long descriptive prose chunks the operation is *redundant* —
//! matching already works. See [`VOCABULARY_ENRICH`].
//!
//! ## Two arms
//!
//! - **A — raw chunks (no enrichment).** The query
//!   `"monthly billing run"` has zero overlap with `calcAmt`'s bare
//!   signature; BM25 doesn't surface it.
//! - **B — enriched chunks.** Each function chunk gets its plain-language
//!   synonyms appended at ingest. The same query now finds `calcAmt`
//!   on its enriched `"calculate amount billing total line-item sum"`
//!   tokens.
//!
//! Selection is observably different between the two arms — that's the
//! point. We print both top hits so the contrast is visible at a glance.
//!
//! Run: cargo run -p redhop-examples --example enrich_code_search --release
//!
//! [`VOCABULARY_ENRICH`]: ../../../docs/findings/VOCABULARY_ENRICH.md

use redhop::core::{Chunk, TokenCount};
use redhop::{Document, Vocabulary};

/// Build a `Chunk` from a short id + text. Source is the same for all
/// chunks in this toy example (one logical "file").
fn chunk(id: &str, text: &str) -> Chunk {
    Chunk::new(
        id,
        text,
        "codebase.rs",
        TokenCount(text.split_whitespace().count()),
    )
}

/// Hand-rolled toy code corpus — short, opaque function signatures
/// with cryptic identifiers. Each chunk is one function-definition
/// "card."
fn code_chunks() -> Vec<String> {
    // Deliberately *no* doc comments — the failure mode this example
    // demonstrates is bare-symbol legacy code where the names are the
    // only surface text. If you have descriptive comments already,
    // enrich is redundant (one of the failure modes documented in
    // VOCABULARY_ENRICH.md).
    vec![
        "fn usrSvc(req: UserReq) -> Resp { /* … */ }",
        "fn chkInv(id: OrderId) -> Inventory { /* … */ }",
        "fn calcAmt(items: &[Item]) -> Cents { /* … */ }",
        "fn chargeCust(cust: &Cust, amt: Cents) -> Result<Receipt> { /* … */ }",
        "fn shipOrd(ord: &Order) -> Tracking { /* … */ }",
        "fn dbInit() -> Db { /* … */ }",
        "fn parseCfg(path: &Path) -> Config { /* … */ }",
        "fn rndId() -> Id { /* … */ }",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

/// Hand-curated symbol → plain-language synonyms. **This is the
/// workload-specific data the library does not ship.** A real codebase
/// would build this from doc comments / function summaries / external
/// docs; here we hand-write it for the eight functions above.
///
/// Note the structural rule: each value list is *term-specific* signal,
/// not generic boilerplate. Bolting the same description ("a function")
/// onto every chunk would re-create CUAD_PRF_NULL's low-IDF dilution —
/// the symmetric failure mode on the chunk side.
fn code_dictionary() -> Vocabulary {
    Vocabulary::new(&[
        (
            "usrSvc",
            &["user service", "signup", "account creation"][..],
        ),
        (
            "chkInv",
            &["check inventory", "warehouse stock", "availability"],
        ),
        (
            "calcAmt",
            &["calculate amount", "billing total", "line-item sum"],
        ),
        (
            "chargeCust",
            &[
                "charge customer",
                "payment",
                "bill customer",
                "process payment",
            ],
        ),
        ("shipOrd", &["ship order", "delivery", "fulfillment"]),
        ("dbInit", &["database connection", "open db", "pool setup"]),
        (
            "parseCfg",
            &["parse configuration", "load YAML", "read config"],
        ),
        ("rndId", &["random identifier", "generate id", "uuid"]),
    ])
}

/// Pretty-print the top-K hits with their inline rank.
fn print_hits(label: &str, hits: &[String]) {
    println!("    {label}:");
    if hits.is_empty() {
        println!("      (no hits)");
        return;
    }
    for (i, c) in hits.iter().enumerate() {
        let one_line = c.replace('\n', " ");
        let trimmed = if one_line.len() > 110 {
            format!("{}…", &one_line[..110])
        } else {
            one_line
        };
        println!("      {}. {trimmed}", i + 1);
    }
}

fn main() -> anyhow::Result<()> {
    // Pick a query whose words have no surface overlap with any bare
    // symbol — so the contrast between Arm A (no enrichment) and
    // Arm B (enriched) is observable on the tiny toy corpus.
    // "monthly billing run" should match `calcAmt` only via the
    // enriched synonyms ("billing total", "calculate amount").
    let query = "monthly billing run";
    let vocab = code_dictionary();

    println!("# Vocabulary.enrich() — code-symbol retrieval");
    println!();
    println!("Query: {query:?}");
    println!("Corpus: 8 cryptic Rust function signatures (one per chunk)");
    println!();

    // ── Arm A: raw chunks, no enrichment ─────────────────────────────
    let raw = code_chunks();
    let raw_chunks: Vec<Chunk> = raw
        .iter()
        .enumerate()
        .map(|(i, t)| chunk(&format!("fn-{i:02}"), t))
        .collect();
    let mut doc_a = Document::from_chunks(raw_chunks)?;
    let ctx_a = doc_a.context_with(query, Some(512), None)?;
    let hits_a: Vec<String> = ctx_a.chunks.iter().map(|c| c.text.clone()).collect();

    println!("Arm A — raw chunks (no enrichment)");
    print_hits("top hits", &hits_a);
    let found_a = hits_a.iter().any(|h| h.contains("calcAmt"));
    println!(
        "    calcAmt in top hits? {}",
        if found_a { "yes" } else { "no" }
    );
    println!();

    // ── Arm B: enriched chunks ───────────────────────────────────────
    let enriched: Vec<Chunk> = raw
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let r = vocab.enrich(c);
            if !r.record.matched.is_empty() {
                println!(
                    "    enriched: matched={:?} added={:?}",
                    r.record.matched, r.record.added
                );
            }
            chunk(&format!("fn-{i:02}"), &r.query)
        })
        .collect();
    println!();
    let mut doc_b = Document::from_chunks(enriched)?;
    let ctx_b = doc_b.context_with(query, Some(512), None)?;
    let hits_b: Vec<String> = ctx_b.chunks.iter().map(|c| c.text.clone()).collect();

    println!("Arm B — enriched chunks");
    print_hits("top hits", &hits_b);
    let found_b = hits_b.iter().any(|h| h.contains("calcAmt"));
    println!(
        "    calcAmt in top hits? {}",
        if found_b { "yes" } else { "no" }
    );
    println!();

    // ── Verdict ──────────────────────────────────────────────────────
    println!("Verdict:");
    println!(
        "    Arm A surfaced calcAmt: {}; Arm B surfaced calcAmt: {}",
        found_a, found_b
    );
    if found_b && !found_a {
        println!("    Enrichment lifted the answer chunk from miss → found.");
    } else if found_b && found_a {
        println!("    Both arms surface calcAmt on this toy corpus (BM25 partial-matches");
        println!("    on a tiny corpus); the demonstrative artifact is the *audit trail*");
        println!("    and the ingest-time text change visible in Arm B's hit.");
    } else {
        println!("    No enrichment lift on this corpus — the dictionary may not");
        println!("    align with the query, or the corpus is too small to discriminate.");
    }
    println!();
    println!("Mechanism: the query \"monthly billing run\" shares no surface");
    println!("forms with the bare symbol `calcAmt`. Arm B appends");
    println!("\"calculate amount billing total line-item sum\" to the chunk");
    println!("at ingest, so the same query now matches on high-IDF tokens");
    println!("(`billing`, `calculate`) and the chunk surfaces.");
    println!();
    println!("Regime: short, opaque chunks + a decoding dictionary. On long");
    println!("descriptive prose chunks this would be redundant. Bolting the");
    println!("same boilerplate onto every chunk would re-create the low-IDF");
    println!("dilution from CUAD_PRF_NULL — enrich must add term-specific");
    println!("signal, not repeated filler. Full rule + use cases:");
    println!("docs/findings/VOCABULARY_ENRICH.md");

    Ok(())
}
