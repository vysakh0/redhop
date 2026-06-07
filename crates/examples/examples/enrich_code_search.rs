//! Code-symbol retrieval with `Vocabulary::enrich(...)` — **a synthetic
//! demo, not a benchmark**. The corpus, the dictionary, and the query
//! were all crafted by hand to make the mechanism's lift visible in a
//! 100-line example. It demonstrates *how* enrich plugs into the
//! pipeline (audit trail, ingest-time text change, downstream
//! retrieval). It does **not** measure whether enrich helps on a real
//! legacy code corpus — there's no eval rig under this; the
//! "monthly billing run" → `calcAmt` lift is engineered, not observed
//! in the wild.
//!
//! For the actual measured datapoints on enrich:
//! - **Negative measured (CUAD, prose):**
//!   [`CUAD_ENRICH_DEFINITIONS_NULL`](../../../docs/findings/CUAD_ENRICH_DEFINITIONS_NULL.md)
//!   regressed −2.0pt on top of the 90.7% workflow baseline.
//! - **Positive measured:** none yet. Spider/BIRD as the schema-regime
//!   probe is the natural positive test; queued, not run.
//!
//! So treat this file as a *demo of the API surface*, not as evidence
//! enrich works for code search. To know if it works on *your*
//! codebase, A/B with `redhop::evaluate(...)` on your own gold.
//!
//! ## The use case it's modeling
//!
//! A legacy codebase where function names are abbreviated (`usrSvc`,
//! `calcAmt`, `chkInv`) and the developer's question is ordinary
//! English ("monthly billing run"). Lexical retrieval has nothing to
//! match — the query and the bare symbol `calcAmt` share zero surface
//! forms — so the right chunk doesn't make it into the BM25 top-K.
//! Enrichment is *predicted* to fix this by appending each symbol's
//! plain-language synonyms to its chunk at ingest time, raising the
//! chunk's matchable surface area without changing how it's authored.
//!
//! ## The mechanism (and how it differs from query-side expand)
//!
//! Query-side [`redhop::Vocabulary::apply`] patches gaps the *author
//! of the query rewriter* anticipated — you enumerate which queries
//! get which synonyms. Chunk-side `enrich` is a different mechanism:
//! you describe the *content* once, and the prediction is that future
//! queries paraphrasing the description will find it. Different jobs:
//!
//! - **apply:** known query reformulations. Surgical, cheap, narrow.
//!   Measured positive on CUAD (+3.0).
//! - **enrich:** the prediction is *raise the content's semantic floor
//!   for queries you can't predict*. Broader (in mechanism), ingest-
//!   time cost, conditional on having a decoding dictionary.
//!   Measured negative on CUAD; positive case unmeasured.
//!
//! ## When this is predicted to earn its keep
//!
//! Mechanism prediction: `value ∝ shortness × opacity × (dictionary
//! exists)`. Function signatures are an extreme case on the shortness
//! and opacity axes. Other predicted-strong cases: SQL schema columns
//! (`emp_compensation`, `ord_dt`), error codes (`ERR_4012`), clinical
//! abbreviations (`MI`, `SOB`). On long descriptive prose chunks the
//! operation is *predicted* to be redundant and was *measured* to
//! regress on CUAD. See
//! [`VOCABULARY_ENRICH`](../../../docs/findings/VOCABULARY_ENRICH.md).
//!
//! ## Two arms (engineered, not eval'd)
//!
//! - **A — raw chunks (no enrichment).** The query
//!   `"monthly billing run"` has zero overlap with `calcAmt`'s bare
//!   signature; BM25 doesn't surface it.
//! - **B — enriched chunks.** Each function chunk gets its hand-
//!   crafted plain-language synonyms appended at ingest. The same
//!   query now finds `calcAmt` on its enriched
//!   `"calculate amount billing total line-item sum"` tokens.
//!
//! Selection differs between arms because the dictionary was engineered
//! to make it differ. That's the demo. Whether the same shape of
//! dictionary helps on *your* code corpus is your A/B to run.
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
            chunk(&format!("fn-{i:02}"), &r.text)
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
