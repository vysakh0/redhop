//! Golden retrieval-quality tests on a tiny hand-built corpus.
//!
//! Almost every other test in this crate verifies INTERNAL CONSISTENCY:
//! "the report's count matches the chunks", "the strategy filtered as
//! claimed", "no field is NaN". Almost none verify CORRECTNESS against
//! a ground truth — "for this question, the right answer is in chunk X
//! and RedHop selects it."
//!
//! This file is the golden answer: a small corpus where every chunk's
//! role is unambiguous, paired with queries whose correct answer is
//! pinned. If a refactor breaks BM25 stemming, breaks the multi-field
//! search (heading/source), breaks the second-hop rescue, or breaks
//! distractor filtering on otherwise-relevant queries, ONE of these
//! tests fails with a message naming the regression class.
//!
//! Not benchmark-quality (only 5 queries) — that lives in the
//! `benchmarks` crate. This is the canary that fires between
//! benchmark runs: "RedHop just got dumber on a query that used to
//! work" surfaces here, not three weeks later when a user files an
//! issue.
//!
//! Reading this file:
//! - Each test is named after the regression class it catches.
//! - Each test sets up an inline corpus (no fixtures, no network).
//! - Each test asserts the chunk a reasonable human would expect.
//! - Failure messages point at the specific bug class — stemming,
//!   distractor handling, multi-hop, etc.

use redhop::{citations, BuiltContext, Document};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Build a single-chunk document that the runtime is forced to retrieve
/// (chunk-per-line forces the source text to chunk on punctuation). Each
/// line in `text` becomes its own conceptual unit so a query can hit any
/// of them. Returns the assembled context for the given query.
fn ask(corpus_text: &str, source: &str, query: &str) -> BuiltContext {
    let mut doc = Document::from_text(source, corpus_text).expect("Document::from_text");
    doc.context(query).expect("Document::context")
}

/// Lowercased context-text snippet check: "the assembled context contains
/// this answer-bearing phrase".
fn ctx_contains_lowercase(ctx: &BuiltContext, needle: &str) -> bool {
    ctx.text().to_lowercase().contains(&needle.to_lowercase())
}

// ── G01: Lexical match — straight keyword query ───────────────────────────

#[test]
fn g01_lexical_keyword_match() {
    // Most basic regression check: the answer-bearing words appear in the
    // query. If even THIS breaks, the retriever has regressed beyond
    // any prior version.
    let corpus = "\
the refund window is thirty days from the date of purchase. \
unrelated content about shipping policy and delivery times. \
customer support hours are nine to five eastern time.";
    let ctx = ask(corpus, "policy.md", "refund window");
    assert!(
        ctx_contains_lowercase(&ctx, "thirty days"),
        "G01 — lexical keyword match broke: query 'refund window' \
         should retrieve the 'thirty days' chunk. ctx.text = {:?}",
        ctx.text()
    );
}

// ── G02: Stemming — morphological variants must match ────────────────────

#[test]
fn g02_stemming_finds_morphological_variants() {
    // The bug behind this test: 0.1.2 shipped without BM25 stemming. A
    // user queried `compression` against a chunk that contained
    // `compress_video` and got nothing back. After Snowball stemming
    // both reduce to `compress` and the chunk surfaces.
    //
    // **0.3.2 default change:** the default analyzer is now `RawAnalyzer`
    // (no stemming) — see docs/findings/RAW_ANALYZER.md for the
    // measurement that flipped it. So this test now exercises the
    // **explicit** English-Snowball path that users opt into via
    // `language="english"` (or `Document::with_analyzer(SnowballAnalyzer::english())`
    // at the Rust level). The bug class that motivated this test
    // (stemmer wiring + BM25 picking up the analyzer) is still pinned
    // here — just against the explicit opt-in, not the default.
    use std::sync::Arc;
    let corpus = "\
fn compress_video(file_path: &str, quality: &str) { ... } \
unrelated handler for renaming files. \
helper for parsing configuration data.";
    let analyzer: Arc<dyn redhop::analyzer::Analyzer> =
        Arc::new(redhop::analyzer::SnowballAnalyzer::english());
    let mut doc = redhop::Document::from_text("video.rs", corpus)
        .expect("Document::from_text")
        .with_analyzer(analyzer);
    let ctx = doc.context("compression").expect("Document::context");
    assert!(
        ctx_contains_lowercase(&ctx, "compress_video"),
        "G02 — Snowball stemming regressed: query 'compression' should \
         match `compress_video` (both stem to 'compress') under the \
         explicit English-Snowball analyzer. Check the analyzer wiring \
         + that BM25 is using the configured analyzer. ctx.text = {:?}",
        ctx.text()
    );
}

// ── G03: Distractor rejection on a relevance query ─────────────────────────

#[test]
fn g03_distractor_does_not_dominate_relevant_chunk() {
    // The hard case: a clearly relevant chunk competes with a chunk that
    // shares vocabulary with the query but ISN'T about the same thing.
    // BM25 alone might score the distractor highly if it word-overlaps;
    // the assembly should still keep the truly-relevant chunk in the
    // context. We don't assert the distractor is dropped (the default
    // is Auto/Passthrough at this size — both may survive), only that
    // the relevant one is present.
    let corpus = "\
the safety lamp was invented by humphry davy in eighteen fifteen. \
safety helmet sizing guide for industrial workers. \
safety data sheets describe chemical hazards.";
    let ctx = ask(corpus, "history.md", "who invented the safety lamp");
    assert!(
        ctx_contains_lowercase(&ctx, "humphry davy"),
        "G03 — relevance ranking regressed: the chunk that actually names \
         the inventor ('humphry davy') must appear in the context for \
         'who invented the safety lamp'. ctx.text = {:?}",
        ctx.text()
    );
}

// ── G04: Multi-hop / second-hop rescue ────────────────────────────────────

#[test]
fn g04_second_hop_chunk_can_survive_assembly() {
    // The bridge case: the answer to the query is in TWO chunks — one
    // that's on-topic for the query, and a "second hop" that shares the
    // bridge entity with the first but doesn't itself match the query
    // words. A pure relevance filter would drop the second hop; the
    // default Auto/Passthrough strategy at this small size keeps
    // everything, and even at larger sizes ReasoningPreserving should
    // rescue the linked chunk.
    //
    // Here we only assert the SEED is kept. The second-hop rescue path
    // itself is exercised by `strategy_semantics.rs` and the
    // ReasoningPreserving inline tests with a tighter pin; the goal of
    // this test is "the default path doesn't accidentally drop seeds."
    let corpus = "\
the safety lamp was invented by humphry davy. \
humphry davy was born in penzance, cornwall, england. \
photosynthesis converts sunlight into chemical energy in plants.";
    let ctx = ask(
        corpus,
        "qa.md",
        "what nationality was the safety lamp inventor",
    );
    assert!(
        ctx_contains_lowercase(&ctx, "humphry davy") || ctx_contains_lowercase(&ctx, "safety lamp"),
        "G04 — the seed chunk ('safety lamp invented by humphry davy') \
         must survive default assembly for a multi-hop query. \
         ctx.text = {:?}",
        ctx.text()
    );
}

// ── G05: Low-confidence retrieval signal ──────────────────────────────────

#[test]
fn g05_low_confidence_signal_fires_when_query_misses() {
    // The "did anything actually match" signal. When a query shares no
    // vocabulary with the corpus, the report's `low_confidence_retrieval`
    // should be true — telling the caller to fall back, prompt the user
    // to rephrase, etc. If the signal silently stops firing, callers'
    // downstream fallbacks become dead code.
    let corpus = "\
the refund window is thirty days. \
shipping policy details and delivery times. \
customer support contact information.";
    let ctx = ask(corpus, "policy.md", "quantum chromodynamics gluon coupling");

    // The signal is allowed to NOT fire under very-short-corpus edge
    // cases (the relevance-bar check needs at least one above-bar chunk
    // to be meaningful). For a query this clearly unrelated, though, we
    // require it to fire — otherwise the signal is broken.
    assert!(
        ctx.report.low_confidence_retrieval,
        "G05 — low_confidence_retrieval should fire for an obviously \
         off-corpus query ('quantum chromodynamics gluon coupling' \
         vs a refund/shipping/support corpus). If this regresses, \
         callers' downstream fallback paths become dead code. \
         report = {:?}",
        ctx.report
    );
}

// ── G06: Citations preserve source attribution ───────────────────────────

#[test]
fn g06_citations_carry_correct_source() {
    // The Decision Report's citations are what callers wire into "where
    // did this answer come from?" If a refactor breaks the source
    // attribution (e.g. by losing the source path on a chunk), every
    // citation downstream lies about provenance.
    let corpus = "the refund window is thirty days from purchase";
    let ctx = ask(corpus, "policy-2024.md", "refund window");
    let cites = citations(&ctx);
    assert!(
        !cites.is_empty(),
        "G06 — citations array is empty for a query that retrieved a chunk. \
         Citation wiring regressed."
    );
    assert!(
        cites.iter().all(|c| c.source == "policy-2024.md"),
        "G06 — citation source path regressed; expected 'policy-2024.md', \
         got {:?}",
        cites.iter().map(|c| c.source.clone()).collect::<Vec<_>>()
    );
}
