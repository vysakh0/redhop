//! Internal consistency invariants for `BuiltContext` + `ContextReport`.
//!
//! Things that should always be true but no other test pins. The wildcard-
//! fallback bug fixed in T51 was the same class: silent-wrong, no error,
//! just a bad output. Each invariant here would have caught a similar bug
//! AT THE LAYER it surfaces, so a future regression fails one specific
//! assertion with an actionable message rather than slipping through.
//!
//! Every invariant runs against multiple corpus shapes + strategies, so a
//! bug that only fires under (say) `DistractorFiltered` + a code-heavy
//! corpus still trips a check.
//!
//! Add a new invariant by:
//! 1. Writing a `check_<invariant_name>(&BuiltContext, &[Chunk])` helper.
//! 2. Adding a `#[test]` that walks the matrix and runs your helper.

use redhop::context::{ContextConfig, ContextStrategy};
use redhop::core::{Chunk, ChunkId, TokenCount};
use redhop::{citations, BuiltContext, Document, DocumentConfig};

// ── Corpus helpers ─────────────────────────────────────────────────────────

fn mk_chunk(id: &str, text: &str, source: &str, kind: &str) -> Chunk {
    let mut c = Chunk::new(
        ChunkId::new(id),
        text,
        source,
        TokenCount(text.split_whitespace().count().max(1)),
    );
    c.metadata.insert("kind".into(), serde_json::json!(kind));
    c
}

/// Small, mixed corpus — code + prose + on-topic + distractor.
fn small_corpus() -> Vec<Chunk> {
    vec![
        mk_chunk(
            "a",
            "the refund window is thirty days from purchase",
            "policy.md",
            "prose",
        ),
        mk_chunk(
            "b",
            "customers may return items within 30 days for a refund",
            "policy.md",
            "prose",
        ),
        mk_chunk("c", "fn compress_video(path: &str)", "video.rs", "code"),
        mk_chunk(
            "d",
            "photosynthesis converts sunlight into glucose in plants",
            "bio.md",
            "prose",
        ),
        mk_chunk("e", "the capital of France is Paris", "trivia.md", "prose"),
    ]
}

/// Medium corpus, all on-topic — exercises the "no distractor" branch.
fn all_on_topic_corpus() -> Vec<Chunk> {
    (0..10)
        .map(|i| {
            mk_chunk(
                &format!("c{i}"),
                &format!("refund window thirty day policy section {i}"),
                "policy.md",
                "prose",
            )
        })
        .collect()
}

/// Corpus where nothing matches the query — exercises low_confidence path.
fn all_off_topic_corpus() -> Vec<Chunk> {
    vec![
        mk_chunk("x", "photosynthesis converts sunlight", "bio.md", "prose"),
        mk_chunk(
            "y",
            "the moon orbits the earth every 27 days",
            "astro.md",
            "prose",
        ),
        mk_chunk("z", "fn parse(input: &str)", "lib.rs", "code"),
    ]
}

/// Build a Document with the requested strategy override.
fn build_doc(chunks: Vec<Chunk>, strategy: ContextStrategy) -> Document {
    let mut cfg = DocumentConfig::default();
    cfg.context = ContextConfig {
        strategy,
        ..cfg.context
    };
    Document::from_chunks_with(chunks, cfg).unwrap()
}

// ── Invariant helpers ──────────────────────────────────────────────────────

/// I-1: `report.n_selected` matches the actual `chunks.len()`.
fn check_n_selected_matches_chunks_len(ctx: &BuiltContext, label: &str) {
    assert_eq!(
        ctx.report.n_selected,
        ctx.chunks.len(),
        "[{label}] I-1: report.n_selected ({}) must equal chunks.len() ({})",
        ctx.report.n_selected,
        ctx.chunks.len()
    );
}

/// I-2: `report.total_tokens` equals the sum of selected chunks' token counts.
fn check_total_tokens_matches_chunks(ctx: &BuiltContext, label: &str) {
    let sum: usize = ctx.chunks.iter().map(|c| c.token_count.0).sum();
    assert_eq!(
        ctx.report.total_tokens, sum,
        "[{label}] I-2: report.total_tokens ({}) must equal sum(chunk.token_count) ({sum})",
        ctx.report.total_tokens
    );
}

/// I-3: `citations(ctx).len()` matches `chunks.len()` (one citation per
/// cited chunk, in order).
fn check_citations_count_and_order(ctx: &BuiltContext, label: &str) {
    let cites = citations(ctx);
    assert_eq!(
        cites.len(),
        ctx.chunks.len(),
        "[{label}] I-3: citations.len() ({}) must equal chunks.len() ({})",
        cites.len(),
        ctx.chunks.len()
    );
    // Order: citations[i].source must come from chunks[i].source.
    for (i, (c, citation)) in ctx.chunks.iter().zip(cites.iter()).enumerate() {
        assert_eq!(
            citation.source, c.source,
            "[{label}] I-3: citation[{i}].source = {:?} but chunks[{i}].source = {:?}",
            citation.source, c.source
        );
    }
}

/// I-4: selected chunks are a subset of input chunks by id (no synthesis).
fn check_selected_subset_of_input(ctx: &BuiltContext, input: &[Chunk], label: &str) {
    let input_ids: std::collections::HashSet<&str> = input.iter().map(|c| c.id.as_str()).collect();
    // Auto-expansion (code neighbors, prose headings) can add NEIGHBORING
    // chunks from the same source — those came from the original chunking,
    // not the retrieved set. So we check membership against ALL Document
    // chunks, but the caller passes a Document via `input` so the input
    // here IS the full chunk list. Selected ⊆ input.
    for c in &ctx.chunks {
        assert!(
            input_ids.contains(c.id.as_str()),
            "[{label}] I-4: selected chunk id={:?} not in input set (synthesis?)",
            c.id.as_str()
        );
    }
}

/// I-5: ratios are all in [0, 1].
fn check_ratios_in_unit_interval(ctx: &BuiltContext, label: &str) {
    let r = &ctx.report;
    for (name, v) in [
        ("input_distractor_ratio", r.input_distractor_ratio),
        ("retained_evidence_ratio", r.retained_evidence_ratio),
        ("token_utilization", r.token_utilization),
    ] {
        assert!(
            (0.0..=1.0).contains(&v),
            "[{label}] I-5: report.{name} = {v} out of [0, 1]"
        );
    }
}

/// I-6: `requested_strategy` is what we asked for; `strategy` is what
/// actually ran. Under non-Auto, they're equal. Under Auto, strategy is
/// the concrete resolution (NOT Auto).
fn check_strategy_resolution(ctx: &BuiltContext, requested: ContextStrategy, label: &str) {
    assert_eq!(
        ctx.report.requested_strategy, requested,
        "[{label}] I-6a: requested_strategy must match what was asked"
    );
    if !matches!(requested, ContextStrategy::Auto) {
        assert_eq!(
            ctx.report.strategy, requested,
            "[{label}] I-6b: for non-Auto, resolved strategy must equal requested"
        );
    } else {
        assert!(
            !matches!(ctx.report.strategy, ContextStrategy::Auto),
            "[{label}] I-6c: under Auto, the report's `strategy` must be the \
             concrete resolution (RawTopK or ReasoningPreserving), never Auto"
        );
    }
}

/// I-7: assembled `text()` is the concatenation of selected `chunk.text`s
/// joined deterministically. Calling `text()` twice returns the same value.
fn check_text_is_deterministic_and_made_of_chunks(ctx: &BuiltContext, label: &str) {
    let a = ctx.text();
    let b = ctx.text();
    assert_eq!(a, b, "[{label}] I-7a: text() must be deterministic");
    // Every selected chunk's text must appear in the assembled text.
    for c in &ctx.chunks {
        assert!(
            a.contains(&c.text),
            "[{label}] I-7b: selected chunk text not in assembled text"
        );
    }
}

/// Run every invariant against `ctx`, producing a labeled assertion-set
/// for the (strategy × corpus) cell.
fn check_all(ctx: &BuiltContext, input: &[Chunk], requested: ContextStrategy, label: &str) {
    check_n_selected_matches_chunks_len(ctx, label);
    check_total_tokens_matches_chunks(ctx, label);
    check_citations_count_and_order(ctx, label);
    check_selected_subset_of_input(ctx, input, label);
    check_ratios_in_unit_interval(ctx, label);
    check_strategy_resolution(ctx, requested, label);
    check_text_is_deterministic_and_made_of_chunks(ctx, label);
}

// ── The matrix: every strategy × every corpus shape ────────────────────────

#[test]
fn invariants_hold_across_strategies_and_corpora() {
    let strategies = [
        ("Auto", ContextStrategy::Auto),
        ("ReasoningPreserving", ContextStrategy::ReasoningPreserving),
        ("DistractorFiltered", ContextStrategy::DistractorFiltered),
        ("MaxDensity", ContextStrategy::MaxDensity),
        ("RawTopK", ContextStrategy::RawTopK),
        ("RedundancyPruned", ContextStrategy::RedundancyPruned),
    ];
    type CorpusFn = fn() -> Vec<Chunk>;
    let corpora: [(&str, CorpusFn); 3] = [
        ("small_mixed", small_corpus),
        ("all_on_topic", all_on_topic_corpus),
        ("all_off_topic", all_off_topic_corpus),
    ];
    let query = "refund window";

    for (strat_name, strat) in &strategies {
        for (corpus_name, corpus_fn) in &corpora {
            let input = corpus_fn();
            let mut doc = build_doc(input.clone(), *strat);
            let ctx = doc.context(query).unwrap();
            let label = format!("{strat_name}/{corpus_name}");
            check_all(&ctx, &input, *strat, &label);
        }
    }
}

#[test]
fn n_input_bounded_by_candidate_k_and_above_n_selected() {
    // I-8 (relaxed): BM25 only returns chunks that match the query, so
    // `n_input_chunks` is the matched-count, not the corpus size. The
    // structural invariant we CAN pin is the chain:
    //
    //   n_selected <= n_input_chunks <= candidate_k
    //
    // (selected ≤ input because assembly is a filter; input ≤ candidate_k
    // because that's the retrieval cap). Walk both ends of the query
    // spectrum — a query that matches most chunks vs one that matches none.
    let input = small_corpus();
    let candidate_k = DocumentConfig::default().candidate_k;

    let mut doc = build_doc(input.clone(), ContextStrategy::RawTopK);
    let ctx = doc.context("refund window thirty days").unwrap();
    assert!(
        ctx.report.n_selected <= ctx.report.n_input_chunks,
        "I-8a: n_selected ({}) must be ≤ n_input_chunks ({})",
        ctx.report.n_selected,
        ctx.report.n_input_chunks
    );
    assert!(
        ctx.report.n_input_chunks <= candidate_k,
        "I-8b: n_input_chunks ({}) must be ≤ candidate_k ({candidate_k})",
        ctx.report.n_input_chunks
    );

    // Zero-match query: input_chunks may be 0 (after the T51 fix); the
    // bound still has to hold trivially.
    let mut doc2 = build_doc(input, ContextStrategy::RawTopK);
    let ctx2 = doc2.context("xyzzy frobnicate quux").unwrap();
    assert!(ctx2.report.n_selected <= ctx2.report.n_input_chunks);
    assert!(ctx2.report.n_input_chunks <= candidate_k);
}

#[test]
fn empty_query_returns_empty_context_with_consistent_report() {
    // After the T51 fix (sanitize_query returning empty), a no-signal
    // query gives back an empty context. All invariants should still hold
    // — `n_selected == 0`, `chunks.len() == 0`, citations empty, etc.
    let input = small_corpus();
    let mut doc = build_doc(input.clone(), ContextStrategy::ReasoningPreserving);
    let ctx = doc.context("    \t\n  ").unwrap();
    assert!(
        ctx.chunks.is_empty(),
        "post-T51: whitespace query should produce empty context"
    );
    check_all(
        &ctx,
        &input,
        ContextStrategy::ReasoningPreserving,
        "empty_query",
    );
    // Specific empty-case asserts beyond the generic invariants:
    assert_eq!(ctx.report.n_selected, 0);
    assert_eq!(ctx.report.total_tokens, 0);
    assert_eq!(citations(&ctx).len(), 0);
    assert_eq!(ctx.text(), "");
}

#[test]
fn token_utilization_matches_definition() {
    // I-extra: report.token_utilization = total_tokens / token_budget,
    // pinned because the field's docstring claims this exactly. A drift
    // in the math (off-by-one, integer division) would silently produce
    // wrong observability numbers downstream.
    let input = small_corpus();
    let mut doc = build_doc(input, ContextStrategy::ReasoningPreserving);
    let ctx = doc.context("refund window").unwrap();
    let r = &ctx.report;
    let expected = r.total_tokens as f32 / r.token_budget as f32;
    let diff = (r.token_utilization - expected).abs();
    assert!(
        diff < 1e-6,
        "token_utilization ({}) must equal total_tokens/token_budget ({}); diff {diff}",
        r.token_utilization,
        expected
    );
}
