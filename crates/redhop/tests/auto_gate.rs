//! Auto-strategy gate boundary tests.
//!
//! The Auto policy's only decision is whether `input_tokens` is at-or-below
//! `auto_passthrough_max_tokens` (default `1500`):
//!
//!   input_tokens <= gate  →  Passthrough (resolved to RawTopK)
//!   input_tokens >  gate  →  Prune       (resolved to ReasoningPreserving)
//!
//! That `<=` is load-bearing — the docstring says "at-or-below". An
//! off-by-one (`<` instead of `<=`) would silently flip the decision class
//! at exactly the calibrated boundary, and the failure mode is hard to
//! debug after-the-fact because the wrong-but-plausible answer comes out
//! without any error.
//!
//! These tests pin the boundary explicitly: 1499 / 1500 / 1501 tokens.

use redhop::context::{analyze_context, AutoDecision, ContextConfig, ContextStrategy};
use redhop::core::{Chunk, ChunkId, Query, RetrievalMethod, RetrievalResult, Score, TokenCount};

/// Build a list of `n` retrieval results each carrying `tokens_per_chunk`
/// tokens in its `TokenCount`, so the input-tokens sum is exactly
/// `n * tokens_per_chunk`. Score is constant so ranking is determined by
/// presentation order, which is enough for the gate decision (which only
/// looks at the total).
fn results_with_total(
    n: usize,
    tokens_per_chunk: usize,
    query_relevant: bool,
) -> Vec<RetrievalResult> {
    let text = if query_relevant {
        // Words that overlap with the query "refund window" so the
        // grounding scorer doesn't trip and dominate the decision.
        "refund window content"
    } else {
        "unrelated chunk text"
    };
    (0..n)
        .map(|i| RetrievalResult {
            chunk: Chunk::new(
                ChunkId::new(format!("c{i}")),
                text,
                "src.md",
                TokenCount(tokens_per_chunk),
            ),
            score: Score {
                value: 1.0,
                method: RetrievalMethod::Lexical,
            },
            breakdown: Default::default(),
        })
        .collect()
}

/// Auto config — `auto_passthrough_max_tokens` at its calibrated default.
fn auto_cfg() -> ContextConfig {
    ContextConfig {
        strategy: ContextStrategy::Auto,
        ..Default::default()
    }
}

#[test]
fn auto_passthrough_at_gate_minus_one() {
    // 1499 input tokens → inside the passthrough band by 1.
    let cfg = auto_cfg();
    let q = Query::new("refund window");
    let results = results_with_total(1, 1499, true);
    let report = analyze_context(&q, &results, &cfg);
    assert_eq!(report.input_tokens, 1499, "construction sanity");
    assert_eq!(
        report.auto_decision(),
        AutoDecision::Passthrough,
        "input 1499 must be below the 1500 gate → Passthrough"
    );
    assert_eq!(
        report.strategy,
        ContextStrategy::RawTopK,
        "Passthrough resolves to RawTopK"
    );
}

#[test]
fn auto_passthrough_at_gate_exactly() {
    // 1500 input tokens — exactly at the gate. The `<=` comparison is
    // inclusive per the docstring "at-or-below", so this MUST resolve to
    // Passthrough, not Prune. An off-by-one (`<` instead of `<=`) would
    // flip this test.
    let cfg = auto_cfg();
    let q = Query::new("refund window");
    let results = results_with_total(1, 1500, true);
    let report = analyze_context(&q, &results, &cfg);
    assert_eq!(report.input_tokens, 1500, "construction sanity");
    assert_eq!(
        report.auto_decision(),
        AutoDecision::Passthrough,
        "input EXACTLY at the gate (1500) must be Passthrough — \
         the gate is inclusive per `auto_passthrough_max_tokens`'s docstring \
         ('at or below'). An off-by-one bug in the comparison would flip this."
    );
}

#[test]
fn auto_prunes_at_gate_plus_one() {
    // 1501 input tokens — one above the gate → Prune.
    let cfg = auto_cfg();
    let q = Query::new("refund window");
    let results = results_with_total(1, 1501, true);
    let report = analyze_context(&q, &results, &cfg);
    assert_eq!(report.input_tokens, 1501, "construction sanity");
    assert_eq!(
        report.auto_decision(),
        AutoDecision::Prune,
        "input 1501 (one above the 1500 gate) must Prune"
    );
    assert_eq!(
        report.strategy,
        ContextStrategy::ReasoningPreserving,
        "Prune resolves to ReasoningPreserving"
    );
}

#[test]
fn auto_decision_respects_custom_gate() {
    // The gate is a per-config knob — pinning a CALLER's custom value
    // works too. Caller sets gate to 100; input 100 passes through, 101
    // prunes. Catches a regression that hardcoded the default constant
    // somewhere it should have read from cfg.
    let mut cfg = auto_cfg();
    cfg.auto_passthrough_max_tokens = 100;
    let q = Query::new("refund window");

    let at_gate = analyze_context(&q, &results_with_total(1, 100, true), &cfg);
    assert_eq!(at_gate.input_tokens, 100);
    assert_eq!(
        at_gate.auto_decision(),
        AutoDecision::Passthrough,
        "custom gate=100, input=100 → Passthrough"
    );

    let just_over = analyze_context(&q, &results_with_total(1, 101, true), &cfg);
    assert_eq!(just_over.input_tokens, 101);
    assert_eq!(
        just_over.auto_decision(),
        AutoDecision::Prune,
        "custom gate=100, input=101 → Prune"
    );
}

#[test]
fn auto_decision_not_auto_for_concrete_strategies() {
    // When the caller explicitly picks a non-Auto strategy, `auto_decision`
    // must surface that as `NotAuto` regardless of input size. Without this
    // pin, a future refactor could quietly mark every report as
    // Passthrough/Prune even when the caller didn't ask for Auto.
    let q = Query::new("refund window");
    let results = results_with_total(1, 500, true);
    for concrete in &[
        ContextStrategy::ReasoningPreserving,
        ContextStrategy::DistractorFiltered,
        ContextStrategy::MaxDensity,
        ContextStrategy::RawTopK,
        ContextStrategy::RedundancyPruned,
    ] {
        let cfg = ContextConfig {
            strategy: *concrete,
            ..Default::default()
        };
        let report = analyze_context(&q, &results, &cfg);
        assert_eq!(
            report.auto_decision(),
            AutoDecision::NotAuto,
            "concrete strategy {concrete:?} must report NotAuto, not a derived Passthrough/Prune"
        );
    }
}
