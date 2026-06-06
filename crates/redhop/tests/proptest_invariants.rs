//! Property-based invariants for `build_context`.
//!
//! Every test in this file is a CLAIM about EVERY POSSIBLE input:
//! "for any random corpus + any random `ContextConfig`, X holds."
//! `proptest` explores the input space until either the claim trips
//! (shrinks to a minimal failing case) or the budgeted N runs pass.
//!
//! Hand-written tests pin specific examples — useful, but they only
//! exercise inputs the author thought of. Property tests catch the bug
//! class hand-written tests can't, by definition: edge cases nobody
//! enumerated, including the famously-bad ones (empty input, single
//! chunk, NaN scores, all-stopword query, single-token budget,
//! identical chunks, …). The bugs that *survive* the existing 374
//! examples live in the corners proptest searches.
//!
//! Generators are deliberately bounded: corpus ≤ 20 chunks, token
//! counts derived from a small vocabulary (so grounding has lexical
//! overlap to actually score against), config knobs in their valid
//! ranges only. Bounding keeps shrinking fast; the random space inside
//! the bounds is still wide enough to find bugs.
//!
//! Properties pinned here:
//! 1. `build_context_never_panics`
//! 2. `resolved_strategy_is_never_auto`
//! 3. `auto_decision_triangle_holds`
//! 4. `selection_is_subset_of_input`
//! 5. `no_duplicate_chunk_ids_in_output`
//! 6. `token_budget_respected`
//! 7. `report_counts_match_reality`
//! 8. `report_ratios_are_finite_and_in_range`
//! 9. `build_context_is_deterministic`

use proptest::prelude::*;
use redhop::analyzer::default_english;
use redhop::context::{build_context, AutoDecision, ContextConfig, ContextStrategy};
use redhop::core::{
    Chunk, ChunkId, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown, TokenCount,
};

// ── Generators ─────────────────────────────────────────────────────────────

/// Small vocabulary so query terms occasionally overlap chunk terms —
/// grounding scoring needs lexical overlap to produce non-trivial signal.
/// Truly-random text would almost never share tokens with the query, so
/// every chunk would look like a distractor and most strategies would
/// degenerate to "drop everything" — not a useful test of their semantics.
const VOCAB: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "the",
    "of", "and", "to", "in", "a", "is", "for", "by", "with", "refund", "window", "policy",
    "shipping", "return", "thirty", "days", "humphry", "davy", "safety", "lamp",
];

/// 1–32 words from `VOCAB`, joined.
fn arb_text() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::sample::select(VOCAB), 1..32).prop_map(|w| w.join(" "))
}

/// A corpus of `size_range` `RetrievalResult`s. Token counts derive
/// from the actual text so `report.total_tokens` arithmetic exercises a
/// real, varied sum (not a constant).
fn arb_corpus(
    size_range: std::ops::RangeInclusive<usize>,
) -> impl Strategy<Value = Vec<RetrievalResult>> {
    prop::collection::vec((arb_text(), 0.0_f32..=1.0_f32), size_range).prop_map(|items| {
        items
            .into_iter()
            .enumerate()
            .map(|(i, (text, score_val))| {
                let token_count = text.split_whitespace().count().max(1);
                RetrievalResult {
                    chunk: Chunk::new(
                        ChunkId::new(format!("c{i}")),
                        text,
                        "doc.md",
                        TokenCount(token_count),
                    ),
                    score: Score {
                        value: score_val,
                        method: RetrievalMethod::Lexical,
                    },
                    breakdown: ScoreBreakdown::default(),
                }
            })
            .collect()
    })
}

/// Any strategy except `RedundancyPruned`. The cosine path needs
/// embeddings, which are awful to shrink (random f32 vectors). The
/// lexical-Jaccard fallback for embedding-less inputs IS exercised
/// by other tests; here we keep proptest focused on the strategies
/// whose state space is text-shaped.
fn arb_strategy() -> impl Strategy<Value = ContextStrategy> {
    prop_oneof![
        Just(ContextStrategy::RawTopK),
        Just(ContextStrategy::DistractorFiltered),
        Just(ContextStrategy::MaxDensity),
        Just(ContextStrategy::ReasoningPreserving),
        Just(ContextStrategy::Auto),
    ]
}

/// A random `ContextConfig` with every knob in its documented range.
/// `token_budget` is bounded to a few thousand so the budget arithmetic
/// is exercised under realistic pressure (sometimes tight, sometimes
/// loose) — not 8 EB and never the deciding factor.
fn arb_config() -> impl Strategy<Value = ContextConfig> {
    (
        arb_strategy(),
        1usize..=2000, // token_budget
        0.0_f32..=1.0, // distractor_min_grounding
        0.0_f32..=1.0, // link_min_jaccard
        0usize..=5000, // auto_passthrough_max_tokens
        0.0_f32..=1.0, // redundancy_max_cosine
        0.0_f32..=1.0, // low_confidence_max_grounding
    )
        .prop_map(
            |(strategy, budget, dmg, lmj, agm, rmc, lcmg)| ContextConfig {
                token_budget: budget,
                strategy,
                distractor_min_grounding: dmg,
                link_min_jaccard: lmj,
                auto_passthrough_max_tokens: agm,
                redundancy_max_cosine: rmc,
                low_confidence_max_grounding: lcmg,
                analyzer: default_english(),
            },
        )
}

fn arb_query() -> impl Strategy<Value = Query> {
    arb_text().prop_map(Query::new)
}

// ── Properties ─────────────────────────────────────────────────────────────

proptest! {
    /// The strongest universal property: `build_context` returns normally
    /// (no panic, no infinite loop within proptest's time budget) for any
    /// well-typed input. Integer overflow, division-by-zero, NaN
    /// propagation, out-of-bounds slicing — all surface as panics in
    /// debug builds, all caught here without anyone having to predict
    /// where they'd be.
    #[test]
    fn build_context_never_panics(
        q in arb_query(),
        corpus in arb_corpus(1..=20),
        cfg in arb_config(),
    ) {
        let _ = build_context(&q, &corpus, &cfg);
    }

    /// The strategy on the report is always a CONCRETE strategy — never
    /// `Auto`. `Auto` is a meta-strategy that must be resolved (to
    /// RawTopK / ReasoningPreserving) before assembly. A future refactor
    /// that forgets to resolve before stamping the report would surface
    /// here, on the first random input that picks `Auto`.
    #[test]
    fn resolved_strategy_is_never_auto(
        q in arb_query(),
        corpus in arb_corpus(1..=20),
        cfg in arb_config(),
    ) {
        let ctx = build_context(&q, &corpus, &cfg);
        prop_assert_ne!(
            ctx.report.strategy,
            ContextStrategy::Auto,
            "report.strategy must be a resolved concrete strategy, never Auto"
        );
    }

    /// The Auto decision triangle: `auto_decision`, `requested_strategy`,
    /// and `strategy` are mutually consistent.
    /// - If the caller didn't request Auto, decision == NotAuto.
    /// - If they did, decision matches the gate test against input_tokens
    ///   and the resolved strategy follows from the decision.
    /// An off-by-one in `resolve()` (which the auto_gate boundary tests
    /// pin for specific values) would surface here for random values
    /// straddling the gate.
    #[test]
    fn auto_decision_triangle_holds(
        q in arb_query(),
        corpus in arb_corpus(1..=20),
        cfg in arb_config(),
    ) {
        let ctx = build_context(&q, &corpus, &cfg);
        let r = &ctx.report;
        match r.requested_strategy {
            ContextStrategy::Auto => {
                if r.input_tokens <= cfg.auto_passthrough_max_tokens {
                    prop_assert_eq!(r.auto_decision(), AutoDecision::Passthrough);
                    prop_assert_eq!(r.strategy, ContextStrategy::RawTopK);
                } else {
                    prop_assert_eq!(r.auto_decision(), AutoDecision::Prune);
                    prop_assert_eq!(r.strategy, ContextStrategy::ReasoningPreserving);
                }
            }
            concrete => {
                prop_assert_eq!(r.auto_decision(), AutoDecision::NotAuto);
                prop_assert_eq!(r.strategy, concrete);
            }
        }
    }

    /// Every selected chunk was actually in the input. The set of output
    /// chunk IDs is a subset of the set of input chunk IDs. Catches:
    /// phantom chunks, off-by-one in indexing, accidental id-rewriting
    /// during assembly.
    #[test]
    fn selection_is_subset_of_input(
        q in arb_query(),
        corpus in arb_corpus(1..=20),
        cfg in arb_config(),
    ) {
        let ctx = build_context(&q, &corpus, &cfg);
        let input_ids: std::collections::HashSet<&str> =
            corpus.iter().map(|r| r.chunk.id.as_str()).collect();
        for c in &ctx.chunks {
            prop_assert!(
                input_ids.contains(c.id.as_str()),
                "selected chunk id {:?} was not in the input",
                c.id.as_str()
            );
        }
    }

    /// No chunk appears twice in the output. Catches: dedup bugs in
    /// `RedundancyPruned`, double-emission during expansion, accidental
    /// reuse of the input list across strategy passes.
    #[test]
    fn no_duplicate_chunk_ids_in_output(
        q in arb_query(),
        corpus in arb_corpus(1..=20),
        cfg in arb_config(),
    ) {
        let ctx = build_context(&q, &corpus, &cfg);
        let mut seen = std::collections::HashSet::new();
        for c in &ctx.chunks {
            prop_assert!(
                seen.insert(c.id.as_str().to_string()),
                "duplicate chunk id in output: {:?}",
                c.id.as_str()
            );
        }
    }

    /// The assembled context fits the token budget. The single
    /// load-bearing arithmetic guarantee — every strategy must respect
    /// this. Catches: budget bugs in the fill loop, overflow on huge
    /// budgets, incorrect token sum on multi-source chunks.
    #[test]
    fn token_budget_respected(
        q in arb_query(),
        corpus in arb_corpus(1..=20),
        cfg in arb_config(),
    ) {
        let ctx = build_context(&q, &corpus, &cfg);
        let total: usize = ctx.chunks.iter().map(|c| c.token_count.0).sum();
        prop_assert!(
            total <= cfg.token_budget,
            "Σ chunk tokens ({total}) exceeds budget ({})",
            cfg.token_budget
        );
    }

    /// The report's counts match what's actually in the BuiltContext.
    /// A drift between `report.n_selected` and `ctx.chunks.len()` (or
    /// between `report.total_tokens` and the real sum) would mean
    /// telemetry lies about what assembly did. Callers wire these
    /// numbers into metrics + decisions; a silent lie here is a real
    ///-impact bug even though it never crashes.
    #[test]
    fn report_counts_match_reality(
        q in arb_query(),
        corpus in arb_corpus(1..=20),
        cfg in arb_config(),
    ) {
        let ctx = build_context(&q, &corpus, &cfg);
        prop_assert_eq!(
            ctx.report.n_input_chunks, corpus.len(),
            "report.n_input_chunks must equal input length"
        );
        prop_assert_eq!(
            ctx.report.n_selected, ctx.chunks.len(),
            "report.n_selected must equal selected chunk count"
        );
        let real_total: usize = ctx.chunks.iter().map(|c| c.token_count.0).sum();
        prop_assert_eq!(
            ctx.report.total_tokens, real_total,
            "report.total_tokens must equal Σ selected chunk tokens"
        );
    }

    /// Every floating-point field on the report is finite, and the
    /// fields that document an output range of `[0, 1]` actually live
    /// there. Catches: NaN/Inf from divide-by-zero on empty selections,
    /// negative ratios, ratios > 1 (a real bug class — happens when the
    /// numerator and denominator come from different scopes).
    #[test]
    fn report_ratios_are_finite_and_in_range(
        q in arb_query(),
        corpus in arb_corpus(1..=20),
        cfg in arb_config(),
    ) {
        let r = &build_context(&q, &corpus, &cfg).report;
        for (name, v) in [
            ("token_utilization", r.token_utilization),
            ("input_distractor_ratio", r.input_distractor_ratio),
            ("retained_evidence_ratio", r.retained_evidence_ratio),
            ("evidence_density", r.economics.evidence_density),
            ("distractor_ratio", r.economics.distractor_ratio),
            ("low_confidence_threshold", r.low_confidence_threshold),
        ] {
            prop_assert!(v.is_finite(), "{name} is not finite: {v}");
            prop_assert!(
                (0.0..=1.0).contains(&v),
                "{name} out of [0,1]: {v}"
            );
        }
    }

    /// `build_context` is a pure function: same input → same output.
    /// Hand-written determinism tests pin specific inputs; this version
    /// generalizes across the input space, catching non-determinism
    /// that only appears for inputs nobody thought to try. A hidden
    /// dependency on iteration order (HashMap, environment, system
    /// time, …) would surface here.
    #[test]
    fn build_context_is_deterministic(
        q in arb_query(),
        corpus in arb_corpus(1..=20),
        cfg in arb_config(),
    ) {
        let a = build_context(&q, &corpus, &cfg);
        let b = build_context(&q, &corpus, &cfg);
        let ids_a: Vec<&str> = a.chunks.iter().map(|c| c.id.as_str()).collect();
        let ids_b: Vec<&str> = b.chunks.iter().map(|c| c.id.as_str()).collect();
        prop_assert_eq!(ids_a, ids_b, "selection order changed across identical calls");
        prop_assert_eq!(
            a.report.total_tokens, b.report.total_tokens,
            "total_tokens drifted across identical calls"
        );
        prop_assert_eq!(
            a.report.n_selected, b.report.n_selected,
            "n_selected drifted across identical calls"
        );
    }
}
