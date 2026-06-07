//! Differential strategy-semantics pinning.
//!
//! Each `ContextStrategy` has a documented behavior that distinguishes it
//! from the others:
//!
//! - `RawTopK`              — keep retrieval order, no filtering by relevance
//! - `DistractorFiltered`   — drop chunks below the grounding bar
//! - `MaxDensity`           — sort by evidence-density (most relevant-tokens-per-token first)
//! - `ReasoningPreserving`  — keep seeds; rescue below-bar chunks LINKED to a seed; drop unlinked junk
//! - `RedundancyPruned`     — skip chunks too similar (cosine on embeddings) to an already-selected one
//!
//! The inline unit tests in `src/context/mod.rs` already cover each
//! strategy in isolation. These tests do something different: they run
//! the **same corpus** through every strategy and pin the **contrasts**.
//! That catches the failure modes per-strategy tests can't:
//!
//! 1. A refactor that accidentally makes two strategies behave the same
//!    on the same input (e.g. ReasoningPreserving silently degrading to
//!    DistractorFiltered when the link-Jaccard threshold drifts).
//! 2. A regression where one strategy's filter leaks into another.
//!
//! The corpus is hand-designed so the right-vs-wrong output is sharply
//! distinguishable for each strategy. See per-test comments for the
//! per-chunk role.

use redhop::context::{build_context, ContextConfig, ContextStrategy};
use redhop::core::{
    Chunk, ChunkId, Embedding, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenCount,
};

// ── Shared corpus ──────────────────────────────────────────────────────────
//
// One query, five chunks designed so each strategy's documented behavior
// produces a distinguishable selection. Roles (the query is "what
// nationality was the safety lamp inventor"):
//
//   id      | role                                | grounding | linked-to-seed | density        | embedding
//   --------|-------------------------------------|-----------|----------------|----------------|------------------
//   hop1    | seed (clearly relevant)             | HIGH      | (is the seed)  | medium         | [1,0,0]
//   hop2    | reasoning-critical second hop —     | LOW       | YES (shares    | low            | [0,1,0]
//           |   shares bridge entity "humphry     |           |  "humphry"+    |                |
//           |   davy" with the seed               |           |  "davy" w/seed)|                |
//   junk    | true junk — unrelated to both       | LOW       | NO             | low            | [0,0,1]
//   dense   | high evidence density: every token  | HIGH      | (is also seed) | HIGHEST        | (none)
//           |   matches the query                 |           |                |                |
//   dup     | near-duplicate of hop1 (embedding   | HIGH      | (is a seed-    | medium         | [0.99,0.01,0] —
//           |   cosine ≈ 1.0 with hop1)           |           |  similar dup)  |                |  redundant w/ hop1
//
// Chunks are presented in this RETRIEVAL ORDER: hop1, hop2, junk, dense, dup.

fn rr_with_embedding(id: &str, text: &str, embedding: Option<Vec<f32>>) -> RetrievalResult {
    let token_count = text.split_whitespace().count();
    let mut c = Chunk::new(ChunkId::new(id), text, "doc", TokenCount(token_count));
    if let Some(e) = embedding {
        c = c.with_embedding(Embedding::from(e));
    }
    RetrievalResult {
        chunk: c,
        score: Score {
            value: 1.0,
            method: RetrievalMethod::Lexical,
        },
        breakdown: ScoreBreakdown::default(),
    }
}

fn corpus() -> Vec<RetrievalResult> {
    vec![
        // hop1: seed — shares {safety, lamp, was, the} with the query.
        rr_with_embedding(
            "hop1",
            "the safety lamp was invented by humphry davy",
            Some(vec![1.0, 0.0, 0.0]),
        ),
        // hop2: low query overlap, but shares "humphry"+"davy" w/ hop1.
        rr_with_embedding(
            "hop2",
            "humphry davy born penzance cornwall england chemist",
            Some(vec![0.0, 1.0, 0.0]),
        ),
        // junk: unrelated to both query and hop1's bridge entity.
        rr_with_embedding(
            "junk",
            "photosynthesis converts sunlight chemical energy green plants",
            Some(vec![0.0, 0.0, 1.0]),
        ),
        // dense: highest evidence density (every token query-relevant).
        rr_with_embedding("dense", "safety lamp inventor", None),
        // dup: lexically and embedding-wise near-identical to hop1.
        rr_with_embedding(
            "dup",
            "the safety lamp was invented by humphry davy",
            Some(vec![0.99, 0.01, 0.0]),
        ),
    ]
}

fn query() -> Query {
    Query::new("what nationality was the safety lamp inventor")
}

/// Shared knobs used across strategies so the comparison is fair. The
/// only thing that changes between tests is `strategy`.
fn cfg_with(strategy: ContextStrategy) -> ContextConfig {
    ContextConfig {
        // Plenty of headroom: budget never becomes the deciding factor.
        token_budget: 1000,
        strategy,
        // Bar high enough that hop2 and junk both fall below it (so
        // DistractorFiltered drops both, ReasoningPreserving can rescue
        // hop2 via the link, and the two strategies' outputs differ).
        distractor_min_grounding: 0.25,
        // Low link bar so the {humphry,davy} overlap between hop2 and
        // hop1 clears it — exercises the rescue path.
        link_min_jaccard: 0.05,
        // Aggressive enough to catch hop1↔dup (cosine ≈ 0.9999) but not
        // so aggressive that everything looks redundant.
        redundancy_max_cosine: 0.9,
        // Disable the Auto gate (won't apply — we pass concrete strategies — but
        // belt-and-braces in case the test corpus tokens grow).
        auto_passthrough_max_tokens: 1_000_000,
        low_confidence_max_grounding: 0.10,
        analyzer: redhop::analyzer::default_english(),
            preserve_order: false,
    }
}

fn ids(rs: &[Chunk]) -> Vec<&str> {
    rs.iter().map(|c| c.id.as_str()).collect()
}

// ── Per-strategy semantics ─────────────────────────────────────────────────

#[test]
fn raw_topk_preserves_retrieval_order_and_keeps_everything() {
    // RawTopK: no relevance filtering, no reordering, no dedup. Everything
    // fits in the budget so everything must survive in the exact order it
    // arrived. If a future refactor accidentally adds filtering to the
    // baseline strategy, this trips.
    let ctx = build_context(&query(), &corpus(), &cfg_with(ContextStrategy::RawTopK));
    assert_eq!(
        ids(&ctx.chunks),
        vec!["hop1", "hop2", "junk", "dense", "dup"],
        "RawTopK must keep retrieval order and never filter — \
         a refactor that adds relevance filtering would break this"
    );
    assert_eq!(
        ctx.report.removed.total, 0,
        "RawTopK never drops chunks under headroom"
    );
}

#[test]
fn distractor_filtered_drops_below_grounding_chunks() {
    // DistractorFiltered: every chunk below `distractor_min_grounding`
    // gets dropped — including the reasoning-critical hop2. Compare with
    // the ReasoningPreserving test below: same corpus, hop2 survives
    // there. That contrast is the semantic differentiator.
    let ctx = build_context(
        &query(),
        &corpus(),
        &cfg_with(ContextStrategy::DistractorFiltered),
    );
    let kept = ids(&ctx.chunks);
    assert!(
        kept.contains(&"hop1"),
        "seed (hop1) must be kept (above bar)"
    );
    assert!(
        !kept.contains(&"junk"),
        "junk (below bar, unlinked) must be dropped"
    );
    assert!(
        !kept.contains(&"hop2"),
        "DistractorFiltered MUST drop hop2 — it's below the grounding bar. \
         If hop2 survives, the strategy is silently behaving like \
         ReasoningPreserving. That's a strategy-leak regression."
    );
    assert!(
        ctx.report.removed.distractor >= 2,
        "report.removed.distractor must count BOTH below-bar chunks (hop2 + junk)"
    );
}

#[test]
fn max_density_packs_densest_chunk_first() {
    // MaxDensity: sort by evidence-density (relevant-tokens / total-tokens),
    // not by retrieval order. The `dense` chunk ("safety lamp inventor")
    // is 100% query-relevant — it MUST come first regardless of where it
    // sat in retrieval. The retrieval order is hop1, hop2, junk, dense,
    // dup → so a RawTopK-style strategy would put `dense` 4th. Position 0
    // ≠ position 3, sharp distinguishing assertion.
    let ctx = build_context(&query(), &corpus(), &cfg_with(ContextStrategy::MaxDensity));
    let kept = ids(&ctx.chunks);
    assert_eq!(
        kept.first().copied(),
        Some("dense"),
        "MaxDensity must put the highest-density chunk first; if `dense` \
         isn't position 0 the sort isn't running. Retrieval-order would have \
         placed `dense` 4th — this is the sharp contrast."
    );
}

#[test]
fn reasoning_preserving_rescues_linked_second_hop_drops_only_unlinked_junk() {
    // ReasoningPreserving: keeps seeds (hop1, dense, dup — all above bar),
    // RESCUES hop2 because it links to hop1 via shared bridge entity
    // "humphry davy" (Jaccard ≥ link_min_jaccard), DROPS junk (below bar
    // AND no link to any seed).
    //
    // The critical contrast with DistractorFiltered: same chunks, same
    // grounding bar, but hop2 survives here because the linkage check
    // rescues it. If hop2 doesn't survive, the rescue path is broken.
    let ctx = build_context(
        &query(),
        &corpus(),
        &cfg_with(ContextStrategy::ReasoningPreserving),
    );
    let kept = ids(&ctx.chunks);
    assert!(kept.contains(&"hop1"), "seed (hop1) must be kept");
    assert!(
        kept.contains(&"hop2"),
        "ReasoningPreserving MUST rescue hop2 — it shares the bridge entity \
         'humphry davy' with the seed (Jaccard ≥ link_min_jaccard). \
         If hop2 is dropped, the linkage rescue path is broken."
    );
    assert!(
        !kept.contains(&"junk"),
        "true junk (below bar AND unlinked to any seed) must be dropped — \
         if junk survives, the strategy is silently behaving like RawTopK"
    );
    assert!(
        ctx.report.second_hop_rescue_count >= 1,
        "report.second_hop_rescue_count must record the rescue (≥1 for hop2)"
    );
}

#[test]
fn redundancy_pruned_drops_near_duplicate_by_cosine() {
    // RedundancyPruned: walks chunks in retrieval order and skips any
    // whose embedding cosine to an already-selected chunk exceeds
    // `redundancy_max_cosine`. hop1 is selected first; dup arrives later
    // with cosine ≈ 0.9999 to hop1's embedding → dup MUST be dropped.
    // hop2 (orthogonal embedding) and junk (orthogonal embedding) survive
    // because their cosine to anything selected is ~0.
    let ctx = build_context(
        &query(),
        &corpus(),
        &cfg_with(ContextStrategy::RedundancyPruned),
    );
    let kept = ids(&ctx.chunks);
    assert!(kept.contains(&"hop1"), "first-arriving chunk hop1 is kept");
    assert!(
        !kept.contains(&"dup"),
        "dup MUST be pruned — its embedding cosine to hop1 is ≈ 1.0, well \
         above redundancy_max_cosine=0.9. If dup survives, redundancy \
         detection is broken."
    );
    assert!(
        ctx.report.removed.redundant >= 1,
        "report.removed.redundant must record the dup drop"
    );
}

// ── Differential contrasts ────────────────────────────────────────────────

#[test]
fn reasoning_preserving_and_distractor_filtered_diverge_on_second_hop() {
    // The single most important contrast in the whole strategy taxonomy:
    // on the SAME corpus with the SAME grounding bar, ReasoningPreserving
    // KEEPS hop2 and DistractorFiltered DROPS it. If these two strategies
    // ever produce identical outputs on this corpus, ReasoningPreserving
    // has collapsed into DistractorFiltered — exactly the "second-hop
    // tax" failure the strategy exists to mitigate.
    let q = query();
    let chunks = corpus();
    let preserving = build_context(&q, &chunks, &cfg_with(ContextStrategy::ReasoningPreserving));
    let filtered = build_context(&q, &chunks, &cfg_with(ContextStrategy::DistractorFiltered));

    let preserving_ids = ids(&preserving.chunks);
    let filtered_ids = ids(&filtered.chunks);

    assert!(
        preserving_ids.contains(&"hop2"),
        "ReasoningPreserving must keep hop2"
    );
    assert!(
        !filtered_ids.contains(&"hop2"),
        "DistractorFiltered must drop hop2"
    );
    assert_ne!(
        preserving_ids, filtered_ids,
        "the two strategies MUST produce different selections on this corpus — \
         this is the second-hop tax contrast made executable. If they tie, one \
         strategy has silently collapsed into the other."
    );
}

#[test]
fn raw_topk_and_max_density_diverge_on_ordering() {
    // RawTopK keeps retrieval order; MaxDensity sorts by density. On a
    // corpus where the densest chunk is NOT first-retrieved, the two must
    // produce different orderings. `dense` is retrieved 4th but is the
    // densest → MaxDensity puts it first, RawTopK doesn't.
    let q = query();
    let chunks = corpus();
    let raw = build_context(&q, &chunks, &cfg_with(ContextStrategy::RawTopK));
    let dense = build_context(&q, &chunks, &cfg_with(ContextStrategy::MaxDensity));
    assert_ne!(
        ids(&raw.chunks),
        ids(&dense.chunks),
        "RawTopK and MaxDensity must produce different orderings on a corpus \
         where the densest chunk isn't first-retrieved. If they match, the \
         density sort isn't running."
    );
}
