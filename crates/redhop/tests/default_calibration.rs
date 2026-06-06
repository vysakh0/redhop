//! Default-value calibration pins.
//!
//! Every tuned constant on `ContextConfig::default()` /
//! `DocumentConfig::default()` exists because a specific finding measured
//! that value to be the right one. If someone changes a default in a
//! refactor without going back to the calibration, no other test fails —
//! everything still type-checks, the suite still passes, but the runtime
//! silently behaves differently than what the evidence layer documents.
//!
//! This file pins each tuned default to its documented value and points
//! at the finding that calibrates it. Changing a default now requires:
//!
//!   1. Editing the constant in `src/{context,document}/mod.rs`.
//!   2. Editing the assertion here.
//!   3. Updating the linked finding (or writing a new one explaining why
//!      the calibration shifted).
//!
//! That three-step ratchet is the whole point. Silent drift becomes
//! impossible; intentional drift is documented in the same commit that
//! makes it.
//!
//! Not pinned here:
//! - **Non-tuned constants** (e.g. `min_candidates: 0` — chosen as the
//!   safe default, not calibrated against a measurement). Those can move
//!   on judgment without a corresponding finding update.
//! - **Type-level defaults** like `strategy: ReasoningPreserving` —
//!   pinned by separate tests (`invariant_default_strategy_is_not_*`).

use redhop::context::{ContextConfig, ContextStrategy};
use redhop::DocumentConfig;

// ── ContextConfig defaults ────────────────────────────────────────────────

/// 8192-token budget aligns Rust/Node with the long-shipping Python
/// default (PyPI users have used this since the wheel shipped). Most
/// production LLMs have ≥32k usable context; a smaller default leaves
/// capacity on the table. Documented in 0.2.0 release notes
/// (`CHANGELOG.md`) — the 2048 → 8192 bump was a breaking change for
/// Rust callers and is the single most user-visible default in the
/// crate.
#[test]
fn default_token_budget_is_8192() {
    assert_eq!(
        ContextConfig::default().token_budget,
        8192,
        "token_budget default drifted from the cross-binding contract. \
         If this is intentional, update CHANGELOG.md (it's a public-API \
         break) and adjust the Python binding's advertised default."
    );
}

/// 1500 — calibrated by the size sweep in
/// `docs/findings/CONTEXT_DILUTION.md`. Pruning helps at every measured
/// size from ~1.5k tokens up on gpt-4o-mini (monotonic, all
/// CI-significant), with no harmful regime above. 1500 is the
/// conservative low edge of the measured-benefit range: prune where
/// evidence shows it helps, pass through below it where evidence is
/// absent. Moving this is moving the calibrated dilution gate —
/// re-run the size sweep before doing so.
#[test]
fn default_auto_passthrough_max_tokens_is_1500() {
    assert_eq!(
        ContextConfig::default().auto_passthrough_max_tokens,
        1_500,
        "auto_passthrough_max_tokens is the calibrated dilution gate. \
         See docs/findings/CONTEXT_DILUTION.md — moving this without \
         re-running the size sweep ships a silently-different runtime."
    );
}

/// 0.10 — a low absolute bar: only near-zero-overlap junk is below it.
/// Used both for `DistractorFiltered` (drop below-bar chunks) and as
/// the seed/junk separator for `ReasoningPreserving`'s linkage rescue.
/// The choice of "low absolute bar" is the safety property: we'd
/// rather rescue a borderline chunk than drop a relevant one. See
/// reasoning in the `ContextConfig::default()` block + the second-hop
/// findings.
#[test]
fn default_distractor_min_grounding_is_010() {
    assert_eq!(
        ContextConfig::default().distractor_min_grounding,
        0.10,
        "distractor_min_grounding is the seed/junk separator. Lowering \
         drops the rescue safety net; raising aggressively prunes \
         linked second hops (the bug the strategy exists to prevent)."
    );
}

/// 0.92 — cosine ceiling for `RedundancyPruned`. Tight enough to flag
/// near-duplicates but loose enough that two genuinely-similar chunks
/// (same topic, different phrasing) survive both. Not aggressively
/// calibrated by a finding — but is the historical default callers
/// have depended on; moving it silently changes redundancy behavior
/// across every embedding-using corpus.
#[test]
fn default_redundancy_max_cosine_is_092() {
    assert_eq!(
        ContextConfig::default().redundancy_max_cosine,
        0.92,
        "redundancy_max_cosine is the dedup ceiling. Moving silently \
         changes RedundancyPruned behavior on every embedded corpus."
    );
}

/// 0.12 — Jaccard floor below which a low-relevance chunk is treated
/// as unlinked junk rather than a rescuable second hop. The single
/// load-bearing knob for `ReasoningPreserving`'s linkage rescue
/// behavior. Calibrated alongside `distractor_min_grounding=0.10` so
/// the seed/junk separation and the bridge-entity rescue land in the
/// same regime.
#[test]
fn default_link_min_jaccard_is_012() {
    assert_eq!(
        ContextConfig::default().link_min_jaccard,
        0.12,
        "link_min_jaccard is the bridge-entity rescue floor for \
         ReasoningPreserving. Drift here moves the second-hop \
         retention rate documented in SECOND_HOP_TAX.md."
    );
}

/// 0.10 — matches `distractor_min_grounding`. The `low_confidence_retrieval`
/// signal fires when every selected chunk is at-or-below distractor
/// relevance, i.e. the assembled context is all noise. Tying the two
/// defaults together means "we're returning nothing above the relevance
/// bar" is the canonical low-confidence trigger.
#[test]
fn default_low_confidence_max_grounding_matches_distractor_bar() {
    let cfg = ContextConfig::default();
    assert_eq!(
        cfg.low_confidence_max_grounding, cfg.distractor_min_grounding,
        "low_confidence_max_grounding should default to the distractor \
         bar — they're calibrated together. If decoupling is intentional, \
         document why in the `ContextConfig::default()` block."
    );
}

// ── DocumentConfig defaults ───────────────────────────────────────────────

/// 128-token chunks — sweep across budgets/datasets in
/// `docs/findings/CHUNK_GRANULARITY.md` showed finer chunks pack better
/// under tight budgets (multi-hop ≥0.8 retention 54%→77%) and tie at
/// large budgets. 128 is the robust default over the previous 256.
#[test]
fn default_chunk_target_tokens_is_128() {
    assert_eq!(
        DocumentConfig::default().target_tokens,
        128,
        "target_tokens is calibrated by CHUNK_GRANULARITY.md. Moving \
         shifts the entire chunk-size economy; rerun the sweep first."
    );
}

/// 20 candidates retrieved before assembly. Calibrated so the assembled
/// context has enough material to express selectivity but not so much
/// that BM25 noise dominates. The lower bound below which strategies
/// can't meaningfully discriminate.
#[test]
fn default_candidate_k_is_20() {
    assert_eq!(
        DocumentConfig::default().candidate_k,
        20,
        "candidate_k below 20 starves the strategy of material; well \
         above 20 trades determinism for marginal recall. Drift here \
         changes the retrieval/assembly trade-off all callers depend on."
    );
}

/// `DocumentConfig` defaults to `ContextStrategy::Auto`. The runtime's
/// shipped philosophy: size-gated decision (prune large inputs,
/// passthrough small ones), no per-call surprises. The `ContextConfig`
/// inside `DocumentConfig` overrides the bare `ContextConfig::default()`
/// strategy (`ReasoningPreserving`) so document-level callers get the
/// gate.
#[test]
fn default_document_strategy_is_auto() {
    assert_eq!(
        DocumentConfig::default().context.strategy,
        ContextStrategy::Auto,
        "DocumentConfig wires Auto so document.context() takes the \
         size-gated decision by default. If this changes, the runtime \
         philosophy documented in README + API_STABILITY.md changes too."
    );
}
