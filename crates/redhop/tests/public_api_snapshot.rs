//! Public API surface snapshot — compile-time guard against silent renames.
//!
//! Every public symbol promised by `docs/API_STABILITY.md` is imported by
//! name and (where applicable) bound at a fixed signature below. If a
//! future refactor renames or removes a public item, **this file fails to
//! compile** with `error[E0432]: unresolved import` or
//! `error[E0308]: mismatched types`, pointing at exactly the contract
//! that changed.
//!
//! Why a test file instead of `cargo-public-api` or a doctool:
//!
//! 1. No external binary or CI plumbing required — just `cargo test`.
//! 2. The failure mode is a clean Rust compile error naming the missing
//!    symbol, not a diff against a snapshot file someone has to interpret.
//! 3. Adds zero ongoing maintenance: the test is purely additive (new
//!    public items don't break it; only renames/removals do, which is the
//!    correct failure semantics for stability).
//!
//! Reading this file:
//! - Each `use` statement is a CLAIM: "this path resolves to a stable
//!   public item."
//! - Each type-bound `let` is a CLAIM: "this function/method exists with
//!   THIS exact signature."
//! - A bare `use` catches renames + removals; a typed `let` also catches
//!   silent signature changes (arg-order swaps, return-type changes).

#![allow(dead_code, unused_imports)]

// ── Top-level re-exports promised in API_STABILITY.md ──────────────────────

// High-level Document surface.
use redhop::{Document, DocumentConfig, RetrievalMode, Section};

// Built-context types + lower-level entry points.
use redhop::{
    analyze_context, build_context, context_economics, filter_context, grounding_score,
    link_strength, AutoDecision, BuiltContext, ContextConfig, ContextReport, ContextStrategy,
};

// Core types callers handle directly.
use redhop::{
    Chunk, ChunkId, Embedding, Error, Query, Result, RetrievalMethod, RetrievalResult, Score,
    ScoreBreakdown, TokenCount,
};

// Citations + loader options.
use redhop::{
    chunks as chunks_fn, citations, retrieval_from_str, strategy_from_str, text as text_fn,
    Citation, FolderOptions, LoadOptions,
};

// `files` feature loaders (gated — only available when the feature is on).
#[cfg(feature = "files")]
use redhop::{
    read_bytes, read_bytes_with, read_file, read_file_with, read_folder, read_folder_with,
};

// Pluggable abstractions for advanced callers.
use redhop::traits::{Chunker, EmbeddingProvider, Retriever, TokenizerBackend};

// Analyzer module surface.
use redhop::analyzer::{default_english, Analyzer, SnowballAnalyzer};

// Stable module paths — these are referenced by the documented surface
// even if most callers use the top-level re-exports.
use redhop::context as _context;
use redhop::core as _core;
use redhop::document as _document;

// ── Signature pins for the load-bearing public functions ───────────────────
//
// Each `const _: SomeFnType = redhop::fn` would fail to compile if the
// function's signature drifted. A bare `use` only catches renames;
// these catch silent signature changes — adding an arg, swapping order,
// changing the return type.
//
// Functions that take `impl Trait` arguments (e.g. `Document::from_text`)
// can't be bound to fn-pointer types — they're called inside the
// `document_methods_exist` test instead, where the call expression
// type-checks the signature.

#[allow(non_upper_case_globals)]
const grounding_score_signature: fn(&str, &str) -> f32 = redhop::grounding_score;

#[allow(non_upper_case_globals)]
const link_strength_signature: fn(&str, &str) -> f32 = redhop::link_strength;

#[allow(non_upper_case_globals)]
const strategy_from_str_signature: fn(&str) -> Result<ContextStrategy> = redhop::strategy_from_str;

#[allow(non_upper_case_globals)]
const build_context_signature: fn(&Query, &[RetrievalResult], &ContextConfig) -> BuiltContext =
    redhop::build_context;

#[allow(non_upper_case_globals)]
const analyze_context_signature: fn(&Query, &[RetrievalResult], &ContextConfig) -> ContextReport =
    redhop::analyze_context;

#[allow(non_upper_case_globals)]
const filter_context_signature: fn(&Query, &[RetrievalResult], &ContextConfig) -> BuiltContext =
    redhop::filter_context;

// ── Method-presence + signature checks for `Document` ─────────────────────
//
// Call-expression type-checks: a rename fails with E0599 (no method named
// X); an arg-order swap or type change fails with E0308 (mismatched
// types). Wrapped in a #[test] so cargo treats it as part of the harness;
// the function body is never executed at runtime (the early return is
// just to avoid panicking on no-chunk inputs).

#[test]
fn document_constructor_signatures_pinned() {
    // Bail out before doing any real work — the body exists purely so
    // the COMPILER type-checks the call expressions below. Each call
    // pins the constructor's positional argument order + types.
    if std::env::var("REDHOP_PUBLIC_API_SNAPSHOT_NEVER_RUN_THIS").is_ok() {
        let _d: Result<Document> = Document::from_text("source.md", "the document text");
        let _d: Result<Document> =
            Document::from_text_with("source.md", "the document text", DocumentConfig::default());
        let chunks: Vec<Chunk> = vec![Chunk::new(
            ChunkId::new("c1"),
            "hello",
            "src",
            TokenCount(1),
        )];
        let _d: Result<Document> = Document::from_chunks(chunks.clone());
        let _d: Result<Document> = Document::from_chunks_with(chunks, DocumentConfig::default());
    }
}

// ── Enum-variant pins for stable string mappings ───────────────────────────
//
// The string forms of `ContextStrategy` and `AutoDecision` are part of the
// stable surface (see API_STABILITY.md "Stable semantics"). The compiler
// catches a variant rename via exhaustive `match`; if a variant is added,
// this test prompts the author to consider whether the new variant has a
// stable string form.

#[test]
fn context_strategy_variants_are_exhaustive() {
    fn _exhaustive(s: ContextStrategy) -> &'static str {
        match s {
            ContextStrategy::RawTopK => "raw_topk",
            ContextStrategy::DistractorFiltered => "distractor_filtered",
            ContextStrategy::RedundancyPruned => "redundancy_pruned",
            ContextStrategy::MaxDensity => "max_density",
            ContextStrategy::ReasoningPreserving => "reasoning_preserving",
            ContextStrategy::Auto => "auto",
        }
    }
}

#[test]
fn auto_decision_variants_are_exhaustive() {
    fn _exhaustive(d: AutoDecision) -> &'static str {
        match d {
            AutoDecision::NotAuto => "not_auto",
            AutoDecision::Passthrough => "passthrough",
            AutoDecision::Prune => "prune",
        }
    }
}

#[test]
fn retrieval_method_variants_are_exhaustive() {
    fn _exhaustive(m: RetrievalMethod) -> &'static str {
        match m {
            RetrievalMethod::Lexical => "lexical",
            RetrievalMethod::Dense => "dense",
            RetrievalMethod::Hybrid => "hybrid",
            RetrievalMethod::Rerank => "rerank",
            RetrievalMethod::External => "external",
        }
    }
}

// ── Sanity: the snapshot itself compiles into one actual test runtime ─────

/// The presence of any `#[test]` ensures cargo treats this as a runnable
/// integration test rather than a compile-only file. The other tests
/// above each cover a distinct class of public-API contract; this final
/// no-op asserts the surface is internally consistent enough to import
/// at runtime (catches a class of broken re-export that compiles but
/// can't actually be linked).
#[test]
fn public_surface_loads_at_runtime() {
    let _ = ContextConfig::default();
    let _ = DocumentConfig::default();
    let _ = ContextStrategy::Auto;
    let _ = AutoDecision::NotAuto;
    let _ = ChunkId::new("smoke");
    let _ = Query::new("smoke");
}
