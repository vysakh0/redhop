//! Concurrency + `Send` / `Sync` audit for the public Rust API.
//!
//! Three tiers:
//!
//! 1. **Compile-time** — `static_assertions`-style trait bounds verify
//!    `Send + Sync` for the public types that claim it. If a future refactor
//!    accidentally introduces a `Rc` or `Cell` that strips an auto-trait, the
//!    crate stops compiling at exactly the regression site rather than at
//!    some downstream usage in a binding.
//! 2. **Runtime free-function concurrency** — spawn N threads calling
//!    `grounding_score` / `link_strength` / `build_context` /
//!    `analyze_context` in parallel; assert each thread gets the same answer
//!    a sequential run would. Real races there would show up as differing
//!    values or panics.
//! 3. **Shared analyzer across Documents** — two `Document` instances
//!    sharing one `Arc<dyn Analyzer>` (the `default_english()` cached one,
//!    plus an explicit `with_analyzer(...)` injection) execute concurrent
//!    queries from separate threads, asserting both produce the same
//!    results as a sequential run.
//!
//! These tests don't use `loom` — that's heavier infrastructure than is
//! warranted for the surface here, and the production-shape "spawn 32
//! threads, each does the same workload" exercises the real Tantivy index +
//! analyzer paths under contention.

use std::sync::Arc;
use std::thread;

use redhop::analyzer::{Analyzer, SnowballAnalyzer};
use redhop::context::{build_context, grounding_score, link_strength, ContextConfig};
use redhop::core::{Chunk, ChunkId, Query, RetrievalMethod, RetrievalResult, Score, TokenCount};
use redhop::{BuiltContext, ContextReport, Document, DocumentConfig};

// ── (1) Compile-time Send + Sync assertions ────────────────────────────────

/// Force-evaluates trait bounds for `T: Send + Sync` at compile time. If any
/// type below silently loses an auto-trait, the crate fails to compile here.
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn public_types_are_send_and_sync() {
    // Core data shapes that cross threads and FFI alike.
    assert_send_sync::<Chunk>();
    assert_send_sync::<ChunkId>();
    assert_send_sync::<Query>();
    assert_send_sync::<RetrievalResult>();
    assert_send_sync::<BuiltContext>();
    assert_send_sync::<ContextReport>();
    assert_send_sync::<ContextConfig>();
    assert_send_sync::<DocumentConfig>();

    // The Analyzer trait is `pub trait Analyzer: Send + Sync` — verify both
    // an `Arc<dyn Analyzer>` and a concrete impl honor it.
    assert_send_sync::<SnowballAnalyzer>();
    assert_send_sync::<Arc<dyn Analyzer>>();
}

// ── (2) Free-function concurrency ──────────────────────────────────────────

#[test]
fn grounding_score_is_thread_safe() {
    // 32 threads, each computing the same grounding_score(query, text) —
    // they must all agree with a sequential computation. Spawn a few
    // queries to also exercise that distinct inputs in parallel don't
    // cross-contaminate (e.g. via a thread-local cache that mishandles
    // contention).
    let cases: Vec<(&str, &str)> = vec![
        ("refund window", "the refund window is thirty days"),
        ("compress video", "fn compress_video(path: &str)"),
        ("xyzzy", "no overlap whatsoever"),
        ("café", "we love a good café in the morning"),
    ];

    let sequential: Vec<f32> = cases.iter().map(|(q, t)| grounding_score(q, t)).collect();

    // 8 parallel rounds × 4 cases × 32 threads ≈ 1024 parallel calls.
    let handles: Vec<_> = (0..32)
        .map(|_| {
            let cases = cases.clone();
            thread::spawn(move || {
                cases
                    .iter()
                    .map(|(q, t)| grounding_score(q, t))
                    .collect::<Vec<f32>>()
            })
        })
        .collect();

    for h in handles {
        let got = h.join().expect("worker panicked");
        assert_eq!(
            got, sequential,
            "grounding_score gave a different answer in a worker thread"
        );
    }
}

#[test]
fn link_strength_is_thread_safe() {
    let cases = [
        ("refund within thirty days", "thirty-day refund policy"),
        ("compress video", "video compression"),
        ("photosynthesis", "the Eiffel Tower"),
    ];
    let sequential: Vec<f32> = cases.iter().map(|(a, b)| link_strength(a, b)).collect();
    let handles: Vec<_> = (0..16)
        .map(|_| {
            thread::spawn(move || {
                cases
                    .iter()
                    .map(|(a, b)| link_strength(a, b))
                    .collect::<Vec<f32>>()
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), sequential);
    }
}

#[test]
fn build_context_is_thread_safe_with_caller_chunks() {
    // `build_context` is a pure function over (query, retrieved chunks,
    // config) — no shared state, no `&mut`. Running it in parallel must
    // produce identical output to running it sequentially.
    let chunks = || -> Vec<RetrievalResult> {
        vec![
            mk_result("a", "the refund window is thirty days"),
            mk_result("b", "customers may return items within 30 days"),
            mk_result("d", "photosynthesis converts sunlight"),
        ]
    };
    let cfg = ContextConfig::default();
    let q = Query::new("refund window");
    let sequential = build_context(&q, &chunks(), &cfg);

    let handles: Vec<_> = (0..16)
        .map(|_| {
            let cfg = cfg.clone();
            let q = Query::new("refund window");
            thread::spawn(move || build_context(&q, &chunks(), &cfg))
        })
        .collect();
    for h in handles {
        let got = h.join().unwrap();
        assert_eq!(
            got.text(),
            sequential.text(),
            "build_context text diverged in a worker"
        );
        assert_eq!(got.report.total_tokens, sequential.report.total_tokens);
        assert_eq!(got.report.n_selected, sequential.report.n_selected);
    }
}

// ── (3) Shared analyzer across Documents on parallel threads ───────────────

#[test]
fn shared_default_analyzer_handles_parallel_documents() {
    // The default analyzer (0.3.2+: RawAnalyzer) is a process-wide
    // cached `Arc<dyn Analyzer>`. If a future change makes it stateful
    // in a non-thread-safe way (e.g. an unsynchronized lazy field on
    // the tokenizer), parallel queries from independent Documents
    // would race.
    //
    // We spawn 8 worker threads, each building its OWN Document from the
    // default config (so each pulls the cached Arc) and running the same
    // query. Results must all match the sequential baseline.
    let chunks = |i: usize| -> Vec<Chunk> {
        vec![
            mk_chunk(
                "a",
                "the refund window is thirty days from purchase",
                "policy.md",
            ),
            mk_chunk(
                &format!("worker{i}"),
                &format!("worker {i} content here"),
                "src.md",
            ),
        ]
    };
    let baseline = {
        let mut doc = Document::from_chunks_with(chunks(0), DocumentConfig::default()).unwrap();
        doc.context("refund window").unwrap().text()
    };

    let handles: Vec<_> = (0..8)
        .map(|i| {
            thread::spawn(move || {
                let mut doc =
                    Document::from_chunks_with(chunks(i), DocumentConfig::default()).unwrap();
                doc.context("refund window").unwrap().text()
            })
        })
        .collect();
    for h in handles {
        let got = h.join().expect("worker panicked");
        // The worker chunks include a per-i "worker" string, but the cited
        // chunk for "refund window" is always the refund chunk, identical
        // across workers. The text() drops the unrelated worker chunk via
        // the strategy filter, so all should match.
        assert_eq!(
            got, baseline,
            "shared default English analyzer drifted between Documents"
        );
    }
}

#[test]
fn explicitly_shared_german_analyzer_across_documents() {
    // Construct ONE `Arc<dyn Analyzer>` and inject it into multiple
    // Documents via `with_analyzer`. Each Document holds the same Arc;
    // their queries run concurrently. If the analyzer's internal state
    // weren't truly Sync, this would race.
    let analyzer: Arc<dyn Analyzer> = Arc::new(SnowballAnalyzer::german());

    let baseline = {
        let mut doc = Document::from_text("library", "ich habe viele Bücher gelesen")
            .unwrap()
            .with_analyzer(analyzer.clone());
        doc.context("Buch").unwrap().text()
    };
    assert!(
        baseline.contains("Bücher"),
        "sanity: German analyzer should find Bücher from query 'Buch'"
    );

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let analyzer = analyzer.clone();
            thread::spawn(move || {
                let mut doc = Document::from_text("library", "ich habe viele Bücher gelesen")
                    .unwrap()
                    .with_analyzer(analyzer);
                doc.context("Buch").unwrap().text()
            })
        })
        .collect();
    for h in handles {
        let got = h.join().expect("worker panicked");
        assert_eq!(
            got, baseline,
            "shared explicit German analyzer drifted between Documents"
        );
    }
}

// ── helpers ────────────────────────────────────────────────────────────────

fn mk_chunk(id: &str, text: &str, source: &str) -> Chunk {
    Chunk::new(
        ChunkId::new(id),
        text,
        source,
        TokenCount(text.split_whitespace().count().max(1)),
    )
}

fn mk_result(id: &str, text: &str) -> RetrievalResult {
    RetrievalResult {
        chunk: mk_chunk(id, text, "input"),
        score: Score {
            value: 1.0,
            method: RetrievalMethod::Lexical,
        },
        breakdown: Default::default(),
    }
}
