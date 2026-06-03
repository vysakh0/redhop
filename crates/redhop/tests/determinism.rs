//! Determinism guarantees — same query + same corpus must produce the same
//! result on repeat runs. Cross-binding parity tests already assume this
//! (Python and Node are diffed against each other, which presumes both are
//! deterministic); this file is the missing direct check.
//!
//! Three deterministic invariants we pin:
//!
//! 1. **Chunk IDs** — `Document::from_chunks_with` over the same input must
//!    produce the same `Chunk.id` sequence, run after run. If not, the
//!    persisted-folder cache can never hit (the fingerprint matches but the
//!    chunks are different) and the cross-binding parity tests break.
//! 2. **Retrieval order** — `Document.context(query)` must return cited
//!    chunks in the same order across runs (BM25 is deterministic; the
//!    test pins that we haven't accidentally introduced non-determinism
//!    elsewhere — HashMap iteration, sort-by-key with ties, etc.).
//! 3. **Report totals** — `report.total_tokens`, `n_selected`,
//!    `retained_evidence_ratio`, `auto_decision` must match bit-for-bit on
//!    repeat runs.
//!
//! All three are run on a **mixed** corpus (code + prose, multiple files)
//! to exercise the sort/dedup paths that are most prone to HashMap-order
//! drift.

use redhop::core::{Chunk, ChunkId, TokenCount};
use redhop::{Document, DocumentConfig};

fn chunk(id: &str, text: &str, source: &str, kind: &str) -> Chunk {
    let mut c = Chunk::new(
        ChunkId::new(id),
        text,
        source,
        TokenCount(text.split_whitespace().count().max(1)),
    );
    c.metadata.insert("kind".into(), serde_json::json!(kind));
    c
}

/// A corpus shaped to exercise the sort/dedup paths: multiple sources,
/// mixed code + prose, on-topic + off-topic, near-duplicate text.
fn mixed_corpus() -> Vec<Chunk> {
    vec![
        chunk(
            "a",
            "the refund window is thirty days from purchase",
            "policy.md",
            "prose",
        ),
        chunk(
            "b",
            "customers may return items within 30 days",
            "policy.md",
            "prose",
        ),
        chunk("c", "fn compress_video(path: &str)", "video.rs", "code"),
        chunk("d", "fn decompress_video(path: &str)", "video.rs", "code"),
        chunk(
            "e",
            "photosynthesis converts sunlight into glucose",
            "bio.md",
            "prose",
        ),
        chunk(
            "f",
            "thirty-day return policy with full refund",
            "faq.md",
            "prose",
        ),
    ]
}

#[test]
fn determinism_chunk_ids_stable_across_runs() {
    // `from_chunks_with` over the same input twice — the assigned chunk ids
    // (which the persisted-folder cache and the cross-binding parity tests
    // both rely on as a stable handle) must match.
    let cfg = DocumentConfig::default();
    let doc1 = Document::from_chunks_with(mixed_corpus(), cfg.clone()).unwrap();
    let doc2 = Document::from_chunks_with(mixed_corpus(), cfg).unwrap();
    let ids1: Vec<&str> = doc1.chunks().iter().map(|c| c.id.as_str()).collect();
    let ids2: Vec<&str> = doc2.chunks().iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids1, ids2,
        "chunk ids must be deterministic across `from_chunks_with` runs"
    );
}

#[test]
fn determinism_retrieval_order_stable_across_runs() {
    // Same query, same corpus, two fresh Documents → same cited chunks in
    // the same order. The interesting case: queries with score ties (which
    // HashMap-order would scramble). "refund" matches both a and f equally
    // and b similarly; the order must still be reproducible.
    let cfg = DocumentConfig::default();
    let mut doc1 = Document::from_chunks_with(mixed_corpus(), cfg.clone()).unwrap();
    let mut doc2 = Document::from_chunks_with(mixed_corpus(), cfg).unwrap();
    let ctx1 = doc1.context("refund window").unwrap();
    let ctx2 = doc2.context("refund window").unwrap();
    let ids1: Vec<&str> = ctx1.chunks.iter().map(|c| c.id.as_str()).collect();
    let ids2: Vec<&str> = ctx2.chunks.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids1, ids2,
        "context() cited-chunk order must be deterministic; got run1={ids1:?} run2={ids2:?}"
    );
}

#[test]
fn determinism_report_totals_stable_across_runs() {
    // Reports also have to be byte-equal on repeats — otherwise downstream
    // metrics dashboards see noise. The fields covered here are the ones
    // the cross-binding parity tests diff on, plus a few more.
    let cfg = DocumentConfig::default();
    let mut doc1 = Document::from_chunks_with(mixed_corpus(), cfg.clone()).unwrap();
    let mut doc2 = Document::from_chunks_with(mixed_corpus(), cfg).unwrap();
    let r1 = doc1.context("refund window").unwrap().report;
    let r2 = doc2.context("refund window").unwrap().report;
    assert_eq!(r1.total_tokens, r2.total_tokens, "total_tokens drift");
    assert_eq!(r1.n_selected, r2.n_selected, "n_selected drift");
    assert_eq!(r1.input_tokens, r2.input_tokens, "input_tokens drift");
    assert_eq!(
        r1.retained_evidence_ratio, r2.retained_evidence_ratio,
        "retained_evidence_ratio drift"
    );
    assert_eq!(
        r1.auto_decision(),
        r2.auto_decision(),
        "auto_decision drift"
    );
    assert_eq!(
        r1.input_distractor_ratio, r2.input_distractor_ratio,
        "input_distractor_ratio drift"
    );
    assert_eq!(
        r1.second_hop_rescue_count, r2.second_hop_rescue_count,
        "second_hop_rescue_count drift"
    );
}

#[test]
fn determinism_repeated_context_calls_on_same_doc() {
    // Same Document instance, same query called twice — must produce the
    // same result. This catches state-mutating bugs (cache invalidation
    // gone wrong, lazy-init that reorders on second call) which the
    // two-fresh-Document tests above wouldn't.
    let mut doc = Document::from_chunks_with(mixed_corpus(), DocumentConfig::default()).unwrap();
    let ctx1 = doc.context("refund window").unwrap();
    let ctx2 = doc.context("refund window").unwrap();
    let ids1: Vec<&str> = ctx1.chunks.iter().map(|c| c.id.as_str()).collect();
    let ids2: Vec<&str> = ctx2.chunks.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids1, ids2,
        "repeated context() on the same Document must be deterministic"
    );
    assert_eq!(ctx1.report.total_tokens, ctx2.report.total_tokens);
    assert_eq!(ctx1.text(), ctx2.text(), "assembled text must match");
}
