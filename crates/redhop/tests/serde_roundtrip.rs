//! Serde round-trip integrity for types that cross FFI.
//!
//! `Chunk`, `RetrievalResult`, `Score`, `ScoreBreakdown`, and
//! `ContextReport` all derive `Serialize`/`Deserialize` because the Python
//! and Node bindings move them across language boundaries as JSON. The
//! load-bearing invariant: **serialize → deserialize must return an equal
//! value**. If a new field gets added without `#[serde(default)]`,
//! deserializing a payload produced by an older binary errors instead of
//! degrading gracefully — a real cross-version-compatibility footgun.
//!
//! These tests serialize representative values to JSON, parse them back,
//! and assert equality. They also exercise the "older payload missing a
//! newer field" path: serialize to a minimal subset, deserialize the
//! superset, and assert the missing field gets its default.

use redhop::context::{analyze_context, build_context, ContextConfig, ContextReport};
use redhop::core::{
    Chunk, ChunkId, Embedding, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenCount,
};

// ── Round-trip helpers ─────────────────────────────────────────────────────

/// Generic round-trip: serialize → parse → serialize, and compare the
/// resulting JSON values. Avoids requiring `PartialEq` on the value type
/// (most of these don't derive it — `Chunk` carries an `Embedding` whose
/// `PartialEq` is intentionally absent because float equality is meaningless
/// for embeddings) while still catching "deserialize succeeds but
/// reconstructs differently" drift.
fn roundtrip<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let json1 = serde_json::to_string(value).expect("serialize must succeed");
    let parsed: T = serde_json::from_str(&json1)
        .unwrap_or_else(|e| panic!("deserialize must succeed; json was {json1:?}, error: {e}"));
    let json2 = serde_json::to_string(&parsed).expect("re-serialize must succeed");
    let v1: serde_json::Value = serde_json::from_str(&json1).unwrap();
    let v2: serde_json::Value = serde_json::from_str(&json2).unwrap();
    assert_eq!(
        v1, v2,
        "round-trip changed the serialized JSON shape:\n  before: {json1}\n  after:  {json2}"
    );
}

// ── Chunk + ChunkId ────────────────────────────────────────────────────────

#[test]
fn chunk_id_roundtrips() {
    let ids = [
        ChunkId::new("a"),
        ChunkId::new("very-long-id-with-special.chars_and_dashes-123"),
        ChunkId::new(""),     // empty id — degenerate but must round-trip
        ChunkId::new("café"), // unicode
    ];
    for id in &ids {
        roundtrip(id);
    }
}

#[test]
fn chunk_without_embedding_roundtrips() {
    let mut c = Chunk::new(
        ChunkId::new("c1"),
        "the refund window is thirty days",
        "policy.md",
        TokenCount(7),
    );
    c.metadata.insert("kind".into(), serde_json::json!("prose"));
    c.metadata.insert("page".into(), serde_json::json!(3));
    roundtrip(&c);
}

#[test]
fn chunk_with_embedding_roundtrips() {
    // Embedding is a Vec<f32> — serde JSON might silently lose precision on
    // f32; pin that the values round-trip exactly. Construct with
    // representable values so the comparison is bit-equal.
    let mut c = Chunk::new(
        ChunkId::new("e1"),
        "semantic match query content",
        "input",
        TokenCount(4),
    );
    c = c.with_embedding(Embedding::from(vec![0.5_f32, 0.25, 0.125, -0.0625]));
    roundtrip(&c);
}

// ── Score + ScoreBreakdown ─────────────────────────────────────────────────

#[test]
fn score_roundtrips_for_every_retrieval_method() {
    for method in [
        RetrievalMethod::Lexical,
        RetrievalMethod::Dense,
        RetrievalMethod::Hybrid,
    ] {
        let s = Score {
            value: 0.875,
            method,
        };
        roundtrip(&s);
    }
}

#[test]
fn score_breakdown_roundtrips_with_partial_fields() {
    // ScoreBreakdown's fields are all Option<f32>. Test the common shapes
    // (all None, just lexical, lexical+dense, all set) so a future field
    // addition needs both serde defaults AND a test entry.
    let cases = [
        ScoreBreakdown::default(),
        ScoreBreakdown {
            lexical: Some(0.5),
            ..Default::default()
        },
        ScoreBreakdown {
            lexical: Some(0.5),
            dense: Some(0.7),
            ..Default::default()
        },
        ScoreBreakdown {
            lexical: Some(0.5),
            dense: Some(0.7),
            fused: Some(0.6),
            rerank: Some(0.8),
        },
    ];
    for b in &cases {
        roundtrip(b);
    }
}

// ── RetrievalResult ────────────────────────────────────────────────────────

#[test]
fn retrieval_result_roundtrips() {
    let r = RetrievalResult {
        chunk: Chunk::new(
            ChunkId::new("r1"),
            "the refund window is thirty days",
            "policy.md",
            TokenCount(7),
        ),
        score: Score {
            value: 0.95,
            method: RetrievalMethod::Hybrid,
        },
        breakdown: ScoreBreakdown {
            lexical: Some(0.4),
            dense: Some(0.8),
            fused: Some(0.6),
            rerank: None,
        },
    };
    roundtrip(&r);
}

// ── ContextReport ──────────────────────────────────────────────────────────

#[test]
fn context_report_roundtrips_after_real_assembly() {
    // Build a real ContextReport via the actual context path so we
    // exercise every field the assembler populates, then round-trip the
    // whole thing. Catches a new field added without serde defaults.
    let q = Query::new("refund window");
    let results = vec![
        RetrievalResult {
            chunk: Chunk::new(
                ChunkId::new("a"),
                "the refund window is thirty days",
                "policy.md",
                TokenCount(7),
            ),
            score: Score {
                value: 1.0,
                method: RetrievalMethod::Lexical,
            },
            breakdown: ScoreBreakdown::default(),
        },
        RetrievalResult {
            chunk: Chunk::new(
                ChunkId::new("b"),
                "unrelated content here",
                "other.md",
                TokenCount(3),
            ),
            score: Score {
                value: 0.5,
                method: RetrievalMethod::Lexical,
            },
            breakdown: ScoreBreakdown::default(),
        },
    ];
    let cfg = ContextConfig::default();
    let ctx = build_context(&q, &results, &cfg);
    roundtrip(&ctx.report);

    // analyze_context also emits a ContextReport — different code path,
    // same round-trip contract.
    let r = analyze_context(&q, &results, &cfg);
    roundtrip(&r);
}

// ── Forward-compatibility: older payload missing a newer field ────────────

#[test]
fn context_report_deserializes_when_low_confidence_fields_missing() {
    // The 0.1.3 release added `low_confidence_retrieval` +
    // `low_confidence_threshold` to ContextReport. Those fields carry
    // `#[serde(default)]` so a payload from before they existed still
    // deserializes cleanly. Pin that contract so a future "remove the
    // default attr" PR is caught at test time, not by a binding crash
    // months later.
    // Minimal payload: ONLY the fields a pre-0.1.3 binary would have
    // produced. `low_confidence_*`, `removed`, and `economics` are all
    // intentionally omitted — they should each fall back to their
    // `#[serde(default)]` value.
    let minimal_json = r#"{
        "strategy": "reasoning_preserving",
        "requested_strategy": "reasoning_preserving",
        "token_budget": 8192,
        "input_tokens": 100,
        "auto_gate_tokens": 1500,
        "total_tokens": 50,
        "token_utilization": 0.006,
        "n_input_chunks": 5,
        "n_selected": 3,
        "input_distractor_ratio": 0.2,
        "retained_evidence_ratio": 1.0,
        "second_hop_rescue_count": 0,
        "reasoning_preservation_delta": 0,
        "n_expanded": 0
    }"#;
    let parsed: Result<ContextReport, _> = serde_json::from_str(minimal_json);
    let report = parsed.unwrap_or_else(|e| {
        panic!(
            "ContextReport must deserialize cleanly from a minimal pre-0.1.3 \
             payload (no low_confidence_* fields). Error: {e}. If you removed \
             #[serde(default)] from a field, that's a binding-compatibility \
             break — restore the attr or version the schema."
        )
    });
    // The missing fields take their defaults:
    assert!(
        !report.low_confidence_retrieval,
        "missing field should default to false"
    );
    assert_eq!(
        report.low_confidence_threshold, 0.0,
        "missing f32 field should default to 0.0"
    );
    // `removed` and `economics` also have #[serde(default)] (added when
    // this test caught their absence). All-zero struct, default ContextEconomics.
    assert_eq!(
        report.removed.total, 0,
        "missing `removed` should default to zeros"
    );
    assert_eq!(
        report.economics.evidence_density, 0.0,
        "missing `economics` should default to ContextEconomics::default()"
    );
}
