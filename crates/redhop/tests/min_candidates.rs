//! Issue #1, Phase 3: `min_candidates` floor with lexical fallback.
//!
//! Verifies the contract that when `Hybrid`/`Dense` retrieval returns fewer
//! candidates than `cfg.min_candidates`, a BM25 fallback over the same chunks
//! tops the result up — and that the floor never kicks in under `Lexical`
//! mode or when set to its default of `0`.
//!
//! Uses a tiny in-process `StubEmbedder` so the test runs without any model
//! download and without the `semantic` feature.

use std::sync::Arc;

use async_trait::async_trait;
use redhop::traits::EmbeddingProvider;
use redhop::{
    core::{Chunk, ChunkId, Embedding, TokenCount},
    Document, DocumentConfig, RetrievalMode,
};

/// 3-dim presence vector over {alpha, beta, gamma}. No model needed.
struct StubEmbedder;

#[async_trait]
impl EmbeddingProvider for StubEmbedder {
    async fn embed(&self, texts: &[String]) -> redhop::core::Result<Vec<Embedding>> {
        Ok(texts
            .iter()
            .map(|t| {
                Embedding::from(vec![
                    t.matches("alpha").count() as f32,
                    t.matches("beta").count() as f32,
                    t.matches("gamma").count() as f32,
                ])
            })
            .collect())
    }
    fn dim(&self) -> usize {
        3
    }
    fn name(&self) -> &'static str {
        "stub"
    }
}

fn chunks() -> Vec<Chunk> {
    // Five prose chunks all about "alpha" to varying degrees. With
    // candidate_k=1 the primary retriever returns just the BM25 winner; the
    // fallback (when min_candidates fires) should top up from BM25's deeper
    // ranking against the same chunks.
    vec![
        Chunk::new(ChunkId::new("a0"), "alpha alpha alpha alpha", "doc", TokenCount(4)),
        Chunk::new(ChunkId::new("a1"), "alpha alpha alpha", "doc", TokenCount(3)),
        Chunk::new(ChunkId::new("a2"), "alpha alpha", "doc", TokenCount(2)),
        Chunk::new(ChunkId::new("a3"), "alpha", "doc", TokenCount(1)),
        Chunk::new(ChunkId::new("nope"), "completely unrelated content here", "doc", TokenCount(4)),
    ]
}

fn hybrid_doc(min_candidates: usize, candidate_k: usize) -> Document {
    let cfg = DocumentConfig {
        candidate_k,
        retrieval_mode: RetrievalMode::Hybrid { candidate_pool: 10 },
        min_candidates,
        ..Default::default()
    };
    Document::from_chunks_with(chunks(), cfg)
        .unwrap()
        .with_embedder(Arc::new(StubEmbedder))
}

#[test]
fn min_candidates_default_is_off() {
    // With min_candidates=0 (the default) and candidate_k=1, the primary
    // hybrid retriever delivers exactly 1 chunk. Fallback must not fire.
    let mut doc = hybrid_doc(0, 1);
    let ctx = doc.context("alpha").unwrap();
    assert_eq!(
        ctx.chunks.len(),
        1,
        "default min_candidates=0 must not trigger fallback; got {:?}",
        ctx.chunks.iter().map(|c| c.id.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn min_candidates_fallback_tops_up_hybrid_results() {
    // candidate_k=1 → primary returns 1 chunk; min_candidates=3 → BM25
    // fallback adds 2 more (the deeper "alpha" BM25 ranking).
    let mut doc = hybrid_doc(3, 1);
    let ctx = doc.context("alpha").unwrap();
    assert_eq!(
        ctx.chunks.len(),
        3,
        "min_candidates=3 must top hybrid up to 3 chunks; got {:?}",
        ctx.chunks.iter().map(|c| c.id.as_str()).collect::<Vec<_>>()
    );
    // The unrelated chunk doesn't match "alpha" in BM25, so the fallback
    // shouldn't pull it in either.
    assert!(
        ctx.chunks.iter().all(|c| c.id.as_str() != "nope"),
        "fallback must not surface a non-matching chunk: got {:?}",
        ctx.chunks.iter().map(|c| c.id.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn min_candidates_noop_under_lexical_mode() {
    // Under Lexical, the primary already IS BM25 — the fallback path is
    // skipped entirely. Asking for min_candidates=5 still returns just the
    // 4 BM25-matching chunks (no spurious top-up from re-querying).
    let cfg = DocumentConfig {
        candidate_k: 2,
        retrieval_mode: RetrievalMode::Lexical,
        min_candidates: 5,
        ..Default::default()
    };
    let mut doc = Document::from_chunks_with(chunks(), cfg).unwrap();
    let ctx = doc.context("alpha").unwrap();
    // Lexical with candidate_k=2 returns 2 BM25 winners; the floor is ignored.
    assert_eq!(
        ctx.chunks.len(),
        2,
        "Lexical mode must ignore min_candidates; got {:?}",
        ctx.chunks.iter().map(|c| c.id.as_str()).collect::<Vec<_>>()
    );
}
