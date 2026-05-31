//! Verify that an indexed Document's embeddings survive a JSON round-trip
//! and are reused on rebuild — the contract `read_folder_with(persist=true)`
//! relies on under semantic / hybrid retrieval. If a counting embedder logs
//! an embed call on the second round, every cold start of a persisted
//! semantic Document re-pays the embedding cost (~30-60 sec for a 1000-chunk
//! codebase with bge-small).
//!
//! The 0.1.3 audit suspected this was broken; the code reads as if it's
//! plumbed end-to-end but had no test pinning the behavior. This file is
//! that test.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use redhop::core::{Chunk, ChunkId, Embedding, TokenCount};
use redhop::traits::EmbeddingProvider;
use redhop::{Document, DocumentConfig, RetrievalMode};

/// Embeds each text into a presence-vector over {alpha, beta, gamma}, and
/// counts every embed call so the test can assert the cache is honored.
struct CountingEmbedder {
    calls: Arc<AtomicUsize>,
    items_embedded: Arc<AtomicUsize>,
}

#[async_trait]
impl EmbeddingProvider for CountingEmbedder {
    async fn embed(&self, texts: &[String]) -> redhop::core::Result<Vec<Embedding>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.items_embedded.fetch_add(texts.len(), Ordering::SeqCst);
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
        "counting-stub"
    }
}

fn chunks() -> Vec<Chunk> {
    vec![
        Chunk::new(ChunkId::new("a"), "alpha alpha alpha", "doc", TokenCount(3)),
        Chunk::new(ChunkId::new("b"), "beta beta", "doc", TokenCount(2)),
        Chunk::new(ChunkId::new("c"), "gamma gamma gamma", "doc", TokenCount(3)),
        Chunk::new(ChunkId::new("d"), "alpha beta", "doc", TokenCount(2)),
    ]
}

fn hybrid_cfg() -> DocumentConfig {
    DocumentConfig {
        retrieval_mode: RetrievalMode::Hybrid { candidate_pool: 10 },
        ..Default::default()
    }
}

#[test]
fn round_trip_chunks_through_serde_keeps_embeddings_reusable() {
    // ── Round 1: index a hybrid Document with a counting embedder ───────────
    let calls = Arc::new(AtomicUsize::new(0));
    let items = Arc::new(AtomicUsize::new(0));
    let mut doc = Document::from_chunks_with(chunks(), hybrid_cfg())
        .unwrap()
        .with_embedder(Arc::new(CountingEmbedder {
            calls: calls.clone(),
            items_embedded: items.clone(),
        }));

    // Trigger indexing + a query; both pay embedder cost.
    let _ = doc.context("alpha").unwrap();
    let chunk_embeds_round1 = items.load(Ordering::SeqCst);
    assert!(
        chunk_embeds_round1 >= 4,
        "round 1 must have embedded all 4 chunks (+ the query), got {chunk_embeds_round1}"
    );

    // Snapshot chunks WITH embeddings — what the persist path writes to disk.
    let embedded = doc.embedded_chunks().unwrap();
    assert!(
        embedded.iter().all(|c| c.embedding.is_some()),
        "embedded_chunks() must populate the embedding field on every chunk; \
         missing on {:?}",
        embedded
            .iter()
            .filter(|c| c.embedding.is_none())
            .map(|c| c.id.as_str())
            .collect::<Vec<_>>(),
    );

    // ── Persist proxy: serialize to JSON, deserialize back. This is exactly
    // what nodejs/load.rs::persisted() does via index.json. ───────────────
    let json = serde_json::to_string(&embedded).expect("serialize chunks");
    let reloaded: Vec<Chunk> = serde_json::from_str(&json).expect("deserialize chunks");
    assert!(
        reloaded.iter().all(|c| c.embedding.is_some()),
        "embeddings must round-trip through serde JSON; missing on {:?}",
        reloaded
            .iter()
            .filter(|c| c.embedding.is_none())
            .map(|c| c.id.as_str())
            .collect::<Vec<_>>(),
    );

    // ── Round 2: rebuild Document with the reloaded chunks. A FRESH counting
    // embedder so we can tell if any chunk gets re-embedded. ─────────────
    let calls2 = Arc::new(AtomicUsize::new(0));
    let items2 = Arc::new(AtomicUsize::new(0));
    let mut doc2 = Document::from_chunks_with(reloaded, hybrid_cfg())
        .unwrap()
        .with_embedder(Arc::new(CountingEmbedder {
            calls: calls2.clone(),
            items_embedded: items2.clone(),
        }));
    let _ = doc2.context("alpha").unwrap();

    // The contract: ZERO chunks re-embedded (they came in pre-loaded). The
    // query still pays one embed call — that's expected.
    let chunk_embeds_round2 = items2.load(Ordering::SeqCst);
    assert_eq!(
        chunk_embeds_round2, 1,
        "round 2 should only embed the query (1 item), got {chunk_embeds_round2}. \
         If >1, the persist→reload cycle isn't reusing cached chunk embeddings — \
         every cold start of a persisted semantic Document re-pays the full \
         embedding cost."
    );
}
