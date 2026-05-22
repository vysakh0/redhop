//! Integration test: BM25 + dense + hybrid + reranker + diagnostics, end to
//! end, using a small fake "embedding provider" that just hashes terms into
//! a fixed-dimensional vector. This deliberately avoids any model dependence
//! so the test is hermetic and fast.

use std::sync::Arc;

use redhop_chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop_core::{Chunk, Chunker, Document, Embedding, Query, Retriever, TokenizerBackend, VectorIndex};
use redhop_reranking::LexicalGroundingReranker;
use redhop_retrieval::{Bm25Retriever, DenseRetriever, HybridRetriever};
use redhop_storage::{ChunkStore, FlatVectorIndex};
use parking_lot::RwLock;
use unicode_segmentation::UnicodeSegmentation;

const DIM: usize = 64;

/// Hash-based fake embedder: deterministic, no model dependency.
fn fake_embed(text: &str) -> Embedding {
    let mut v = vec![0f32; DIM];
    for w in text.unicode_words().map(|s| s.to_lowercase()) {
        let mut h: u64 = 1469598103934665603;
        for b in w.bytes() {
            h = h.wrapping_mul(1099511628211).wrapping_add(b as u64);
        }
        let i = (h as usize) % DIM;
        v[i] += 1.0;
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    Embedding(v)
}

fn embed_chunks(mut chunks: Vec<Chunk>) -> Vec<Chunk> {
    for c in &mut chunks {
        c.embedding = Some(fake_embed(&c.text));
    }
    chunks
}

#[tokio::test(flavor = "multi_thread")]
async fn hybrid_pipeline_end_to_end() {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0).unwrap();

    let docs = vec![
        Document::new(
            "tokio",
            "Tokio is an asynchronous runtime for the Rust programming language. \
             It powers many production async applications.",
        ),
        Document::new(
            "rust",
            "Rust is a systems programming language focused on safety. \
             Ownership prevents data races at compile time.",
        ),
        Document::new(
            "django",
            "Django is a high-level Python web framework. It encourages rapid \
             development and clean design.",
        ),
        Document::new(
            "postgres",
            "Postgres is a relational database with strong ACID semantics. \
             It supports MVCC for concurrent transactions.",
        ),
    ];
    let chunks = embed_chunks(chunker.chunk_batch(&docs).unwrap());

    // BM25 retriever
    let mut bm25 = Bm25Retriever::new().unwrap();
    bm25.index(&chunks).await.unwrap();

    // Dense retriever
    let idx: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
    let store = Arc::new(ChunkStore::new());
    let mut dense = DenseRetriever::new(idx, store);
    dense.index(&chunks).await.unwrap();

    // Hybrid
    let hybrid = HybridRetriever::rrf(
        vec![Arc::new(bm25), Arc::new(dense)],
        16,
    );

    let q = Query::new("rust async runtime").with_embedding(fake_embed("rust async runtime"));
    let cand = hybrid.retrieve(&q, 4).await.unwrap();
    assert!(!cand.is_empty());

    // The "tokio" doc contains every query term verbatim and should dominate.
    assert_eq!(cand[0].chunk.source, "tokio");

    // Rerank with lexical grounding — order should remain stable here since
    // the top result already has full grounding, but we exercise the path.
    let reranker = LexicalGroundingReranker::default();
    let reranked = redhop_core::Reranker::rerank(&reranker, &q, cand.clone(), 4)
        .await
        .unwrap();
    assert_eq!(reranked[0].chunk.source, "tokio");

    // Diagnostics
    let engine = redhop_diagnostics::DefaultDiagnosticsEngine::new();
    let report = redhop_core::DiagnosticsEngine::diagnose(&engine, &q, &reranked).unwrap();
    assert!(report.lexical_grounding.unwrap() > 0.0);
    assert!(report.retrieval_confidence.is_some());
}
