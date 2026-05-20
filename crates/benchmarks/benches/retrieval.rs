//! Retrieval microbenchmarks.

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
use neorag_core::{Chunker, Document, Query, Retriever, TokenizerBackend};
use neorag_retrieval::Bm25Retriever;

fn corpus(n_docs: usize) -> Vec<Document> {
    let templates = [
        "Rust is a systems programming language focused on safety and concurrency.",
        "Tokio is an asynchronous runtime for the Rust programming language.",
        "Django is a high-level Python web framework for rapid development.",
        "TensorFlow is an open source library for numerical computation and machine learning.",
        "Postgres is a relational database with strong ACID semantics.",
        "Kubernetes orchestrates containerized workloads and services declaratively.",
    ];
    (0..n_docs)
        .map(|i| Document::new(format!("doc-{i}"), templates[i % templates.len()]))
        .collect()
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn bench_bm25(c: &mut Criterion) {
    let runtime = rt();
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 64, 96, 0).unwrap();
    let docs = corpus(1024);
    let chunks = chunker.chunk_batch(&docs).unwrap();

    let mut retriever = Bm25Retriever::new().unwrap();
    runtime.block_on(async { retriever.index(&chunks).await.unwrap() });

    let q = Query::new("rust async runtime");
    c.bench_function("bm25_retrieve_top10", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let _ = black_box(retriever.retrieve(black_box(&q), 10).await.unwrap());
            });
        });
    });
}

criterion_group!(benches, bench_bm25);
criterion_main!(benches);
