//! Embedding-backend microbenchmarks.
//!
//! Measures the hermetic pieces: hashing-provider throughput and the
//! cache hit/miss path. The ONNX backend's latency is measured
//! separately on a machine with model files (see
//! `docs/EMBEDDING_RUNTIME.md`); it is not benchmarked here because the
//! sandbox has no model.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use neorag_core::EmbeddingProvider;
use neorag_embeddings::{CachedEmbedder, HashingProvider};

fn corpus(n: usize) -> Vec<String> {
    let templates = [
        "rust is a systems programming language focused on memory safety and concurrency",
        "tokio provides an asynchronous runtime with a work stealing scheduler for tasks",
        "postgres offers acid transactions and mvcc for concurrent read write workloads",
        "retrieval augmented generation grounds language models in external evidence passages",
        "dense retrieval encodes queries and documents into a shared vector space for ranking",
    ];
    (0..n).map(|i| templates[i % templates.len()].to_string()).collect()
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread().build().unwrap()
}

fn bench_hashing(c: &mut Criterion) {
    let runtime = rt();
    let provider = HashingProvider::with_dim(256);
    let texts = corpus(512);

    let mut g = c.benchmark_group("embeddings");
    g.throughput(Throughput::Elements(texts.len() as u64));
    g.bench_function("hashing_embed_512", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let _ = black_box(provider.embed(black_box(&texts)).await.unwrap());
            });
        });
    });

    // Cache: warm it, then measure the all-hit path vs the cold path.
    let cached = Arc::new(CachedEmbedder::new(HashingProvider::with_dim(256), 1024));
    runtime.block_on(async {
        let _ = cached.embed(&texts).await.unwrap();
    });
    g.bench_function("cached_embed_512_all_hit", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let _ = black_box(cached.embed(black_box(&texts)).await.unwrap());
            });
        });
    });
    g.finish();
}

criterion_group!(benches, bench_hashing);
criterion_main!(benches);
