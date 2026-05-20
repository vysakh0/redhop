//! Chunking microbenchmarks.

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use neorag_chunking::{AdaptiveChunker, FixedChunker, SentenceChunker, WhitespaceTokenizer};
use neorag_core::{Chunker, Document, TokenizerBackend};

fn doc(n_sentences: usize) -> Document {
    let one = "The quick brown fox jumps over the lazy dog and other animals nearby. ";
    let text: String = std::iter::repeat(one).take(n_sentences).collect();
    Document::new("bench", text)
}

fn bench_chunkers(c: &mut Criterion) {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let d = doc(200);
    let mut g = c.benchmark_group("chunking");
    g.throughput(Throughput::Bytes(d.text.len() as u64));

    let fixed = FixedChunker::new(tok.clone(), 128, 16).unwrap();
    g.bench_function("fixed_128_16", |b| {
        b.iter(|| {
            let _ = black_box(fixed.chunk(black_box(&d)).unwrap());
        });
    });

    let sentence = SentenceChunker::new(tok.clone(), 128, 192, 1).unwrap();
    g.bench_function("sentence_128_192", |b| {
        b.iter(|| {
            let _ = black_box(sentence.chunk(black_box(&d)).unwrap());
        });
    });

    let adaptive = AdaptiveChunker::new(tok, 128, 192, 0.15).unwrap();
    g.bench_function("adaptive_128_192", |b| {
        b.iter(|| {
            let _ = black_box(adaptive.chunk(black_box(&d)).unwrap());
        });
    });

    g.finish();
}

criterion_group!(benches, bench_chunkers);
criterion_main!(benches);
