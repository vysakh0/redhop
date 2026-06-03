//! Real embedding bakeoff: BGE-small (ONNX) vs the hashing baseline,
//! on real HotpotQA data. Requires `--features onnx` and a downloaded
//! BGE-small ONNX model.
//!
//! This is the Phase-1 "operational truth" run: real model inference,
//! real recall numbers, real per-query latency. It answers:
//!
//!   1. How much retrieval recall does a real semantic embedder buy
//!      over the lexical hashing baseline on multi-hop QA?
//!   2. What does that cost in per-query embedding latency and memory?
//!
//! Setup (one-time): download the BGE-small ONNX model + tokenizer to any
//! directory and point `REDHOP_MODELS_DIR` at its parent (or set
//! `REDHOP_BGE_MODEL` / `REDHOP_BGE_TOKENIZER` per-file):
//!   python -c "
//!   from huggingface_hub import hf_hub_download
//!   for f in ['onnx/model.onnx','tokenizer.json']:
//!       hf_hub_download('BAAI/bge-small-en-v1.5', f,
//!           local_dir='./models/bge-small-en-v1.5')"
//!   export REDHOP_MODELS_DIR=$PWD/models
//!
//! Run:
//!   cargo run -p redhop-examples --example real_embedding_bakeoff \
//!       --features onnx --release

use std::sync::Arc;

use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{ChunkId, Chunker, EmbeddingProvider, TokenizerBackend};
use redhop::embeddings::{EmbedderConfig, HashingProvider, OnnxEmbedder};
use redhop_calibration::{
    embedder_bench::{compare_embedders, render_comparison},
    loaders::hotpotqa::{default_regime, HotpotQADataset},
};
const SAMPLE_SIZE: usize = 50;
const TOP_K: usize = 4;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Real embedding bakeoff — BGE-small (ONNX) vs hashing baseline   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // ── Load HotpotQA sample → LabeledCorpus ──
    let mut dataset = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
    dataset.examples.truncate(SAMPLE_SIZE);
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;
    // No query embedding in the corpus itself — compare_embedders embeds
    // with each provider.
    let corpus = dataset.to_labeled_corpus(&chunker, |_| None, default_regime)?;

    // Chunk the same docs with the same chunker so chunk ids match the
    // gold ids the loader synthesized.
    let chunks = chunker.chunk_batch(&corpus.docs)?;
    let chunk_texts: Vec<(ChunkId, String)> = chunks
        .iter()
        .map(|c| (c.id.clone(), c.text.clone()))
        .collect();
    println!(
        "corpus: {} chunks, {} queries (gold = supporting-fact chunks)\n",
        chunk_texts.len(),
        corpus.queries.len()
    );

    // ── Providers ──
    let hashing: Arc<dyn EmbeddingProvider> = Arc::new(HashingProvider::with_dim(384));
    let (bge_model, bge_tokenizer) = redhop_examples::bge_small_paths();
    println!("loading BGE-small ONNX model ({})...", bge_model.display());
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &bge_model,
        &bge_tokenizer,
        EmbedderConfig::bge(384),
    )?);
    println!("  loaded. dim={}\n", bge.dim());

    // ── Bakeoff ──
    println!("running bakeoff (hashing = baseline, BGE = candidate)...\n");
    let cmp = compare_embedders(hashing, bge, &corpus, &chunk_texts, TOP_K).await?;
    println!("{}", render_comparison(&cmp));

    println!("\n──── headline ────");
    println!(
        "BGE-small recall@{}: {:.3}   hashing recall@{}: {:.3}",
        TOP_K, cmp.candidate.mean_recall, TOP_K, cmp.baseline.mean_recall
    );
    println!(
        "recall lift from real semantic embedder: {:+.3} ({:+.0}%)",
        cmp.recall_delta,
        if cmp.baseline.mean_recall > 0.0 {
            cmp.recall_delta / cmp.baseline.mean_recall * 100.0
        } else {
            0.0
        }
    );
    println!(
        "latency cost: BGE {:.1} us/query vs hashing {:.1} us/query ({:.0}x)",
        cmp.candidate.query_embed_us, cmp.baseline.query_embed_us, cmp.latency_multiple
    );
    println!(
        "memory: {} bytes/vector (both, 384-dim f32)",
        cmp.candidate.bytes_per_vector
    );
    Ok(())
}
