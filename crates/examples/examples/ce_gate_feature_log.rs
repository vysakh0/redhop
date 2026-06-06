//! Phase A of the CE-helps-vs-hurts gate investigation: log per-query
//! features + the ground-truth "did CE help or hurt this query" so
//! Phase B (EDA) can probe whether a hand-tuned threshold gate is
//! buildable.
//!
//! Setup is IDENTICAL to ce_type_gate_economics (now deleted): 100
//! bridge + 100 comparison stratified HotpotQA queries, dense BGE wide
//! net = 20, k_final = 4, ms-marco MiniLM-L-6 cross-encoder. Same
//! corpus, same retrieval call path — so the headline numbers
//! reproduce and the per-query labels are computed against the same
//! reality both experiments measured.
//!
//! Output: writes `target/ce_gate_features.csv` with one row per query
//! and columns:
//!   id, kind, dense_top1_cos, margin, score_spread, grounding_top1,
//!   query_len, pool_entropy, dense_r4, ce_r4, delta, helped, hurt
//!
//! The CSV is NOT committed; it's measurement data the EDA pass
//! consumes locally.
//!
//! Run: cargo run -p redhop-examples --example ce_gate_feature_log \
//!          --features onnx --release

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use parking_lot::RwLock;
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::context::grounding_score;
use redhop::core::{
    ChunkId, Chunker, Embedding, EmbeddingProvider, Query, Reranker, RetrievalResult, Retriever,
    TokenizerBackend, VectorIndex,
};
use redhop::embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop::reranking::OnnxCrossEncoder;
use redhop::retrieval::DenseRetriever;
use redhop::storage::{ChunkStore, FlatVectorIndex};
use redhop_calibration::loaders::hotpotqa::{default_regime, HotpotQADataset};

const N_BRIDGE: usize = 100;
const N_COMPARISON: usize = 100;
const WIDE_N: usize = 20;
const K_FINAL: usize = 4;
const DIM: usize = 384;

fn recall(results: &[RetrievalResult], gold: &[ChunkId]) -> f32 {
    if gold.is_empty() {
        return 1.0;
    }
    let ids: Vec<&ChunkId> = results.iter().map(|r| &r.chunk.id).collect();
    let found = gold.iter().filter(|g| ids.contains(g)).count();
    found as f32 / gold.len() as f32
}

/// Shannon entropy of a softmaxed score distribution. Higher = the
/// retriever isn't sure (scores are uniform); lower = one chunk
/// dominates the pool.
fn pool_entropy(scores: &[f32]) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let p: Vec<f32> = exp.iter().map(|e| e / sum.max(1e-9)).collect();
    -p.iter()
        .filter(|&&pi| pi > 0.0)
        .map(|&pi| pi * pi.ln())
        .sum::<f32>()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("Phase A: feature log for CE-helps-vs-hurts gate investigation\n");

    let mut dataset = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
    let (bridge, comparison): (Vec<_>, Vec<_>) = dataset
        .examples
        .iter()
        .cloned()
        .partition(|e| e.kind == "bridge");
    let mut bridge = bridge;
    let mut comparison = comparison;
    bridge.truncate(N_BRIDGE);
    comparison.truncate(N_COMPARISON);
    let examples: Vec<_> = bridge.into_iter().chain(comparison).collect();
    let kind_by_id: HashMap<String, String> = examples
        .iter()
        .map(|e| (e.id.clone(), e.kind.clone()))
        .collect();
    dataset.examples = examples;
    println!(
        "Loaded {} bridge + {} comparison examples.",
        N_BRIDGE, N_COMPARISON
    );

    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;

    println!("Loading BGE-small + ms-marco cross-encoder...");
    let (bge_model, bge_tokenizer) = redhop_examples::bge_small_paths();
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &bge_model,
        &bge_tokenizer,
        EmbedderConfig::bge(DIM),
    )?);
    let (ce_model, ce_tokenizer) = redhop_examples::ms_marco_paths();
    let ce = OnnxCrossEncoder::load(&ce_model, &ce_tokenizer, 256)?;

    let q_texts: Vec<String> = dataset
        .examples
        .iter()
        .map(|e| e.question.clone())
        .collect();
    println!("Embedding {} queries...", q_texts.len());
    let q_vecs = bge.embed(&q_texts).await?;
    let q_map: HashMap<String, Embedding> = q_texts.into_iter().zip(q_vecs).collect();
    let corpus = dataset.to_labeled_corpus(&chunker, |q| q_map.get(q).cloned(), default_regime)?;

    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    println!("Embedding {} chunks with BGE...", chunks.len());
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = bge.embed(&texts).await?;
    for (c, v) in chunks.iter_mut().zip(vecs.iter()) {
        c.embedding = Some(v.clone());
    }
    let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
    let store = Arc::new(ChunkStore::new());
    let mut dense = DenseRetriever::new(index, store);
    dense.index(&chunks).await?;

    // CSV: target/ce_gate_features.csv (not committed; measurement data).
    std::fs::create_dir_all("target")?;
    let path = "target/ce_gate_features.csv";
    let mut f = File::create(path)?;
    writeln!(
        f,
        "id,kind,dense_top1_cos,margin,score_spread,grounding_top1,query_len,pool_entropy,dense_r4,ce_r4,delta,helped,hurt"
    )?;

    let mut n_logged = 0;
    for lq in &corpus.queries {
        if lq.gold_chunk_ids.is_empty() {
            continue;
        }
        let kind = kind_by_id.get(&lq.id).cloned().unwrap_or_default();
        if kind != "bridge" && kind != "comparison" {
            continue;
        }

        let mut query = Query::new(&lq.text);
        query.embedding = lq.embedding.clone();
        let wide = dense.retrieve(&query, WIDE_N).await?;
        if wide.is_empty() {
            continue;
        }
        let static_top: Vec<RetrievalResult> = wide.iter().take(K_FINAL).cloned().collect();
        let ce_top = ce.rerank(&query, wide.clone(), K_FINAL).await?;

        let rec_static = recall(&static_top, &lq.gold_chunk_ids);
        let rec_ce = recall(&ce_top, &lq.gold_chunk_ids);

        // Per-query features.
        let scores: Vec<f32> = wide.iter().map(|r| r.score.value).collect();
        let top1 = scores[0];
        let top2 = scores.get(1).copied().unwrap_or(top1);
        let topn = scores.last().copied().unwrap_or(top1);
        let margin = top1 - top2;
        let score_spread = top1 - topn;
        let pool_e = pool_entropy(&scores);

        let grounding_top1 = grounding_score(&lq.text, &wide[0].chunk.text);
        let query_len = lq.text.split_whitespace().count();

        let delta = rec_ce - rec_static;
        let helped = (delta > 1e-6) as u8;
        let hurt = (delta < -1e-6) as u8;

        writeln!(
            f,
            "{},{},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.4},{:.4},{:.4},{},{}",
            lq.id,
            kind,
            top1,
            margin,
            score_spread,
            grounding_top1,
            query_len,
            pool_e,
            rec_static,
            rec_ce,
            delta,
            helped,
            hurt
        )?;
        n_logged += 1;
    }

    println!("\nLogged {n_logged} rows to {path}");
    println!(
        "Phase B (EDA): inspect with the python script in target/ or load in your tool of choice."
    );
    Ok(())
}
