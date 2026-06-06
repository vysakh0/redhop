//! MuSiQue cross-corpus validation: same Phase A harness as
//! ce_gate_feature_log but on MuSiQue's multi-hop QA data. Tests
//! whether the `grounding_top1 <= 0.35` gate found on HotpotQA
//! generalizes, or whether it was HotpotQA-specific.
//!
//! MuSiQue is harder than HotpotQA: more hops (2/3/4-hop), more
//! distractors per item, more compositional questions. If the gate
//! holds here, we ship with confidence. If it doesn't, the gate is
//! corpus-specific and we either calibrate per-corpus or look for a
//! richer signal.
//!
//! Setup: 200 answerable MuSiQue dev examples, dense BGE wide-net = 20,
//! k_final = 4, ms-marco MiniLM-L-6. Identical retrieval call path to
//! the HotpotQA harness; differences (sample size, k_final) match the
//! prior run so the numbers are directly comparable.
//!
//! Output: target/ce_gate_features_musique.csv.
//!
//! Run: cargo run -p redhop-examples --example ce_gate_feature_log_musique \
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
use redhop_calibration::loaders::musique::{default_regime, MuSiQueDataset};

const SAMPLE: usize = 200;
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
    println!("Phase A on MuSiQue: cross-corpus validation of the grounding_top1 gate.\n");

    let mut dataset = MuSiQueDataset::from_path(redhop_examples::data_path("musique/dev.jsonl"))?;
    // Keep answerable examples only — unanswerable has no gold, so the
    // CE-helps/hurts label would be undefined.
    dataset.examples.retain(|e| e.answerable);
    dataset.examples.truncate(SAMPLE);
    let n_loaded = dataset.examples.len();
    println!("Loaded {n_loaded} answerable MuSiQue examples.");

    // Hop counts (from question_decomposition length) so we can split the
    // result post-hoc. MuSiQue spans 2/3/4-hop.
    let hops_by_id: HashMap<String, usize> = dataset
        .examples
        .iter()
        .map(|e| (e.id.clone(), e.question_decomposition.len()))
        .collect();

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

    std::fs::create_dir_all("target")?;
    let path = "target/ce_gate_features_musique.csv";
    let mut f = File::create(path)?;
    writeln!(
        f,
        "id,hops,dense_top1_cos,margin,score_spread,grounding_top1,query_len,pool_entropy,dense_r4,ce_r4,delta,helped,hurt"
    )?;

    let mut n_logged = 0;
    for lq in &corpus.queries {
        if lq.gold_chunk_ids.is_empty() {
            continue;
        }
        let hops = hops_by_id.get(&lq.id).copied().unwrap_or(0);

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
            hops,
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
    Ok(())
}
