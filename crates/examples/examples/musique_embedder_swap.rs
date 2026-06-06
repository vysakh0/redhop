//! BGE-small vs BGE-base on MuSiQue: does a bigger embedder break the
//! 0.51 recall@50 ceiling?
//!
//! Diagnostic established the ceiling; chunk sweep ruled out chunking
//! as the lever. The remaining candidate is the embedder itself.
//! BGE-small is 384-dim (24M params); BGE-base is 768-dim (110M params).
//! Bigger = more representational capacity for compositional questions.
//!
//! Same setup as the diagnostic: 200 answerable MuSiQue + 200
//! stratified HotpotQA, default chunker (40,60), recall@K for
//! K=4,10,20,50.
//!
//! Pre-registered success criterion: MuSiQue dense recall@50 climbs by
//! ≥ +0.05 (from 0.51 baseline to ≥ 0.56). Anything less is "more
//! capacity didn't help" — a meaningful negative result.
//!
//! Run:
//!   REDHOP_BGE_SMALL_MODEL=... REDHOP_BGE_SMALL_TOKENIZER=... \
//!   REDHOP_BGE_BASE_MODEL=... REDHOP_BGE_BASE_TOKENIZER=... \
//!   cargo run -p redhop-examples --example musique_embedder_swap \
//!         --features onnx --release

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    ChunkId, Chunker, Embedding, EmbeddingProvider, Query, Retriever, TokenizerBackend, VectorIndex,
};
use redhop::embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop::retrieval::{Bm25Retriever, DenseRetriever};
use redhop::storage::{ChunkStore, FlatVectorIndex};
use redhop_calibration::dataset::LabeledCorpus;
use redhop_calibration::loaders::hotpotqa::{default_regime as hotpot_regime, HotpotQADataset};
use redhop_calibration::loaders::musique::{default_regime as musique_regime, MuSiQueDataset};

const SAMPLE: usize = 200;
const KS: &[usize] = &[4, 10, 20, 50];

fn recall_at_k(ids: &[String], gold: &[ChunkId], k: usize) -> f32 {
    if gold.is_empty() {
        return 1.0;
    }
    let set: HashSet<&str> = ids.iter().take(k).map(|s| s.as_str()).collect();
    let found = gold.iter().filter(|g| set.contains(g.as_str())).count();
    found as f32 / gold.len() as f32
}

struct Row {
    corpus: String,
    embedder: String,
    dim: usize,
    bm25: HashMap<usize, f32>,
    dense: HashMap<usize, f32>,
}

async fn run_one(
    corpus_name: &str,
    corpus: &LabeledCorpus,
    embedder: Arc<dyn EmbeddingProvider>,
    embedder_name: &str,
    dim: usize,
) -> anyhow::Result<Row> {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;
    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    println!(
        "  [{corpus_name} {embedder_name}] embedding {} chunks (dim={})...",
        chunks.len(),
        dim
    );
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = embedder.embed(&texts).await?;
    for (c, v) in chunks.iter_mut().zip(vecs.iter()) {
        c.embedding = Some(v.clone());
    }

    let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(dim)));
    let store = Arc::new(ChunkStore::new());
    let mut dense = DenseRetriever::new(index, store);
    dense.index(&chunks).await?;
    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;

    let max_k = *KS.iter().max().unwrap();
    let mut bm25_acc: HashMap<usize, f32> = KS.iter().map(|&k| (k, 0.0)).collect();
    let mut dense_acc: HashMap<usize, f32> = KS.iter().map(|&k| (k, 0.0)).collect();
    let mut n = 0usize;

    // Re-embed queries with the SAME embedder so the query vector and
    // chunk vectors live in the same space.
    let q_texts: Vec<String> = corpus.queries.iter().map(|q| q.text.clone()).collect();
    let q_vecs = embedder.embed(&q_texts).await?;
    let q_map: HashMap<&str, Embedding> = corpus
        .queries
        .iter()
        .zip(q_vecs.iter())
        .map(|(q, v)| (q.text.as_str(), v.clone()))
        .collect();

    for lq in &corpus.queries {
        if lq.gold_chunk_ids.is_empty() {
            continue;
        }
        n += 1;
        let mut q = Query::new(&lq.text);
        q.embedding = q_map.get(lq.text.as_str()).cloned();
        let dense_hits = dense.retrieve(&q, max_k).await?;
        let bm25_hits = bm25.retrieve(&q, max_k).await?;
        let dense_ids: Vec<String> = dense_hits
            .iter()
            .map(|r| r.chunk.id.as_str().to_string())
            .collect();
        let bm25_ids: Vec<String> = bm25_hits
            .iter()
            .map(|r| r.chunk.id.as_str().to_string())
            .collect();
        for &k in KS {
            *bm25_acc.get_mut(&k).unwrap() += recall_at_k(&bm25_ids, &lq.gold_chunk_ids, k);
            *dense_acc.get_mut(&k).unwrap() += recall_at_k(&dense_ids, &lq.gold_chunk_ids, k);
        }
    }
    let nf = n.max(1) as f32;
    for &k in KS {
        *bm25_acc.get_mut(&k).unwrap() /= nf;
        *dense_acc.get_mut(&k).unwrap() /= nf;
    }

    Ok(Row {
        corpus: corpus_name.to_string(),
        embedder: embedder_name.to_string(),
        dim,
        bm25: bm25_acc,
        dense: dense_acc,
    })
}

fn paths_for(env_model: &str, env_tok: &str, fallback_dir: &str) -> (PathBuf, PathBuf) {
    let model = std::env::var(env_model)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(fallback_dir).join("model_optimized.onnx"));
    let tokenizer = std::env::var(env_tok)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(fallback_dir).join("tokenizer.json"));
    (model, tokenizer)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("BGE-small vs BGE-base on MuSiQue + HotpotQA — does more embedder capacity help?\n");

    let tok_for_corpus: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let base_chunker = SentenceChunker::new(tok_for_corpus.clone(), 40, 60, 0)?;

    // ── Load embedders ──
    let home = std::env::var("HOME").unwrap_or_default();
    let small_default = format!(
        "{home}/.cache/huggingface/hub/models--Qdrant--bge-small-en-v1.5-onnx-Q/snapshots/52398278842ec682c6f32300af41344b1c0b0bb2"
    );
    let base_default = format!("{home}/.cache/huggingface/hub/manual-bge-base");

    let (small_model, small_tok) = paths_for(
        "REDHOP_BGE_SMALL_MODEL",
        "REDHOP_BGE_SMALL_TOKENIZER",
        &small_default,
    );
    let (base_model, base_tok) = paths_for(
        "REDHOP_BGE_BASE_MODEL",
        "REDHOP_BGE_BASE_TOKENIZER",
        &base_default,
    );

    println!("Loading BGE-small (dim=384) from {}", small_model.display());
    let bge_small: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &small_model,
        &small_tok,
        EmbedderConfig::bge(384),
    )?);
    println!("Loading BGE-base (dim=768) from {}", base_model.display());
    let bge_base: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &base_model,
        &base_tok,
        EmbedderConfig::bge(768),
    )?);

    // ── HotpotQA ──
    let mut hotpot = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
    let (bridge, comparison): (Vec<_>, Vec<_>) = hotpot
        .examples
        .iter()
        .cloned()
        .partition(|e| e.kind == "bridge");
    let mut bridge = bridge;
    let mut comparison = comparison;
    bridge.truncate(100);
    comparison.truncate(100);
    hotpot.examples = bridge.into_iter().chain(comparison).collect();
    // Embed queries with bge-small just for the LabeledCorpus builder; we
    // re-embed inside run_one with the right model anyway.
    let qtexts: Vec<String> = hotpot.examples.iter().map(|e| e.question.clone()).collect();
    let qvecs = bge_small.embed(&qtexts).await?;
    let qmap: HashMap<String, Embedding> = qtexts.into_iter().zip(qvecs).collect();
    let hotpot_corpus =
        hotpot.to_labeled_corpus(&base_chunker, |q| qmap.get(q).cloned(), hotpot_regime)?;

    // ── MuSiQue ──
    let mut musique = MuSiQueDataset::from_path(redhop_examples::data_path("musique/dev.jsonl"))?;
    musique.examples.retain(|e| e.answerable);
    musique.examples.truncate(SAMPLE);
    let qtexts: Vec<String> = musique
        .examples
        .iter()
        .map(|e| e.question.clone())
        .collect();
    let qvecs = bge_small.embed(&qtexts).await?;
    let qmap: HashMap<String, Embedding> = qtexts.into_iter().zip(qvecs).collect();
    let musique_corpus =
        musique.to_labeled_corpus(&base_chunker, |q| qmap.get(q).cloned(), musique_regime)?;

    let mut rows = Vec::new();
    rows.push(
        run_one(
            "HotpotQA",
            &hotpot_corpus,
            bge_small.clone(),
            "bge-small",
            384,
        )
        .await?,
    );
    rows.push(
        run_one(
            "HotpotQA",
            &hotpot_corpus,
            bge_base.clone(),
            "bge-base",
            768,
        )
        .await?,
    );
    rows.push(
        run_one(
            "MuSiQue",
            &musique_corpus,
            bge_small.clone(),
            "bge-small",
            384,
        )
        .await?,
    );
    rows.push(
        run_one(
            "MuSiQue",
            &musique_corpus,
            bge_base.clone(),
            "bge-base",
            768,
        )
        .await?,
    );

    // ── Tables ──
    for corpus_name in ["HotpotQA", "MuSiQue"] {
        println!("\n══ {corpus_name} ══");
        println!(
            "  {:<12} {:>6} {:>10} {:>10} {:>10} {:>10} {:>14}",
            "embedder", "dim", "dense@4", "dense@10", "dense@20", "dense@50", "Δ@50 vs small"
        );
        let baseline = rows
            .iter()
            .find(|r| r.corpus == corpus_name && r.embedder == "bge-small")
            .map(|r| r.dense[&50])
            .unwrap_or(0.0);
        for r in rows.iter().filter(|r| r.corpus == corpus_name) {
            let d = r.dense[&50] - baseline;
            let marker = if d > 0.02 {
                "  ✓"
            } else if d < -0.02 {
                "  ✗"
            } else if r.embedder == "bge-small" {
                "   "
            } else {
                "   "
            };
            println!(
                "  {:<12} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>+14.4}{}",
                r.embedder, r.dim, r.dense[&4], r.dense[&10], r.dense[&20], r.dense[&50], d, marker
            );
        }
    }

    // ── Verdict ──
    let m_small = rows
        .iter()
        .find(|r| r.corpus == "MuSiQue" && r.embedder == "bge-small")
        .map(|r| r.dense[&50])
        .unwrap_or(0.0);
    let m_base = rows
        .iter()
        .find(|r| r.corpus == "MuSiQue" && r.embedder == "bge-base")
        .map(|r| r.dense[&50])
        .unwrap_or(0.0);
    let lift = m_base - m_small;
    println!("\n── verdict ──");
    println!(
        "  MuSiQue dense recall@50: bge-small = {:.4} → bge-base = {:.4} (Δ = {:+.4})",
        m_small, m_base, lift
    );
    if lift >= 0.05 {
        println!("  ✓ Stronger embedder breaks the 0.51 ceiling — bigger model is the lever.");
    } else if lift >= 0.02 {
        println!("  ~ Modest lift; possibly worth shipping bge-base as default for hard corpora.");
    } else {
        println!("  ✗ Embedder size is NOT the bottleneck. The 0.51 ceiling is structural to the corpus.");
    }
    Ok(())
}
