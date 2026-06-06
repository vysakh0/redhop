//! Chunk-size sweep on MuSiQue: does smaller chunking pull recall@50
//! above the 0.51 ceiling we found with the default (40, 60) chunker?
//!
//! The recall-gap diagnostic showed pool recall@50 caps at 0.51 on
//! MuSiQue — even a wide net misses half the gold. One hypothesis:
//! with `SentenceChunker(40, 60, 0)`, gold sentences get packed into
//! chunks alongside enough distractor content that no single chunk
//! looks like a strong match. Smaller chunks = each gold sentence
//! is its own chunk = higher hit rate at fixed K.
//!
//! Counter-hypothesis: smaller chunks dilute the gold across MORE
//! chunks per gold paragraph, so each gold chunk has less retrieval
//! weight individually and recall doesn't improve.
//!
//! This sweep settles it. Vary `target_tokens` in (16, 24, 40, 64,
//! 96) — keeping max_tokens at 1.5× target, overlap at 0 — and
//! measure pool recall@K for BM25, dense, RRF on both corpora.
//!
//! Pre-registered success criterion: pool recall@50 climbs by ≥ +0.05
//! on MuSiQue at SOME chunk size, without regressing HotpotQA pool
//! recall by more than 0.02.
//!
//! Run:
//!   REDHOP_BGE_MODEL=... REDHOP_BGE_TOKENIZER=... \
//!   cargo run -p redhop-examples --example musique_chunk_sweep \
//!         --features onnx --release

use std::collections::{HashMap, HashSet};
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
const DIM: usize = 384;
const KS: &[usize] = &[4, 10, 50];
const POOL_K: usize = 50;
const RRF_K: f32 = 60.0;
const CHUNK_TARGETS: &[usize] = &[16, 24, 40, 64, 96];

fn recall_at_k(ids: &[String], gold: &[ChunkId], k: usize) -> f32 {
    if gold.is_empty() {
        return 1.0;
    }
    let set: HashSet<&str> = ids.iter().take(k).map(|s| s.as_str()).collect();
    let found = gold.iter().filter(|g| set.contains(g.as_str())).count();
    found as f32 / gold.len() as f32
}

fn rrf_fuse(lists: &[Vec<String>], k: f32) -> Vec<String> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            let s = 1.0 / (k + (rank as f32 + 1.0));
            *scores.entry(id.clone()).or_default() += s;
        }
    }
    let mut v: Vec<(String, f32)> = scores.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.into_iter().map(|(id, _)| id).collect()
}

#[derive(Clone)]
struct Row {
    corpus: String,
    target: usize,
    n_chunks: usize,
    mean_gold_per_query: f32,
    bm25: HashMap<usize, f32>,
    dense: HashMap<usize, f32>,
    rrf: HashMap<usize, f32>,
}

async fn run_one(
    corpus_name: &str,
    corpus: &LabeledCorpus,
    bge: Arc<dyn EmbeddingProvider>,
    target_tokens: usize,
) -> anyhow::Result<Row> {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let max_tokens = (target_tokens * 3) / 2;
    let chunker = SentenceChunker::new(tok, target_tokens, max_tokens, 0)?;
    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    let n_chunks = chunks.len();
    println!("  [{corpus_name} target={target_tokens}] embedding {n_chunks} chunks...");
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = bge.embed(&texts).await?;
    for (c, v) in chunks.iter_mut().zip(vecs.iter()) {
        c.embedding = Some(v.clone());
    }

    let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
    let store = Arc::new(ChunkStore::new());
    let mut dense = DenseRetriever::new(index, store);
    dense.index(&chunks).await?;
    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;

    let mut bm25_acc: HashMap<usize, f32> = KS.iter().map(|&k| (k, 0.0)).collect();
    let mut dense_acc: HashMap<usize, f32> = KS.iter().map(|&k| (k, 0.0)).collect();
    let mut rrf_acc: HashMap<usize, f32> = KS.iter().map(|&k| (k, 0.0)).collect();
    let mut n = 0usize;
    let mut gold_sum = 0usize;

    for lq in &corpus.queries {
        if lq.gold_chunk_ids.is_empty() {
            continue;
        }
        n += 1;
        gold_sum += lq.gold_chunk_ids.len();
        let mut q = Query::new(&lq.text);
        q.embedding = lq.embedding.clone();
        let dense_hits = dense.retrieve(&q, POOL_K).await?;
        let bm25_hits = bm25.retrieve(&q, POOL_K).await?;
        let dense_ids: Vec<String> = dense_hits
            .iter()
            .map(|r| r.chunk.id.as_str().to_string())
            .collect();
        let bm25_ids: Vec<String> = bm25_hits
            .iter()
            .map(|r| r.chunk.id.as_str().to_string())
            .collect();
        let rrf_ids = rrf_fuse(&[dense_ids.clone(), bm25_ids.clone()], RRF_K);
        for &k in KS {
            *bm25_acc.get_mut(&k).unwrap() += recall_at_k(&bm25_ids, &lq.gold_chunk_ids, k);
            *dense_acc.get_mut(&k).unwrap() += recall_at_k(&dense_ids, &lq.gold_chunk_ids, k);
            *rrf_acc.get_mut(&k).unwrap() += recall_at_k(&rrf_ids, &lq.gold_chunk_ids, k);
        }
    }
    let nf = n.max(1) as f32;
    for &k in KS {
        *bm25_acc.get_mut(&k).unwrap() /= nf;
        *dense_acc.get_mut(&k).unwrap() /= nf;
        *rrf_acc.get_mut(&k).unwrap() /= nf;
    }

    Ok(Row {
        corpus: corpus_name.to_string(),
        target: target_tokens,
        n_chunks,
        mean_gold_per_query: gold_sum as f32 / nf,
        bm25: bm25_acc,
        dense: dense_acc,
        rrf: rrf_acc,
    })
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("Chunk-size sweep on MuSiQue + HotpotQA — does smaller chunking break the 0.51 pool ceiling?\n");

    let (bge_model, bge_tokenizer) = redhop_examples::bge_small_paths();
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &bge_model,
        &bge_tokenizer,
        EmbedderConfig::bge(DIM),
    )?);

    let tok_for_corpus: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let base_chunker = SentenceChunker::new(tok_for_corpus, 40, 60, 0)?;

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
    let qtexts: Vec<String> = hotpot.examples.iter().map(|e| e.question.clone()).collect();
    println!("Embedding HotpotQA queries...");
    let qvecs = bge.embed(&qtexts).await?;
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
    println!("Embedding MuSiQue queries...");
    let qvecs = bge.embed(&qtexts).await?;
    let qmap: HashMap<String, Embedding> = qtexts.into_iter().zip(qvecs).collect();
    let musique_corpus =
        musique.to_labeled_corpus(&base_chunker, |q| qmap.get(q).cloned(), musique_regime)?;

    let mut rows: Vec<Row> = Vec::new();
    for &target in CHUNK_TARGETS {
        println!("\n── target_tokens = {target} ──");
        rows.push(run_one("HotpotQA", &hotpot_corpus, bge.clone(), target).await?);
        rows.push(run_one("MuSiQue", &musique_corpus, bge.clone(), target).await?);
    }

    // ── Print sweep tables ──
    for corpus_name in ["HotpotQA", "MuSiQue"] {
        println!("\n══ {corpus_name} ══");
        println!(
            "  {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}",
            "target",
            "chunks",
            "gold/q",
            "BM25@4",
            "BM25@50",
            "dense@4",
            "dense@50",
            "RRF@4",
            "RRF@50",
            "Δ(RRF@50)"
        );
        let baseline_rrf_50 = rows
            .iter()
            .find(|r| r.corpus == corpus_name && r.target == 40)
            .map(|r| r.rrf[&50])
            .unwrap_or(0.0);
        for r in rows.iter().filter(|r| r.corpus == corpus_name) {
            let d = r.rrf[&50] - baseline_rrf_50;
            let marker = if d > 0.02 {
                "  ✓"
            } else if d < -0.02 {
                "  ✗"
            } else {
                "   "
            };
            println!(
                "  {:<10} {:<10} {:<10.2} {:<10.4} {:<10.4} {:<10.4} {:<10.4} {:<10.4} {:<10.4} {:>+10.4}{}",
                r.target,
                r.n_chunks,
                r.mean_gold_per_query,
                r.bm25[&4],
                r.bm25[&50],
                r.dense[&4],
                r.dense[&50],
                r.rrf[&4],
                r.rrf[&50],
                d,
                marker
            );
        }
    }

    // ── Verdict ──
    let musique_best_50 = rows
        .iter()
        .filter(|r| r.corpus == "MuSiQue")
        .map(|r| (r.target, r.rrf[&50]))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let musique_baseline_50 = rows
        .iter()
        .find(|r| r.corpus == "MuSiQue" && r.target == 40)
        .map(|r| r.rrf[&50])
        .unwrap();
    let hotpot_baseline_50 = rows
        .iter()
        .find(|r| r.corpus == "HotpotQA" && r.target == 40)
        .map(|r| r.rrf[&50])
        .unwrap();
    let hotpot_at_best = rows
        .iter()
        .find(|r| r.corpus == "HotpotQA" && r.target == musique_best_50.0)
        .map(|r| r.rrf[&50])
        .unwrap();

    println!("\n── verdict ──");
    println!(
        "  Best MuSiQue RRF@50: target_tokens = {} → {:.4} (baseline @40 = {:.4}; Δ = {:+.4})",
        musique_best_50.0,
        musique_best_50.1,
        musique_baseline_50,
        musique_best_50.1 - musique_baseline_50
    );
    println!(
        "  HotpotQA RRF@50 at same target: {:.4} (baseline @40 = {:.4}; Δ = {:+.4})",
        hotpot_at_best,
        hotpot_baseline_50,
        hotpot_at_best - hotpot_baseline_50
    );
    let lift = musique_best_50.1 - musique_baseline_50;
    let hot_regression = hotpot_baseline_50 - hotpot_at_best;
    if lift >= 0.05 && hot_regression <= 0.02 {
        println!(
            "  ✓ Chunk-size sweep IDENTIFIES a better default: target = {}",
            musique_best_50.0
        );
    } else if lift >= 0.02 {
        println!("  ~ Modest lift on MuSiQue; check whether it's worth a default change.");
    } else {
        println!("  ✗ Chunking is NOT the bottleneck on MuSiQue — the 0.51 ceiling is structural.");
    }
    Ok(())
}
