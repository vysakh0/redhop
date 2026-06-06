//! Does BM25 + dense fusion (Reciprocal Rank Fusion) beat either alone
//! on MuSiQue, and does it regress HotpotQA?
//!
//! The diagnostic established: BM25 wins on MuSiQue (lexical
//! compositional questions), dense wins on HotpotQA (semantic-friendly
//! questions). The natural next probe is FUSION — combine the two
//! ranked lists via RRF, take the top-K of the fused score.
//!
//! RRF (Cormack, Clarke, Buettcher 2009): for each candidate doc d,
//!     fused_score(d) = Σ_r 1 / (k + rank_r(d))
//! where rank_r is d's rank in retriever r's ranked list (1-indexed),
//! and k is a constant (canonical: 60). Higher = better.
//!
//! RRF is a no-ML, no-knob, no-training fusion method — exactly the
//! kind of operator that stays inside RedHop's bounded-architecture
//! constraint. If it beats the better-of-(BM25, dense) on BOTH corpora,
//! it's directly shippable (or already shippable, depending on what
//! RetrievalMode::Hybrid currently does).
//!
//! Comparison:
//!   BM25          : lexical only
//!   dense         : semantic only
//!   RRF(60)       : BM25 + dense rank-fused with k=60
//!
//! On each of HotpotQA (200 stratified) and MuSiQue (200 answerable),
//! recall@4, recall@10, recall@20, recall@50.
//!
//! Run:
//!   REDHOP_BGE_MODEL=... REDHOP_BGE_TOKENIZER=... \
//!   cargo run -p redhop-examples --example musique_hybrid_recall \
//!         --features onnx --release

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    ChunkId, Chunker, Embedding, EmbeddingProvider, Query, RetrievalResult, Retriever,
    TokenizerBackend, VectorIndex,
};
use redhop::embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop::retrieval::{Bm25Retriever, DenseRetriever};
use redhop::storage::{ChunkStore, FlatVectorIndex};
use redhop_calibration::dataset::LabeledCorpus;
use redhop_calibration::loaders::hotpotqa::{default_regime as hotpot_regime, HotpotQADataset};
use redhop_calibration::loaders::musique::{default_regime as musique_regime, MuSiQueDataset};

const SAMPLE: usize = 200;
const DIM: usize = 384;
const KS: &[usize] = &[4, 10, 20, 50];
const POOL_K: usize = 50; // depth of each retriever's pool before fusion
const RRF_K: f32 = 60.0;

fn recall_at_k(ids: &[String], gold: &[ChunkId], k: usize) -> f32 {
    if gold.is_empty() {
        return 1.0;
    }
    let set: HashSet<&str> = ids.iter().take(k).map(|s| s.as_str()).collect();
    let found = gold.iter().filter(|g| set.contains(g.as_str())).count();
    found as f32 / gold.len() as f32
}

/// RRF fusion of N ranked lists. Returns fused (id, score) sorted desc.
fn rrf_fuse(lists: &[Vec<String>], k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            // RRF: 1 / (k + rank_1_indexed)
            let s = 1.0 / (k + (rank as f32 + 1.0));
            *scores.entry(id.clone()).or_default() += s;
        }
    }
    let mut v: Vec<(String, f32)> = scores.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v
}

struct Summary {
    name: String,
    n_queries: usize,
    bm25: HashMap<usize, f32>,
    dense: HashMap<usize, f32>,
    rrf: HashMap<usize, f32>,
}

async fn evaluate(
    name: &str,
    corpus: LabeledCorpus,
    bge: Arc<dyn EmbeddingProvider>,
) -> anyhow::Result<Summary> {
    let chunker_tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(chunker_tok, 40, 60, 0)?;
    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    println!("[{name}] embedding {} chunks...", chunks.len());
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
    let mut n = 0;

    for lq in &corpus.queries {
        if lq.gold_chunk_ids.is_empty() {
            continue;
        }
        n += 1;
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
        let rrf_ids: Vec<String> = rrf_fuse(&[dense_ids.clone(), bm25_ids.clone()], RRF_K)
            .into_iter()
            .map(|(id, _)| id)
            .collect();

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

    Ok(Summary {
        name: name.to_string(),
        n_queries: n,
        bm25: bm25_acc,
        dense: dense_acc,
        rrf: rrf_acc,
    })
}

fn print_table(s: &Summary) {
    println!("\n── {} (n={}) ──", s.name, s.n_queries);
    println!(
        "  {:<8} {:>10} {:>10} {:>10} {:>14}",
        "K", "BM25", "dense", "RRF", "RRF Δ vs best"
    );
    println!("  {}", "─".repeat(56));
    for &k in KS {
        let b = s.bm25[&k];
        let d = s.dense[&k];
        let r = s.rrf[&k];
        let best_solo = b.max(d);
        let delta = r - best_solo;
        let marker = if delta > 0.005 {
            "  ✓"
        } else if delta < -0.005 {
            "  ✗"
        } else {
            "   "
        };
        println!(
            "  {:<8} {:>10.4} {:>10.4} {:>10.4} {:>+14.4}{}",
            k, b, d, r, delta, marker
        );
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("Hybrid (RRF) vs BM25 vs dense — does fusion beat the better-of-both?\n");

    let (bge_model, bge_tokenizer) = redhop_examples::bge_small_paths();
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &bge_model,
        &bge_tokenizer,
        EmbedderConfig::bge(DIM),
    )?);

    let chunker_tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(chunker_tok, 40, 60, 0)?;

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
        hotpot.to_labeled_corpus(&chunker, |q| qmap.get(q).cloned(), hotpot_regime)?;
    let hotpot_sum = evaluate("HotpotQA", hotpot_corpus, bge.clone()).await?;

    // ── MuSiQue ──
    let mut musique = MuSiQueDataset::from_path(redhop_examples::data_path("musique/dev.jsonl"))?;
    musique.examples.retain(|e| e.answerable);
    musique.examples.truncate(SAMPLE);
    let qtexts: Vec<String> = musique
        .examples
        .iter()
        .map(|e| e.question.clone())
        .collect();
    println!("\nEmbedding MuSiQue queries...");
    let qvecs = bge.embed(&qtexts).await?;
    let qmap: HashMap<String, Embedding> = qtexts.into_iter().zip(qvecs).collect();
    let musique_corpus =
        musique.to_labeled_corpus(&chunker, |q| qmap.get(q).cloned(), musique_regime)?;
    let musique_sum = evaluate("MuSiQue", musique_corpus, bge.clone()).await?;

    print_table(&hotpot_sum);
    print_table(&musique_sum);

    println!("\n── verdict ──");
    let print_verdict = |s: &Summary| {
        let r4 = s.rrf[&4];
        let best4 = s.bm25[&4].max(s.dense[&4]);
        let d4 = r4 - best4;
        if d4 > 0.01 {
            println!(
                "  {}: RRF beats better-of-both at K=4 by Δ = {:+.4}. Fusion wins.",
                s.name, d4
            );
        } else if d4 > 0.005 {
            println!(
                "  {}: RRF marginally beats better-of-both at K=4 (Δ = {:+.4}). Above noise but small.",
                s.name, d4
            );
        } else if d4.abs() <= 0.005 {
            println!(
                "  {}: RRF ties better-of-both at K=4 (Δ = {:+.4}). Pick whichever is operationally cheaper.",
                s.name, d4
            );
        } else {
            println!(
                "  {}: RRF UNDERPERFORMS better-of-both at K=4 (Δ = {:+.4}). Fusion hurts.",
                s.name, d4
            );
        }
    };
    print_verdict(&hotpot_sum);
    print_verdict(&musique_sum);

    Ok(())
}
