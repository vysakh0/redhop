//! Retrieval-runtime HYPOTHESIS (not an algorithm): "lexical topology first,
//! semantic refinement second." BM25 prunes a large corpus to a candidate pool;
//! dense reranks *only within that pool* (local), instead of embedding the whole
//! corpus (global). The question this measures:
//!
//!   Does local dense-rerank over BM25 candidates recover most of GLOBAL dense's
//!   gains — or is it capped by BM25's candidate recall?
//!
//! The crux is **BM25 recall@K_cand**: local rerank can only promote gold that
//! BM25 already surfaced. If gold isn't in the top-K_cand (pure synonym, no
//! lexical overlap), local rerank cannot recover it — only global dense can.
//!
//! Global corpus: all HotpotQA paragraphs pooled + deduped (so BM25 top-K is a
//! real prune of thousands, not a per-item 10-paragraph set). Arms: bm25 / global
//! dense / local rerank / hybrid (RRF, control). Tier-1 + geometry (recall@K
//! ceiling, rerank movement, semantic recovery), split lexical vs semantic. No
//! ANN — FlatVectorIndex is exact cosine.
//!
//! Run:  cargo run -p redhop-examples --example semantic_local_rerank --features onnx --release

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use redhop_calibration::loaders::hotpotqa::HotpotQADataset;
use redhop_context::grounding_score;
use redhop_core::{
    Chunk, ChunkId, Embedding, EmbeddingProvider, Query, Retriever, TokenCount, VectorIndex,
};
use redhop_embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop_retrieval::{Bm25Retriever, DenseRetriever, HybridRetriever};
use redhop_storage::{ChunkStore, FlatVectorIndex};

const DIM: usize = 384;
const SAMPLE: usize = 400;
const K_CAND: usize = 50; // BM25 candidate pool depth (the prune)
const TOP_K: usize = 3; // final cut
const DEFAULT_MODEL: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/onnx/model.onnx";
const DEFAULT_TOKENIZER: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/tokenizer.json";

fn cosine(a: &Embedding, b: &Embedding) -> f32 {
    let (x, y) = (a.as_slice(), b.as_slice());
    let n = x.len().min(y.len());
    let (mut d, mut nx, mut ny) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        d += x[i] * y[i];
        nx += x[i] * x[i];
        ny += y[i] * y[i];
    }
    d / (nx.sqrt() * ny.sqrt()).max(1e-9)
}

async fn embed_batched(
    bge: &Arc<dyn EmbeddingProvider>,
    texts: &[String],
) -> anyhow::Result<Vec<Embedding>> {
    let mut out = Vec::with_capacity(texts.len());
    for batch in texts.chunks(64) {
        out.extend(bge.embed(batch).await?);
    }
    Ok(out)
}

#[derive(Default, Clone)]
struct Acc {
    n: usize,
    sum: f64,
}
impl Acc {
    fn add(&mut self, v: f64) {
        self.n += 1;
        self.sum += v;
    }
    fn mean(&self) -> f64 {
        self.sum / self.n.max(1) as f64
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = std::env::var("REDHOP_BGE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let tokenizer =
        std::env::var("REDHOP_BGE_TOKENIZER").unwrap_or_else(|_| DEFAULT_TOKENIZER.into());

    let mut ds = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
    ds.examples.truncate(SAMPLE);

    // ── Build ONE global corpus: dedupe paragraphs by title across all items. ──
    let mut by_title: HashMap<String, String> = HashMap::new();
    for ex in &ds.examples {
        for (title, sents) in &ex.context {
            by_title
                .entry(title.clone())
                .or_insert_with(|| sents.join(" "));
        }
    }
    let titles: Vec<String> = by_title.keys().cloned().collect();
    let title_id: HashMap<&str, usize> = titles
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();
    let chunks: Vec<Chunk> = titles
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let text = &by_title[t];
            Chunk::new(
                ChunkId::new(format!("c{i}")),
                text,
                "pool",
                TokenCount(text.split_whitespace().count().max(1)),
            )
        })
        .collect();

    // Queries with gold chunk ids (supporting titles present in the pool).
    struct QItem {
        question: String,
        gold: HashSet<String>,
        overlap: f32,
    }
    let mut qs: Vec<QItem> = Vec::new();
    for ex in &ds.examples {
        let gold: HashSet<String> = ex
            .supporting_facts
            .iter()
            .filter_map(|(t, _)| title_id.get(t.as_str()).map(|i| format!("c{i}")))
            .collect();
        if gold.is_empty() {
            continue;
        }
        let gold_text: String = ex
            .supporting_facts
            .iter()
            .filter_map(|(t, _)| by_title.get(t).cloned())
            .collect::<Vec<_>>()
            .join(" ");
        qs.push(QItem {
            question: ex.question.clone(),
            overlap: grounding_score(&ex.question, &gold_text),
            gold,
        });
    }

    println!(
        "global corpus: {} unique paragraphs; {} queries; K_cand={K_CAND}, top-{TOP_K}",
        chunks.len(),
        qs.len()
    );

    println!("loading BGE-small ONNX + embedding corpus...");
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &model,
        &tokenizer,
        EmbedderConfig::bge(DIM),
    )?);
    let corpus_vecs = embed_batched(
        &bge,
        &chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>(),
    )
    .await?;
    let emb_by_id: HashMap<String, Embedding> = chunks
        .iter()
        .zip(&corpus_vecs)
        .map(|(c, v)| (c.id.as_str().to_string(), v.clone()))
        .collect();
    let q_vecs = embed_batched(
        &bge,
        &qs.iter().map(|q| q.question.clone()).collect::<Vec<_>>(),
    )
    .await?;

    // Embed chunks into the chunk objects for the dense retriever.
    let mut chunks_emb = chunks.clone();
    for (c, v) in chunks_emb.iter_mut().zip(&corpus_vecs) {
        c.embedding = Some(v.clone());
    }

    // Global retrievers (built once over the whole corpus).
    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;
    let bm25: Arc<dyn Retriever> = Arc::new(bm25);
    let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
    let mut dr = DenseRetriever::new(index, Arc::new(ChunkStore::new()));
    dr.index(&chunks_emb).await?;
    let dense: Arc<dyn Retriever> = Arc::new(dr);
    let hybrid: Arc<dyn Retriever> = Arc::new(HybridRetriever::rrf(
        vec![bm25.clone(), dense.clone()],
        K_CAND,
    ));

    let median = {
        let mut o: Vec<f32> = qs.iter().map(|q| q.overlap).collect();
        o.sort_by(|a, b| a.partial_cmp(b).unwrap());
        o[o.len() / 2]
    };

    let mut rec: HashMap<(&str, &str), Acc> = HashMap::new(); // (subset, arm) -> recall@TOP_K
    let mut bm25_ceiling: HashMap<&str, Acc> = HashMap::new(); // (subset) -> recall@K_cand
                                                               // semantic recovery: among queries where global_dense hit gold@TOP_K but
                                                               // bm25@TOP_K missed, how often does local_rerank recover?
    let (mut recov_total, mut recov_local, mut recov_global_only) = (0usize, 0usize, 0usize);
    let (mut t_bm25, mut t_dense, mut t_local) = (0.0f64, 0.0f64, 0.0f64);

    for (qi, q) in qs.iter().enumerate() {
        let subset = if q.overlap <= median {
            "semantic"
        } else {
            "lexical"
        };
        let qv = &q_vecs[qi];
        let query = Query::new(&q.question).with_embedding(qv.clone());
        let recall_at = |res: &[redhop_core::RetrievalResult], k: usize| -> f64 {
            let hit = res
                .iter()
                .take(k)
                .filter(|r| q.gold.contains(r.chunk.id.as_str()))
                .count();
            hit as f64 / q.gold.len() as f64
        };

        // BM25 candidate pool (depth K_cand) — and its recall ceiling.
        let t = Instant::now();
        let cand = bm25.retrieve(&query, K_CAND).await?;
        t_bm25 += t.elapsed().as_secs_f64() * 1000.0;
        let bm25_r3 = recall_at(&cand, TOP_K);
        bm25_ceiling
            .entry(subset)
            .or_default()
            .add(recall_at(&cand, K_CAND));
        bm25_ceiling
            .entry("ALL")
            .or_default()
            .add(recall_at(&cand, K_CAND));

        // Local rerank: cosine over ONLY the BM25 candidates.
        let t = Instant::now();
        let mut scored: Vec<(&str, f32)> = cand
            .iter()
            .map(|r| {
                let id = r.chunk.id.as_str();
                (id, cosine(qv, &emb_by_id[id]))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        t_local += t.elapsed().as_secs_f64() * 1000.0;
        let local_hit = scored
            .iter()
            .take(TOP_K)
            .filter(|(id, _)| q.gold.contains(*id))
            .count();
        let local_r3 = local_hit as f64 / q.gold.len() as f64;

        // Global dense.
        let t = Instant::now();
        let gd = dense.retrieve(&query, TOP_K).await?;
        t_dense += t.elapsed().as_secs_f64() * 1000.0;
        let gdense_r3 = recall_at(&gd, TOP_K);

        let hyb = hybrid.retrieve(&query, TOP_K).await?;

        for (arm, r) in [
            ("bm25", bm25_r3),
            ("global_dense", gdense_r3),
            ("local_rerank", local_r3),
            ("hybrid", recall_at(&hyb, TOP_K)),
        ] {
            rec.entry((subset, arm)).or_default().add(r);
            rec.entry(("ALL", arm)).or_default().add(r);
        }

        // Semantic-recovery accounting (per-query, full-gold).
        if gdense_r3 > bm25_r3 {
            recov_total += 1;
            if local_r3 >= gdense_r3 {
                recov_local += 1;
            } else {
                recov_global_only += 1;
            }
        }
    }

    // ── Tables ──
    println!("\nRecall@{TOP_K} by subset and arm (median overlap split {median:.3})");
    println!(
        "  {:<14} {:>8} {:>12} {:>12} {:>8}",
        "subset", "bm25", "global_dense", "local_rerank", "hybrid"
    );
    println!("  {}", "─".repeat(58));
    for subset in ["lexical", "semantic", "ALL"] {
        let g = |a: &str| rec.get(&(subset, a)).map(|x| x.mean()).unwrap_or(0.0);
        let nn = rec.get(&(subset, "bm25")).map(|x| x.n).unwrap_or(0);
        println!(
            "  {:<8} (n={:>4}) {:>8.2} {:>12.2} {:>12.2} {:>8.2}",
            subset,
            nn,
            g("bm25"),
            g("global_dense"),
            g("local_rerank"),
            g("hybrid")
        );
    }

    println!("\nThe crux — BM25 recall@{K_CAND} (the ceiling local rerank can reach):");
    for subset in ["lexical", "semantic", "ALL"] {
        println!(
            "  {:<10} {:.3}",
            subset,
            bm25_ceiling.get(subset).map(|x| x.mean()).unwrap_or(0.0)
        );
    }

    println!("\nSemantic recovery (queries where global dense beat bm25@{TOP_K}): n={recov_total}");
    if recov_total > 0 {
        println!(
            "  local rerank recovered: {}/{} ({:.0}%)   only global dense could: {} ({:.0}%)",
            recov_local,
            recov_total,
            100.0 * recov_local as f64 / recov_total as f64,
            recov_global_only,
            100.0 * recov_global_only as f64 / recov_total as f64,
        );
    }

    let nq = qs.len() as f64;
    println!("\nLatency / compute (per query, mean):");
    println!("  bm25 retrieve:        {:.3} ms", t_bm25 / nq);
    println!(
        "  global dense search:  {:.3} ms (over {} chunks)",
        t_dense / nq,
        chunks.len()
    );
    println!(
        "  local rerank:         {:.3} ms (cosine over {K_CAND} candidates)",
        t_local / nq
    );
    println!(
        "  embedding calls/query: local rerank needs {K_CAND} chunk-embeds if NOT precomputed"
    );
    println!("    (global dense needs the whole corpus pre-embedded + stored — the infra local rerank avoids)");
    Ok(())
}
