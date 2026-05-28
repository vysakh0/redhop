//! Semantic-mismatch retrieval probe: BM25 vs dense (BGE, exact cosine) vs
//! hybrid (RRF), on controlled lexical-vs-semantic regimes.
//!
//! The question is NOT "are embeddings better" — it's *where does lexical
//! retrieval fail, and when does semantic retrieval materially help?* Each item
//! has a GOLD passage (semantically right, often low lexical overlap), a TRAP
//! passage (high lexical overlap, wrong meaning — a BM25 attractor), and
//! distractors. All passages are pooled into one corpus; per query we measure
//! whether each retriever ranks the gold passage at the top.
//!
//! Tier-1, retrieval quality (recall@1/@3, MRR, trap-beats-gold rate), per
//! category, plus latency (BM25 vs dense, and the query-embedding tax).
//! No ANN — `FlatVectorIndex` is exact cosine. No vector DB.
//!
//! Run:  cargo run -p redhop-examples --example semantic_mismatch --features onnx --release
//! Override the model with REDHOP_BGE_MODEL / REDHOP_BGE_TOKENIZER.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use redhop::chunking::WhitespaceTokenizer;
use redhop::core::{
    Chunk, ChunkId, EmbeddingProvider, Query, Retriever, TokenCount, TokenizerBackend, VectorIndex,
};
use redhop::embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop::retrieval::{Bm25Retriever, DenseRetriever, HybridRetriever};
use redhop::storage::{ChunkStore, FlatVectorIndex};
use serde::Deserialize;

const DIM: usize = 384;
const TOP_K: usize = 5;
const CANDIDATE_K: usize = 10;
const DEFAULT_MODEL: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/onnx/model.onnx";
const DEFAULT_TOKENIZER: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/tokenizer.json";

#[derive(Deserialize)]
struct Data {
    items: Vec<Item>,
}
#[derive(Deserialize)]
struct Item {
    id: String,
    category: String,
    query: String,
    gold: String,
    trap: String,
    distractors: Vec<String>,
}

fn tok_count(s: &str) -> usize {
    s.split_whitespace().count().max(1)
}

#[derive(Default, Clone)]
struct Stat {
    n: usize,
    r1: usize,
    r3: usize,
    mrr: f64,
    trap_over_gold: usize,
}
impl Stat {
    fn add(&mut self, gold_rank: Option<usize>, trap_rank: Option<usize>) {
        self.n += 1;
        if let Some(r) = gold_rank {
            if r == 1 {
                self.r1 += 1;
            }
            if r <= 3 {
                self.r3 += 1;
            }
            self.mrr += 1.0 / r as f64;
        }
        // Trap ranked above gold (or gold absent while trap present).
        let beats = match (trap_rank, gold_rank) {
            (Some(t), Some(g)) => t < g,
            (Some(_), None) => true,
            _ => false,
        };
        if beats {
            self.trap_over_gold += 1;
        }
    }
    fn line(&self, label: &str) {
        let n = self.n.max(1) as f64;
        println!(
            "  {:<22} {:>5.0}% {:>5.0}% {:>6.2} {:>10.0}%",
            label,
            100.0 * self.r1 as f64 / n,
            100.0 * self.r3 as f64 / n,
            self.mrr / n,
            100.0 * self.trap_over_gold as f64 / n,
        );
    }
}

fn rank_of(results: &[redhop::core::RetrievalResult], id: &str) -> Option<usize> {
    results
        .iter()
        .position(|r| r.chunk.id.as_str() == id)
        .map(|p| p + 1)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = std::env::var("REDHOP_BGE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let tokenizer =
        std::env::var("REDHOP_BGE_TOKENIZER").unwrap_or_else(|_| DEFAULT_TOKENIZER.into());

    let raw = std::fs::read_to_string(redhop_examples::data_path("semantic_mismatch.json"))?;
    let data: Data = serde_json::from_str(&raw)?;

    // Pool every passage into one corpus. Chunk ids encode item + role.
    let _tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let mut chunks: Vec<Chunk> = Vec::new();
    let mk = |id: String, text: &str| {
        Chunk::new(ChunkId::new(id), text, "probe", TokenCount(tok_count(text)))
    };
    for it in &data.items {
        chunks.push(mk(format!("{}::gold", it.id), &it.gold));
        chunks.push(mk(format!("{}::trap", it.id), &it.trap));
        for (i, d) in it.distractors.iter().enumerate() {
            chunks.push(mk(format!("{}::d{i}", it.id), d));
        }
    }
    println!(
        "loading BGE-small ONNX ({} passages, {} queries)...",
        chunks.len(),
        data.items.len()
    );
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &model,
        &tokenizer,
        EmbedderConfig::bge(DIM),
    )?);

    // Embed passages.
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = bge.embed(&texts).await?;
    for (c, v) in chunks.iter_mut().zip(vecs) {
        c.embedding = Some(v);
    }

    // Build the three retrievers.
    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;
    let bm25: Arc<dyn Retriever> = Arc::new(bm25);

    let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
    let mut dense = DenseRetriever::new(index, Arc::new(ChunkStore::new()));
    dense.index(&chunks).await?;
    let dense: Arc<dyn Retriever> = Arc::new(dense);

    let hybrid: Arc<dyn Retriever> = Arc::new(HybridRetriever::rrf(
        vec![bm25.clone(), dense.clone()],
        CANDIDATE_K,
    ));

    // Per-(category, mode) stats; "ALL" aggregates across categories.
    let modes = ["bm25", "dense", "hybrid"];
    let mut stats: HashMap<(String, &str), Stat> = HashMap::new();
    let mut cats: Vec<String> = Vec::new();
    let (mut embed_ms, mut bm25_ms, mut dense_ms) = (0.0f64, 0.0f64, 0.0f64);

    for it in &data.items {
        if !cats.contains(&it.category) {
            cats.push(it.category.clone());
        }
        // Query embedding (the semantic tax) — timed.
        let t = Instant::now();
        let qv = bge
            .embed(std::slice::from_ref(&it.query))
            .await?
            .pop()
            .unwrap();
        embed_ms += t.elapsed().as_secs_f64() * 1000.0;
        let q = Query::new(&it.query).with_embedding(qv);

        let gold = format!("{}::gold", it.id);
        let trap = format!("{}::trap", it.id);

        for mode in modes {
            let retr = match mode {
                "bm25" => &bm25,
                "dense" => &dense,
                _ => &hybrid,
            };
            let t = Instant::now();
            let res = retr.retrieve(&q, TOP_K).await?;
            let dt = t.elapsed().as_secs_f64() * 1000.0;
            match mode {
                "bm25" => bm25_ms += dt,
                "dense" => dense_ms += dt,
                _ => {}
            }
            let s = stats.entry((it.category.clone(), mode)).or_default();
            s.add(rank_of(&res, &gold), rank_of(&res, &trap));
            let a = stats.entry(("ALL".into(), mode)).or_default();
            a.add(rank_of(&res, &gold), rank_of(&res, &trap));
        }
    }

    println!(
        "\nSemantic-mismatch retrieval (n={} queries, top-{TOP_K})",
        data.items.len()
    );
    println!(
        "  {:<22} {:>6} {:>6} {:>6} {:>11}",
        "category / mode", "R@1", "R@3", "MRR", "trap>gold"
    );
    for cat in cats.iter().chain(std::iter::once(&"ALL".to_string())) {
        println!("  {}", "─".repeat(54));
        for mode in modes {
            if let Some(s) = stats.get(&(cat.clone(), mode)) {
                s.line(&format!("{cat} / {mode}"));
            }
        }
    }

    let nq = data.items.len() as f64;
    println!("\nLatency (per query, mean):");
    println!(
        "  query embedding (BGE):  {:.2} ms   ← the semantic tax",
        embed_ms / nq
    );
    println!("  bm25 retrieve:          {:.3} ms", bm25_ms / nq);
    println!("  dense retrieve (exact): {:.3} ms", dense_ms / nq);
    Ok(())
}
