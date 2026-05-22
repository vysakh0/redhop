//! Token-efficiency curves: pruned context vs raw top-k stuffing.
//!
//! On real HotpotQA + BGE dense retrieval (wide net = top-20), sweep the
//! token budget across four `build_context` strategies and measure, per
//! (strategy, budget):
//!
//!   - gold retained   : fraction of gold chunks present in the context
//!   - tokens used     : actual context size
//!   - distractor ratio: fraction of context chunks below grounding cutoff
//!   - evidence density: query-relevant tokens / total tokens
//!
//! Two predictions in tension:
//!   - context-economics: pruning concentrates evidence → fewer tokens,
//!     lower distractor ratio, higher density at equal gold retention.
//!   - reranking-limits : query-relevance pruning (DistractorFiltered,
//!     MaxDensity) drops the low-query-relevance SECOND HOP → LOWER gold
//!     retention on multi-hop questions.
//!
//! The experiment shows which dominates — and that is the honest answer
//! to "is density pruning safe on multi-hop?".
//!
//! Requires `--features onnx` + BGE-small.
//!
//! Run:
//!   cargo run -p neorag-examples --example context_economics --features onnx --release

use std::collections::HashMap;
use std::sync::Arc;

use neorag_calibration::loaders::hotpotqa::{default_regime, HotpotQADataset};
use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
use neorag_context::{build_context, ContextConfig, ContextStrategy};
use neorag_core::{
    Chunker, ChunkId, Embedding, EmbeddingProvider, Query, Retriever, TokenizerBackend,
    VectorIndex,
};
use neorag_embeddings::{EmbedderConfig, OnnxEmbedder};
use neorag_retrieval::DenseRetriever;
use neorag_storage::{ChunkStore, FlatVectorIndex};
use parking_lot::RwLock;

const HOTPOTQA_PATH: &str =
    "/Users/vysakh/projects/neorag/data/hotpotqa/hotpot_dev_distractor_v1.json";
const BGE_MODEL: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/onnx/model.onnx";
const BGE_TOKENIZER: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/tokenizer.json";
const SAMPLE_SIZE: usize = 60;
const WIDE_N: usize = 20;
const DIM: usize = 384;

struct Row {
    gold_retained: f32,
    tokens: f32,
    distractor: f32,
    density: f32,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Context economics: token-efficiency of pruned vs raw top-k      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let mut dataset = HotpotQADataset::from_path(HOTPOTQA_PATH)?;
    dataset.examples.truncate(SAMPLE_SIZE);
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;

    println!("loading BGE-small...");
    let bge: Arc<dyn EmbeddingProvider> =
        Arc::new(OnnxEmbedder::load(BGE_MODEL, BGE_TOKENIZER, EmbedderConfig::bge(DIM))?);

    let q_texts: Vec<String> = dataset.examples.iter().map(|e| e.question.clone()).collect();
    let q_vecs = bge.embed(&q_texts).await?;
    let q_map: HashMap<String, Embedding> = q_texts.into_iter().zip(q_vecs).collect();
    let corpus = dataset.to_labeled_corpus(&chunker, |q| q_map.get(q).cloned(), default_regime)?;

    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    println!("embedding {} chunks with BGE...", chunks.len());
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = bge.embed(&texts).await?;
    for (c, v) in chunks.iter_mut().zip(vecs.iter()) {
        c.embedding = Some(v.clone());
    }
    let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
    let store = Arc::new(ChunkStore::new());
    let mut dense = DenseRetriever::new(index, store);
    dense.index(&chunks).await?;

    // Pre-retrieve the wide net for each query once.
    let mut wide_nets: Vec<(Vec<ChunkId>, Vec<neorag_core::RetrievalResult>)> = Vec::new();
    for lq in &corpus.queries {
        if lq.gold_chunk_ids.is_empty() {
            continue;
        }
        let mut query = Query::new(&lq.text);
        query.embedding = lq.embedding.clone();
        let wide = dense.retrieve(&query, WIDE_N).await?;
        wide_nets.push((lq.gold_chunk_ids.clone(), wide));
    }
    println!("evaluating {} queries\n", wide_nets.len());

    let budgets = [80usize, 150, 250, 400, 800];
    let strategies = [
        ("raw_topk", ContextStrategy::RawTopK),
        ("distractor_filtered", ContextStrategy::DistractorFiltered),
        ("redundancy_pruned", ContextStrategy::RedundancyPruned),
        ("max_density", ContextStrategy::MaxDensity),
    ];

    for (sname, strat) in strategies {
        println!("──── strategy: {sname} ────");
        println!(
            "  {:<8} {:>13} {:>10} {:>12} {:>10}",
            "budget", "gold_retained", "tokens", "distractor", "density"
        );
        for &budget in &budgets {
            let mut agg = Row { gold_retained: 0.0, tokens: 0.0, distractor: 0.0, density: 0.0 };
            let mut n: f32 = 0.0;
            for (lq, (gold, wide)) in corpus
                .queries
                .iter()
                .filter(|q| !q.gold_chunk_ids.is_empty())
                .zip(wide_nets.iter())
            {
                let mut query = Query::new(&lq.text);
                query.embedding = lq.embedding.clone();
                let ctx = build_context(
                    &query,
                    wide,
                    &ContextConfig {
                        token_budget: budget,
                        strategy: strat,
                        distractor_min_grounding: 0.10,
                        link_min_jaccard: 0.12,
                        redundancy_max_cosine: 0.92,
                    },
                );
                let found = gold.iter().filter(|g| ctx.contains(g)).count();
                agg.gold_retained += found as f32 / gold.len() as f32;
                agg.tokens += ctx.total_tokens as f32;
                agg.distractor += ctx.economics.distractor_ratio;
                agg.density += ctx.economics.evidence_density;
                n += 1.0;
            }
            let nf = n.max(1.0);
            println!(
                "  {:<8} {:>13.3} {:>10.0} {:>12.3} {:>10.3}",
                budget,
                agg.gold_retained / nf,
                agg.tokens / nf,
                agg.distractor / nf,
                agg.density / nf
            );
        }
        println!();
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("Reading: at a FIXED budget, compare strategies. If max_density /");
    println!("distractor_filtered show HIGHER density + LOWER distractor but LOWER");
    println!("gold_retained than raw_topk, that is the reranking-limits geometry");
    println!("repeating: query-relevance pruning concentrates evidence but drops the");
    println!("low-query-relevance second hop. Context pruning is an evidence-");
    println!("CONCENTRATION tool, honest about what it discards — not a multi-hop fix.");
    println!("════════════════════════════════════════════════════════════════════════");
    Ok(())
}
