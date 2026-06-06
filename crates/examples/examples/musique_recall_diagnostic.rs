//! MuSiQue recall-gap diagnostic — where is the 0.28 vs 0.76 recall
//! actually being lost?
//!
//! The CE-gate investigation surfaced a striking finding: same BGE-small,
//! same retrieval pipeline, recall@4 = 0.76 on HotpotQA and 0.28 on
//! MuSiQue. Before "improving" anything we need to know WHICH lever in
//! the pipeline is the bottleneck. This harness runs both corpora
//! through the SAME measurement and breaks recall down into:
//!
//! 1. **Pool recall (recall@K for K=4,10,20,50)** — if recall@50 is
//!    high, the gold is in the candidate pool and the bottleneck is
//!    RANKING (top-K cutoff). If recall@50 is also low, the bottleneck
//!    is RETRIEVAL MISSES — the gold isn't even surfacing.
//! 2. **Lexical (BM25) vs dense (BGE) baseline** — rules out
//!    embedder-vs-corpus mismatch. If BM25 beats dense on MuSiQue,
//!    the issue isn't the corpus difficulty but the lexical-vs-semantic
//!    fit.
//! 3. **Gold density (gold chunks per query)** — recall@4 has a
//!    different mathematical ceiling for queries that need 4 gold
//!    chunks vs 2. If MuSiQue queries need 3-4 gold chunks in their
//!    top-4 vs HotpotQA's 2, the gap is partly metric, not method.
//! 4. **Per-hop breakdown** (MuSiQue only) — does the gap concentrate
//!    on 3-hop / 4-hop queries vs 2-hop? Tells us whether hop count
//!    drives the failure, or it's something else.
//! 5. **Chunk count per document** — if MuSiQue's gold paragraphs
//!    get sliced into many chunks by the chunker, gold is diluted in
//!    the top-K (each gold chunk has lower retrieval weight).
//!
//! Output: a single comparison table printed to stdout. The numbers
//! tell us where to focus next.
//!
//! Run:
//!   REDHOP_BGE_MODEL=... REDHOP_BGE_TOKENIZER=... \
//!   cargo run -p redhop-examples --example musique_recall_diagnostic \
//!         --features onnx --release

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    Chunk, ChunkId, Chunker, Embedding, EmbeddingProvider, Query, RetrievalResult, Retriever,
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

fn recall_at_k(results: &[RetrievalResult], gold: &[ChunkId], k: usize) -> f32 {
    if gold.is_empty() {
        return 1.0;
    }
    let ids: HashSet<&ChunkId> = results.iter().take(k).map(|r| &r.chunk.id).collect();
    let found = gold.iter().filter(|g| ids.contains(g)).count();
    found as f32 / gold.len() as f32
}

struct Summary {
    name: String,
    n_queries: usize,
    n_chunks: usize,
    mean_chunks_per_doc: f32,
    mean_gold_per_query: f32,
    bm25_recall: HashMap<usize, f32>,
    dense_recall: HashMap<usize, f32>,
    /// Recall@4 conditional on hop count, if available. HotpotQA: always 2.
    per_hop_recall_at_4: HashMap<usize, (f32, usize)>, // hops -> (dense recall, count)
}

async fn diagnose(
    name: &str,
    corpus: LabeledCorpus,
    bge: Arc<dyn EmbeddingProvider>,
    hops_by_id: HashMap<String, usize>,
) -> anyhow::Result<Summary> {
    let chunker_tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(chunker_tok, 40, 60, 0)?;

    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    let n_chunks = chunks.len();
    let n_docs = corpus.docs.len();
    let mean_chunks_per_doc = n_chunks as f32 / n_docs.max(1) as f32;

    println!("[{name}] embedding {n_chunks} chunks (from {n_docs} docs)...");
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = bge.embed(&texts).await?;
    for (c, v) in chunks.iter_mut().zip(vecs.iter()) {
        c.embedding = Some(v.clone());
    }

    // Build dense + BM25 retrievers over the SAME chunk set.
    let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
    let store = Arc::new(ChunkStore::new());
    let mut dense = DenseRetriever::new(index, store);
    dense.index(&chunks).await?;
    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;

    let max_k = *KS.iter().max().unwrap();
    let mut bm25_recall: HashMap<usize, f32> = KS.iter().map(|&k| (k, 0.0)).collect();
    let mut dense_recall: HashMap<usize, f32> = KS.iter().map(|&k| (k, 0.0)).collect();
    let mut per_hop: HashMap<usize, (f32, usize)> = HashMap::new();
    let mut n_queries = 0;
    let mut gold_sum = 0usize;

    for lq in &corpus.queries {
        if lq.gold_chunk_ids.is_empty() {
            continue;
        }
        n_queries += 1;
        gold_sum += lq.gold_chunk_ids.len();

        let mut q = Query::new(&lq.text);
        q.embedding = lq.embedding.clone();
        let dense_hits = dense.retrieve(&q, max_k).await?;
        let bm25_hits = bm25.retrieve(&q, max_k).await?;
        for &k in KS {
            *bm25_recall.entry(k).or_default() += recall_at_k(&bm25_hits, &lq.gold_chunk_ids, k);
            *dense_recall.entry(k).or_default() += recall_at_k(&dense_hits, &lq.gold_chunk_ids, k);
        }
        let r4 = recall_at_k(&dense_hits, &lq.gold_chunk_ids, 4);
        let hops = hops_by_id.get(&lq.id).copied().unwrap_or(2);
        let e = per_hop.entry(hops).or_insert((0.0, 0));
        e.0 += r4;
        e.1 += 1;
    }
    let nq = n_queries.max(1) as f32;
    for &k in KS {
        *bm25_recall.get_mut(&k).unwrap() /= nq;
        *dense_recall.get_mut(&k).unwrap() /= nq;
    }
    for (_, (sum, count)) in per_hop.iter_mut() {
        *sum /= (*count).max(1) as f32;
    }

    Ok(Summary {
        name: name.to_string(),
        n_queries,
        n_chunks,
        mean_chunks_per_doc,
        mean_gold_per_query: gold_sum as f32 / nq,
        bm25_recall,
        dense_recall,
        per_hop_recall_at_4: per_hop,
    })
}

fn print_summary(s: &Summary) {
    println!("\n── {} ──", s.name);
    println!("  queries: {}", s.n_queries);
    println!(
        "  chunks: {}  (mean {:.2} chunks/doc)",
        s.n_chunks, s.mean_chunks_per_doc
    );
    println!("  mean gold chunks per query: {:.2}", s.mean_gold_per_query);
    println!("  recall by K and retriever:");
    println!("    {:<8} {:>10} {:>10}", "K", "BM25", "dense");
    for &k in KS {
        println!(
            "    {:<8} {:>10.4} {:>10.4}",
            k,
            s.bm25_recall.get(&k).copied().unwrap_or(0.0),
            s.dense_recall.get(&k).copied().unwrap_or(0.0)
        );
    }
    println!("  per-hop recall@4 (dense):");
    let mut keys: Vec<usize> = s.per_hop_recall_at_4.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let (r, c) = s.per_hop_recall_at_4[&k];
        println!("    {}-hop  n={:<4} recall@4 = {:.4}", k, c, r);
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("MuSiQue recall-gap diagnostic: decomposing 0.28 vs 0.76\n");

    let chunker_tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(chunker_tok, 40, 60, 0)?;

    println!("Loading BGE-small...");
    let (bge_model, bge_tokenizer) = redhop_examples::bge_small_paths();
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &bge_model,
        &bge_tokenizer,
        EmbedderConfig::bge(DIM),
    )?);

    // ── HotpotQA: stratified 100 bridge + 100 comparison, matching the
    // CE-gate setup so the 0.76 baseline reproduces.
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
    let hotpot_examples: Vec<_> = bridge.into_iter().chain(comparison).collect();
    let hotpot_hops: HashMap<String, usize> = hotpot_examples
        .iter()
        .map(|e| (e.id.clone(), 2)) // HotpotQA is always 2-hop
        .collect();
    hotpot.examples = hotpot_examples;
    let hotpot_qtexts: Vec<String> = hotpot.examples.iter().map(|e| e.question.clone()).collect();
    println!("Embedding HotpotQA queries...");
    let hotpot_qvecs = bge.embed(&hotpot_qtexts).await?;
    let hotpot_qmap: HashMap<String, Embedding> =
        hotpot_qtexts.into_iter().zip(hotpot_qvecs).collect();
    let hotpot_corpus =
        hotpot.to_labeled_corpus(&chunker, |q| hotpot_qmap.get(q).cloned(), hotpot_regime)?;
    let hotpot_sum = diagnose("HotpotQA", hotpot_corpus, bge.clone(), hotpot_hops).await?;

    // ── MuSiQue: 200 answerable examples.
    let mut musique = MuSiQueDataset::from_path(redhop_examples::data_path("musique/dev.jsonl"))?;
    musique.examples.retain(|e| e.answerable);
    musique.examples.truncate(SAMPLE);
    let musique_hops: HashMap<String, usize> = musique
        .examples
        .iter()
        .map(|e| (e.id.clone(), e.question_decomposition.len()))
        .collect();
    let musique_qtexts: Vec<String> = musique
        .examples
        .iter()
        .map(|e| e.question.clone())
        .collect();
    println!("\nEmbedding MuSiQue queries...");
    let musique_qvecs = bge.embed(&musique_qtexts).await?;
    let musique_qmap: HashMap<String, Embedding> =
        musique_qtexts.into_iter().zip(musique_qvecs).collect();
    let musique_corpus =
        musique.to_labeled_corpus(&chunker, |q| musique_qmap.get(q).cloned(), musique_regime)?;
    let musique_sum = diagnose("MuSiQue", musique_corpus, bge.clone(), musique_hops).await?;

    print_summary(&hotpot_sum);
    print_summary(&musique_sum);

    // ── Headline comparison table ──
    println!("\n══ HEADLINE COMPARISON ══");
    println!("  {:<28} {:>14} {:>14}", "metric", "HotpotQA", "MuSiQue");
    println!("  {}", "─".repeat(58));
    println!(
        "  {:<28} {:>14} {:>14}",
        "mean gold chunks/query",
        format!("{:.2}", hotpot_sum.mean_gold_per_query),
        format!("{:.2}", musique_sum.mean_gold_per_query),
    );
    println!(
        "  {:<28} {:>14} {:>14}",
        "mean chunks/doc",
        format!("{:.2}", hotpot_sum.mean_chunks_per_doc),
        format!("{:.2}", musique_sum.mean_chunks_per_doc),
    );
    for &k in KS {
        println!(
            "  {:<28} {:>14} {:>14}",
            format!("BM25 recall@{}", k),
            format!("{:.4}", hotpot_sum.bm25_recall[&k]),
            format!("{:.4}", musique_sum.bm25_recall[&k]),
        );
    }
    for &k in KS {
        println!(
            "  {:<28} {:>14} {:>14}",
            format!("dense recall@{}", k),
            format!("{:.4}", hotpot_sum.dense_recall[&k]),
            format!("{:.4}", musique_sum.dense_recall[&k]),
        );
    }

    // ── Diagnostic interpretation ──
    println!("\n── interpretation (preliminary) ──");
    let pool_recall_50_musique = musique_sum.dense_recall[&50];
    let pool_recall_50_hotpot = hotpot_sum.dense_recall[&50];
    let ranking_problem_share = if pool_recall_50_musique > 0.5 {
        format!(
            "pool recall@50 = {:.2} on MuSiQue — gold is mostly findable; bottleneck is RANKING/top-K cutoff",
            pool_recall_50_musique
        )
    } else {
        format!(
            "pool recall@50 = {:.2} on MuSiQue — gold isn't even surfacing; bottleneck is RETRIEVAL MISSES",
            pool_recall_50_musique
        )
    };
    println!("  • {ranking_problem_share}");

    let bm25_vs_dense_musique = musique_sum.bm25_recall[&4] - musique_sum.dense_recall[&4];
    if bm25_vs_dense_musique > 0.02 {
        println!(
            "  • BM25 beats dense on MuSiQue@4 by Δ = {:+.4}. The dense embedder is the wrong tool here.",
            bm25_vs_dense_musique
        );
    } else if bm25_vs_dense_musique < -0.02 {
        println!(
            "  • Dense beats BM25 on MuSiQue@4 by Δ = {:+.4}. The embedder is doing its job; the issue isn't BGE.",
            -bm25_vs_dense_musique
        );
    } else {
        println!(
            "  • BM25 ≈ dense on MuSiQue@4 (Δ = {:+.4}). Both methods hit the same ceiling.",
            bm25_vs_dense_musique
        );
    }

    let gold_ratio = musique_sum.mean_gold_per_query / hotpot_sum.mean_gold_per_query;
    if gold_ratio > 1.3 {
        println!(
            "  • MuSiQue requires {:.1}× more gold chunks per query than HotpotQA ({:.2} vs {:.2}). \
             At fixed k_final, the recall@4 ceiling is mathematically lower — this is partly a \
             metric artifact, not a method gap.",
            gold_ratio, musique_sum.mean_gold_per_query, hotpot_sum.mean_gold_per_query
        );
    }

    println!(
        "  • HotpotQA reproduces 0.76 dense recall@4 baseline: {:.4}",
        hotpot_sum.dense_recall[&4]
    );
    println!(
        "  • MuSiQue reproduces ~0.28 dense recall@4 baseline: {:.4}",
        musique_sum.dense_recall[&4]
    );
    println!(
        "  • Pool-recall headroom on MuSiQue: dense@4 = {:.2}, dense@50 = {:.2} (gap = {:+.2})",
        musique_sum.dense_recall[&4],
        musique_sum.dense_recall[&50],
        musique_sum.dense_recall[&50] - musique_sum.dense_recall[&4]
    );

    let _ = pool_recall_50_hotpot;
    Ok(())
}
