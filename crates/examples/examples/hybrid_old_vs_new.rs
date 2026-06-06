//! Direct A/B benchmark: OLD `RetrievalMode::Hybrid` (BM25-prune-then-
//! dense-rerank, the previous default) vs NEW `RetrievalMode::Hybrid`
//! (BM25 + global dense + RRF, the post-2026-06-06 default).
//!
//! The `musique_hybrid_recall` experiment that motivated the refactor
//! compared THREE retrievers (BM25-alone, dense-alone, RRF-of-two) on
//! the question "does RRF win over single retrievers." It did NOT
//! directly compare the old and new Hybrid compositions, which is the
//! honest test for "is the runtime change worth shipping."
//!
//! This harness runs that direct comparison. Same corpus, same queries,
//! same chunker, same models — the only variable is the Hybrid
//! composition. Recall is measured at K ∈ {4, 10, 20, 50} so we can see
//! whether the wide-net win (the +0.07 claim) actually carries through
//! to the candidate_k=20 default users actually consume, or whether it
//! only shows up at K=50 (academic).
//!
//! Pre-registered success criterion: NEW Hybrid beats OLD Hybrid at
//! K=20 (the default candidate_k) by Δ ≥ +0.02 on at least one corpus
//! without regressing the other by more than 0.01. Anything weaker is
//! "the change didn't carry through to the user-facing recall."
//!
//! Run:
//!   REDHOP_BGE_MODEL=... REDHOP_BGE_TOKENIZER=... \
//!   cargo run -p redhop-examples --example hybrid_old_vs_new \
//!         --features onnx --release

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    ChunkId, Chunker, Embedding, EmbeddingProvider, Query, Retriever, TokenizerBackend, VectorIndex,
};
use redhop::embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop::retrieval::{Bm25Retriever, DenseRetriever, HybridRetriever, LocalRerankRetriever};
use redhop::storage::{ChunkStore, FlatVectorIndex};
use redhop_calibration::dataset::LabeledCorpus;
use redhop_calibration::loaders::hotpotqa::{default_regime as hotpot_regime, HotpotQADataset};
use redhop_calibration::loaders::musique::{default_regime as musique_regime, MuSiQueDataset};

const SAMPLE: usize = 200;
const DIM: usize = 384;
const KS: &[usize] = &[4, 10, 20, 50];
const CANDIDATE_POOL: usize = 50;

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
    arm: String, // "old_hybrid" | "new_hybrid"
    recall: HashMap<usize, f32>,
}

async fn evaluate(
    corpus_name: &str,
    corpus: &LabeledCorpus,
    bge: Arc<dyn EmbeddingProvider>,
) -> anyhow::Result<Vec<Row>> {
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;
    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    println!("  [{corpus_name}] embedding {} chunks...", chunks.len());
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = bge.embed(&texts).await?;
    for (c, v) in chunks.iter_mut().zip(vecs.iter()) {
        c.embedding = Some(v.clone());
    }

    // OLD Hybrid: LocalRerankRetriever (BM25-prune-then-dense-rerank +
    // RRF-fuse the two rankings of the SAME pool). Constructed manually
    // from the public retrieval surface — this is what
    // `RetrievalMode::Hybrid` did before 2026-06-06 and what
    // LocalRerankRetriever does today.
    let mut old_hybrid = LocalRerankRetriever::new(bge.clone(), CANDIDATE_POOL)?
        .with_analyzer(redhop::analyzer::default_english())?;
    old_hybrid.index(&chunks).await?;

    // NEW Hybrid: BM25 + global Dense, RRF-fused. Constructed using the
    // same building blocks `Document::ensure_indexed()` now uses for
    // `RetrievalMode::Hybrid`.
    let mut bm25 = Bm25Retriever::new()?;
    bm25.index(&chunks).await?;
    let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
    let store = Arc::new(ChunkStore::new());
    let mut dense = DenseRetriever::new(index, store);
    dense.index(&chunks).await?;
    // For an apples-to-apples comparison, the NEW arm uses DenseRetriever
    // directly rather than LocalRerankRetriever.global() — both produce a
    // global dense ranking; DenseRetriever is the cleaner building block
    // and the same one a downstream HybridRetriever consumes.
    let new_hybrid = HybridRetriever::rrf(
        vec![Arc::new(bm25) as Arc<dyn Retriever>, Arc::new(dense)],
        CANDIDATE_POOL,
    );

    let mut old_acc: HashMap<usize, f32> = KS.iter().map(|&k| (k, 0.0)).collect();
    let mut new_acc: HashMap<usize, f32> = KS.iter().map(|&k| (k, 0.0)).collect();
    let mut n = 0usize;
    let max_k = *KS.iter().max().unwrap();

    for lq in &corpus.queries {
        if lq.gold_chunk_ids.is_empty() {
            continue;
        }
        n += 1;
        let mut q = Query::new(&lq.text);
        q.embedding = lq.embedding.clone();

        let old_hits = old_hybrid.retrieve(&q, max_k).await?;
        let new_hits = new_hybrid.retrieve(&q, max_k).await?;
        let old_ids: Vec<String> = old_hits
            .iter()
            .map(|r| r.chunk.id.as_str().to_string())
            .collect();
        let new_ids: Vec<String> = new_hits
            .iter()
            .map(|r| r.chunk.id.as_str().to_string())
            .collect();
        for &k in KS {
            *old_acc.get_mut(&k).unwrap() += recall_at_k(&old_ids, &lq.gold_chunk_ids, k);
            *new_acc.get_mut(&k).unwrap() += recall_at_k(&new_ids, &lq.gold_chunk_ids, k);
        }
    }
    let nf = n.max(1) as f32;
    for &k in KS {
        *old_acc.get_mut(&k).unwrap() /= nf;
        *new_acc.get_mut(&k).unwrap() /= nf;
    }

    Ok(vec![
        Row {
            corpus: corpus_name.to_string(),
            arm: "old_hybrid".to_string(),
            recall: old_acc,
        },
        Row {
            corpus: corpus_name.to_string(),
            arm: "new_hybrid".to_string(),
            recall: new_acc,
        },
    ])
}

fn print_corpus(rows: &[Row], corpus_name: &str) {
    println!("\n══ {corpus_name} ══");
    println!(
        "  {:<14} {:>10} {:>10} {:>10} {:>10}",
        "arm",
        format!("@{}", KS[0]),
        format!("@{}", KS[1]),
        format!("@{}", KS[2]),
        format!("@{}", KS[3])
    );
    for arm in ["old_hybrid", "new_hybrid"] {
        if let Some(r) = rows
            .iter()
            .find(|r| r.corpus == corpus_name && r.arm == arm)
        {
            print!("  {:<14}", arm);
            for &k in KS {
                print!(" {:>10.4}", r.recall[&k]);
            }
            println!();
        }
    }
    let old = rows
        .iter()
        .find(|r| r.corpus == corpus_name && r.arm == "old_hybrid");
    let new = rows
        .iter()
        .find(|r| r.corpus == corpus_name && r.arm == "new_hybrid");
    if let (Some(o), Some(n)) = (old, new) {
        print!("  {:<14}", "Δ (new-old)");
        for &k in KS {
            let d = n.recall[&k] - o.recall[&k];
            let marker = if d > 0.005 {
                "+"
            } else if d < -0.005 {
                "-"
            } else {
                "≈"
            };
            print!(" {:>+9.4}{}", d, marker);
        }
        println!();
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("Direct A/B: OLD Hybrid (LocalRerank) vs NEW Hybrid (BM25 + global Dense + RRF)\n");

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
    let hotpot_rows = evaluate("HotpotQA", &hotpot_corpus, bge.clone()).await?;

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
    let musique_rows = evaluate("MuSiQue", &musique_corpus, bge.clone()).await?;

    let mut all = hotpot_rows;
    all.extend(musique_rows);
    print_corpus(&all, "HotpotQA");
    print_corpus(&all, "MuSiQue");

    // ── Verdict at the OPERATIVE candidate_k = 20 ──
    let pick = |corpus: &str, arm: &str, k: usize| {
        all.iter()
            .find(|r| r.corpus == corpus && r.arm == arm)
            .map(|r| r.recall[&k])
            .unwrap_or(0.0)
    };
    println!("\n──── verdict at candidate_k = 20 (the default) ────");
    for corpus in ["HotpotQA", "MuSiQue"] {
        let o = pick(corpus, "old_hybrid", 20);
        let n = pick(corpus, "new_hybrid", 20);
        let d = n - o;
        let tag = if d > 0.02 {
            "✓ ship (clear win)"
        } else if d > 0.005 {
            "~ marginal win — re-check at K=4 too"
        } else if d.abs() <= 0.005 {
            "≈ tie — runtime change has no user-facing impact at default K"
        } else {
            "✗ regression at default K — DO NOT SHIP"
        };
        println!("  {corpus}: old = {o:.4}, new = {n:.4}, Δ = {d:+.4}  → {tag}");
    }
    println!("\n──── verdict at K=4 (cutoff after assembly) ────");
    for corpus in ["HotpotQA", "MuSiQue"] {
        let o = pick(corpus, "old_hybrid", 4);
        let n = pick(corpus, "new_hybrid", 4);
        let d = n - o;
        let tag = if d > 0.01 {
            "✓ improves top-K"
        } else if d > -0.005 {
            "≈ neutral at top-K"
        } else if d > -0.02 {
            "~ small top-K regression — check whether the wide-K win compensates"
        } else {
            "✗ TOP-K REGRESSION — users consuming top-4 directly will see worse recall"
        };
        println!("  {corpus}: old = {o:.4}, new = {n:.4}, Δ = {d:+.4}  → {tag}");
    }
    Ok(())
}
