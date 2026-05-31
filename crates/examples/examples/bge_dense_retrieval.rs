//! BGE in the ACTION path: dense retrieval vs BM25.
//!
//! The recalibration experiment showed that with BGE only in the
//! *sensing* path (diagnostics), the controller's economics didn't move
//! — because the retriever (BM25) and reranker (lexical) were
//! embedding-blind. This experiment puts BGE in the *action* path by
//! swapping the retriever to dense BGE retrieval, holding everything
//! else constant:
//!
//!   both arms: BGE diagnostics, same classifier, same policy
//!   arm A retriever: BM25            (embedding-blind action path)
//!   arm B retriever: dense BGE       (embedding-driven action path)
//!
//! Now the substrate genuinely changes what the controller can DO. The
//! questions this can finally answer:
//!   - does strong first-stage retrieval raise STATIC recall?
//!   - does the controller then escalate LESS (less to fix)?
//!   - is the recall lift from intervention smaller (less headroom)?
//!
//! Requires `--features onnx` + the BGE-small model.
//!
//! Run:
//!   cargo run -p redhop-examples --example bge_dense_retrieval \
//!       --features onnx --release

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    Chunk, ChunkId, Chunker, DiagnosticsEngine, Embedding, EmbeddingProvider, Query,
    RegimeClassifier, Reranker, RerankerLevel, Result as CoreResult, RetrievalResult, Retriever,
    TokenizerBackend, VectorIndex,
};
use redhop::embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop::reranking::LexicalGroundingReranker;
use redhop::retrieval::{Bm25Retriever, DenseRetriever};
use redhop::storage::{ChunkStore, FlatVectorIndex};
use redhop_calibration::{
    analysis::regret_summary,
    economics::{economics, CostModel},
    loaders::hotpotqa::{default_regime, HotpotQADataset},
    reliability::reliability_diagram,
    runner::{run_query, QueryOutcome, RunnerConfig},
};
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use redhop_orchestration::{ConservativeRulePolicy, Policy, RuleBasedClassifier};

const HOTPOTQA_PATH: &str =
    "/Users/vysakh/projects/neorag/data/hotpotqa/hotpot_dev_distractor_v1.json";
const BGE_MODEL: &str = "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/onnx/model.onnx";
const BGE_TOKENIZER: &str = "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/tokenizer.json";
const SAMPLE_SIZE: usize = 60;
const TOP_K: usize = 4;
const DIM: usize = 384;

struct EmbedAttachingRetriever {
    inner: Arc<dyn Retriever>,
    by_id: HashMap<ChunkId, Embedding>,
}
#[async_trait]
impl Retriever for EmbedAttachingRetriever {
    async fn retrieve(&self, q: &Query, top_k: usize) -> CoreResult<Vec<RetrievalResult>> {
        let mut results = self.inner.retrieve(q, top_k).await?;
        for r in &mut results {
            if r.chunk.embedding.is_none() {
                if let Some(e) = self.by_id.get(&r.chunk.id) {
                    r.chunk.embedding = Some(e.clone());
                }
            }
        }
        Ok(results)
    }
    async fn index(&mut self, _c: &[Chunk]) -> CoreResult<()> {
        Ok(())
    }
    fn name(&self) -> &'static str {
        "embed_attaching"
    }
}

struct Metrics {
    static_recall: f32,
    adaptive_recall: f32,
    recall_lift: f32,
    intervention_rate: f32,
    useful: f32,
    rerank_calls: f32,
    harmful_lift: f32,
    wasted: usize,
    ece: f32,
}

fn aggregate(outcomes: &[QueryOutcome]) -> Metrics {
    let n = outcomes.len().max(1) as f32;
    let intervened: Vec<&QueryOutcome> = outcomes.iter().filter(|o| o.intervened).collect();
    let useful = if intervened.is_empty() {
        0.0
    } else {
        intervened.iter().filter(|o| o.recall_lift > 0.0).count() as f32 / intervened.len() as f32
    };
    let regret = regret_summary(outcomes);
    let econ = economics(outcomes, &CostModel::default());
    let diag = reliability_diagram(outcomes, 10);
    Metrics {
        static_recall: outcomes.iter().map(|o| o.gold_recall_static).sum::<f32>() / n,
        adaptive_recall: outcomes.iter().map(|o| o.gold_recall_adaptive).sum::<f32>() / n,
        recall_lift: outcomes.iter().map(|o| o.recall_lift).sum::<f32>() / n,
        intervention_rate: intervened.len() as f32 / n,
        useful,
        rerank_calls: econ.mean_rerank_calls,
        harmful_lift: regret.mean_harmful_lift,
        wasted: regret.n_wasted_interventions,
        ece: diag.ece,
    }
}

async fn run_eval(
    retriever: Arc<dyn Retriever>,
    diagnostics: Arc<dyn DiagnosticsEngine>,
    classifier: Arc<dyn RegimeClassifier>,
    policy: Arc<dyn Policy>,
    corpus: &redhop_calibration::dataset::LabeledCorpus,
) -> anyhow::Result<Metrics> {
    let rerankers: Vec<(RerankerLevel, Arc<dyn Reranker>)> = vec![(
        RerankerLevel::Lexical,
        Arc::new(LexicalGroundingReranker::default()),
    )];
    let cfg = RunnerConfig {
        retriever,
        diagnostics,
        classifier,
        policy,
        rerankers,
        top_k: TOP_K,
    };
    let mut outcomes = Vec::with_capacity(corpus.queries.len());
    for q in &corpus.queries {
        outcomes.push(run_query(q, &cfg).await?);
    }
    Ok(aggregate(&outcomes))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  BGE in the ACTION path: dense retrieval vs BM25                 ║");
    println!("║  (BGE diagnostics + controller held constant; only retriever varies)║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let mut dataset = HotpotQADataset::from_path(HOTPOTQA_PATH)?;
    dataset.examples.truncate(SAMPLE_SIZE);
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;

    // BGE provider — used for query embeddings, chunk embeddings, and
    // (via attach) the semantic diagnostics in both arms.
    println!("loading BGE-small ONNX...");
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        BGE_MODEL,
        BGE_TOKENIZER,
        EmbedderConfig::bge(DIM),
    )?);

    // Pre-embed queries with BGE → LabeledCorpus.
    let q_texts: Vec<String> = dataset
        .examples
        .iter()
        .map(|e| e.question.clone())
        .collect();
    let q_vecs = bge.embed(&q_texts).await?;
    let q_map: HashMap<String, Embedding> = q_texts.into_iter().zip(q_vecs).collect();
    let corpus = dataset.to_labeled_corpus(&chunker, |q| q_map.get(q).cloned(), default_regime)?;

    // Chunk + embed corpus with BGE (chunk ids match gold ids).
    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    println!("embedding {} chunks with BGE...", chunks.len());
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = bge.embed(&texts).await?;
    for (c, v) in chunks.iter_mut().zip(vecs.iter()) {
        c.embedding = Some(v.clone());
    }
    let by_id: HashMap<ChunkId, Embedding> = chunks
        .iter()
        .map(|c| (c.id.clone(), c.embedding.clone().unwrap()))
        .collect();

    // Shared diagnostics (BGE-fed) + controller + policy.
    let diagnostics: Arc<dyn DiagnosticsEngine> =
        Arc::new(LayeredDiagnosticsEngine::lexical_and_semantic(
            Arc::new(DefaultDiagnosticsEngine::new()),
            Arc::new(SemanticDiagnosticsEngine::new()),
        ));
    let classifier: Arc<dyn RegimeClassifier> = Arc::new(RuleBasedClassifier::new());
    let policy: Arc<dyn Policy> = Arc::new(ConservativeRulePolicy::new());

    // ── Arm A: BM25 (embedding-blind action path) ──
    println!("\narm A: BM25 retrieval (embedding-blind action path)...");
    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;
    let bm25_arm: Arc<dyn Retriever> = Arc::new(EmbedAttachingRetriever {
        inner: Arc::new(bm25),
        by_id: by_id.clone(),
    });
    let a = run_eval(
        bm25_arm,
        diagnostics.clone(),
        classifier.clone(),
        policy.clone(),
        &corpus,
    )
    .await?;

    // ── Arm B: dense BGE (embedding-driven action path) ──
    println!("arm B: dense BGE retrieval (embedding-driven action path)...");
    let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
    let store = Arc::new(ChunkStore::new());
    let mut dense = DenseRetriever::new(index, store);
    dense.index(&chunks).await?;
    let dense_arm: Arc<dyn Retriever> = Arc::new(dense);
    let b = run_eval(dense_arm, diagnostics, classifier, policy, &corpus).await?;

    // ── Compare ──
    println!("\n──── controller economics by retriever (BGE diagnostics held constant) ────");
    let f3 = |v: f32| format!("{v:.3}");
    let p0 = |v: f32| format!("{:.0}%", v * 100.0);
    let row = |name: &str, x: f32, y: f32, f: &dyn Fn(f32) -> String| {
        println!("  {:<24} {:>14} {:>16}", name, f(x), f(y));
    };
    println!(
        "  {:<24} {:>14} {:>16}",
        "metric", "BM25 (blind)", "dense BGE (action)"
    );
    println!("  {}", "─".repeat(56));
    row("static recall", a.static_recall, b.static_recall, &f3);
    row("adaptive recall", a.adaptive_recall, b.adaptive_recall, &f3);
    row("recall lift (adaptive)", a.recall_lift, b.recall_lift, &f3);
    row(
        "intervention rate",
        a.intervention_rate,
        b.intervention_rate,
        &p0,
    );
    row("useful %", a.useful, b.useful, &p0);
    row("rerank calls/query", a.rerank_calls, b.rerank_calls, &f3);
    row("mean harmful lift", a.harmful_lift, b.harmful_lift, &f3);
    println!(
        "  {:<24} {:>14} {:>16}",
        "wasted interventions", a.wasted, b.wasted
    );
    row("ECE", a.ece, b.ece, &f3);

    // ── Verdict ──
    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("CAUSAL TEST: substrate now in the retrieval action path");
    println!(
        "  Δ static recall (dense − BM25):     {:+.3}",
        b.static_recall - a.static_recall
    );
    println!(
        "  Δ intervention rate:                {:+.1} pts",
        (b.intervention_rate - a.intervention_rate) * 100.0
    );
    println!(
        "  Δ recall lift (from intervention):  {:+.3}",
        b.recall_lift - a.recall_lift
    );
    println!();
    let strong_static = b.static_recall > a.static_recall + 0.05;
    let less_interv = b.intervention_rate < a.intervention_rate - 0.02;
    let safe = b.harmful_lift <= 1e-6;
    if strong_static && less_interv && safe {
        println!("  → STRONG RETRIEVAL REDUCES ESCALATION NEED: dense BGE lifts static");
        println!("    recall and the controller correctly escalates LESS — substrate in");
        println!("    the action path genuinely changes the economics. Zero harm holds.");
    } else if strong_static && !less_interv {
        println!("  → dense BGE lifts static recall but the controller escalates similarly");
        println!("    — escalation is driven by signals orthogonal to first-stage recall.");
    } else {
        println!("  → report the deltas above; don't force a story.");
    }
    println!("════════════════════════════════════════════════════════════════════════");
    Ok(())
}
