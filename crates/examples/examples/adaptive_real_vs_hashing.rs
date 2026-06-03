//! The central operational experiment:
//! does a stronger semantic substrate sharpen the adaptive controller?
//!
//! We run the adaptive threshold sweep TWICE on the same HotpotQA
//! sample, holding the candidate retriever CONSTANT (BM25) and varying
//! ONLY the embedding backend that feeds the semantic-tier diagnostics:
//!
//!   arm A: hashing embeddings → semantic diagnostics
//!   arm B: BGE-small ONNX     → semantic diagnostics
//!
//! Because retrieval is identical across arms (BM25 ignores
//! embeddings), the *static* recall is identical too. Any difference in
//! intervention rate, useful%, regret, ECE, escalation frequency, or
//! recall lift is attributable purely to the controller making better
//! decisions off sharper semantic sensing — which is exactly the
//! question:
//!
//!   Do better embeddings let the controller intervene less and more
//!   precisely — i.e. reduce the need for expensive reranking?
//!
//! Requires `--features onnx` and the BGE-small model (see
//! docs/findings/EMBEDDING_BAKEOFF.md for the one-time download).
//!
//! Run:
//!   cargo run -p redhop-examples --example adaptive_real_vs_hashing \
//!       --features onnx --release

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    Chunk, ChunkId, Chunker, DiagnosticsEngine, Embedding, EmbeddingProvider, Query,
    RegimeClassifier, Reranker, RerankerLevel, Result as CoreResult, RetrievalResult, Retriever,
    TokenizerBackend,
};
use redhop::embeddings::{EmbedderConfig, HashingProvider, OnnxEmbedder};
use redhop::reranking::LexicalGroundingReranker;
use redhop::retrieval::Bm25Retriever;
use redhop_calibration::{
    analysis::{confusion_matrix, regret_summary},
    economics::{economics, CostModel},
    loaders::hotpotqa::{default_regime, HotpotQADataset},
    reliability::reliability_diagram,
    sweep::{SweepReport, ThresholdSweep},
};
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use redhop_orchestration::RuleBasedClassifier;

const SAMPLE_SIZE: usize = 60;
const TOP_K: usize = 4;
const DISTRACTOR_GRID: [f32; 2] = [0.40, 0.50];
const AMBIGUOUS_GRID: [f32; 2] = [0.30, 0.40];

/// BM25 retriever that re-attaches precomputed embeddings to results so
/// the semantic-tier diagnostics have vectors to read. Retrieval order
/// is pure BM25 — identical across embedding arms.
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

/// Aggregated metrics for one embedding arm.
struct ArmMetrics {
    label: String,
    intervention_rate: f32,
    mean_recall_lift: f32,
    fraction_useful: f32,
    mean_rerank_calls: f32,
    rerank_compute_reduction: f32,
    mean_useful_lift: f32,
    mean_harmful_lift: f32,
    wasted: usize,
    classifier_accuracy: f32,
    ece: f32,
}

fn summarize(label: &str, report: &SweepReport) -> ArmMetrics {
    // Best-lift row for the operating-point figures.
    let best = report.argmax_lift().cloned().unwrap();
    // Aggregate analyses across all settings' outcomes.
    let all: Vec<_> = report
        .outcomes
        .iter()
        .flat_map(|v| v.iter().cloned())
        .collect();
    let regret = regret_summary(&all);
    let cm = confusion_matrix(&all);
    let diag = reliability_diagram(&all, 10);
    let econ = economics(&all, &CostModel::default());
    ArmMetrics {
        label: label.to_string(),
        intervention_rate: best.intervention_rate,
        mean_recall_lift: best.mean_recall_lift,
        fraction_useful: best.fraction_useful_interventions,
        mean_rerank_calls: econ.mean_rerank_calls,
        rerank_compute_reduction: econ.rerank_compute_reduction,
        mean_useful_lift: regret.mean_useful_lift,
        mean_harmful_lift: regret.mean_harmful_lift,
        wasted: regret.n_wasted_interventions,
        classifier_accuracy: cm.accuracy,
        ece: diag.ece,
    }
}

async fn embed_map(
    provider: &Arc<dyn EmbeddingProvider>,
    chunks: &[Chunk],
) -> CoreResult<HashMap<ChunkId, Embedding>> {
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = provider.embed(&texts).await?;
    Ok(chunks
        .iter()
        .zip(vecs)
        .map(|(c, v)| (c.id.clone(), v))
        .collect())
}

async fn run_arm(
    label: &str,
    provider: Arc<dyn EmbeddingProvider>,
    dataset: &HotpotQADataset,
    chunker: &SentenceChunker,
) -> anyhow::Result<ArmMetrics> {
    // Pre-embed queries with this arm's provider.
    let q_texts: Vec<String> = dataset
        .examples
        .iter()
        .map(|e| e.question.clone())
        .collect();
    let q_vecs = provider.embed(&q_texts).await?;
    let q_map: HashMap<String, Embedding> = q_texts.into_iter().zip(q_vecs).collect();

    let corpus = dataset.to_labeled_corpus(chunker, |q| q_map.get(q).cloned(), default_regime)?;

    // Chunk + embed corpus (chunk ids match the loader's gold ids).
    let chunks = chunker.chunk_batch(&corpus.docs)?;
    let by_id = embed_map(&provider, &chunks).await?;

    // BM25 retriever (identical retrieval order across arms) + attach.
    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;
    let retriever: Arc<dyn Retriever> = Arc::new(EmbedAttachingRetriever {
        inner: Arc::new(bm25),
        by_id,
    });

    let lexical: Arc<dyn DiagnosticsEngine> = Arc::new(DefaultDiagnosticsEngine::new());
    let semantic: Arc<dyn DiagnosticsEngine> = Arc::new(SemanticDiagnosticsEngine::new());
    let diagnostics: Arc<dyn DiagnosticsEngine> = Arc::new(
        LayeredDiagnosticsEngine::lexical_and_semantic(lexical, semantic),
    );
    let classifier: Arc<dyn RegimeClassifier> = Arc::new(RuleBasedClassifier::new());
    let rerankers: Vec<(RerankerLevel, Arc<dyn Reranker>)> = vec![(
        RerankerLevel::Lexical,
        Arc::new(LexicalGroundingReranker::default()),
    )];

    let sweep = ThresholdSweep {
        min_p_distractor_grid: DISTRACTOR_GRID.to_vec(),
        min_p_ambiguous_grid: AMBIGUOUS_GRID.to_vec(),
        top_k: TOP_K,
        static_thresholds: Default::default(),
    };
    let report = sweep
        .run(&corpus, retriever, diagnostics, classifier, rerankers)
        .await?;
    Ok(summarize(label, &report))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Adaptive controller: BGE vs hashing semantic sensing            ║");
    println!("║  (BM25 retrieval held constant — isolates diagnostic sharpness)  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let hotpot_path = redhop_examples::data_path("hotpotqa/hotpot_dev_distractor_v1.json");
    let mut dataset = HotpotQADataset::from_path(&hotpot_path)?;
    dataset.examples.truncate(SAMPLE_SIZE);
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;
    println!(
        "HotpotQA sample: {} items, top_k={}\n",
        dataset.examples.len(),
        TOP_K
    );

    println!("arm A: hashing embeddings (CI baseline substrate)...");
    let hashing: Arc<dyn EmbeddingProvider> = Arc::new(HashingProvider::with_dim(384));
    let a = run_arm("hashing", hashing, &dataset, &chunker).await?;

    println!("arm B: BGE-small ONNX (real semantic substrate)...");
    let (bge_model, bge_tokenizer) = redhop_examples::bge_small_paths();
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &bge_model,
        &bge_tokenizer,
        EmbedderConfig::bge(384),
    )?);
    let b = run_arm("BGE-small", bge, &dataset, &chunker).await?;

    // ── Side-by-side ──
    println!("\n──── adaptive-controller metrics by semantic substrate ────");
    let row = |name: &str, x: f32, y: f32, fmt: &dyn Fn(f32) -> String| {
        println!("  {:<26} {:>12} {:>12}", name, fmt(x), fmt(y));
    };
    let f3 = |v: f32| format!("{v:.3}");
    let pct = |v: f32| format!("{:.0}%", v * 100.0);
    println!("  {:<26} {:>12} {:>12}", "metric", a.label, b.label);
    println!("  {}", "─".repeat(52));
    row(
        "intervention rate",
        a.intervention_rate,
        b.intervention_rate,
        &pct,
    );
    row("useful %", a.fraction_useful, b.fraction_useful, &pct);
    row(
        "mean recall lift",
        a.mean_recall_lift,
        b.mean_recall_lift,
        &f3,
    );
    row(
        "mean rerank calls/q",
        a.mean_rerank_calls,
        b.mean_rerank_calls,
        &f3,
    );
    row(
        "rerank compute avoided",
        a.rerank_compute_reduction,
        b.rerank_compute_reduction,
        &pct,
    );
    row(
        "mean useful lift",
        a.mean_useful_lift,
        b.mean_useful_lift,
        &f3,
    );
    row(
        "mean harmful lift",
        a.mean_harmful_lift,
        b.mean_harmful_lift,
        &f3,
    );
    println!(
        "  {:<26} {:>12} {:>12}",
        "wasted interventions", a.wasted, b.wasted
    );
    row(
        "classifier accuracy",
        a.classifier_accuracy,
        b.classifier_accuracy,
        &pct,
    );
    row("ECE (calibration)", a.ece, b.ece, &f3);

    // ── Headline ──
    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("CENTRAL QUESTION: do better embeddings reduce the need for reranking?");
    let d_interv = b.intervention_rate - a.intervention_rate;
    let d_useful = b.fraction_useful - a.fraction_useful;
    let d_rerank = b.mean_rerank_calls - a.mean_rerank_calls;
    let d_ece = b.ece - a.ece;
    println!(
        "  Δ intervention rate (BGE − hashing):  {:+.1} pts",
        d_interv * 100.0
    );
    println!(
        "  Δ useful %        (BGE − hashing):    {:+.1} pts",
        d_useful * 100.0
    );
    println!("  Δ rerank calls/q  (BGE − hashing):    {:+.3}", d_rerank);
    println!("  Δ ECE             (BGE − hashing):    {:+.3}", d_ece);
    println!();
    if d_rerank < -0.005 || (d_interv < -0.02 && d_useful >= -0.02) {
        println!("  → BGE LETS THE CONTROLLER DO LESS: stronger semantic sensing");
        println!("    reduces escalation while holding (or improving) precision.");
    } else if d_useful > 0.02 {
        println!("  → BGE SHARPENS PRECISION: similar escalation, more of it useful.");
    } else {
        println!("  → No clear reduction on this sample; report the raw deltas above.");
    }
    println!("════════════════════════════════════════════════════════════════════════");
    Ok(())
}
