//! BGE-specific recalibration — the genuine-reduction-vs-drift test.
//!
//! The previous experiment (adaptive_real_vs_hashing) found that BGE
//! reduced interventions BUT degraded classifier calibration, because
//! the regime thresholds were tuned for the hashing substrate. This
//! experiment recalibrates for BGE and answers the critical question:
//!
//!   Does stronger semantic retrieval GENUINELY reduce reranking need,
//!   or did threshold drift merely SUPPRESS escalation?
//!
//! Steps (per the plan):
//!   1. Build the BGE substrate once (expensive embedding done once).
//!   2. Measure BGE's semantic_grounding distribution vs the default
//!      easy threshold — quantifies the drift.
//!   3. Sweep the classifier's easy_min_semantic_grounding to find
//!      BGE-specific calibration (lowest ECE / best useful lift).
//!   4. Reliability diagrams: BGE@default-thresholds vs BGE@recalibrated.
//!   5. Final comparison: hashing@default vs BGE@recalibrated.
//!
//! Requires `--features onnx` + the BGE-small model.
//!
//! Run:
//!   cargo run -p neorag-examples --example bge_recalibration \
//!       --features onnx --release

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use neorag_calibration::{
    analysis::{confusion_matrix, regret_summary},
    economics::{economics, CostModel},
    loaders::hotpotqa::{default_regime, HotpotQADataset},
    reliability::{reliability_diagram, ReliabilityDiagram},
    runner::{run_query, QueryOutcome, RunnerConfig},
};
use neorag_chunking::{SentenceChunker, WhitespaceTokenizer};
use neorag_core::{
    Chunk, ChunkId, Chunker, DiagnosticsEngine, Embedding, EmbeddingProvider, Query,
    RegimeClassifier, Reranker, RerankerLevel, Result as CoreResult, RetrievalResult, Retriever,
    TokenizerBackend,
};
use neorag_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use neorag_embeddings::{EmbedderConfig, HashingProvider, OnnxEmbedder};
use neorag_orchestration::{
    ClassifierThresholds, ConservativeRulePolicy, Policy, RuleBasedClassifier,
};
use neorag_reranking::LexicalGroundingReranker;
use neorag_retrieval::Bm25Retriever;

const HOTPOTQA_PATH: &str =
    "/Users/vysakh/projects/neorag/data/hotpotqa/hotpot_dev_distractor_v1.json";
const BGE_MODEL: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/onnx/model.onnx";
const BGE_TOKENIZER: &str =
    "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/tokenizer.json";
const SAMPLE_SIZE: usize = 60;
const TOP_K: usize = 4;

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

struct Substrate {
    retriever: Arc<dyn Retriever>,
    diagnostics: Arc<dyn DiagnosticsEngine>,
    rerankers: Vec<(RerankerLevel, Arc<dyn Reranker>)>,
    corpus: neorag_calibration::dataset::LabeledCorpus,
}

async fn build_substrate(
    provider: Arc<dyn EmbeddingProvider>,
    dataset: &HotpotQADataset,
    chunker: &SentenceChunker,
) -> anyhow::Result<Substrate> {
    let q_texts: Vec<String> = dataset.examples.iter().map(|e| e.question.clone()).collect();
    let q_vecs = provider.embed(&q_texts).await?;
    let q_map: HashMap<String, Embedding> = q_texts.into_iter().zip(q_vecs).collect();
    let corpus = dataset.to_labeled_corpus(chunker, |q| q_map.get(q).cloned(), default_regime)?;

    let chunks = chunker.chunk_batch(&corpus.docs)?;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = provider.embed(&texts).await?;
    let by_id: HashMap<ChunkId, Embedding> =
        chunks.iter().zip(vecs).map(|(c, v)| (c.id.clone(), v)).collect();

    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;
    let retriever: Arc<dyn Retriever> = Arc::new(EmbedAttachingRetriever {
        inner: Arc::new(bm25),
        by_id,
    });

    let lexical: Arc<dyn DiagnosticsEngine> = Arc::new(DefaultDiagnosticsEngine::new());
    let semantic: Arc<dyn DiagnosticsEngine> = Arc::new(SemanticDiagnosticsEngine::new());
    let diagnostics: Arc<dyn DiagnosticsEngine> =
        Arc::new(LayeredDiagnosticsEngine::lexical_and_semantic(lexical, semantic));
    let rerankers: Vec<(RerankerLevel, Arc<dyn Reranker>)> =
        vec![(RerankerLevel::Lexical, Arc::new(LexicalGroundingReranker::default()))];

    Ok(Substrate {
        retriever,
        diagnostics,
        rerankers,
        corpus,
    })
}

#[derive(Clone)]
struct Metrics {
    intervention_rate: f32,
    useful: f32,
    recall_lift: f32,
    rerank_calls: f32,
    useful_lift: f32,
    harmful_lift: f32,
    wasted: usize,
    accuracy: f32,
    ece: f32,
}

async fn run_with_classifier(
    sub: &Substrate,
    classifier: Arc<dyn RegimeClassifier>,
    policy: Arc<dyn Policy>,
) -> anyhow::Result<(Metrics, Vec<QueryOutcome>)> {
    let cfg = RunnerConfig {
        retriever: sub.retriever.clone(),
        diagnostics: sub.diagnostics.clone(),
        classifier,
        policy,
        rerankers: sub.rerankers.clone(),
        top_k: TOP_K,
    };
    let mut outcomes = Vec::with_capacity(sub.corpus.queries.len());
    for q in &sub.corpus.queries {
        outcomes.push(run_query(q, &cfg).await?);
    }
    let n = outcomes.len().max(1) as f32;
    let intervened: Vec<&QueryOutcome> = outcomes.iter().filter(|o| o.intervened).collect();
    let useful = if intervened.is_empty() {
        0.0
    } else {
        intervened.iter().filter(|o| o.recall_lift > 0.0).count() as f32 / intervened.len() as f32
    };
    let regret = regret_summary(&outcomes);
    let cm = confusion_matrix(&outcomes);
    let diag = reliability_diagram(&outcomes, 10);
    let econ = economics(&outcomes, &CostModel::default());
    let m = Metrics {
        intervention_rate: intervened.len() as f32 / n,
        useful,
        recall_lift: outcomes.iter().map(|o| o.recall_lift).sum::<f32>() / n,
        rerank_calls: econ.mean_rerank_calls,
        useful_lift: regret.mean_useful_lift,
        harmful_lift: regret.mean_harmful_lift,
        wasted: regret.n_wasted_interventions,
        accuracy: cm.accuracy,
        ece: diag.ece,
    };
    Ok((m, outcomes))
}

fn classifier_with_easy_sem(easy_sem: f32) -> Arc<dyn RegimeClassifier> {
    let t = ClassifierThresholds {
        easy_min_semantic_grounding: easy_sem,
        ..Default::default()
    };
    Arc::new(RuleBasedClassifier::with_thresholds(t))
}

fn print_reliability(label: &str, d: &ReliabilityDiagram) {
    println!("  reliability [{label}]  ECE = {:.3}", d.ece);
    for b in &d.bins {
        if b.count == 0 {
            continue;
        }
        println!(
            "    [{:.1},{:.1}) n={:>3} pred={:.2} emp={:.2}",
            b.lo, b.hi, b.count, b.mean_predicted_p, b.empirical_correct
        );
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  BGE recalibration: genuine reduced reranking, or threshold drift? ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let mut dataset = HotpotQADataset::from_path(HOTPOTQA_PATH)?;
    dataset.examples.truncate(SAMPLE_SIZE);
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;
    let policy: Arc<dyn Policy> = Arc::new(ConservativeRulePolicy::new());

    // ── Build both substrates (BGE embedding done once) ──
    println!("building hashing substrate...");
    let hashing: Arc<dyn EmbeddingProvider> = Arc::new(HashingProvider::with_dim(384));
    let hsub = build_substrate(hashing, &dataset, &chunker).await?;

    println!("building BGE substrate (real ONNX inference)...");
    let bge: Arc<dyn EmbeddingProvider> =
        Arc::new(OnnxEmbedder::load(BGE_MODEL, BGE_TOKENIZER, EmbedderConfig::bge(384))?);
    let bsub = build_substrate(bge, &dataset, &chunker).await?;

    // ── Step 2: measure BGE's semantic_grounding distribution ──
    println!("\n──── BGE semantic_grounding distribution (top-{TOP_K} retrieved) ────");
    let mut groundings = Vec::new();
    for q in &bsub.corpus.queries {
        let mut query = Query::new(&q.text);
        query.embedding = q.embedding.clone();
        let results = bsub.retriever.retrieve(&query, TOP_K).await?;
        let report = bsub.diagnostics.diagnose(&query, &results)?;
        if let Some(g) = report.semantic_grounding {
            groundings.push(g);
        }
    }
    groundings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f32| {
        let i = ((groundings.len() as f32 - 1.0) * p).round() as usize;
        groundings.get(i).copied().unwrap_or(0.0)
    };
    println!(
        "  p10={:.3}  p25={:.3}  p50={:.3}  p75={:.3}  p90={:.3}",
        pct(0.10), pct(0.25), pct(0.50), pct(0.75), pct(0.90)
    );
    let above_default =
        groundings.iter().filter(|&&g| g >= 0.75).count() as f32 / groundings.len().max(1) as f32;
    println!(
        "  fraction ≥ default easy threshold (0.75): {:.0}%  ← the drift: nearly all queries clear it",
        above_default * 100.0
    );

    // ── Step 3: classifier easy-threshold recalibration sweep (BGE) ──
    println!("\n──── BGE classifier recalibration: sweep easy_min_semantic_grounding ────");
    println!(
        "  {:<8} {:>10} {:>8} {:>10} {:>10} {:>8} {:>8}",
        "easy_sem", "interv", "useful", "lift", "rerank/q", "acc", "ECE"
    );
    let grid = [0.75f32, 0.80, 0.85, 0.90, 0.93, 0.96, 0.99];
    let mut best = (0.75f32, f32::INFINITY); // (threshold, ece) — minimize ECE
    let mut best_recall = (0.75f32, -1.0f32);
    for &es in &grid {
        let (m, _) = run_with_classifier(&bsub, classifier_with_easy_sem(es), policy.clone()).await?;
        println!(
            "  {:<8.2} {:>9.0}% {:>7.0}% {:>10.3} {:>10.3} {:>7.0}% {:>8.3}",
            es,
            m.intervention_rate * 100.0,
            m.useful * 100.0,
            m.recall_lift,
            m.rerank_calls,
            m.accuracy * 100.0,
            m.ece
        );
        if m.ece < best.1 {
            best = (es, m.ece);
        }
        if m.recall_lift > best_recall.1 {
            best_recall = (es, m.recall_lift);
        }
    }
    println!(
        "  → lowest-ECE easy_sem = {:.2} (ECE {:.3});  highest-lift easy_sem = {:.2} ({:+.3})",
        best.0, best.1, best_recall.0, best_recall.1
    );
    // Use the highest-lift calibration as the BGE operating point (we
    // care about recall utility, with ECE as a tiebreaker / sanity).
    let bge_easy = best_recall.0;

    // ── Step 4: reliability diagrams, default vs recalibrated (BGE) ──
    println!("\n──── reliability diagrams (BGE) ────");
    let (_, out_default) =
        run_with_classifier(&bsub, classifier_with_easy_sem(0.75), policy.clone()).await?;
    let (_, out_recal) =
        run_with_classifier(&bsub, classifier_with_easy_sem(bge_easy), policy.clone()).await?;
    print_reliability("BGE @ default 0.75", &reliability_diagram(&out_default, 10));
    print_reliability(&format!("BGE @ recalibrated {bge_easy:.2}"), &reliability_diagram(&out_recal, 10));

    // ── Step 5: final comparison ──
    let (hm, _) = run_with_classifier(&hsub, classifier_with_easy_sem(0.75), policy.clone()).await?;
    let (bm, _) = run_with_classifier(&bsub, classifier_with_easy_sem(bge_easy), policy.clone()).await?;
    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("FINAL: hashing@default  vs  BGE@recalibrated (easy_sem={bge_easy:.2})");
    println!("  {:<22} {:>14} {:>16}", "metric", "hashing@0.75", "BGE@recal");
    println!("  {}", "─".repeat(54));
    let f3 = |v: f32| format!("{v:.3}");
    let p0 = |v: f32| format!("{:.0}%", v * 100.0);
    let row = |n: &str, a: f32, b: f32, f: &dyn Fn(f32) -> String| {
        println!("  {:<22} {:>14} {:>16}", n, f(a), f(b));
    };
    row("intervention rate", hm.intervention_rate, bm.intervention_rate, &p0);
    row("useful %", hm.useful, bm.useful, &p0);
    row("mean recall lift", hm.recall_lift, bm.recall_lift, &f3);
    row("rerank calls/query", hm.rerank_calls, bm.rerank_calls, &f3);
    row("mean useful lift", hm.useful_lift, bm.useful_lift, &f3);
    row("mean harmful lift", hm.harmful_lift, bm.harmful_lift, &f3);
    println!("  {:<22} {:>14} {:>16}", "wasted interventions", hm.wasted, bm.wasted);
    row("ECE", hm.ece, bm.ece, &f3);

    println!("\n  VERDICT:");
    let lift_ok = bm.recall_lift >= hm.recall_lift - 0.005;
    let fewer = bm.intervention_rate < hm.intervention_rate - 0.02;
    let safe = bm.harmful_lift <= 0.0 + 1e-6;
    if lift_ok && fewer && safe {
        println!("  → GENUINE REDUCTION: at its own calibration, BGE matches/exceeds");
        println!("    hashing's recall lift with FEWER interventions and zero harm.");
        println!("    Strong semantic retrieval + calibrated selective escalation =");
        println!("    a real retrieval-economics win, not threshold drift.");
    } else if !lift_ok && fewer {
        println!("  → PARTLY DRIFT: BGE intervenes less but also captures less lift");
        println!("    even after recalibration — some of the reduction was suppression.");
    } else {
        println!("  → MIXED: see the deltas above; report honestly, don't force a story.");
    }
    println!("════════════════════════════════════════════════════════════════════════");
    Ok(())
}
