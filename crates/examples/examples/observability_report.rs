//! Observability + economics demonstration (Phases D + E).
//!
//! Runs the Rust adaptive controller over a HotpotQA sample and produces:
//!
//!   1. Per-query CLI traces (a few) — the live, per-query observability
//!      a production deployment emits.
//!   2. A JSONL trace stream for all queries (machine-readable).
//!   3. A self-contained HTML "moat report" written to disk — the
//!      aggregate artifact a buyer opens to see WHERE compute went and
//!      WHY.
//!   4. A cost-economics summary printed to the terminal, including the
//!      selective-escalation ROI vs uniform reranking.
//!
//! Run with:
//!     cargo run -p redhop-examples --example observability_report --release

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    Chunk, ChunkId, Chunker, DiagnosticsEngine, Embedding, Query, RegimeClassifier, Reranker,
    RerankerLevel, Result as CoreResult, RetrievalResult, Retriever, TokenizerBackend,
};
use redhop::reranking::LexicalGroundingReranker;
use redhop::retrieval::Bm25Retriever;
use redhop_calibration::{
    economics::{economics, selective_escalation_roi, CostModel},
    embedder::HashingEmbedder,
    htmlreport::{render_html, ReportOptions},
    loaders::hotpotqa::{default_regime, HotpotQADataset},
    runner::{run_query, RunnerConfig},
};
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use redhop_observability::{render::cli, render::json, RetrievalTrace};
use redhop_orchestration::RuleBasedClassifier;
use redhop_orchestration::{
    AdaptiveOrchestrator, ConservativeRulePolicy, DefaultActuator, Policy, PolicyThresholds,
};
const SAMPLE_SIZE: usize = 150;
const TOP_K: usize = 4;
// Uniform-rerank lift baseline measured by method_pair_regret on this corpus.
const UNIFORM_RERANK_LIFT: f32 = 0.046;

struct EmbedAttachingRetriever {
    inner: Arc<dyn Retriever>,
    by_id: HashMap<ChunkId, Embedding>,
}
impl EmbedAttachingRetriever {
    fn new(inner: Arc<dyn Retriever>, chunks: &[Chunk]) -> Self {
        let by_id = chunks
            .iter()
            .filter_map(|c| c.embedding.clone().map(|e| (c.id.clone(), e)))
            .collect();
        Self { inner, by_id }
    }
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

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    // ── Setup ─────────────────────────────────────────────────────
    let mut dataset = HotpotQADataset::from_path(redhop_examples::data_path(
        "hotpotqa/hotpot_dev_distractor_v1.json",
    ))?;
    dataset.examples.truncate(SAMPLE_SIZE);
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 60, 90, 0)?;
    let embedder = HashingEmbedder::with_dim(256);
    let corpus =
        dataset.to_labeled_corpus(&chunker, |q| Some(embedder.embed(q)), default_regime)?;

    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    for c in &mut chunks {
        c.embedding = Some(embedder.embed(&c.text));
    }
    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;
    let retriever: Arc<dyn Retriever> =
        Arc::new(EmbedAttachingRetriever::new(Arc::new(bm25), &chunks));

    let lexical: Arc<dyn DiagnosticsEngine> = Arc::new(DefaultDiagnosticsEngine::new());
    let semantic: Arc<dyn DiagnosticsEngine> = Arc::new(SemanticDiagnosticsEngine::new());
    let diagnostics: Arc<dyn DiagnosticsEngine> = Arc::new(
        LayeredDiagnosticsEngine::lexical_and_semantic(lexical, semantic),
    );
    let classifier: Arc<dyn RegimeClassifier> = Arc::new(RuleBasedClassifier::new());
    let lex_reranker: Arc<dyn Reranker> = Arc::new(LexicalGroundingReranker::default());
    let rerankers = vec![(RerankerLevel::Lexical, lex_reranker.clone())];

    // Best policy setting from the real-workload findings: ambiguous=0.30.
    let policy: Arc<dyn Policy> =
        Arc::new(ConservativeRulePolicy::with_thresholds(PolicyThresholds {
            min_p_ambiguous: 0.30,
            ..Default::default()
        }));

    println!(
        "running adaptive controller over {} HotpotQA queries...",
        corpus.queries.len()
    );

    // ── 1+2. Per-query traces via the orchestrator ────────────────
    let actuator = Arc::new(DefaultActuator::new(retriever.clone(), rerankers.clone()));
    let orchestrator = AdaptiveOrchestrator::new(
        diagnostics.clone(),
        classifier.clone(),
        policy.clone(),
        actuator,
    )
    .with_initial_top_k(TOP_K);

    let mut traces: Vec<RetrievalTrace> = Vec::with_capacity(corpus.queries.len());
    for lq in &corpus.queries {
        let mut q = Query::new(&lq.text);
        q.embedding = lq.embedding.clone();
        let state = orchestrator.run(q).await?;
        traces.push(RetrievalTrace::from_state(&state));
    }

    // Show the first intervened trace and the first abstain/no-op trace.
    println!("\n──────────── sample per-query traces ────────────\n");
    if let Some(t) = traces.iter().find(|t| t.intervened) {
        println!("AN INTERVENED QUERY:\n{}", cli::render(t));
    }
    if let Some(t) = traces.iter().find(|t| !t.intervened) {
        println!(
            "A NO-OP QUERY (controller chose to do nothing):\n{}",
            cli::render(t)
        );
    }

    // Write all traces as JSONL.
    let trace_out = redhop_examples::exports_path("redhop_traces.jsonl");
    std::fs::create_dir_all(trace_out.parent().unwrap()).ok();
    std::fs::write(&trace_out, json::render_jsonl(&traces))?;
    println!(
        "wrote {} per-query traces → {}",
        traces.len(),
        trace_out.display()
    );

    // ── 3+4. Aggregate outcomes for the HTML report + economics ───
    let cfg = RunnerConfig {
        retriever,
        diagnostics,
        classifier,
        policy,
        rerankers,
        top_k: TOP_K,
    };
    let mut outcomes = Vec::with_capacity(corpus.queries.len());
    for lq in &corpus.queries {
        outcomes.push(run_query(lq, &cfg).await?);
    }

    let cost = CostModel::default();
    let econ = economics(&outcomes, &cost);
    let roi = selective_escalation_roi(&econ, UNIFORM_RERANK_LIFT, &cost);

    println!("\n──────────── cost economics ────────────");
    println!("mean recall lift:          {:+.3}", econ.mean_recall_lift);
    println!("intervention (rerank/q):   {:.2}", econ.mean_rerank_calls);
    println!(
        "rerank compute avoided:    {:.0}%  (vs uniform rerank-everything)",
        econ.rerank_compute_reduction * 100.0
    );
    println!(
        "adaptive cost / query:     {:.2}  (uniform would be {:.2})",
        econ.mean_adaptive_cost, econ.uniform_cost
    );
    println!(
        "cost fraction vs uniform:  {:.0}%",
        econ.cost_fraction_vs_uniform * 100.0
    );
    if let Some(cpl) = econ.cost_per_unit_lift {
        println!("cost per unit recall-lift: {:.1}", cpl);
    }
    if let Some(roi) = roi {
        println!(
            "selective-escalation ROI:  {:.1}× uniform reranking efficiency",
            roi
        );
    }

    // ── HTML report ───────────────────────────────────────────────
    let opts = ReportOptions {
        title: "RedHop — HotpotQA Adaptive Retrieval Report".into(),
        workload: format!(
            "HotpotQA dev (distractor), first {} items",
            corpus.queries.len()
        ),
        cost,
        uniform_rerank_lift: Some(UNIFORM_RERANK_LIFT),
    };
    let html = render_html(&outcomes, &opts);
    let html_out = redhop_examples::exports_path("redhop_report.html");
    std::fs::create_dir_all(html_out.parent().unwrap()).ok();
    std::fs::write(&html_out, &html)?;
    println!(
        "\nwrote self-contained HTML moat report → {}",
        html_out.display()
    );
    println!("  ({} bytes, open it in any browser)", html.len());

    Ok(())
}
