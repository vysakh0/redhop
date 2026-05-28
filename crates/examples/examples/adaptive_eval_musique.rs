//! Real-workload adaptive evaluation: MuSiQue dev set.
//!
//! MuSiQue is multi-hop (2-, 3-, 4-hop questions) with 20 paragraphs
//! per item, including a mix of supporting and distractor paragraphs.
//! Expected regime profile leans toward `Ambiguous` and
//! `DistractorHeavy` more than HotpotQA does.
//!
//! Same pipeline as `adaptive_eval_hotpotqa.rs`. See that file for
//! the methodological caveats.
//!
//! Run with:
//!     cargo run -p redhop-examples --example adaptive_eval_musique --release

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use redhop_calibration::{
    analysis::{bootstrap_stability, confusion_matrix, regret_summary},
    embedder::HashingEmbedder,
    loaders::musique::{default_regime, MuSiQueDataset},
    reliability::reliability_diagram,
    report::{render_pareto, render_reliability, render_sweep_table},
    ThresholdSweep,
};
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    Chunk, ChunkId, Chunker, DiagnosticsEngine, Embedding, Query, RerankerLevel,
    Result as CoreResult, RetrievalRegime, RetrievalResult, Retriever, TokenizerBackend,
};
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use redhop_orchestration::RuleBasedClassifier;
use redhop::reranking::LexicalGroundingReranker;
use redhop::retrieval::Bm25Retriever;

const MUSIQUE_PATH: &str = "/Users/vysakh/projects/neorag/data/musique/dev.jsonl";
const SAMPLE_SIZE: usize = 200;
const SWEEP_TOP_K: usize = 6;
const BOOTSTRAP_B: usize = 200;
const DISTRACTOR_GRID: [f32; 3] = [0.30, 0.40, 0.50];
const AMBIGUOUS_GRID: [f32; 3] = [0.30, 0.40, 0.50];

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
    let t_total = Instant::now();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  MuSiQue adaptive evaluation — Rust controller, real workload    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let t = Instant::now();
    println!("loading MuSiQue dev from {MUSIQUE_PATH}");
    let mut dataset = MuSiQueDataset::from_path(MUSIQUE_PATH)?;
    let full_size = dataset.len();
    dataset.examples.truncate(SAMPLE_SIZE);
    println!(
        "  → {} examples loaded, sampled first {} ({:.1}s)",
        full_size,
        dataset.len(),
        t.elapsed().as_secs_f32()
    );

    let t = Instant::now();
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 60, 90, 0)?;
    let embedder = HashingEmbedder::with_dim(256);
    let corpus =
        dataset.to_labeled_corpus(&chunker, |q| Some(embedder.embed(q)), default_regime)?;
    println!(
        "built LabeledCorpus: {} docs, {} queries ({:.1}s)",
        corpus.docs.len(),
        corpus.queries.len(),
        t.elapsed().as_secs_f32()
    );

    let mut regime_counts: HashMap<&'static str, usize> = HashMap::new();
    for q in &corpus.queries {
        *regime_counts.entry(q.true_regime.code()).or_insert(0) += 1;
    }
    println!("  true-regime distribution:");
    for r in RetrievalRegime::all() {
        let c = regime_counts.get(r.code()).copied().unwrap_or(0);
        if c > 0 {
            println!("    {:<18} {}", r.code(), c);
        }
    }
    println!();

    let t = Instant::now();
    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    for c in &mut chunks {
        c.embedding = Some(embedder.embed(&c.text));
    }
    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;
    let retriever: Arc<dyn Retriever> =
        Arc::new(EmbedAttachingRetriever::new(Arc::new(bm25), &chunks));
    println!(
        "indexed {} chunks ({:.1}s)",
        chunks.len(),
        t.elapsed().as_secs_f32()
    );

    let lexical: Arc<dyn DiagnosticsEngine> = Arc::new(DefaultDiagnosticsEngine::new());
    let semantic: Arc<dyn DiagnosticsEngine> = Arc::new(SemanticDiagnosticsEngine::new());
    let diagnostics: Arc<dyn DiagnosticsEngine> = Arc::new(
        LayeredDiagnosticsEngine::lexical_and_semantic(lexical, semantic),
    );
    let classifier = Arc::new(RuleBasedClassifier::new());
    let rerankers = vec![(
        RerankerLevel::Lexical,
        Arc::new(LexicalGroundingReranker::default()) as _,
    )];

    let t = Instant::now();
    let sweep = ThresholdSweep {
        min_p_distractor_grid: DISTRACTOR_GRID.to_vec(),
        min_p_ambiguous_grid: AMBIGUOUS_GRID.to_vec(),
        top_k: SWEEP_TOP_K,
        static_thresholds: Default::default(),
    };
    println!(
        "running threshold sweep: {} settings × {} queries",
        DISTRACTOR_GRID.len() * AMBIGUOUS_GRID.len(),
        corpus.queries.len()
    );
    let report = sweep
        .run(&corpus, retriever, diagnostics, classifier, rerankers)
        .await?;
    println!("  sweep complete ({:.1}s)", t.elapsed().as_secs_f32());
    println!();

    println!("{}", render_sweep_table(&report));
    println!();
    println!("{}", render_pareto(&report));

    let all_outcomes: Vec<_> = report
        .outcomes
        .iter()
        .flat_map(|v| v.iter().cloned())
        .collect();
    let diag = reliability_diagram(&all_outcomes, 10);
    println!();
    println!("{}", render_reliability(&diag));

    let cm = confusion_matrix(&all_outcomes);
    println!();
    println!("─── regime confusion matrix ───");
    println!(
        "accuracy = {:.3}  n_predicted = {}  n_unpredicted = {}",
        cm.accuracy, cm.n_predicted, cm.n_unpredicted
    );
    print!("\n{:<18}", "true \\ pred");
    for r in RetrievalRegime::all() {
        print!(" {:>10}", r.code());
    }
    println!();
    for r_true in RetrievalRegime::all() {
        print!("{:<18}", r_true.code());
        let row = cm.matrix.get(r_true).cloned().unwrap_or_default();
        for r_pred in RetrievalRegime::all() {
            print!(" {:>10}", row.get(r_pred).copied().unwrap_or(0));
        }
        println!();
    }

    let regret = regret_summary(&all_outcomes);
    println!();
    println!("─── intervention regret ───");
    println!("n_interventions:               {}", regret.n_interventions);
    println!(
        "mean lift when useful:         {:+.3}",
        regret.mean_useful_lift
    );
    println!(
        "mean lift when harmful:        {:+.3}",
        regret.mean_harmful_lift
    );
    println!(
        "wasted interventions:          {}",
        regret.n_wasted_interventions
    );

    let stability = bootstrap_stability(&report, BOOTSTRAP_B, 0xC0FFEE);
    println!();
    println!(
        "─── threshold stability (B = {} bootstrap resamples) ───",
        stability.n_bootstrap
    );
    println!(
        "{:<14} {:<14} {:>10} {:>11} {:>12} {:>12}",
        "min_p_distr.", "min_p_amb.", "lift_stddev", "argmax_freq", "ci90_low", "ci90_high"
    );
    println!("{}", "─".repeat(80));
    for (i, row) in report.rows.iter().enumerate() {
        let (lo, hi) = stability.ci90.get(i).copied().unwrap_or((0.0, 0.0));
        let sd = stability.lift_stddev.get(i).copied().unwrap_or(0.0);
        let af = stability.argmax_frequency.get(i).copied().unwrap_or(0.0);
        println!(
            "{:<14} {:<14} {:>10.3} {:>11.3} {:>+12.3} {:>+12.3}",
            format!("{:.2}", row.min_p_distractor),
            format!("{:.2}", row.min_p_ambiguous),
            sd,
            af,
            lo,
            hi
        );
    }

    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!(
        "HEADLINE — MuSiQue dev, first {} items, hashing-trick TF embedder",
        corpus.queries.len()
    );
    if let Some(best) = report.argmax_lift() {
        println!(
            "  argmax sweep setting:   min_p_distractor={:.2}, min_p_ambiguous={:.2}",
            best.min_p_distractor, best.min_p_ambiguous
        );
        println!("  mean_recall_lift:       {:+.3}", best.mean_recall_lift);
        println!("  intervention_rate:      {:.2}", best.intervention_rate);
        println!(
            "  fraction_useful:        {:.2}",
            best.fraction_useful_interventions
        );
    }
    println!("  classifier accuracy:    {:.1}%", cm.accuracy * 100.0);
    println!("  ECE (calibration):      {:.3}", diag.ece);
    println!(
        "  total runtime:          {:.1}s",
        t_total.elapsed().as_secs_f32()
    );
    println!("════════════════════════════════════════════════════════════════════════");

    Ok(())
}
