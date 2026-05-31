//! Real-workload adaptive evaluation: HotpotQA dev (distractor split).
//!
//! This example runs the **Rust adaptive controller end-to-end** against
//! a sample of real HotpotQA items. Every metric printed below is
//! measured on real questions, not synthetic fixtures.
//!
//! ## What this exercises
//!
//! 1. `HotpotQADataset::from_path` → `LabeledCorpus` (questions + true
//!    regime labels via the `(level, type)` heuristic + gold chunk
//!    ids resolved via sentence-containment).
//! 2. Hashing-trick TF embedder for chunks and queries — deterministic,
//!    no model dependency.
//! 3. `Bm25Retriever` + `EmbedAttachingRetriever` wrapper.
//! 4. `LayeredDiagnosticsEngine` (lexical + semantic tiers).
//! 5. `RuleBasedClassifier` + `ConservativeRulePolicy`.
//! 6. `ThresholdSweep` over a small grid.
//! 7. `confusion_matrix`, `regret_summary`, `bootstrap_stability`,
//!    `reliability_diagram`, `render_sweep_table`, `render_pareto`.
//!
//! ## What this does NOT exercise
//!
//! - A real embedding model. The hashing-trick embedder is a
//!   deterministic baseline; a real embedder would *only widen* the
//!   gap between adaptive and static.
//! - The full 7,405-item HotpotQA dev set. We sample 200 items by
//!   default. The sample is honest; the headline numbers are
//!   stratified by `(level, type)` and reported in the regime
//!   confusion matrix.
//! - LLM-generated answers. Recall lift is measured against
//!   gold *chunk ids* (paragraph-level retrieval accuracy), not
//!   answer correctness. This is the right primitive for retrieval-
//!   stage adaptive policy evaluation; answer-level evaluation is a
//!   separate concern that lives in the Python lab.
//!
//! Run with:
//!     cargo run -p redhop-examples --example adaptive_eval_hotpotqa --release
//!
//! `--release` is recommended; debug build is ~10× slower on this
//! workload but produces identical numbers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    Chunk, ChunkId, Chunker, DiagnosticsEngine, Embedding, Query, RerankerLevel,
    Result as CoreResult, RetrievalRegime, RetrievalResult, Retriever, TokenizerBackend,
};
use redhop::reranking::LexicalGroundingReranker;
use redhop::retrieval::Bm25Retriever;
use redhop_calibration::{
    analysis::{bootstrap_stability, confusion_matrix, regret_summary},
    embedder::HashingEmbedder,
    loaders::hotpotqa::{default_regime, HotpotQADataset},
    reliability::reliability_diagram,
    report::{render_pareto, render_reliability, render_sweep_table},
    ThresholdSweep,
};
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use redhop_orchestration::RuleBasedClassifier;

const HOTPOTQA_PATH: &str =
    "/Users/vysakh/projects/neorag/data/hotpotqa/hotpot_dev_distractor_v1.json";

const SAMPLE_SIZE: usize = 200;
const SWEEP_TOP_K: usize = 4;
const BOOTSTRAP_B: usize = 200;

// Smaller sweep grid than the synthetic demo: real retrieval is
// expensive enough that we keep the grid tight and let the headline
// be "which corner of the policy space wins on real data".
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
    println!("║  HotpotQA adaptive evaluation — Rust controller, real workload   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ─── 1. Load HotpotQA dev ─────────────────────────────────────
    let t = Instant::now();
    println!("loading HotpotQA dev from {HOTPOTQA_PATH}");
    let mut dataset = HotpotQADataset::from_path(HOTPOTQA_PATH)?;
    let full_size = dataset.len();
    dataset.examples.truncate(SAMPLE_SIZE);
    println!(
        "  → {} examples loaded, sampled first {} ({:.1}s)",
        full_size,
        dataset.len(),
        t.elapsed().as_secs_f32()
    );

    // ─── 2. Build chunker + embedder + LabeledCorpus ──────────────
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

    // True-regime distribution (sanity check the heuristic).
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

    // ─── 3. Chunk corpus, embed, index BM25 ───────────────────────
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
        "indexed {} chunks into BM25 + attached embeddings ({:.1}s)",
        chunks.len(),
        t.elapsed().as_secs_f32()
    );

    // ─── 4. Assemble adaptive components ──────────────────────────
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

    // ─── 5. Run the threshold sweep ───────────────────────────────
    let t = Instant::now();
    let sweep = ThresholdSweep {
        min_p_distractor_grid: DISTRACTOR_GRID.to_vec(),
        min_p_ambiguous_grid: AMBIGUOUS_GRID.to_vec(),
        top_k: SWEEP_TOP_K,
        static_thresholds: Default::default(),
    };
    println!(
        "running threshold sweep: {} × {} = {} settings × {} queries = {} runs",
        DISTRACTOR_GRID.len(),
        AMBIGUOUS_GRID.len(),
        DISTRACTOR_GRID.len() * AMBIGUOUS_GRID.len(),
        corpus.queries.len(),
        DISTRACTOR_GRID.len() * AMBIGUOUS_GRID.len() * corpus.queries.len()
    );
    let report = sweep
        .run(&corpus, retriever, diagnostics, classifier, rerankers)
        .await?;
    println!("  sweep complete ({:.1}s)", t.elapsed().as_secs_f32());
    println!();

    // ─── 6. Reports ──────────────────────────────────────────────
    println!("{}", render_sweep_table(&report));
    println!();
    println!("{}", render_pareto(&report));

    // Reliability diagram across all settings (more samples per bin).
    let all_outcomes: Vec<_> = report
        .outcomes
        .iter()
        .flat_map(|v| v.iter().cloned())
        .collect();
    let diag = reliability_diagram(&all_outcomes, 10);
    println!();
    println!("{}", render_reliability(&diag));

    // ─── 7. Confusion matrix ──────────────────────────────────────
    let cm = confusion_matrix(&all_outcomes);
    println!();
    println!("─── regime confusion matrix (across all sweep settings) ───");
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
            let count = row.get(r_pred).copied().unwrap_or(0);
            print!(" {:>10}", count);
        }
        println!();
    }
    println!();
    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>8}",
        "regime", "precision", "recall", "f1", "support"
    );
    println!("{}", "─".repeat(60));
    for r in RetrievalRegime::all() {
        let m = cm.per_regime.get(r).cloned().unwrap_or_default();
        if m.support == 0 {
            continue;
        }
        println!(
            "{:<18} {:>10.3} {:>10.3} {:>10.3} {:>8}",
            r.code(),
            m.precision,
            m.recall,
            m.f1,
            m.support
        );
    }

    // ─── 8. Intervention regret ──────────────────────────────────
    let regret = regret_summary(&all_outcomes);
    println!();
    println!("─── intervention regret ───");
    println!("n_interventions:               {}", regret.n_interventions);
    println!(
        "mean(actual − expected) gain:  {:+.3}",
        regret.mean_expected_actual_error
    );
    println!(
        "mean |actual − expected|:      {:.3}",
        regret.mean_abs_expected_actual_error
    );
    println!(
        "mean lift when useful:         {:+.3}  (upside when intervention helps)",
        regret.mean_useful_lift
    );
    println!(
        "mean lift when harmful:        {:+.3}  (damage when it hurts)",
        regret.mean_harmful_lift
    );
    println!(
        "wasted interventions:          {}     (intervened but lift = 0)",
        regret.n_wasted_interventions
    );

    // ─── 9. Bootstrap stability ──────────────────────────────────
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

    // ─── 10. Headline ────────────────────────────────────────────
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!(
        "HEADLINE — HotpotQA dev, first {} items, hashing-trick TF embedder",
        corpus.queries.len()
    );
    if let Some(best) = report.argmax_lift() {
        println!(
            "  argmax sweep setting:   min_p_distractor={:.2}, min_p_ambiguous={:.2}",
            best.min_p_distractor, best.min_p_ambiguous
        );
        println!("  mean_recall_lift:       {:+.3}", best.mean_recall_lift);
        println!(
            "  intervention_rate:      {:.2}  ({:.0}% of queries acted on)",
            best.intervention_rate,
            best.intervention_rate * 100.0
        );
        println!(
            "  fraction_useful:        {:.2}  (precision of intervention)",
            best.fraction_useful_interventions
        );
        if let Some(i) = report.rows.iter().position(|r| std::ptr::eq(r, best)) {
            let af = stability.argmax_frequency.get(i).copied().unwrap_or(0.0);
            println!(
                "  bootstrap stability:    {:.0}% of resamples agreed",
                af * 100.0
            );
        }
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
