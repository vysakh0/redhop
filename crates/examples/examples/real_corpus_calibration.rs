//! Real-corpus calibration demonstration.
//!
//! Demonstrates the full calibration pipeline against a HotpotQA-shaped
//! JSON fixture (embedded as a constant for hermeticity):
//!
//!   1. Load via `HotpotQADataset::from_json` and convert to
//!      LabeledCorpus.
//!   2. Run the threshold sweep.
//!   3. Compute regime confusion matrix, intervention regret, and
//!      bootstrap threshold stability.
//!   4. Print ALL of them side by side as ASCII reports.
//!
//! For real production calibration, replace the embedded JSON with
//! `HotpotQADataset::from_path("path/to/hotpot_dev_v1.json")`. The
//! analysis is identical; only the numbers change.
//!
//! Run with:
//!     cargo run -p redhop-examples --example real_corpus_calibration

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use redhop_calibration::{
    analysis::{bootstrap_stability, confusion_matrix, regret_summary},
    fixtures::embed,
    loaders::hotpotqa::{default_regime, HotpotQADataset},
    reliability::reliability_diagram,
    report::{render_pareto, render_reliability, render_sweep_table},
    ThresholdSweep,
};
use redhop_chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop_core::{
    Chunk, ChunkId, Chunker, DiagnosticsEngine, Embedding, Query, RerankerLevel,
    Result as CoreResult, RetrievalRegime, RetrievalResult, Retriever, TokenizerBackend,
};
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use redhop_orchestration::RuleBasedClassifier;
use redhop_reranking::LexicalGroundingReranker;
use redhop_retrieval::Bm25Retriever;

/// Mini HotpotQA-shaped fixture. Real users replace this with a path to
/// the actual HotpotQA dev set. The shape matches the canonical
/// distribution exactly.
const MINI_HOTPOTQA: &str = r#"[
{"_id": "e1", "question": "What runtime powers async Rust applications?",
 "answer": "Tokio", "type": "comparison", "level": "easy",
 "supporting_facts": [["Tokio", 0]],
 "context": [
   ["Tokio", ["Tokio is an asynchronous runtime for Rust applications.", "It uses a work-stealing scheduler."]],
   ["Django", ["Django is a Python web framework."]]
 ]},
{"_id": "e2", "question": "Which scheduler does Tokio use to distribute work?",
 "answer": "work-stealing", "type": "bridge", "level": "medium",
 "supporting_facts": [["Tokio", 1]],
 "context": [
   ["Tokio", ["Tokio is an asynchronous runtime for Rust applications.", "It uses a work-stealing scheduler."]],
   ["Postgres", ["Postgres provides ACID transactions and MVCC."]]
 ]},
{"_id": "e3", "question": "Which database provides ACID guarantees?",
 "answer": "Postgres", "type": "comparison", "level": "easy",
 "supporting_facts": [["Postgres", 0]],
 "context": [
   ["Postgres", ["Postgres provides ACID transactions and MVCC."]],
   ["Redis", ["Redis is an in-memory key-value store."]]
 ]},
{"_id": "e4", "question": "Was the scheduler used by Tokio designed before MVCC was added to Postgres?",
 "answer": "no", "type": "bridge", "level": "hard",
 "supporting_facts": [["Tokio", 1], ["Postgres", 0]],
 "context": [
   ["Tokio", ["Tokio is an asynchronous runtime for Rust applications.", "It uses a work-stealing scheduler."]],
   ["Postgres", ["Postgres provides ACID transactions and MVCC."]],
   ["Redis", ["Redis is an in-memory key-value store."]],
   ["Django", ["Django is a Python web framework."]]
 ]},
{"_id": "e5", "question": "Compare Redis and Postgres for transactional workloads.",
 "answer": "Postgres", "type": "comparison", "level": "hard",
 "supporting_facts": [["Postgres", 0], ["Redis", 0]],
 "context": [
   ["Postgres", ["Postgres provides ACID transactions and MVCC."]],
   ["Redis", ["Redis is an in-memory key-value store."]]
 ]}
]"#;

/// Same embedding-attaching retriever pattern as the other examples —
/// BM25 strips chunk embeddings, so re-attach them post-retrieval for
/// the semantic-tier diagnostics.
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ---- 1. Load the HotpotQA-shaped fixture ----
    let dataset = HotpotQADataset::from_json(MINI_HOTPOTQA)?;
    println!("loaded HotpotQA-shaped fixture: {} examples", dataset.len());

    // ---- 2. Build the chunker and convert to LabeledCorpus ----
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;
    let corpus = dataset.to_labeled_corpus(&chunker, |q| Some(embed(q)), default_regime)?;
    println!(
        "→ LabeledCorpus: {} documents, {} queries",
        corpus.docs.len(),
        corpus.queries.len()
    );
    println!();

    // ---- 3. Index + assemble adaptive components ----
    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    for c in &mut chunks {
        c.embedding = Some(embed(&c.text));
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
    let classifier = Arc::new(RuleBasedClassifier::new());
    let rerankers = vec![(
        RerankerLevel::Lexical,
        Arc::new(LexicalGroundingReranker::default()) as _,
    )];

    // ---- 4. Run the sweep ----
    let sweep = ThresholdSweep::default_grid(3);
    let report = sweep
        .run(&corpus, retriever, diagnostics, classifier, rerankers)
        .await?;

    // ---- 5. Render sweep table ----
    println!("{}", render_sweep_table(&report));
    println!();
    println!("{}", render_pareto(&report));

    // ---- 6. Reliability diagram aggregated across all settings ----
    let all_outcomes: Vec<_> = report
        .outcomes
        .iter()
        .flat_map(|v| v.iter().cloned())
        .collect();
    let diag = reliability_diagram(&all_outcomes, 10);
    println!();
    println!("{}", render_reliability(&diag));

    // ---- 7. Regime confusion matrix ----
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
            let count = row.get(r_pred).copied().unwrap_or(0);
            print!(" {:>10}", count);
        }
        println!();
    }
    println!();
    println!("per-regime metrics:");
    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>8}",
        "regime", "precision", "recall", "f1", "support"
    );
    println!("{}", "─".repeat(60));
    for r in RetrievalRegime::all() {
        let m = cm.per_regime.get(r).cloned().unwrap_or_default();
        println!(
            "{:<18} {:>10.3} {:>10.3} {:>10.3} {:>8}",
            r.code(),
            m.precision,
            m.recall,
            m.f1,
            m.support
        );
    }

    // ---- 8. Intervention regret ----
    let regret = regret_summary(&all_outcomes);
    println!();
    println!("─── intervention regret ───");
    println!("n_interventions:               {}", regret.n_interventions);
    println!(
        "mean(actual − expected) gain:  {:+.3}   (positive = policy UNDERESTIMATES gains)",
        regret.mean_expected_actual_error
    );
    println!(
        "mean |actual − expected|:      {:.3}",
        regret.mean_abs_expected_actual_error
    );
    println!(
        "mean lift when useful:         {:+.3}   (upside on the cases where intervention helps)",
        regret.mean_useful_lift
    );
    println!(
        "mean lift when harmful:        {:+.3}   (damage on the cases where it hurts)",
        regret.mean_harmful_lift
    );
    println!(
        "wasted interventions:          {}     (intervened but lift = 0)",
        regret.n_wasted_interventions
    );
    println!(
        "unused useful opportunities:   {}     (no intervention but adaptive ≠ static)",
        regret.n_unused_useful_opportunities
    );

    // ---- 9. Bootstrap threshold stability ----
    let stability = bootstrap_stability(&report, 200, 0xC0FFEE);
    println!();
    println!(
        "─── threshold stability (B = {} bootstrap resamples) ───",
        stability.n_bootstrap
    );
    println!(
        "{:<14} {:<14} {:>10} {:>10} {:>12} {:>12}",
        "min_p_distr.", "min_p_amb.", "lift_stddev", "argmax_freq", "ci90_low", "ci90_high"
    );
    println!("{}", "─".repeat(80));
    for (i, row) in report.rows.iter().enumerate() {
        let (lo, hi) = stability.ci90.get(i).copied().unwrap_or((0.0, 0.0));
        let sd = stability.lift_stddev.get(i).copied().unwrap_or(0.0);
        let af = stability.argmax_frequency.get(i).copied().unwrap_or(0.0);
        println!(
            "{:<14} {:<14} {:>10.3} {:>10.3} {:>+12.3} {:>+12.3}",
            format!("{:.2}", row.min_p_distractor),
            format!("{:.2}", row.min_p_ambiguous),
            sd,
            af,
            lo,
            hi
        );
    }

    // ---- 10. Headline ----
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("HEADLINE (this fixture — replace with real HotpotQA dev set):");
    if let Some(best) = report.argmax_lift() {
        println!(
            "  argmax sweep setting:  min_p_distractor={:.2}, min_p_ambiguous={:.2}",
            best.min_p_distractor, best.min_p_ambiguous
        );
        println!("  mean_recall_lift:      {:+.3}", best.mean_recall_lift);
        // Is the argmax stable?
        if let Some(i) = report.rows.iter().position(|r| std::ptr::eq(r, best)) {
            let af = stability.argmax_frequency.get(i).copied().unwrap_or(0.0);
            println!(
                "  argmax stability:      {:.0}% of bootstrap resamples agreed",
                af * 100.0
            );
            if af < 0.5 {
                println!("  → WARNING: best setting is NOT stable. Multiple settings tied.");
            } else if af < 0.8 {
                println!("  → caution: best setting is somewhat stable but not decisive.");
            } else {
                println!("  → best setting is stable.");
            }
        }
    }
    println!("  classifier accuracy:   {:.1}%", cm.accuracy * 100.0);
    println!("  ECE (calibration):     {:.3}", diag.ece);
    println!("════════════════════════════════════════════════════════════════════════");

    Ok(())
}
