//! Calibration-phase demo.
//!
//! Runs the full calibration harness against the synthetic fixture
//! (`redhop-calibration::fixtures::synthetic_dataset`) and prints:
//!
//!   1. The per-threshold sweep table — intervention rate, recall lift,
//!      latency, cost, regime accuracy at each policy threshold.
//!   2. The Pareto comparison — flags dominated settings.
//!   3. The reliability diagram for the predicted-argmax regime —
//!      whether `p` actually predicts correctness.
//!   4. A headline line telling you which threshold setting maximizes
//!      recall lift on THIS dataset.
//!
//! The synthetic numbers are not the answer; they only demonstrate the
//! harness. Real calibration uses a `LabeledCorpus` built from HotpotQA
//! traces or your production workload.
//!
//! Run with:
//!     cargo run -p redhop-examples --example calibration_sweep

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use redhop_calibration::{
    fixtures::{embed, synthetic_dataset},
    reliability::reliability_diagram,
    report::{render_pareto, render_reliability, render_sweep_table},
    ThresholdSweep,
};
use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{
    Chunk, ChunkId, Chunker, DiagnosticsEngine, Embedding, Query, RerankerLevel,
    Result as CoreResult, RetrievalResult, Retriever, TokenizerBackend,
};
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use redhop_orchestration::RuleBasedClassifier;
use redhop::reranking::LexicalGroundingReranker;
use redhop::retrieval::Bm25Retriever;

/// Same EmbedAttachingRetriever pattern as the adaptive_loop example.
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
    let corpus = synthetic_dataset();
    println!(
        "calibration corpus: {} documents, {} queries\n",
        corpus.docs.len(),
        corpus.queries.len()
    );

    // Index the corpus.
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;
    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    for c in &mut chunks {
        c.embedding = Some(embed(&c.text));
    }
    let mut bm25 = Bm25Retriever::new()?;
    Retriever::index(&mut bm25, &chunks).await?;
    let retriever: Arc<dyn Retriever> =
        Arc::new(EmbedAttachingRetriever::new(Arc::new(bm25), &chunks));

    // Layered diagnostics + classifier.
    let lexical: Arc<dyn DiagnosticsEngine> = Arc::new(DefaultDiagnosticsEngine::new());
    let semantic: Arc<dyn DiagnosticsEngine> = Arc::new(SemanticDiagnosticsEngine::new());
    let diagnostics: Arc<dyn DiagnosticsEngine> = Arc::new(
        LayeredDiagnosticsEngine::lexical_and_semantic(lexical, semantic),
    );
    let classifier = Arc::new(RuleBasedClassifier::new());

    // One reranker registered at the Lexical tier so EscalateReranker has
    // something to dispatch to.
    let rerankers = vec![(
        RerankerLevel::Lexical,
        Arc::new(LexicalGroundingReranker::default()) as _,
    )];

    // Run the sweep. We deliberately use a small initial top_k (3) so
    // that adaptive ExpandTopK has room to pull in additional gold
    // chunks; at top_k=10 against an 18-doc corpus, BM25 already finds
    // everything and intervention can't lift recall.
    let sweep = ThresholdSweep::default_grid(3);
    let report = sweep
        .run(&corpus, retriever, diagnostics, classifier, rerankers)
        .await?;

    // Print the sweep table.
    println!("{}", render_sweep_table(&report));
    println!();
    println!("{}", render_pareto(&report));

    // Reliability diagram for the predicted argmax. We use 10 bins.
    // Aggregate across all swept settings to get more samples per bin.
    let mut all_outcomes = Vec::new();
    for outs in &report.outcomes {
        for o in outs {
            all_outcomes.push(o.clone());
        }
    }
    let diagram = reliability_diagram(&all_outcomes, 10);
    println!();
    println!("{}", render_reliability(&diagram));

    // Headline: where's the best setting on THIS dataset?
    if let Some(best) = report.argmax_lift() {
        println!();
        println!("════════════════════════════════════════════════════════════════════════");
        println!("HEADLINE: best (min_p_distractor, min_p_ambiguous) on this dataset");
        println!(
            "         = ({:.2}, {:.2})  with mean_recall_lift = {:+.3}",
            best.min_p_distractor, best.min_p_ambiguous, best.mean_recall_lift
        );
        println!(
            "         intervention_rate = {:.2}, mean_rerank_calls = {:.2}, regime_argmax_accuracy = {:.1}%",
            best.intervention_rate,
            best.mean_rerank_calls,
            best.regime_argmax_accuracy * 100.0,
        );
        println!("════════════════════════════════════════════════════════════════════════");
    }

    // Diagnostic interpretation of the reliability diagram. This is the
    // direct answer to the user's central calibration question: "is
    // p∈[0.30, 0.45] genuinely weak signal, or is the classifier
    // underconfident?"
    println!();
    println!("──── calibration interpretation ────");
    println!(
        "ECE = {:.3}  (0 = perfectly calibrated, >0.10 is notable)",
        diagram.ece
    );
    let mut underconfident_bins = 0;
    let mut overconfident_bins = 0;
    for b in &diagram.bins {
        if b.count == 0 {
            continue;
        }
        let gap = b.empirical_correct - b.mean_predicted_p;
        if gap > 0.10 {
            underconfident_bins += 1;
        } else if gap < -0.10 {
            overconfident_bins += 1;
        }
    }
    println!("underconfident bins (empirical > predicted + 0.10): {underconfident_bins}");
    println!("overconfident bins  (empirical < predicted - 0.10): {overconfident_bins}");
    if underconfident_bins > overconfident_bins {
        println!("→ the classifier appears UNDERCONFIDENT on this dataset.");
        println!("  Lowering policy thresholds in the underconfident range would convert");
        println!("  conservative no-ops into useful interventions. Cross-check by sweeping");
        println!("  the relevant policy threshold (see the row that maximizes mean_recall_lift).");
    } else if overconfident_bins > underconfident_bins {
        println!(
            "→ the classifier appears OVERCONFIDENT on this dataset. Conservatism is justified."
        );
    } else {
        println!("→ calibration is balanced on this dataset.");
    }

    // Pareto-optimal (non-dominated) rows, surfaced explicitly.
    println!();
    println!(
        "──── non-dominated threshold settings (the empirical answer to ‘when to intervene?’) ────"
    );
    let mut shown = 0;
    for (i, r) in report.rows.iter().enumerate() {
        let dominated = report.rows.iter().enumerate().any(|(j, other)| {
            j != i
                && other.mean_latency_ms <= r.mean_latency_ms
                && other.mean_recall_lift >= r.mean_recall_lift
                && (other.mean_latency_ms < r.mean_latency_ms
                    || other.mean_recall_lift > r.mean_recall_lift)
        });
        if dominated {
            continue;
        }
        if r.mean_recall_lift <= 0.0 && r.intervention_rate > 0.0 {
            // Settings that intervene but produce no lift — also surface,
            // they're the "useful no-op" benchmark (high intervention with
            // zero net effect = controller wasting compute).
        }
        if shown == 0 {
            println!(
                "{:<14} {:<14} {:>5} {:>10} {:>+10} {:>10}",
                "min_p_distr.", "min_p_amb.", "n", "interv", "lift", "useful%"
            );
        }
        println!(
            "{:<14} {:<14} {:>5} {:>10.2} {:>+10.3} {:>9.0}%",
            format!("{:.2}", r.min_p_distractor),
            format!("{:.2}", r.min_p_ambiguous),
            r.n,
            r.intervention_rate,
            r.mean_recall_lift,
            r.fraction_useful_interventions * 100.0,
        );
        shown += 1;
    }
    if shown == 0 {
        println!("(every row dominated by another — no clear winner)");
    }

    Ok(())
}
