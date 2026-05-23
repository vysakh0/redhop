//! Cross-encoder escalation economics — the aligned-geometry experiment.
//!
//! The action-path finding: dense retrieval fails by returning a
//! semantically tight cluster that misses the orthogonal second hop;
//! ExpandTopK (more similar neighbors) can't fix that. A cross-encoder
//! re-scoring a WIDER net is the action whose geometry matches the
//! failure: it can pull a dissimilar-but-relevant chunk up from deep in
//! the candidate pool into the final top-k.
//!
//! This experiment measures, on dense BGE retrieval (wide net = top-N),
//! recall@k_final for three strategies:
//!
//!   static       : dense top-k_final, no rerank.            (0 CE calls)
//!   uniform CE   : CE-rerank the top-N, keep k_final.       (every query)
//!   selective CE : CE only when the controller flags the     (M < N queries)
//!                  dense top-k_final as not-Easy.
//!   oracle       : CE only when it actually helps.           (upper bound)
//!
//! The questions (all five priorities):
//!   1. real cross-encoder escalation — does CE recover dense's missed recall?
//!   2. intervention precision under dense retrieval — useful% of CE firings
//!   3. wasted intervention reduction — selective vs uniform CE calls
//!   4. recall-per-rerank economics — recall gain per CE invocation
//!   5. adaptive-vs-static under semantic reranking
//!
//! Requires `--features onnx` + BGE-small + ms-marco cross-encoder.
//!
//! Run:
//!   cargo run -p redhop-examples --example ce_escalation_economics \
//!       --features onnx --release

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use redhop_calibration::loaders::hotpotqa::{default_regime, HotpotQADataset};
use redhop_chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop_core::{
    ChunkId, Chunker, DiagnosticsEngine, Embedding, EmbeddingProvider, Query, RegimeClassifier,
    Reranker, RetrievalRegime, RetrievalResult, Retriever, TokenizerBackend, VectorIndex,
};
use redhop_diagnostics::{
    DefaultDiagnosticsEngine, LayeredDiagnosticsEngine, SemanticDiagnosticsEngine,
};
use redhop_embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop_orchestration::{compute_confidence, RuleBasedClassifier};
use redhop_reranking::OnnxCrossEncoder;
use redhop_retrieval::DenseRetriever;
use redhop_storage::{ChunkStore, FlatVectorIndex};

const HOTPOTQA_PATH: &str =
    "/Users/vysakh/projects/neorag/data/hotpotqa/hotpot_dev_distractor_v1.json";
const BGE_MODEL: &str = "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/onnx/model.onnx";
const BGE_TOKENIZER: &str = "/Users/vysakh/projects/neorag/models/bge-small-en-v1.5/tokenizer.json";
const CE_MODEL: &str =
    "/Users/vysakh/projects/neorag/models/ms-marco-MiniLM-L-6-v2/onnx/model.onnx";
const CE_TOKENIZER: &str =
    "/Users/vysakh/projects/neorag/models/ms-marco-MiniLM-L-6-v2/tokenizer.json";
const SAMPLE_SIZE: usize = 60;
const WIDE_N: usize = 20; // candidate pool the CE re-scores
const K_FINAL: usize = 4; // final answer size; recall measured here
const DIM: usize = 384;
const EASY_GATE: f32 = 0.50; // fire CE when p(Easy) < this

fn recall(results: &[RetrievalResult], gold: &[ChunkId]) -> f32 {
    if gold.is_empty() {
        return 1.0;
    }
    let ids: Vec<&ChunkId> = results.iter().map(|r| &r.chunk.id).collect();
    let found = gold.iter().filter(|g| ids.contains(g)).count();
    found as f32 / gold.len() as f32
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Cross-encoder escalation economics on dense BGE retrieval       ║");
    println!("║  (aligned geometry: CE re-scores a wide net to reach the 2nd hop)║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let mut dataset = HotpotQADataset::from_path(HOTPOTQA_PATH)?;
    dataset.examples.truncate(SAMPLE_SIZE);
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 40, 60, 0)?;

    println!("loading BGE-small + ms-marco cross-encoder...");
    let bge: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        BGE_MODEL,
        BGE_TOKENIZER,
        EmbedderConfig::bge(DIM),
    )?);
    let ce = OnnxCrossEncoder::load(CE_MODEL, CE_TOKENIZER, 256)?;

    // Build corpus with BGE query embeddings.
    let q_texts: Vec<String> = dataset
        .examples
        .iter()
        .map(|e| e.question.clone())
        .collect();
    let q_vecs = bge.embed(&q_texts).await?;
    let q_map: HashMap<String, Embedding> = q_texts.into_iter().zip(q_vecs).collect();
    let corpus = dataset.to_labeled_corpus(&chunker, |q| q_map.get(q).cloned(), default_regime)?;

    // Chunk + embed corpus with BGE, build dense retriever.
    let mut chunks = chunker.chunk_batch(&corpus.docs)?;
    println!("embedding {} chunks with BGE...", chunks.len());
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vecs = bge.embed(&texts).await?;
    for (c, v) in chunks.iter_mut().zip(vecs.iter()) {
        c.embedding = Some(v.clone());
    }
    let index: Arc<RwLock<dyn VectorIndex>> = Arc::new(RwLock::new(FlatVectorIndex::new(DIM)));
    let store = Arc::new(ChunkStore::new());
    let mut dense = DenseRetriever::new(index, store);
    dense.index(&chunks).await?;

    let diagnostics = LayeredDiagnosticsEngine::lexical_and_semantic(
        Arc::new(DefaultDiagnosticsEngine::new()),
        Arc::new(SemanticDiagnosticsEngine::new()),
    );
    let classifier = RuleBasedClassifier::new();

    // Accumulators.
    let mut r_static = 0f32;
    let mut r_uniform = 0f32;
    let mut r_selective = 0f32;
    let mut r_oracle = 0f32;
    let mut n = 0f32;

    let mut ce_calls_uniform = 0usize;
    let mut ce_calls_selective = 0usize;
    let mut ce_calls_oracle = 0usize;

    // Intervention quality bookkeeping (selective firings).
    let mut sel_fired = 0usize;
    let mut sel_useful = 0usize; // fired AND ce helped
    let mut sel_wasted = 0usize; // fired AND ce didn't help
    let mut uniform_useful = 0usize; // CE helped (any query)
    let mut harmful = 0usize; // CE hurt recall vs static

    let mut ce_latency_ms = 0f64;

    for lq in &corpus.queries {
        if lq.gold_chunk_ids.is_empty() {
            continue;
        }
        n += 1.0;
        let mut query = Query::new(&lq.text);
        query.embedding = lq.embedding.clone();

        // Dense wide net.
        let wide = dense.retrieve(&query, WIDE_N).await?;
        let static_top: Vec<RetrievalResult> = wide.iter().take(K_FINAL).cloned().collect();

        // CE rerank the wide net → top-k_final.
        let t = Instant::now();
        let ce_top = ce.rerank(&query, wide.clone(), K_FINAL).await?;
        ce_latency_ms += t.elapsed().as_secs_f64() * 1000.0;

        let rec_static = recall(&static_top, &lq.gold_chunk_ids);
        let rec_ce = recall(&ce_top, &lq.gold_chunk_ids);

        // Selective gate: diagnose+classify the dense top-k, fire CE if
        // the controller is not confident retrieval is Easy.
        let diag = diagnostics.diagnose(&query, &static_top)?;
        let conf = compute_confidence(&static_top);
        let regime = classifier.classify(&diag, &conf);
        let p_easy = regime.p(RetrievalRegime::Easy);
        let fire = p_easy < EASY_GATE;

        // Tally.
        r_static += rec_static;
        r_uniform += rec_ce;
        ce_calls_uniform += 1;
        if rec_ce > rec_static + 1e-6 {
            uniform_useful += 1;
        }
        if rec_ce < rec_static - 1e-6 {
            harmful += 1;
        }
        r_oracle += rec_static.max(rec_ce);
        if rec_ce > rec_static + 1e-6 {
            ce_calls_oracle += 1;
        }

        if fire {
            sel_fired += 1;
            ce_calls_selective += 1;
            r_selective += rec_ce;
            if rec_ce > rec_static + 1e-6 {
                sel_useful += 1;
            } else if (rec_ce - rec_static).abs() <= 1e-6 {
                sel_wasted += 1;
            }
        } else {
            r_selective += rec_static;
        }
    }

    let nq = n.max(1.0);
    let mean = |x: f32| x / nq;
    println!("\n──── recall@{K_FINAL} by strategy (dense BGE, wide net = {WIDE_N}) ────");
    println!("  {:<26} {:>10} {:>14}", "strategy", "recall", "CE calls");
    println!("  {}", "─".repeat(52));
    println!(
        "  {:<26} {:>10.3} {:>14}",
        "static (no CE)",
        mean(r_static),
        0
    );
    println!(
        "  {:<26} {:>10.3} {:>14}",
        "uniform CE (every query)",
        mean(r_uniform),
        ce_calls_uniform
    );
    println!(
        "  {:<26} {:>10.3} {:>14}",
        "selective CE (controller)",
        mean(r_selective),
        ce_calls_selective
    );
    println!(
        "  {:<26} {:>10.3} {:>14}",
        "oracle (CE iff it helps)",
        mean(r_oracle),
        ce_calls_oracle
    );

    // Economics.
    let gain_uniform = mean(r_uniform) - mean(r_static);
    let gain_selective = mean(r_selective) - mean(r_static);
    let rpr_uniform = if ce_calls_uniform > 0 {
        gain_uniform / ce_calls_uniform as f32
    } else {
        0.0
    };
    let rpr_selective = if ce_calls_selective > 0 {
        gain_selective / ce_calls_selective as f32
    } else {
        0.0
    };

    println!("\n──── escalation economics ────");
    println!(
        "  uniform CE:   +{:.3} recall for {} calls   (recall/call = {:.5})",
        gain_uniform, ce_calls_uniform, rpr_uniform
    );
    println!(
        "  selective CE: +{:.3} recall for {} calls   (recall/call = {:.5})",
        gain_selective, ce_calls_selective, rpr_selective
    );
    if gain_uniform > 0.0 {
        println!(
            "  selective captured {:.0}% of uniform's recall gain with {:.0}% of the CE calls",
            gain_selective / gain_uniform * 100.0,
            ce_calls_selective as f32 / ce_calls_uniform as f32 * 100.0
        );
    }
    println!(
        "  recall-per-rerank efficiency multiple (selective / uniform): {:.2}x",
        if rpr_uniform > 0.0 {
            rpr_selective / rpr_uniform
        } else {
            0.0
        }
    );

    println!("\n──── intervention precision under dense retrieval ────");
    println!(
        "  uniform CE useful on:    {}/{} queries ({:.0}%)",
        uniform_useful,
        ce_calls_uniform,
        uniform_useful as f32 / nq * 100.0
    );
    println!(
        "  CE harmful on:           {}/{} queries ({:.0}%)  (CE hurt recall vs static)",
        harmful,
        ce_calls_uniform,
        harmful as f32 / nq * 100.0
    );
    if sel_fired > 0 {
        println!("  selective fired:         {} queries", sel_fired);
        println!(
            "  selective useful%:       {:.0}%  (fired AND helped)",
            sel_useful as f32 / sel_fired as f32 * 100.0
        );
        println!(
            "  selective wasted:        {}  (fired, no recall change)",
            sel_wasted
        );
    }
    println!(
        "  mean CE latency:         {:.1} ms/query (over {} candidates)",
        ce_latency_ms / nq as f64,
        WIDE_N
    );

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("VERDICT — aligned geometry: CE re-scores a wide net");
    if gain_uniform > 0.01 {
        println!("  ✓ Cross-encoder RECOVERS dense retrieval's missed recall:");
        println!(
            "    +{:.3} recall@{} over static dense top-{} — the action geometry",
            gain_uniform, K_FINAL, K_FINAL
        );
        println!("    (re-score a wide net) finally matches dense's failure geometry.");
        let eff = if rpr_uniform > 0.0 {
            rpr_selective / rpr_uniform
        } else {
            0.0
        };
        if eff > 1.2 && gain_selective > 0.01 {
            println!(
                "  ✓ Selective escalation is MORE ECONOMICAL: {:.1}x recall-per-rerank,",
                eff
            );
            println!("    capturing meaningful recall at a fraction of the CE compute.");
        } else if gain_selective > 0.01 {
            println!("  ~ Selective captures recall but the controller's gate needs tuning");
            println!("    to beat uniform on recall-per-rerank (see selective useful%).");
        } else {
            println!(
                "  ✗ The controller's gate (p_easy<{:.2}) mis-selects on dense — it",
                EASY_GATE
            );
            println!("    doesn't fire CE on the queries that need it. Gate needs recalibration.");
        }
    } else {
        println!(
            "  ✗ Cross-encoder did NOT recover recall on this sample (+{:.3}).",
            gain_uniform
        );
        println!(
            "    Either gold is outside the top-{} net, or dense top-{} already had it.",
            WIDE_N, K_FINAL
        );
    }
    println!("════════════════════════════════════════════════════════════════════════");
    Ok(())
}
