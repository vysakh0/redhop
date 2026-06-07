//! CUAD hybrid + cross-encoder rerank probe.
//!
//! Tests whether switching CUAD from BM25 (the default) to
//! `retrieval="hybrid"` with BGE-small dense rerank, or to hybrid +
//! ms-marco cross-encoder rerank, lifts ≥0.8 retention past the 90.3%
//! plateau established by template-strip + clause-expansion (see
//! [CUAD_CLAUSE_EXPANSION](docs/findings/CUAD_CLAUSE_EXPANSION.md)).
//!
//! The mechanism contrast: the four-corner rule from
//! [SUB_IDF_AUTO_DROP_NULL](docs/findings/SUB_IDF_AUTO_DROP_NULL.md) said
//! corpus-only IDF manipulation fails; this probe tests a different
//! mechanism entirely — **semantic similarity over chunk embeddings**, which
//! reads the content of the chunk rather than counting tokens. If hybrid /
//! CE help on CUAD that's evidence that some of the residual gap is
//! lexical-mismatch (gold span uses terms the query doesn't), not retrieval
//! ranking under BM25 IDF noise.
//!
//! Two axes, 3×2 = 6 arms so we can read off whether dense helps the raw
//! template query (gold-vs-query lexical mismatch) or the stripped query
//! (where the noise is already low):
//!
//!         retrieval       × query
//!   A. lexical (BM25)     × raw, stripped   ← 81.3%, 87.7% baselines
//!   B. hybrid (BGE-small) × raw, stripped
//!   C. hybrid + CE        × raw, stripped
//!
//! Median per-query latency tracked per arm — the cost is the other side
//! of the tradeoff users care about.
//!
//! Same n=300, BUDGET=2000, CANDIDATE_K=40, RawTopK, set-based span_recall
//! as the other CUAD harnesses.
//!
//! Run: cargo run -p redhop-examples --example cuad_hybrid_rerank --features onnx --release

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use redhop::context::{ContextConfig, ContextStrategy};
use redhop::core::{EmbeddingProvider, Reranker};
use redhop::document::{Document, DocumentConfig, RetrievalMode};
use redhop::embeddings::{EmbedderConfig, OnnxEmbedder};
use redhop::reranking::OnnxCrossEncoder;
use serde::Deserialize;

#[derive(Deserialize)]
struct Cuad {
    data: Vec<Contract>,
}
#[derive(Deserialize)]
struct Contract {
    title: String,
    paragraphs: Vec<Paragraph>,
}
#[derive(Deserialize)]
struct Paragraph {
    context: String,
    qas: Vec<Qa>,
}
#[derive(Deserialize)]
struct Qa {
    question: String,
    answers: Vec<Answer>,
}
#[derive(Deserialize)]
struct Answer {
    text: String,
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .map(|w| w.to_string())
        .collect()
}

fn span_recall(gold: &str, ctx_words: &HashSet<String>) -> f32 {
    let g: HashSet<String> = words(gold).into_iter().collect();
    if g.is_empty() {
        return 1.0;
    }
    let hit = g.iter().filter(|w| ctx_words.contains(*w)).count();
    hit as f32 / g.len() as f32
}

fn extract_cuad_signal(q: &str) -> String {
    let mut clause = String::new();
    if let Some(start) = q.find('"') {
        if let Some(end_rel) = q[start + 1..].find('"') {
            clause = q[start + 1..start + 1 + end_rel].to_string();
        }
    }
    let details = q
        .find("Details:")
        .map(|i| q[i + "Details:".len()..].trim().to_string())
        .unwrap_or_default();
    if clause.is_empty() && details.is_empty() {
        return q.to_string();
    }
    format!("{clause} {details}").trim().to_string()
}

#[derive(Default, Clone)]
struct Cell {
    n: usize,
    sum_recall: f64,
    retained_80: usize,
    latencies_ms: Vec<f64>,
}
impl Cell {
    fn add(&mut self, r: f32, ms: f64) {
        self.n += 1;
        self.sum_recall += r as f64;
        if r >= 0.8 {
            self.retained_80 += 1;
        }
        self.latencies_ms.push(ms);
    }
    fn r80(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            100.0 * self.retained_80 as f64 / self.n as f64
        }
    }
    fn mean_recall(&self) -> f64 {
        self.sum_recall / self.n.max(1) as f64
    }
    fn median_ms(&self) -> f64 {
        if self.latencies_ms.is_empty() {
            return 0.0;
        }
        let mut v = self.latencies_ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }
    fn p95_ms(&self) -> f64 {
        if self.latencies_ms.is_empty() {
            return 0.0;
        }
        let mut v = self.latencies_ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(v.len() * 95 / 100).min(v.len() - 1)]
    }
}

const BUDGET: usize = 2000;
const CANDIDATE_K: usize = 40;
const LIMIT_Q: usize = 300;
const RERANK_POOL: usize = 40;
const DIM: usize = 384;

#[derive(Copy, Clone, PartialEq)]
enum Retrieval {
    Lexical,
    Hybrid,
    HybridCe,
}

#[derive(Copy, Clone, PartialEq)]
enum QueryPrep {
    Raw,
    Stripped,
}

fn arm_label(r: Retrieval, q: QueryPrep) -> &'static str {
    match (r, q) {
        (Retrieval::Lexical, QueryPrep::Raw) => "A1: lexical / raw",
        (Retrieval::Lexical, QueryPrep::Stripped) => "A2: lexical / stripped",
        (Retrieval::Hybrid, QueryPrep::Raw) => "B1: hybrid / raw",
        (Retrieval::Hybrid, QueryPrep::Stripped) => "B2: hybrid / stripped",
        (Retrieval::HybridCe, QueryPrep::Raw) => "C1: hybrid+CE / raw",
        (Retrieval::HybridCe, QueryPrep::Stripped) => "C2: hybrid+CE / stripped",
    }
}

fn build_cfg(retrieval: Retrieval) -> DocumentConfig {
    let retrieval_mode = match retrieval {
        Retrieval::Lexical => RetrievalMode::Lexical,
        Retrieval::Hybrid | Retrieval::HybridCe => RetrievalMode::Hybrid { candidate_pool: 40 },
    };
    DocumentConfig {
        candidate_k: CANDIDATE_K,
        retrieval_mode,
        rerank_pool: RERANK_POOL,
        context: ContextConfig {
            strategy: ContextStrategy::RawTopK,
            token_budget: BUDGET,
            ..DocumentConfig::default().context
        },
        ..DocumentConfig::default()
    }
}

fn run_arm(
    cuad: &Cuad,
    retrieval: Retrieval,
    qprep: QueryPrep,
    embedder: &Arc<dyn EmbeddingProvider>,
    ce: Option<&Arc<dyn Reranker>>,
) -> anyhow::Result<Cell> {
    let mut acc = Cell::default();
    let mut q_count = 0usize;

    'outer: for c in &cuad.data {
        for para in &c.paragraphs {
            let cfg = build_cfg(retrieval);
            let mut doc = match Document::from_text_with(&c.title, &para.context, cfg) {
                Ok(d) => d,
                Err(_) => continue,
            };
            if retrieval != Retrieval::Lexical {
                doc = doc.with_embedder(embedder.clone());
            }
            if retrieval == Retrieval::HybridCe {
                if let Some(ce_arc) = ce {
                    doc = doc.with_reranker(ce_arc.clone());
                }
            }
            for qa in &para.qas {
                if q_count >= LIMIT_Q {
                    break 'outer;
                }
                let gold = qa
                    .answers
                    .first()
                    .map(|a| a.text.as_str())
                    .unwrap_or_default();
                if gold.is_empty() {
                    continue;
                }
                let q = match qprep {
                    QueryPrep::Raw => qa.question.clone(),
                    QueryPrep::Stripped => extract_cuad_signal(&qa.question),
                };
                let t0 = Instant::now();
                let ctx = match doc.context(&q) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let assembled = ctx.text();
                let ctx_words: HashSet<String> = words(&assembled).into_iter().collect();
                let recall = span_recall(gold, &ctx_words);
                acc.add(recall, dt_ms);
                q_count += 1;
            }
        }
    }
    Ok(acc)
}

fn print_table(cells: &[(Retrieval, QueryPrep, Cell)]) {
    println!(
        "  {:<28} {:>6} {:>8} {:>10} {:>10} {:>10}",
        "arm", "n", "≥0.8", "mean rec", "p50 ms", "p95 ms"
    );
    for (r, q, cell) in cells {
        println!(
            "  {:<28} {:>6} {:>7.1}% {:>10.3} {:>10.1} {:>10.1}",
            arm_label(*r, *q),
            cell.n,
            cell.r80(),
            cell.mean_recall(),
            cell.median_ms(),
            cell.p95_ms(),
        );
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::var("REDHOP_CUAD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("cuad/cuad_sample.json"));
    let raw = std::fs::read_to_string(&path)?;
    let cuad: Cuad = serde_json::from_str(&raw)?;

    println!("CUAD hybrid + cross-encoder rerank probe");
    println!(
        "  config: n={LIMIT_Q}, BM25/hybrid/hybrid+CE, budget={BUDGET}, candidate_k={CANDIDATE_K}"
    );
    println!("         rerank_pool={RERANK_POOL}, RawTopK, set-based span_recall");
    println!();

    println!("loading BGE-small embedder...");
    let (model, tokenizer) = redhop_examples::bge_small_paths();
    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(OnnxEmbedder::load(
        &model,
        &tokenizer,
        EmbedderConfig::bge(DIM),
    )?);

    println!("loading ms-marco cross-encoder...");
    let (ce_model, ce_tok) = redhop_examples::ms_marco_paths();
    let ce: Arc<dyn Reranker> = Arc::new(OnnxCrossEncoder::load(&ce_model, &ce_tok, 256)?);
    println!();

    let mut cells: Vec<(Retrieval, QueryPrep, Cell)> = Vec::new();
    for (r, q) in [
        (Retrieval::Lexical, QueryPrep::Raw),
        (Retrieval::Lexical, QueryPrep::Stripped),
        (Retrieval::Hybrid, QueryPrep::Raw),
        (Retrieval::Hybrid, QueryPrep::Stripped),
        (Retrieval::HybridCe, QueryPrep::Raw),
        (Retrieval::HybridCe, QueryPrep::Stripped),
    ] {
        print!("running {} ... ", arm_label(r, q));
        std::io::Write::flush(&mut std::io::stdout())?;
        let cell = run_arm(&cuad, r, q, &embedder, Some(&ce))?;
        println!(
            "n={} ≥0.8={:.1}% p50={:.1}ms",
            cell.n,
            cell.r80(),
            cell.median_ms()
        );
        cells.push((r, q, cell));
    }
    println!();
    println!("══ results ══");
    print_table(&cells);
    println!();

    // ── verdict ───────────────────────────────────────────────────────────
    let by_arm = |r: Retrieval, q: QueryPrep| -> f64 {
        cells
            .iter()
            .find(|(rr, qq, _)| *rr == r && *qq == q)
            .map(|(_, _, c)| c.r80())
            .unwrap_or(0.0)
    };
    let a1 = by_arm(Retrieval::Lexical, QueryPrep::Raw);
    let a2 = by_arm(Retrieval::Lexical, QueryPrep::Stripped);
    let b1 = by_arm(Retrieval::Hybrid, QueryPrep::Raw);
    let b2 = by_arm(Retrieval::Hybrid, QueryPrep::Stripped);
    let c1 = by_arm(Retrieval::HybridCe, QueryPrep::Raw);
    let c2 = by_arm(Retrieval::HybridCe, QueryPrep::Stripped);

    println!(
        "Δ on raw query:      hybrid={:+.1}  hybrid+CE={:+.1}",
        b1 - a1,
        c1 - a1
    );
    println!(
        "Δ on stripped query: hybrid={:+.1}  hybrid+CE={:+.1}",
        b2 - a2,
        c2 - a2
    );
    println!();

    let best = [a1, a2, b1, b2, c1, c2]
        .iter()
        .copied()
        .fold(f64::MIN, f64::max);
    let prior_plateau = 90.3; // CUAD_CLAUSE_EXPANSION ceiling
    println!("highest cell on this probe: {best:.1}%");
    println!("prior CUAD plateau (template strip + clause expand): {prior_plateau:.1}%");
    println!();

    let semantic_helps_raw = (b1 - a1) >= 1.5 || (c1 - a1) >= 1.5;
    let semantic_helps_stripped = (b2 - a2) >= 1.5 || (c2 - a2) >= 1.5;
    if best > prior_plateau + 1.0 {
        println!("  ✓ Semantic retrieval lifts past the strip+expand plateau. Worth shipping");
        println!("    as a `retrieval=\"hybrid\"` recommendation for contract workloads.");
    } else if semantic_helps_raw && !semantic_helps_stripped {
        println!("  ~ Semantic helps the RAW query but not the stripped one. Suggests dense");
        println!("    compensates for the boilerplate-dilution effect we already fix with");
        println!("    template stripping; not a separate lift.");
    } else if semantic_helps_stripped && !semantic_helps_raw {
        println!("  ~ Semantic helps the STRIPPED query but the lift is small or smaller than");
        println!("    what we already get from clause expansion. Marginal call.");
    } else {
        println!("  ✗ Semantic doesn't meaningfully help CUAD beyond what BM25 + strip + expand");
        println!("    already deliver. Latency cost is real; this is not the lever for");
        println!("    contract workloads.");
    }
    Ok(())
}
