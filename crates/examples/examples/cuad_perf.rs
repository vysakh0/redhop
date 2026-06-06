//! CUAD performance benchmark — Rust API path, per-stage timing.
//!
//! All the CUAD recall work in this branch ran through `redhop` directly
//! (`Document::from_text_with(...) → doc.context(query)`), no Python. This
//! companion harness reports the latency the Rust path actually delivers:
//!
//!   - Document build time per contract (~9k tokens of legal text)
//!   - Per-query `doc.context(query)` latency (median, p95, mean)
//!   - Throughput (queries per second) over the full 300-question slice
//!   - Same comparison as cuad_query_preprocessing.rs: original 24-word
//!     template vs the ~5-word stripped query
//!
//! No external dependencies, no criterion — just `std::time::Instant`
//! around real product calls. The latencies are wall-clock, single-threaded,
//! release-build, on whatever box this runs on (note that in the README).
//!
//! Run: cargo run -p redhop-examples --example cuad_perf --release

use std::collections::HashSet;
use std::time::Instant;

use redhop::context::{ContextConfig, ContextStrategy};
use redhop::document::{Document, DocumentConfig};
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
    let g = words(gold);
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

const BUDGET: usize = 2000;
const LIMIT_Q: usize = 300;
const CANDIDATE_K: usize = 40;

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

struct PerfReport {
    arm: &'static str,
    build_ms: Vec<f64>,
    query_ms: Vec<f64>,
    sum_recall: f64,
    retained_80: usize,
    n: usize,
}

impl PerfReport {
    fn new(arm: &'static str) -> Self {
        Self {
            arm,
            build_ms: Vec::new(),
            query_ms: Vec::new(),
            sum_recall: 0.0,
            retained_80: 0,
            n: 0,
        }
    }

    fn print(&self) {
        let mut bs: Vec<f64> = self.build_ms.clone();
        bs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut qs: Vec<f64> = self.query_ms.clone();
        qs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let total_query_ms: f64 = self.query_ms.iter().sum();
        let qps = if total_query_ms > 0.0 {
            1000.0 * self.n as f64 / total_query_ms
        } else {
            0.0
        };
        println!("── {} ──", self.arm);
        println!("  n queries: {}", self.n);
        println!(
            "  recall@≥0.8: {:.1}%",
            100.0 * self.retained_80 as f64 / self.n as f64
        );
        println!("  mean recall:   {:.3}", self.sum_recall / self.n as f64);
        println!();
        println!(
            "  Document build (per contract):  p50={:.2}ms  p95={:.2}ms  mean={:.2}ms  (n={} contracts)",
            percentile(&bs, 0.50),
            percentile(&bs, 0.95),
            bs.iter().sum::<f64>() / bs.len() as f64,
            bs.len()
        );
        println!(
            "  context() per query:            p50={:.2}ms  p95={:.2}ms  mean={:.2}ms",
            percentile(&qs, 0.50),
            percentile(&qs, 0.95),
            qs.iter().sum::<f64>() / qs.len() as f64,
        );
        println!("  throughput (queries / sec):     {:.0} qps", qps);
        println!();
    }
}

fn run(cuad: &Cuad, strip_template: bool, arm: &'static str) -> anyhow::Result<PerfReport> {
    let mut rep = PerfReport::new(arm);
    let mut q_count = 0usize;
    'outer: for c in &cuad.data {
        for para in &c.paragraphs {
            let cfg = DocumentConfig {
                candidate_k: CANDIDATE_K,
                context: ContextConfig {
                    strategy: ContextStrategy::RawTopK,
                    token_budget: BUDGET,
                    ..DocumentConfig::default().context
                },
                ..DocumentConfig::default()
            };
            let t_build = Instant::now();
            let mut doc = match Document::from_text_with(&c.title, &para.context, cfg) {
                Ok(d) => d,
                Err(_) => continue,
            };
            // Warm the retriever cache by issuing a no-op query — keeps the
            // build measurement clean of first-query-side index work.
            let _ = doc.context("warmup");
            let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;
            rep.build_ms.push(build_ms);

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
                let query_text = if strip_template {
                    extract_cuad_signal(&qa.question)
                } else {
                    qa.question.clone()
                };
                let t_query = Instant::now();
                let ctx = match doc.context(&query_text) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let query_ms = t_query.elapsed().as_secs_f64() * 1000.0;
                rep.query_ms.push(query_ms);

                let assembled = ctx.text();
                let ctx_words: HashSet<String> = words(&assembled).into_iter().collect();
                let recall = span_recall(gold, &ctx_words);
                rep.sum_recall += recall as f64;
                if recall >= 0.8 {
                    rep.retained_80 += 1;
                }
                rep.n += 1;
                q_count += 1;
            }
        }
    }
    Ok(rep)
}

fn main() -> anyhow::Result<()> {
    let path = std::env::var("REDHOP_CUAD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("cuad/cuad_sample.json"));
    let raw = std::fs::read_to_string(&path)?;
    let cuad: Cuad = serde_json::from_str(&raw)?;

    println!("CUAD performance benchmark — Rust API path, release build");
    println!(
        "  budget={}  candidate_k={}  strategy=RawTopK  default chunker (target=128, max=256)",
        BUDGET, CANDIDATE_K
    );
    println!(
        "  data: cuad_sample.json (50 contracts), first {} questions",
        LIMIT_Q
    );
    println!();

    let baseline = run(
        &cuad,
        false,
        "arm A: original CUAD queries (24-word template)",
    )?;
    baseline.print();

    let stripped = run(&cuad, true, "arm B: template stripped (~5 words)")?;
    stripped.print();

    println!("══ headline ══");
    println!(
        "  Per-query context() latency: ~{:.1}ms median, ~{:.1}ms p95.",
        percentile(
            &{
                let mut v = baseline.query_ms.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            },
            0.50,
        ),
        percentile(
            &{
                let mut v = baseline.query_ms.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            },
            0.95,
        ),
    );
    println!(
        "  Document build (one ~9k-token contract): ~{:.1}ms median.",
        percentile(
            &{
                let mut v = baseline.build_ms.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            },
            0.50,
        ),
    );
    println!(
        "  Throughput across the full 300-question slice: ~{:.0} queries/sec.",
        1000.0 * baseline.n as f64 / baseline.query_ms.iter().sum::<f64>(),
    );
    println!();
    println!("Rust API, BM25 lexical retrieval, no embeddings, no LLM calls — fully in-process.");
    Ok(())
}
