//! CUAD recall-gap diagnostic — sweep chunk size × ContextStrategy on the
//! same harness FRAMEWORK_COMPARISON.md used, to find what (if anything)
//! closes the 4-point retention gap vs LlamaIndex (RedHop 82% vs LlamaIndex
//! 86% at ≥0.8 word-recall, budget=2000, BM25).
//!
//! Hypothesis (from the headline numbers):
//!   LangChain  : 1813 tokens used, recall 0.87, 73% ≥0.8
//!   LlamaIndex : 1806 tokens used, recall 0.93, 86% ≥0.8  ← winner
//!   RedHop[topk]: 1894 tokens used, recall 0.91, 82% ≥0.8
//!
//! LlamaIndex packs MORE answer-bearing content per token than RedHop —
//! same retrieval (BM25), same budget. That points at chunking
//! (granularity / boundary alignment) and/or strategy choice. The default
//! chunker (target=128, max=256) was calibrated on HotpotQA in
//! CHUNK_GRANULARITY.md; CUAD is structurally different (single-document
//! answer-span extraction, not multi-hop), so the optimum may shift.
//!
//! Sweep:
//!   target_tokens ∈ {32, 48, 64, 96, 128, 192}  (× max = target * 2)
//!   strategy      ∈ {RawTopK, MaxDensity, DistractorFiltered,
//!                    ReasoningPreserving}
//!
//! Budget = 2000 (matches FRAMEWORK_COMPARISON.md's setup). Metric =
//! word-recall of the gold answer span against the assembled context
//! (same `span_recall` definition the existing eval harness uses).
//!
//! Run: cargo run -p redhop-examples --example cuad_chunk_strategy_sweep --release

use std::collections::HashSet;

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

#[derive(Default, Clone)]
struct Cell {
    n: usize,
    sum_recall: f64,
    retained_50: usize,
    retained_80: usize,
    sum_final_tokens: f64,
}

impl Cell {
    fn add(&mut self, r: f32, final_tokens: usize) {
        self.n += 1;
        self.sum_recall += r as f64;
        self.sum_final_tokens += final_tokens as f64;
        if r >= 0.5 {
            self.retained_50 += 1;
        }
        if r >= 0.8 {
            self.retained_80 += 1;
        }
    }

    fn mean_recall(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum_recall / self.n as f64
        }
    }
    fn pct_50(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            100.0 * self.retained_50 as f64 / self.n as f64
        }
    }
    fn pct_80(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            100.0 * self.retained_80 as f64 / self.n as f64
        }
    }
    fn mean_tokens(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum_final_tokens / self.n as f64
        }
    }
}

fn strategy_name(s: ContextStrategy) -> &'static str {
    match s {
        ContextStrategy::RawTopK => "raw_topk",
        ContextStrategy::DistractorFiltered => "distractor_filtered",
        ContextStrategy::RedundancyPruned => "redundancy_pruned",
        ContextStrategy::MaxDensity => "max_density",
        ContextStrategy::ReasoningPreserving => "reasoning_preserving",
        ContextStrategy::Auto => "auto",
    }
}

const CHUNK_TARGETS: &[usize] = &[32, 48, 64, 96, 128, 192];
const BUDGET: usize = 2000;
/// Match `bench/compare.py`'s `cuad_items(300)` slice — sample the FIRST 300
/// QA pairs in document order. cuad_sample.json has ~950 QAs across 50
/// contracts, so taking all of them would not be apples-to-apples with the
/// framework comparison's n=300.
const LIMIT_Q: usize = 300;

fn run_cell(cuad: &Cuad, target_tokens: usize, strategy: ContextStrategy) -> anyhow::Result<Cell> {
    let max_tokens = target_tokens * 2;
    let mut acc = Cell::default();
    let mut q_count = 0usize;
    'outer: for c in &cuad.data {
        for para in &c.paragraphs {
            let cfg = DocumentConfig {
                target_tokens,
                max_tokens,
                // Match `bench/compare.py`'s CANDIDATE_K=40, not the
                // DocumentConfig default of 20 — this is the
                // apples-to-apples reproduction setting.
                candidate_k: 40,
                context: ContextConfig {
                    strategy,
                    token_budget: BUDGET,
                    ..DocumentConfig::default().context
                },
                ..DocumentConfig::default()
            };
            let mut doc = match Document::from_text_with(&c.title, &para.context, cfg) {
                Ok(d) => d,
                Err(_) => continue,
            };
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
                let ctx = match doc.context(&qa.question) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let assembled = ctx.text();
                let ctx_words: HashSet<String> = words(&assembled).into_iter().collect();
                let recall = span_recall(gold, &ctx_words);
                acc.add(recall, ctx.report.total_tokens);
                q_count += 1;
            }
        }
    }
    Ok(acc)
}

fn main() -> anyhow::Result<()> {
    let path = std::env::var("REDHOP_CUAD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("cuad/cuad_sample.json"));
    let raw = std::fs::read_to_string(&path)?;
    let cuad: Cuad = serde_json::from_str(&raw)?;

    println!(
        "CUAD chunk-size × strategy sweep  ·  budget={BUDGET}  ·  contracts={}",
        cuad.data.len()
    );
    println!(
        "Baseline to beat (FRAMEWORK_COMPARISON.md): LlamaIndex 86% ≥0.8 @ 1806 tokens, RedHop[topk] 82% @ 1894 tokens."
    );
    println!();

    let strategies = [
        ContextStrategy::RawTopK,
        ContextStrategy::MaxDensity,
        ContextStrategy::DistractorFiltered,
        ContextStrategy::ReasoningPreserving,
    ];

    let mut best: (usize, ContextStrategy, f64) = (0, ContextStrategy::RawTopK, 0.0);

    for &strategy in &strategies {
        println!("── strategy = {} ──", strategy_name(strategy));
        println!(
            "  {:<8} {:>10} {:>10} {:>10} {:>12}",
            "target", "recall", "≥0.5", "≥0.8", "avg_tokens"
        );
        for &t in CHUNK_TARGETS {
            let cell = run_cell(&cuad, t, strategy)?;
            let p80 = cell.pct_80();
            if p80 > best.2 {
                best = (t, strategy, p80);
            }
            let marker = if p80 >= 86.0 {
                " ✓ matches LlamaIndex"
            } else if p80 >= 82.0 {
                " ≈ matches RedHop[topk]"
            } else {
                ""
            };
            println!(
                "  {:<8} {:>10.3} {:>9.0}% {:>9.0}% {:>12.0}{}",
                t,
                cell.mean_recall(),
                cell.pct_50(),
                cell.pct_80(),
                cell.mean_tokens(),
                marker
            );
        }
        println!();
    }

    println!("══ verdict ══");
    println!(
        "  best cell: target_tokens={} · strategy={} · ≥0.8 retention = {:.1}%",
        best.0,
        strategy_name(best.1),
        best.2
    );
    let llamaindex_bar = 86.0;
    let redhop_baseline = 82.0;
    if best.2 >= llamaindex_bar {
        println!("  ✓ MATCHES OR BEATS LlamaIndex baseline (86%).");
    } else if best.2 >= redhop_baseline + 1.0 {
        println!(
            "  ~ Improves over current RedHop baseline (82%) by {:+.1} points but still below LlamaIndex.",
            best.2 - redhop_baseline
        );
    } else {
        println!(
            "  ✗ No combination clears the LlamaIndex bar; current default is approximately optimal."
        );
    }
    Ok(())
}
