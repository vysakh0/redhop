//! CUAD diagnostic: does stripping the boilerplate template from each query
//! close the 4-point gap to LlamaIndex?
//!
//! CUAD questions are 100% identical-length templates:
//!
//!   "Highlight the parts (if any) of this contract related to \"X\"
//!    that should be reviewed by a lawyer. Details: <elaboration>"
//!
//! Every question is exactly 24 words. The actual *discriminating signal* is
//! the quoted clause name (`"X"`) plus the `Details:` elaboration — maybe 5
//! content words. The other 19+ words are boilerplate shared by every query.
//!
//! BM25 computes term-frequency-weighted relevance over the WHOLE query, so
//! the boilerplate dilutes the discriminating signal. CUAD's gold answer
//! spans are specific phrases from contract text; the boilerplate terms
//! ("highlight", "contract", "lawyer", "Details", …) match nothing useful.
//!
//! HotpotQA questions, by contrast, are diverse natural language (15.7 mean
//! word count, max 37) with no shared boilerplate. The "BM25 silent-wildcard
//! fallback" fix that helped HotpotQA by +3 points couldn't help CUAD because
//! CUAD queries DO have signal — it's just buried.
//!
//! Hypothesis: extract just the quoted clause name + Details elaboration as
//! the BM25 query, run the same retrieval + assembly, see if we close the
//! gap. If yes: the gap is dilution, and a small query-preprocessor closes
//! it. If no: the gap is somewhere else (chunking semantics, retrieval
//! algorithm), and we need to look there.
//!
//! Apples-to-apples with bench/compare.py: same 300-question slice, same
//! 2000 token budget, candidate_k=40, RawTopK strategy, default chunker.
//!
//! Run: cargo run -p redhop-examples --example cuad_query_preprocessing --release

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

/// Set-based gold-word recall, matching `bench/compare.py`'s `span_recall`
/// exactly: `len(g & ctx) / len(g)` where both g and ctx are SETS of unique
/// content words. An earlier Vec-based version of this function in this
/// branch double-counted duplicate gold words and inflated recall by ~2
/// points on CUAD (gold spans are legal contract clauses with high
/// repetition). The set-based version is apples-to-apples with the
/// framework comparison.
fn span_recall(gold: &str, ctx_words: &HashSet<String>) -> f32 {
    let g: HashSet<String> = words(gold).into_iter().collect();
    if g.is_empty() {
        return 1.0;
    }
    let hit = g.iter().filter(|w| ctx_words.contains(*w)).count();
    hit as f32 / g.len() as f32
}

/// Extract the discriminating signal from a CUAD template query.
///
/// Template:  `Highlight the parts (if any) of this contract related to
///             "X" that should be reviewed by a lawyer. Details: <elab>`
///
/// We pull out:
///   - the contents of the first quoted string (the clause name, `X`)
///   - everything after `Details:` (the elaboration)
///
/// Returns just those, space-joined. Falls back to the original question if
/// the template doesn't match (so non-CUAD questions pass through unchanged).
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
        // Not the CUAD template — preserve original.
        return q.to_string();
    }
    format!("{clause} {details}").trim().to_string()
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
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

const BUDGET: usize = 2000;
const LIMIT_Q: usize = 300;
const CANDIDATE_K: usize = 40;

fn run(cuad: &Cuad, strip_template: bool) -> anyhow::Result<Cell> {
    let mut acc = Cell::default();
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
                let query_text = if strip_template {
                    extract_cuad_signal(&qa.question)
                } else {
                    qa.question.clone()
                };
                let ctx = match doc.context(&query_text) {
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

    println!("CUAD query-preprocessing diagnostic — does stripping the template close the gap?");
    println!();
    println!("Sample template query (full):");
    let sample = &cuad.data[0].paragraphs[0].qas[0].question;
    println!("  {sample}");
    println!("After extracting clause + details:");
    println!("  \"{}\"", extract_cuad_signal(sample));
    println!();

    let baseline = run(&cuad, false)?;
    println!("── arm A: original CUAD queries (the 24-word template) ──");
    println!(
        "  n={}, mean recall={:.3}, ≥0.5={:.0}%, ≥0.8={:.0}%, avg tokens={:.0}",
        baseline.n,
        baseline.sum_recall / baseline.n as f64,
        pct(baseline.retained_50, baseline.n),
        pct(baseline.retained_80, baseline.n),
        baseline.sum_final_tokens / baseline.n as f64,
    );

    let stripped = run(&cuad, true)?;
    println!();
    println!("── arm B: template stripped (clause name + Details only) ──");
    println!(
        "  n={}, mean recall={:.3}, ≥0.5={:.0}%, ≥0.8={:.0}%, avg tokens={:.0}",
        stripped.n,
        stripped.sum_recall / stripped.n as f64,
        pct(stripped.retained_50, stripped.n),
        pct(stripped.retained_80, stripped.n),
        stripped.sum_final_tokens / stripped.n as f64,
    );

    let base80 = pct(baseline.retained_80, baseline.n);
    let strip80 = pct(stripped.retained_80, stripped.n);
    let delta = strip80 - base80;
    println!();
    println!("── verdict ══");
    println!(
        "  Δ ≥0.8 retention: {:+.1} points  (baseline {:.1}% → stripped {:.1}%)",
        delta, base80, strip80
    );
    if strip80 >= 86.0 {
        println!("  ✓ Stripping the template CLOSES the gap to LlamaIndex (86%).");
        println!("    Implication: CUAD's gap was BM25 template-boilerplate dilution. A small");
        println!(
            "    query-preprocessor would ship the fix; no chunking or strategy change needed."
        );
    } else if delta >= 1.0 {
        println!("  ~ Stripping helps by +{delta:.1} points but doesn't reach LlamaIndex (86%).");
        println!("    Implication: boilerplate dilution is PART of the gap. Other levers remain.");
    } else {
        println!(
            "  ✗ Stripping doesn't move recall. Boilerplate dilution isn't the gap mechanism."
        );
        println!("    Implication: look elsewhere — chunking semantics, retrieval algorithm.");
    }
    Ok(())
}
