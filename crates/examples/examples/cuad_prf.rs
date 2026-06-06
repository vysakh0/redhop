//! CUAD query expansion via pseudo-relevance feedback (PRF) — NULL RESULT.
//!
//! Tests whether PRF on top of template-stripping lifts past the
//! template-stripped baseline from CUAD_RECALL_GAP. It does not. See
//! docs/findings/CUAD_PRF_NULL.md for the full finding, mechanism, and
//! sweep results.
//!
//! PRF mechanism (classic Rocchio-style, BM25-friendly):
//!   1. First pass — retrieve a small context (~500 tok) using the stripped
//!      query → the "expansion pool" of top chunks.
//!   2. Term mining — tokenize the pool, drop stop-words and original-query
//!      terms, take the top-N most frequent content terms.
//!   3. Second pass — augment the stripped query with those terms; retrieve
//!      the full 2000-tok context with the augmented query; measure recall
//!      against the gold span.
//!
//! Three arms so the comparison is apples-to-apples with the rest of the
//! CUAD harnesses:
//!   - arm A: raw 24-word template            (the ~81% baseline)
//!   - arm B: template stripped               (the ~88% baseline from CUAD_RECALL_GAP)
//!   - arm C: template stripped + PRF         (this experiment)
//!
//! Same setup as bench/compare.py + the other CUAD harnesses: n=300, BM25,
//! budget=2000, candidate_k=40, RawTopK, set-based span_recall.
//!
//! Knobs (env vars):
//!   REDHOP_PRF_POOL   first-pass token budget for the expansion pool (default 500)
//!   REDHOP_PRF_N      number of expansion terms to append             (default 8)
//!
//! Run: cargo run -p redhop-examples --example cuad_prf --release

use std::collections::{HashMap, HashSet};

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

/// Set-based, matches bench/compare.py.
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

/// Small English stop list + CUAD-template residuals as a safety net.
const STOP: &[&str] = &[
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "are",
    "from",
    "any",
    "all",
    "not",
    "but",
    "you",
    "your",
    "our",
    "its",
    "his",
    "her",
    "their",
    "they",
    "have",
    "has",
    "had",
    "was",
    "were",
    "been",
    "being",
    "into",
    "such",
    "which",
    "who",
    "whom",
    "what",
    "when",
    "where",
    "why",
    "how",
    "than",
    "then",
    "there",
    "these",
    "those",
    "between",
    "about",
    "above",
    "below",
    "under",
    "over",
    "without",
    "within",
    "upon",
    "out",
    "off",
    "per",
    "via",
    "would",
    "could",
    "should",
    "must",
    "ought",
    "highlight",
    "parts",
    "contract",
    "related",
    "reviewed",
    "lawyer",
    "details",
];

/// Top-N most frequent content terms from the first-pass pool, with
/// stop-words and original-query terms filtered.
fn prf_expansion_terms(pool_text: &str, original_query: &str, n_terms: usize) -> Vec<String> {
    let stop: HashSet<&str> = STOP.iter().copied().collect();
    let query_words: HashSet<String> = words(original_query).into_iter().collect();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for w in words(pool_text) {
        if w.len() < 3 {
            continue;
        }
        if stop.contains(w.as_str()) {
            continue;
        }
        if query_words.contains(&w) {
            continue;
        }
        *counts.entry(w).or_insert(0) += 1;
    }
    let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs.into_iter().take(n_terms).map(|(w, _)| w).collect()
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

fn prf_pool_budget() -> usize {
    std::env::var("REDHOP_PRF_POOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500)
}
fn prf_n_terms() -> usize {
    std::env::var("REDHOP_PRF_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
}

#[derive(Copy, Clone, PartialEq)]
enum Arm {
    Raw,
    Stripped,
    StrippedPrf,
}

fn run(cuad: &Cuad, arm: Arm) -> anyhow::Result<Cell> {
    let mut acc = Cell::default();
    let mut q_count = 0usize;
    let pool_budget = prf_pool_budget();
    let n_terms = prf_n_terms();

    'outer: for c in &cuad.data {
        for para in &c.paragraphs {
            let cfg_full = DocumentConfig {
                candidate_k: CANDIDATE_K,
                context: ContextConfig {
                    strategy: ContextStrategy::RawTopK,
                    token_budget: BUDGET,
                    ..DocumentConfig::default().context
                },
                ..DocumentConfig::default()
            };
            let cfg_pool = DocumentConfig {
                candidate_k: CANDIDATE_K,
                context: ContextConfig {
                    strategy: ContextStrategy::RawTopK,
                    token_budget: pool_budget,
                    ..DocumentConfig::default().context
                },
                ..DocumentConfig::default()
            };

            let mut doc_full = match Document::from_text_with(&c.title, &para.context, cfg_full) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let mut doc_pool = if arm == Arm::StrippedPrf {
                match Document::from_text_with(&c.title, &para.context, cfg_pool) {
                    Ok(d) => d,
                    Err(_) => continue,
                }
            } else {
                Document::from_text("__dummy__", "placeholder").expect("dummy doc")
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

                let stripped = extract_cuad_signal(&qa.question);
                let query_text = match arm {
                    Arm::Raw => qa.question.clone(),
                    Arm::Stripped => stripped.clone(),
                    Arm::StrippedPrf => {
                        let pool = match doc_pool.context(&stripped) {
                            Ok(c) => c.text(),
                            Err(_) => String::new(),
                        };
                        let terms = prf_expansion_terms(&pool, &stripped, n_terms);
                        if terms.is_empty() {
                            stripped.clone()
                        } else {
                            format!("{} {}", stripped, terms.join(" "))
                        }
                    }
                };

                let ctx = match doc_full.context(&query_text) {
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

fn print_arm(label: &str, c: &Cell) {
    println!("── {label} ──");
    println!(
        "  n={}, mean recall={:.3}, ≥0.5={:.0}%, ≥0.8={:.0}%, avg tokens={:.0}",
        c.n,
        c.sum_recall / c.n.max(1) as f64,
        pct(c.retained_50, c.n),
        pct(c.retained_80, c.n),
        c.sum_final_tokens / c.n.max(1) as f64,
    );
}

fn main() -> anyhow::Result<()> {
    let path = std::env::var("REDHOP_CUAD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("cuad/cuad_sample.json"));
    let raw = std::fs::read_to_string(&path)?;
    let cuad: Cuad = serde_json::from_str(&raw)?;

    let pool_budget = prf_pool_budget();
    let n_terms = prf_n_terms();
    println!("CUAD PRF probe — does pseudo-relevance feedback lift past template stripping?");
    println!("  config: n=300, BM25, budget={BUDGET}, candidate_k={CANDIDATE_K}, RawTopK, set-based span_recall");
    println!("  PRF:    first-pass pool budget={pool_budget}, expansion terms N={n_terms}");
    println!("          (override via REDHOP_PRF_POOL and REDHOP_PRF_N env vars)");
    println!();

    let sample_q = &cuad.data[0].paragraphs[0].qas[0].question;
    let sample_stripped = extract_cuad_signal(sample_q);
    println!("sample query:");
    println!("  raw:      {sample_q}");
    println!("  stripped: \"{sample_stripped}\"");
    {
        let cfg_pool = DocumentConfig {
            candidate_k: CANDIDATE_K,
            context: ContextConfig {
                strategy: ContextStrategy::RawTopK,
                token_budget: pool_budget,
                ..DocumentConfig::default().context
            },
            ..DocumentConfig::default()
        };
        let mut d =
            Document::from_text_with("sample", &cuad.data[0].paragraphs[0].context, cfg_pool)?;
        let pool = d.context(&sample_stripped)?.text();
        let terms = prf_expansion_terms(&pool, &sample_stripped, n_terms);
        println!("  prf expansion terms: {:?}", terms);
        println!("  arm-C query: \"{} {}\"", sample_stripped, terms.join(" "));
    }
    println!();

    let arm_a = run(&cuad, Arm::Raw)?;
    print_arm("arm A: raw 24-word template", &arm_a);
    println!();

    let arm_b = run(&cuad, Arm::Stripped)?;
    print_arm("arm B: template stripped", &arm_b);
    println!();

    let arm_c = run(&cuad, Arm::StrippedPrf)?;
    print_arm("arm C: template stripped + PRF", &arm_c);
    println!();

    let a80 = pct(arm_a.retained_80, arm_a.n);
    let b80 = pct(arm_b.retained_80, arm_b.n);
    let c80 = pct(arm_c.retained_80, arm_c.n);
    let delta_ba = b80 - a80;
    let delta_cb = c80 - b80;
    let delta_ca = c80 - a80;
    println!("══ verdict ══");
    println!(
        "  ≥0.8 retention: A={:.1}%  B={:.1}%  C={:.1}%",
        a80, b80, c80
    );
    println!(
        "  ΔB−A = {:+.1}  (template stripping)   ΔC−B = {:+.1}  (PRF on top of strip)   ΔC−A = {:+.1}",
        delta_ba, delta_cb, delta_ca
    );
    if delta_cb >= 1.5 {
        println!(
            "  ✓ PRF gives a meaningful lift on top of template stripping (+{delta_cb:.1} pts)."
        );
    } else if delta_cb >= 0.5 {
        println!(
            "  ~ PRF gives a small lift (+{delta_cb:.1} pts) — not clearly worth the complexity."
        );
    } else if delta_cb > -0.5 {
        println!("  ✗ PRF is flat ({delta_cb:+.1} pts). Mechanism didn't help on this workload.");
    } else {
        println!(
            "  ✗ PRF REGRESSED ({delta_cb:+.1} pts). Expansion terms diluted the signal back."
        );
        println!("    See docs/findings/CUAD_PRF_NULL.md for the full mechanism explanation.");
    }
    Ok(())
}
