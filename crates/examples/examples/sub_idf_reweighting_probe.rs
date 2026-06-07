//! Sub-IDF reweighting probe — can the library auto-drop low-IDF terms?
//!
//! [CUAD_CLAUSE_EXPANSION](docs/findings/CUAD_CLAUSE_EXPANSION.md) showed
//! that manipulating the IDF profile of the query closes the CUAD gap:
//! adding high-IDF synonyms lifts +2.7 points. The natural symmetric
//! follow-up: **can the library automatically drop low-IDF terms from
//! the query using corpus statistics, with no user-supplied dict?**
//!
//! Two failure modes to rule out simultaneously:
//!   - **CUAD true positive.** A templated workload on a boilerplate-
//!     heavy corpus should benefit. If the auto-drop doesn't lift CUAD,
//!     the mechanism is null.
//!   - **HotpotQA / MuSiQue false-positive regression.** Diverse natural-
//!     language queries are short (5-15 words); dropping ANY of them
//!     could destroy retrieval. If diverse workloads regress, the
//!     mechanism has to be a user-opt-in, not an auto-default.
//!
//! Mechanism: for each query, look up each token's **chunk-document
//! frequency** in the corpus (what fraction of chunks contain the token).
//! If a token appears in more than `threshold_share` of chunks, it's low
//! IDF and unlikely to discriminate — drop it. This is a probe-side
//! approximation of true Tantivy IDF (which uses BM25's smoothed log
//! formula); the direction of the signal is the same.
//!
//! Same setup as the other CUAD harnesses: n=300, BM25, budget=2000,
//! candidate_k=40, RawTopK, set-based span_recall (matches
//! bench/compare.py). For HotpotQA + MuSiQue, the per-question
//! distractor pool is concatenated into a single Document; gold span =
//! the supporting paragraph(s).
//!
//! Run: cargo run -p redhop-examples --example sub_idf_reweighting_probe --release

use std::collections::{HashMap, HashSet};

use redhop::context::{ContextConfig, ContextStrategy};
use redhop::document::{Document, DocumentConfig};
use serde::Deserialize;

// ─── shared helpers ────────────────────────────────────────────────────────

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

/// Drop tokens whose corpus document-frequency share is above `cap_share`.
/// "Above cap" means the token appears in too many chunks to discriminate
/// — auto-detected stop word.
fn drop_low_idf(query: &str, chunk_texts: &[String], cap_share: f32) -> String {
    if cap_share >= 1.0 {
        return query.to_string();
    }
    let n = chunk_texts.len();
    if n == 0 {
        return query.to_string();
    }
    let mut df: HashMap<String, usize> = HashMap::new();
    for ct in chunk_texts {
        let seen: HashSet<String> = words(ct).into_iter().collect();
        for w in seen {
            *df.entry(w).or_insert(0) += 1;
        }
    }
    let cap = ((cap_share * n as f32).ceil() as usize).max(1);

    let filtered: Vec<&str> = query
        .split_whitespace()
        .filter(|tok| {
            let key: String = tok
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect();
            if key.len() < 2 {
                return true; // preserve short tokens (likely structural)
            }
            df.get(&key).copied().unwrap_or(0) <= cap
        })
        .collect();
    // Safety: if filtering emptied the query, return original (don't tank
    // the workload by handing BM25 an empty string).
    if filtered.is_empty() {
        return query.to_string();
    }
    filtered.join(" ")
}

const BUDGET: usize = 2000;
const HOTPOT_BUDGET: usize = 400; // matches bench/compare.py
const CANDIDATE_K: usize = 40;
const LIMIT_Q: usize = 300;
const THRESHOLDS: &[(f32, &str)] = &[
    (1.0, "none (control)"),
    (0.70, "drop df > 70%"),
    (0.50, "drop df > 50%"),
    (0.30, "drop df > 30%"),
];

#[derive(Default, Clone, Copy)]
struct Cell {
    n: usize,
    sum_recall: f64,
    retained_80: usize,
    sum_query_len_tokens: f64,
}

impl Cell {
    fn add(&mut self, r: f32, query_text: &str) {
        self.n += 1;
        self.sum_recall += r as f64;
        if r >= 0.8 {
            self.retained_80 += 1;
        }
        self.sum_query_len_tokens += query_text.split_whitespace().count() as f64;
    }
    fn r80(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            100.0 * self.retained_80 as f64 / self.n as f64
        }
    }
    fn mean_q_len(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum_query_len_tokens / self.n as f64
        }
    }
}

// ─── CUAD loader ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct CuadFile {
    data: Vec<CuadContract>,
}
#[derive(Deserialize)]
struct CuadContract {
    title: String,
    paragraphs: Vec<CuadPara>,
}
#[derive(Deserialize)]
struct CuadPara {
    context: String,
    qas: Vec<CuadQa>,
}
#[derive(Deserialize)]
struct CuadQa {
    question: String,
    answers: Vec<CuadAnswer>,
}
#[derive(Deserialize)]
struct CuadAnswer {
    text: String,
}

fn run_cuad(budget: usize) -> anyhow::Result<Vec<Cell>> {
    let path = std::env::var("REDHOP_CUAD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("cuad/cuad_sample.json"));
    let raw = std::fs::read_to_string(&path)?;
    let cuad: CuadFile = serde_json::from_str(&raw)?;

    let mut cells = vec![Cell::default(); THRESHOLDS.len()];
    let mut q_count = 0usize;

    'outer: for c in &cuad.data {
        for para in &c.paragraphs {
            let cfg = DocumentConfig {
                candidate_k: CANDIDATE_K,
                context: ContextConfig {
                    strategy: ContextStrategy::RawTopK,
                    token_budget: budget,
                    ..DocumentConfig::default().context
                },
                ..DocumentConfig::default()
            };
            let mut doc = match Document::from_text_with(&c.title, &para.context, cfg) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let chunk_texts: Vec<String> = doc.chunks().iter().map(|c| c.text.clone()).collect();

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
                for (i, (cap, _)) in THRESHOLDS.iter().enumerate() {
                    let filtered = drop_low_idf(&qa.question, &chunk_texts, *cap);
                    let ctx = match doc.context(&filtered) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    let assembled = ctx.text();
                    let ctx_words: HashSet<String> = words(&assembled).into_iter().collect();
                    let recall = span_recall(gold, &ctx_words);
                    cells[i].add(recall, &filtered);
                }
                q_count += 1;
            }
        }
    }
    Ok(cells)
}

// ─── HotpotQA loader (concatenate the per-question distractor pool) ────────

#[derive(Deserialize)]
struct HotpotItem {
    question: String,
    answer: String,
    context: Vec<HotpotContext>,
}
#[derive(Deserialize)]
struct HotpotContext(String, Vec<String>); // (title, sentences)

fn run_hotpot(budget: usize) -> anyhow::Result<Vec<Cell>> {
    let path = redhop_examples::data_path("hotpotqa/hotpot_dev_distractor_v1.json");
    let raw = std::fs::read_to_string(&path)?;
    let items: Vec<HotpotItem> = serde_json::from_str(&raw)?;

    let mut cells = vec![Cell::default(); THRESHOLDS.len()];

    for item in items.iter().take(LIMIT_Q) {
        // Concatenate all paragraphs into one "document" for the question.
        let mut doc_text = String::new();
        for HotpotContext(title, sentences) in &item.context {
            doc_text.push_str(title);
            doc_text.push_str(". ");
            for s in sentences {
                doc_text.push_str(s);
                doc_text.push(' ');
            }
            doc_text.push('\n');
        }
        let cfg = DocumentConfig {
            candidate_k: CANDIDATE_K,
            context: ContextConfig {
                strategy: ContextStrategy::RawTopK,
                token_budget: budget,
                ..DocumentConfig::default().context
            },
            ..DocumentConfig::default()
        };
        let mut doc = match Document::from_text_with("hotpot", &doc_text, cfg) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let chunk_texts: Vec<String> = doc.chunks().iter().map(|c| c.text.clone()).collect();
        // Use the answer as the gold (HotpotQA gold spans are the answer
        // text, embedded in supporting sentences).
        let gold = item.answer.as_str();
        if gold.is_empty() || gold == "yes" || gold == "no" {
            continue; // boolean answers don't have lexical gold to test
        }
        for (i, (cap, _)) in THRESHOLDS.iter().enumerate() {
            let filtered = drop_low_idf(&item.question, &chunk_texts, *cap);
            let ctx = match doc.context(&filtered) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let assembled = ctx.text();
            let ctx_words: HashSet<String> = words(&assembled).into_iter().collect();
            let recall = span_recall(gold, &ctx_words);
            cells[i].add(recall, &filtered);
        }
    }
    Ok(cells)
}

// ─── MuSiQue loader (JSONL, per-question paragraph pool) ──────────────────

#[derive(Deserialize)]
struct MusiqueItem {
    question: String,
    answer: String,
    paragraphs: Vec<MusiquePara>,
}
#[derive(Deserialize)]
struct MusiquePara {
    title: String,
    paragraph_text: String,
}

fn run_musique(budget: usize) -> anyhow::Result<Vec<Cell>> {
    let path = redhop_examples::data_path("musique/dev.jsonl");
    let raw = std::fs::read_to_string(&path)?;
    let mut cells = vec![Cell::default(); THRESHOLDS.len()];

    let mut n = 0;
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if n >= LIMIT_Q {
            break;
        }
        let item: MusiqueItem = match serde_json::from_str(line) {
            Ok(it) => it,
            Err(_) => continue,
        };
        let gold = item.answer.as_str();
        if gold.is_empty() {
            continue;
        }
        let mut doc_text = String::new();
        for p in &item.paragraphs {
            doc_text.push_str(&p.title);
            doc_text.push_str(". ");
            doc_text.push_str(&p.paragraph_text);
            doc_text.push_str("\n\n");
        }
        let cfg = DocumentConfig {
            candidate_k: CANDIDATE_K,
            context: ContextConfig {
                strategy: ContextStrategy::RawTopK,
                token_budget: budget,
                ..DocumentConfig::default().context
            },
            ..DocumentConfig::default()
        };
        let mut doc = match Document::from_text_with("musique", &doc_text, cfg) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let chunk_texts: Vec<String> = doc.chunks().iter().map(|c| c.text.clone()).collect();
        for (i, (cap, _)) in THRESHOLDS.iter().enumerate() {
            let filtered = drop_low_idf(&item.question, &chunk_texts, *cap);
            let ctx = match doc.context(&filtered) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let assembled = ctx.text();
            let ctx_words: HashSet<String> = words(&assembled).into_iter().collect();
            let recall = span_recall(gold, &ctx_words);
            cells[i].add(recall, &filtered);
        }
        n += 1;
    }
    Ok(cells)
}

fn print_workload(label: &str, cells: &[Cell]) {
    println!("── {label} (n={}) ──", cells[0].n);
    println!(
        "  {:<30}  {:>10}  {:>10}",
        "threshold", "≥0.8", "mean q len"
    );
    for (cell, (_, lbl)) in cells.iter().zip(THRESHOLDS.iter()) {
        println!(
            "  {:<30}  {:>9.1}%  {:>10.1}",
            lbl,
            cell.r80(),
            cell.mean_q_len()
        );
    }
}

fn main() -> anyhow::Result<()> {
    println!("Sub-IDF reweighting probe — does auto-dropping low-IDF terms close the CUAD gap");
    println!("without regressing diverse natural-language workloads?");
    println!("  thresholds: {} cells per workload", THRESHOLDS.len());
    println!("  per workload n: up to {LIMIT_Q}");
    println!();

    println!(":: CUAD (templated workload, boilerplate-heavy corpus, budget={BUDGET}) ::");
    let cuad = run_cuad(BUDGET)?;
    print_workload("CUAD", &cuad);
    println!();

    println!(":: HotpotQA (diverse natural language, n=300, budget={HOTPOT_BUDGET}) ::");
    let hotpot = run_hotpot(HOTPOT_BUDGET)?;
    print_workload("HotpotQA", &hotpot);
    println!();

    println!(":: MuSiQue (diverse natural language, n=300, budget={HOTPOT_BUDGET}) ::");
    let musique = run_musique(HOTPOT_BUDGET)?;
    print_workload("MuSiQue", &musique);
    println!();

    // ── verdict ───────────────────────────────────────────────────────────
    let control_cuad = cuad[0].r80();
    let best_cuad = cuad
        .iter()
        .skip(1)
        .map(|c| c.r80())
        .fold(f64::MIN, f64::max);
    let cuad_lift = best_cuad - control_cuad;

    let control_hot = hotpot[0].r80();
    let worst_hot = hotpot
        .iter()
        .skip(1)
        .map(|c| c.r80())
        .fold(f64::MAX, f64::min);
    let hot_regression = control_hot - worst_hot;

    let control_mus = musique[0].r80();
    let worst_mus = musique
        .iter()
        .skip(1)
        .map(|c| c.r80())
        .fold(f64::MAX, f64::min);
    let mus_regression = control_mus - worst_mus;

    println!("══ summary ══");
    println!(
        "  CUAD     control={:.1}%   best={:.1}%   ΔCUAD = {:+.1}",
        control_cuad, best_cuad, cuad_lift
    );
    println!(
        "  HotpotQA control={:.1}%   worst={:.1}%   max regression = {:.1}",
        control_hot, worst_hot, hot_regression
    );
    println!(
        "  MuSiQue  control={:.1}%   worst={:.1}%   max regression = {:.1}",
        control_mus, worst_mus, mus_regression
    );
    println!();
    let lifts_cuad = cuad_lift >= 1.5;
    let preserves_diverse = hot_regression <= 1.0 && mus_regression <= 1.0;
    if lifts_cuad && preserves_diverse {
        println!("  ✓ CLEAN POSITIVE: sub-IDF auto-drop lifts CUAD without regressing diverse.");
        println!("    Mechanism is general. Worth shipping as an opt-in/auto API.");
    } else if lifts_cuad && !preserves_diverse {
        println!(
            "  ~ CONDITIONAL POSITIVE: lifts CUAD but regresses one or more diverse workloads."
        );
        println!("    The mechanism can't be auto-default; must be opt-in.");
    } else if !lifts_cuad && preserves_diverse {
        println!("  ✗ NULL ON CUAD, BENIGN ON DIVERSE: corpus-side auto-drop doesn't help");
        println!("    where the user-side dict-based stripping did. The win was the query-set");
        println!("    overlap, not the corpus IDF profile.");
    } else {
        println!("  ✗ FALSE BOTH WAYS: doesn't help CUAD, regresses diverse. Don't ship.");
    }
    Ok(())
}
