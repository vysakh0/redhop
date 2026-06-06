//! Isolating the 2-point Rust-vs-Python CUAD parity gap.
//!
//! `bench/compare.py` calls `redhop.Document.from_text(text, strategy=…,
//! token_budget=…, candidate_k=…)` which routes through the pyo3 binding's
//! `build_text_doc` → `RhDocument::from_sources_with(vec![(source, vec![Section
//! { text, … }])], cfg)`.
//!
//! My CUAD sweep calls `Document::from_text_with(source, text, cfg)`
//! directly, which chunks the raw text via `chunker.chunk(&SourceDoc::new(…))`
//! without going through `chunk_sections`.
//!
//! Both produce identical recall ON CHUNKS — the chunk text and counts come
//! out the same. But the *metadata* attached differs:
//!
//!   from_text_with  : chunks have NO `metadata["kind"]`.
//!   from_sources_with: chunks have `metadata["kind"] = "prose"` stamped.
//!
//! Neither `code_neighbors_default` (checks kind=="code") nor
//! `prose_heading_default` (checks heading != empty) should fire on CUAD
//! prose without headings. So this should not matter for retention.
//!
//! This harness puts both paths side-by-side on the same contract + 50
//! queries to isolate exactly where the 2-point difference enters: chunk
//! count, chunk text, retrieval order, or assembled context contents.
//!
//! Run: cargo run -p redhop-examples --example cuad_rust_vs_python_path --release

use std::collections::HashSet;

use redhop::context::{ContextConfig, ContextStrategy};
use redhop::document::{Document, DocumentConfig, Section};
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

/// Vec-based recall — bench/compare.py's `span_recall` calls `words(s)` which
/// returns a SET, so its denominator is the count of UNIQUE gold words. Our
/// `cuad_chunk_strategy_sweep` and friends use this Vec version which counts
/// duplicate-word matches as if they were independent. The result is a
/// 1-3-point discrepancy whenever the gold answer text has repeated content
/// words (extremely common in legal contract spans).
fn span_recall_vec(gold: &str, ctx_words: &HashSet<String>) -> f32 {
    let g = words(gold);
    if g.is_empty() {
        return 1.0;
    }
    let hit = g.iter().filter(|w| ctx_words.contains(*w)).count();
    hit as f32 / g.len() as f32
}

/// Set-based recall — matches bench/compare.py exactly: `len(g & cw) / len(g)`
/// where both `g` and `cw` are sets.
fn span_recall_set(gold: &str, ctx_words: &HashSet<String>) -> f32 {
    let g: HashSet<String> = words(gold).into_iter().collect();
    if g.is_empty() {
        return 1.0;
    }
    let hit = g.iter().filter(|w| ctx_words.contains(*w)).count();
    hit as f32 / g.len() as f32
}

const BUDGET: usize = 2000;
const LIMIT_Q: usize = 300;
const CANDIDATE_K: usize = 40;

fn cfg() -> DocumentConfig {
    DocumentConfig {
        candidate_k: CANDIDATE_K,
        context: ContextConfig {
            strategy: ContextStrategy::RawTopK,
            token_budget: BUDGET,
            ..DocumentConfig::default().context
        },
        ..DocumentConfig::default()
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::var("REDHOP_CUAD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("cuad/cuad_sample.json"));
    let raw = std::fs::read_to_string(&path)?;
    let cuad: Cuad = serde_json::from_str(&raw)?;

    println!("Rust direct (`Document::from_text_with`) vs Python's underlying path");
    println!("(`Document::from_sources_with(vec![(source, vec![Section{{text,…}}])])`)");
    println!();

    // ── Build BOTH paths on the first contract — what differs in chunk counts? ──
    let c = &cuad.data[0];
    let para = &c.paragraphs[0];

    // Path A: from_text_with — what the CUAD sweep uses
    let mut doc_a = Document::from_text_with(&c.title, &para.context, cfg())?;
    // Path B: from_sources_with — what Python's from_text routes through
    let section_b = Section {
        text: para.context.clone(),
        page: None,
        heading: None,
        line: None,
    };
    let mut doc_b = Document::from_sources_with(vec![(c.title.clone(), vec![section_b])], cfg())?;

    println!("── chunk-level comparison on contract: {}", c.title);
    println!("  Path A (from_text_with):     {} chunks", doc_a.len());
    println!("  Path B (from_sources_with):  {} chunks", doc_b.len());
    println!();

    if doc_a.len() == doc_b.len() {
        println!("  Chunk counts match. Inspecting kind/heading/metadata on each chunk…");
        let a_chunks = doc_a.embedded_chunks().unwrap_or_default();
        let b_chunks = doc_b.embedded_chunks().unwrap_or_default();
        let mut text_differs = 0;
        let mut metadata_differs = 0;
        for (a, b) in a_chunks.iter().zip(b_chunks.iter()) {
            if a.text != b.text {
                text_differs += 1;
            }
            if a.metadata != b.metadata {
                metadata_differs += 1;
            }
        }
        println!(
            "    chunks with different TEXT:     {text_differs}/{}",
            a_chunks.len()
        );
        println!(
            "    chunks with different METADATA: {metadata_differs}/{}",
            a_chunks.len()
        );

        if let (Some(a0), Some(b0)) = (a_chunks.first(), b_chunks.first()) {
            println!();
            println!("    sample metadata diff (chunk 0):");
            println!("      Path A: {:?}", a0.metadata);
            println!("      Path B: {:?}", b0.metadata);
        }
    } else {
        println!("  ✗ chunk counts DIFFER — that's the smoking gun.");
    }

    // ── Run the same 300-query slice through both, see if retention differs ──
    println!();
    println!("── 300-question recall comparison ──");

    let mut q_count = 0usize;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut hit_a = 0usize;
    let mut hit_b = 0usize;
    let mut n = 0usize;

    'outer: for c in &cuad.data {
        for para in &c.paragraphs {
            let mut doc_a = match Document::from_text_with(&c.title, &para.context, cfg()) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let section = Section {
                text: para.context.clone(),
                page: None,
                heading: None,
                line: None,
            };
            let mut doc_b =
                match Document::from_sources_with(vec![(c.title.clone(), vec![section])], cfg()) {
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
                let ctx_a = match doc_a.context(&qa.question) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let ctx_b = match doc_b.context(&qa.question) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let words_a: HashSet<String> = words(&ctx_a.text()).into_iter().collect();
                let words_b: HashSet<String> = words(&ctx_b.text()).into_iter().collect();
                let r_a = span_recall_vec(gold, &words_a);
                let r_b = span_recall_vec(gold, &words_b);
                sum_a += r_a as f64;
                sum_b += r_b as f64;
                if r_a >= 0.8 {
                    hit_a += 1;
                }
                if r_b >= 0.8 {
                    hit_b += 1;
                }
                n += 1;
                q_count += 1;
            }
        }
    }

    println!(
        "  Path A (from_text_with)     n={n}  mean recall={:.3}  ≥0.8={:.1}% (Vec metric)",
        sum_a / n as f64,
        100.0 * hit_a as f64 / n as f64
    );
    println!(
        "  Path B (from_sources_with)  n={n}  mean recall={:.3}  ≥0.8={:.1}% (Vec metric)",
        sum_b / n as f64,
        100.0 * hit_b as f64 / n as f64
    );
    let delta = (hit_a as f64 - hit_b as f64) / n as f64 * 100.0;
    println!("  Δ ≥0.8 retention (A − B):  {:+.2} points", delta);

    // ── Apples-to-apples with bench/compare.py: SET-based span_recall ──
    println!();
    println!("── re-run with SET-based span_recall (matches bench/compare.py exactly) ──");
    let mut q_count = 0usize;
    let mut sum_a_set = 0.0f64;
    let mut hit_a_set = 0usize;
    let mut n_set = 0usize;
    'outer2: for c in &cuad.data {
        for para in &c.paragraphs {
            let mut doc = match Document::from_text_with(&c.title, &para.context, cfg()) {
                Ok(d) => d,
                Err(_) => continue,
            };
            for qa in &para.qas {
                if q_count >= LIMIT_Q {
                    break 'outer2;
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
                let cw: HashSet<String> = words(&ctx.text()).into_iter().collect();
                let r = span_recall_set(gold, &cw);
                sum_a_set += r as f64;
                if r >= 0.8 {
                    hit_a_set += 1;
                }
                n_set += 1;
                q_count += 1;
            }
        }
    }
    println!(
        "  Path A with SET metric (apples-to-apples bench): n={n_set}  mean={:.3}  ≥0.8={:.1}%",
        sum_a_set / n_set as f64,
        100.0 * hit_a_set as f64 / n_set as f64
    );
    println!(
        "  bench/compare.py headline (redhop[topk]):                                 ≥0.8=82.0%"
    );

    if delta.abs() <= 0.5 {
        println!();
        println!("  → Both code paths are equivalent on CUAD. The Python-vs-Rust 2-point gap");
        println!("    is somewhere ELSE (probably bench/compare.py's metric impl, or the pyo3");
        println!("    binding's string-to-strategy routing). Worth a separate probe.");
    } else {
        println!();
        println!("  → The two code paths DIVERGE. Investigate the chunk-level diff above.");
    }
    Ok(())
}
