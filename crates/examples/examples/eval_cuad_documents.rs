//! Tier-1 real-document eval: the `Document` runtime on CUAD contracts.
//!
//! CUAD (Contract Understanding Atticus Dataset) is real commercial contracts
//! (~9k tokens each) with clause questions and **gold answer spans**. We run
//! the actual product path — `Document::from_text → context(query)` — and
//! measure, with NO LLM (free, local, deterministic):
//!
//!   - token reduction: full contract → assembled context
//!   - evidence retention: did the assembled context keep the gold span?
//!     (word-recall of the gold answer in the final context — robust to chunk
//!     boundaries splitting a long clause)
//!   - the Auto decision distribution (passthrough vs prune)
//!   - latency (document build + per-query)
//!
//! To separate *retrieval* loss from *pruning* loss we run two configs on the
//! same contracts:
//!   A. retrieval-only ceiling: top-k candidates, no budget pruning
//!   B. default Document:       top-k candidates + Auto pruning to budget
//! The gap between A and B is what pruning costs in evidence retention.
//!
//! Data: `data/cuad/cuad_sample.json` (50 contracts, answerable QAs). Point
//! `REDHOP_CUAD_PATH` at the full `CUADv1.json` to run the whole set.
//!
//! Run:  cargo run -p redhop-examples --example eval_cuad_documents --release

use std::time::Instant;

use redhop_context::{ContextConfig, ContextStrategy};
use redhop_document::{Document, DocumentConfig};
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

/// Tier-2 robustness perturbations applied to the raw contract text before
/// ingestion — messy-corpus stress tests. Deterministic.
///   "dup" : triplicate the document (duplicated corpus)
///   "ocr" : split ~15% of long words mid-word (OCR fragmentation that breaks
///           lexical tokens — the honest worst case for a lexical retriever)
fn perturb(text: &str, mode: &str) -> String {
    match mode {
        "dup" => format!("{text}\n\n{text}\n\n{text}"),
        "ocr" => {
            let mut rng = 0x9E3779B97F4A7C15u64;
            let mut next = || {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                rng >> 33
            };
            text.split(' ')
                .map(|w| {
                    let ch: Vec<char> = w.chars().collect();
                    if ch.len() > 4 && next() % 100 < 15 {
                        let cut = 1 + (next() as usize % (ch.len() - 1));
                        format!(
                            "{} {}",
                            ch[..cut].iter().collect::<String>(),
                            ch[cut..].iter().collect::<String>()
                        )
                    } else {
                        w.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => text.to_string(),
    }
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .map(|w| w.to_string())
        .collect()
}

/// Fraction of the gold span's words present in the assembled context. Robust
/// to a long clause being split across chunk boundaries (a binary substring
/// test would under-count those).
fn span_recall(gold: &str, ctx_words: &std::collections::HashSet<String>) -> f32 {
    let g = words(gold);
    if g.is_empty() {
        return 1.0;
    }
    let hit = g.iter().filter(|w| ctx_words.contains(*w)).count();
    hit as f32 / g.len() as f32
}

fn p(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)]
}

#[derive(Default)]
struct Acc {
    n: usize,
    sum_recall: f64,
    retained_50: usize,
    retained_80: usize,
    passthrough: usize,
    prune: usize,
    sum_final_tokens: f64,
    latencies_ms: Vec<f64>,
}
impl Acc {
    fn add_recall(&mut self, r: f32) {
        self.n += 1;
        self.sum_recall += r as f64;
        if r >= 0.5 {
            self.retained_50 += 1;
        }
        if r >= 0.8 {
            self.retained_80 += 1;
        }
    }
}

fn cfg_retrieval_only(candidate_k: usize) -> DocumentConfig {
    // No pruning: top-k candidates passed straight through (RawTopK, unbounded
    // budget). Measures the retrieval ceiling.
    DocumentConfig {
        candidate_k,
        context: ContextConfig {
            strategy: ContextStrategy::RawTopK,
            token_budget: usize::MAX,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::var("REDHOP_CUAD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("cuad/cuad_sample.json"));
    let raw = std::fs::read_to_string(&path)?;
    let cuad: Cuad = serde_json::from_str(&raw)?;
    let candidate_k = DocumentConfig::default().candidate_k;
    // Tier-2 robustness: REDHOP_CUAD_PERTURB = none (default) | dup | ocr.
    let mode = std::env::var("REDHOP_CUAD_PERTURB").unwrap_or_else(|_| "none".into());
    // Strategy override for the end-to-end path (default = Auto).
    let strategy = match std::env::var("REDHOP_DOC_STRATEGY").as_deref() {
        Ok("raw_topk") => ContextStrategy::RawTopK,
        Ok("distractor_filtered") => ContextStrategy::DistractorFiltered,
        Ok("redundancy_pruned") => ContextStrategy::RedundancyPruned,
        Ok("max_density") => ContextStrategy::MaxDensity,
        Ok("reasoning_preserving") => ContextStrategy::ReasoningPreserving,
        _ => ContextStrategy::Auto,
    };
    let e2e_cfg = || DocumentConfig {
        context: ContextConfig {
            strategy,
            ..DocumentConfig::default().context
        },
        ..DocumentConfig::default()
    };

    let mut end_to_end = Acc::default();
    let mut retrieval_only = Acc::default();
    let mut sum_contract_tokens = 0.0f64;
    let mut n_contracts = 0usize;
    let mut build_ms: Vec<f64> = Vec::new();

    for c in &cuad.data {
        for para in &c.paragraphs {
            let context = perturb(&para.context, &mode);
            // Default Document (Auto + budget) — the product path.
            let t0 = Instant::now();
            let mut doc = match Document::from_text_with(&c.title, &context, e2e_cfg()) {
                Ok(d) => d,
                Err(_) => continue,
            };
            // Build the retrieval-only twin once too.
            let mut doc_ret =
                Document::from_text_with(&c.title, &context, cfg_retrieval_only(candidate_k))?;
            build_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

            let contract_tokens = doc.total_tokens();
            sum_contract_tokens += contract_tokens as f64;
            n_contracts += 1;

            for qa in &para.qas {
                // End-to-end (default Auto).
                let t = Instant::now();
                let ctx = doc.context(&qa.question)?;
                end_to_end
                    .latencies_ms
                    .push(t.elapsed().as_secs_f64() * 1000.0);
                let ctx_words: std::collections::HashSet<String> =
                    words(&ctx.text()).into_iter().collect();
                let r = qa
                    .answers
                    .iter()
                    .map(|a| span_recall(&a.text, &ctx_words))
                    .fold(0.0f32, f32::max);
                end_to_end.add_recall(r);
                end_to_end.sum_final_tokens += ctx.report.total_tokens as f64;
                match ctx.report.auto_decision() {
                    redhop_context::AutoDecision::Passthrough => end_to_end.passthrough += 1,
                    redhop_context::AutoDecision::Prune => end_to_end.prune += 1,
                    redhop_context::AutoDecision::NotAuto => {}
                }

                // Retrieval-only ceiling.
                let ctx_r = doc_ret.context(&qa.question)?;
                let cr_words: std::collections::HashSet<String> =
                    words(&ctx_r.text()).into_iter().collect();
                let rr = qa
                    .answers
                    .iter()
                    .map(|a| span_recall(&a.text, &cr_words))
                    .fold(0.0f32, f32::max);
                retrieval_only.add_recall(rr);
            }
        }
    }

    end_to_end
        .latencies_ms
        .sort_by(|a, b| a.partial_cmp(b).unwrap());
    build_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = end_to_end.n.max(1) as f64;

    println!("CUAD Document eval (no LLM) — {}", path.display());
    println!("  perturbation: {mode}   strategy: {strategy:?}");
    println!(
        "  contracts: {n_contracts}   answerable queries: {}",
        end_to_end.n
    );
    println!(
        "  candidate_k: {candidate_k}   budget: {} tok\n",
        DocumentConfig::default().context.token_budget
    );

    println!("Token economics (end-to-end, the product path)");
    let avg_contract = sum_contract_tokens / n_contracts.max(1) as f64;
    let avg_final = end_to_end.sum_final_tokens / q;
    println!("  avg full-contract tokens:  {avg_contract:.0}");
    println!("  avg assembled tokens:      {avg_final:.0}");
    println!(
        "  end-to-end reduction:      {:+.0}%\n",
        100.0 * (avg_final - avg_contract) / avg_contract
    );

    println!("Auto decisions");
    println!(
        "  passthrough: {}   prune: {}\n",
        end_to_end.passthrough, end_to_end.prune
    );

    println!("Evidence retention (gold-span word-recall in the assembled context)");
    println!(
        "  retrieval ceiling (top-{candidate_k}, no prune):  mean {:.2}   ≥0.5 {:.0}%   ≥0.8 {:.0}%",
        retrieval_only.sum_recall / retrieval_only.n.max(1) as f64,
        100.0 * retrieval_only.retained_50 as f64 / retrieval_only.n.max(1) as f64,
        100.0 * retrieval_only.retained_80 as f64 / retrieval_only.n.max(1) as f64,
    );
    println!(
        "  end-to-end (top-{candidate_k} + Auto prune):      mean {:.2}   ≥0.5 {:.0}%   ≥0.8 {:.0}%",
        end_to_end.sum_recall / q,
        100.0 * end_to_end.retained_50 as f64 / q,
        100.0 * end_to_end.retained_80 as f64 / q,
    );
    println!("  → the gap is what pruning costs in evidence retention\n");

    println!("Latency");
    println!(
        "  doc build (chunk+index):   p50 {:.1}ms   p95 {:.1}ms",
        p(&build_ms, 0.5),
        p(&build_ms, 0.95)
    );
    println!(
        "  per-query context():       p50 {:.1}ms   p95 {:.1}ms",
        p(&end_to_end.latencies_ms, 0.5),
        p(&end_to_end.latencies_ms, 0.95)
    );
    Ok(())
}
