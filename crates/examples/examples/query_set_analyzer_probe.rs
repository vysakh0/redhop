//! Cross-workload probe for [`redhop::analyze_query_set`].
//!
//! The mechanism we want to validate before shipping `analyze_query_set` as
//! public API is the **false-positive vs true-positive trade-off**: on a
//! known-templated workload (CUAD) it should fire; on diverse natural-language
//! workloads (HotpotQA, MuSiQue) it should stay quiet. If both hold we have
//! evidence the heuristic generalizes; if HotpotQA or MuSiQue light up we
//! have a false-positive problem and the analyzer is too aggressive.
//!
//! No models, no embeddings — this is a string-only diagnostic, so the probe
//! parses the dataset JSON / JSONL directly without the
//! `redhop_examples::loaders::*` machinery (which is `required-features =
//! ["onnx"]`).
//!
//! Run: cargo run -p redhop-examples --example query_set_analyzer_probe --release

use std::path::Path;

use redhop::analyzer::{analyze_query_set, DilutionCost, QuerySetReport};
use serde::Deserialize;

// ─── CUAD loader ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct CuadFile {
    data: Vec<CuadContract>,
}
#[derive(Deserialize)]
struct CuadContract {
    paragraphs: Vec<CuadPara>,
}
#[derive(Deserialize)]
struct CuadPara {
    qas: Vec<CuadQa>,
}
#[derive(Deserialize)]
struct CuadQa {
    question: String,
}

fn load_cuad_questions(path: &Path, limit: usize) -> anyhow::Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)?;
    let f: CuadFile = serde_json::from_str(&raw)?;
    let mut out = Vec::new();
    for c in &f.data {
        for p in &c.paragraphs {
            for q in &p.qas {
                out.push(q.question.clone());
                if out.len() >= limit {
                    return Ok(out);
                }
            }
        }
    }
    Ok(out)
}

// ─── HotpotQA loader (the same JSON shape used by adaptive_eval_hotpotqa) ──

#[derive(Deserialize)]
struct HotpotItem {
    question: String,
}

fn load_hotpot_questions(path: &Path, limit: usize) -> anyhow::Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)?;
    let v: Vec<HotpotItem> = serde_json::from_str(&raw)?;
    Ok(v.into_iter().take(limit).map(|i| i.question).collect())
}

// ─── MuSiQue loader (JSONL) ───────────────────────────────────────────────

#[derive(Deserialize)]
struct MusiqueItem {
    question: String,
}

fn load_musique_questions(path: &Path, limit: usize) -> anyhow::Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let item: MusiqueItem = serde_json::from_str(line)?;
        out.push(item.question);
        if out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

// ─── Reporting helpers ────────────────────────────────────────────────────

fn cost_label(c: DilutionCost) -> &'static str {
    match c {
        DilutionCost::High => "High",
        DilutionCost::Medium => "Medium",
        DilutionCost::Low => "Low",
        DilutionCost::None => "None",
    }
}

fn print_workload(label: &str, queries: &[String], report: &QuerySetReport, expected_flag: bool) {
    println!("── {label} (n={}) ──", queries.len());
    println!("  sample queries:");
    for q in queries.iter().take(3) {
        let snippet: String = q.chars().take(110).collect();
        println!("    · {snippet}{}", if q.len() > snippet.len() { "…" } else { "" });
    }
    println!(
        "  template_word_share: {:.3}    cost: {}",
        report.template_word_share,
        cost_label(report.estimated_dilution_cost)
    );
    let top: Vec<String> = report
        .boilerplate_terms
        .iter()
        .take(15)
        .cloned()
        .collect();
    println!("  boilerplate_terms (top {}): {:?}", top.len(), top);
    println!("  is_templated:        {}", report.is_templated);
    println!("  suggested_action:    {}", report.suggested_action);

    // Verdict for the probe: did it match what we expected?
    let pass = report.is_templated == expected_flag;
    let symbol = if pass { "✓" } else { "✗" };
    let verdict = if expected_flag {
        "true positive expected"
    } else {
        "should NOT have flagged this workload"
    };
    println!("  {symbol} probe: {verdict}");
}

const SAMPLE_N: usize = 300;

fn main() -> anyhow::Result<()> {
    println!(
        "redhop::analyze_query_set — cross-workload probe (n={SAMPLE_N} per workload)\n"
    );

    // True-positive workload: CUAD (every query is the 24-word fixed template).
    let cuad_path = std::env::var("REDHOP_CUAD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("cuad/cuad_sample.json"));
    let cuad_q = load_cuad_questions(&cuad_path, SAMPLE_N)?;
    let cuad_report = analyze_query_set(&cuad_q);
    print_workload("CUAD — expected: templated", &cuad_q, &cuad_report, true);
    println!();

    // False-positive checks: diverse natural-language multi-hop workloads.
    let hotpot_path = redhop_examples::data_path("hotpotqa/hotpot_dev_distractor_v1.json");
    let hotpot_q = load_hotpot_questions(&hotpot_path, SAMPLE_N)?;
    let hotpot_report = analyze_query_set(&hotpot_q);
    print_workload(
        "HotpotQA — expected: NOT templated",
        &hotpot_q,
        &hotpot_report,
        false,
    );
    println!();

    let musique_path = redhop_examples::data_path("musique/dev.jsonl");
    let musique_q = load_musique_questions(&musique_path, SAMPLE_N)?;
    let musique_report = analyze_query_set(&musique_q);
    print_workload(
        "MuSiQue — expected: NOT templated",
        &musique_q,
        &musique_report,
        false,
    );
    println!();

    // ── Probe summary ─────────────────────────────────────────────────────
    println!("══ probe summary ══");
    let tp = cuad_report.is_templated;
    let fp_hotpot = hotpot_report.is_templated;
    let fp_musique = musique_report.is_templated;

    println!(
        "  CUAD       (templated):     is_templated={:>5}   share={:.3}   cost={}",
        tp,
        cuad_report.template_word_share,
        cost_label(cuad_report.estimated_dilution_cost)
    );
    println!(
        "  HotpotQA   (NOT templated): is_templated={:>5}   share={:.3}   cost={}",
        fp_hotpot,
        hotpot_report.template_word_share,
        cost_label(hotpot_report.estimated_dilution_cost)
    );
    println!(
        "  MuSiQue    (NOT templated): is_templated={:>5}   share={:.3}   cost={}",
        fp_musique,
        musique_report.template_word_share,
        cost_label(musique_report.estimated_dilution_cost)
    );

    let pass_tp = tp;
    let pass_fp = !fp_hotpot && !fp_musique;
    if pass_tp && pass_fp {
        println!();
        println!("  ✓ Heuristic ships: true-positive on CUAD, no false positives on diverse workloads.");
    } else {
        println!();
        if !pass_tp {
            println!("  ✗ CUAD did NOT fire — the heuristic missed the canonical true positive.");
        }
        if fp_hotpot {
            println!("  ✗ HotpotQA fired as templated (false positive).");
        }
        if fp_musique {
            println!("  ✗ MuSiQue fired as templated (false positive).");
        }
        println!("    Reconsider thresholds before shipping `analyze_query_set` as public API.");
    }
    Ok(())
}
