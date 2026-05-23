//! Does context distractor ratio actually hurt answer quality?
//!
//! The premise behind context economics is "distractors hurt." This
//! tests it on REAL LLM outputs — the NeoTrace export from the Python
//! lab, where each (query, method) row carries both the retrieval
//! distractor_ratio AND the measured answer quality (ans_kw_recall =
//! gold-keyword recall in the answer, ans_similarity = answer↔gold
//! cosine). No model needed here; it reads the recorded measurements.
//!
//! Reports Pearson correlations across the HotpotQA + MuSiQue cross-LLM
//! runs:
//!   distractor_ratio      vs answer quality   (expect NEGATIVE)
//!   answer_span_density   vs answer quality   (expect POSITIVE)
//!
//! Run:
//!   cargo run -p redhop-examples --example distractor_answer_correlation --release

use redhop_calibration::{corruption::pearson, loaders::neotrace::parse_path};

const FILES: &[(&str, &str)] = &[
    (
        "HotpotQA (haiku)",
        "/Users/vysakh/projects/neorag/exports/neotrace/hotpot_full.neotrace.jsonl",
    ),
    (
        "HotpotQA (llama8b)",
        "/Users/vysakh/projects/neorag/exports/neotrace/hotpot_llama8b.neotrace.jsonl",
    ),
    (
        "MuSiQue (haiku)",
        "/Users/vysakh/projects/neorag/exports/neotrace/musique_full.neotrace.jsonl",
    ),
    (
        "MuSiQue (qwen7b)",
        "/Users/vysakh/projects/neorag/exports/neotrace/musique_qwen7b.neotrace.jsonl",
    ),
    (
        "MuSiQue (mistralnemo)",
        "/Users/vysakh/projects/neorag/exports/neotrace/musique_mistralnemo.neotrace.jsonl",
    ),
    (
        "evidence study",
        "/Users/vysakh/projects/neorag/exports/neotrace/evidence_evidence.neotrace.jsonl",
    ),
];

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Distractor ratio & evidence density vs answer quality           ║");
    println!("║  (real LLM outputs from the Python lab, via NeoTrace)            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!(
        "  {:<24} {:>5} {:>16} {:>16} {:>16}",
        "dataset (model)", "n", "distr→kw_recall", "distr→ans_sim", "density→kw_recall"
    );
    println!("  {}", "─".repeat(82));

    let mut all_distr = Vec::new();
    let mut all_kw = Vec::new();
    let mut all_sim = Vec::new();
    let mut all_density = Vec::new();

    for (label, path) in FILES {
        let records = match parse_path(path) {
            Ok(r) => r,
            Err(e) => {
                println!("  {label:<24} (skip: {e})");
                continue;
            }
        };
        let mut distr = Vec::new();
        let mut kw = Vec::new();
        let mut sim = Vec::new();
        let mut density = Vec::new();
        for r in &records {
            // Only rows that carry both a distractor signal and answer
            // quality (Hotpot/MuSiQue/evidence all do).
            if let (Some(d), Some(k)) = (r.distractor_ratio, r.ans_kw_recall) {
                distr.push(d);
                kw.push(k);
            }
            if let (Some(d), Some(s)) = (r.distractor_ratio, r.ans_similarity) {
                let _ = d;
                sim.push(s);
            }
            if let (Some(asd), Some(k)) = (r.answer_span_density, r.ans_kw_recall) {
                density.push((asd, k));
            }
        }
        // distractor → kw_recall (paired)
        let (dx, kwy): (Vec<f32>, Vec<f32>) = records
            .iter()
            .filter_map(|r| match (r.distractor_ratio, r.ans_kw_recall) {
                (Some(d), Some(k)) => Some((d, k)),
                _ => None,
            })
            .unzip();
        let (dx2, simy): (Vec<f32>, Vec<f32>) = records
            .iter()
            .filter_map(|r| match (r.distractor_ratio, r.ans_similarity) {
                (Some(d), Some(s)) => Some((d, s)),
                _ => None,
            })
            .unzip();
        let (denx, deny): (Vec<f32>, Vec<f32>) = records
            .iter()
            .filter_map(|r| match (r.answer_span_density, r.ans_kw_recall) {
                (Some(d), Some(k)) => Some((d, k)),
                _ => None,
            })
            .unzip();

        let c_distr_kw = pearson(&dx, &kwy);
        let c_distr_sim = pearson(&dx2, &simy);
        let c_den_kw = pearson(&denx, &deny);

        println!(
            "  {:<24} {:>5} {:>+16.3} {:>+16.3} {:>+16.3}",
            label,
            dx.len(),
            c_distr_kw,
            c_distr_sim,
            c_den_kw
        );

        all_distr.extend(dx);
        all_kw.extend(kwy);
        all_sim.extend(simy);
        all_density.extend(denx.into_iter().zip(deny));
    }

    // Pooled.
    let pooled_distr_kw = pearson(&all_distr, &all_kw);
    let (den_x, den_y): (Vec<f32>, Vec<f32>) = all_density.into_iter().unzip();
    let pooled_den_kw = pearson(&den_x, &den_y);

    println!("\n──── pooled across all real LLM runs ────");
    println!(
        "  distractor_ratio    → ans_kw_recall:  {:+.3}",
        pooled_distr_kw
    );
    println!(
        "  answer_span_density → ans_kw_recall:  {:+.3}",
        pooled_den_kw
    );

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("VERDICT (real LLM outputs, not synthetic):");
    if pooled_distr_kw < -0.05 {
        println!("  ✓ distractors HURT answer quality: distractor_ratio negatively");
        println!(
            "    correlates with gold-keyword recall in the LLM's answer ({:+.3}).",
            pooled_distr_kw
        );
    } else if pooled_distr_kw.abs() <= 0.05 {
        println!(
            "  ~ distractor_ratio is weakly correlated with answer quality ({:+.3}).",
            pooled_distr_kw
        );
    } else {
        println!(
            "  ✗ distractor_ratio POSITIVELY correlates ({:+.3}) — unexpected; report honestly.",
            pooled_distr_kw
        );
    }
    if pooled_den_kw > 0.05 {
        println!("  ✓ evidence density HELPS: answer_span_density positively correlates");
        println!(
            "    with answer quality ({:+.3}) — confirms the context-economics premise.",
            pooled_den_kw
        );
    }
    println!("════════════════════════════════════════════════════════════════════════");
    Ok(())
}
