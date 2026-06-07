//! CUAD clause-name expansion probe.
//!
//! Tests whether adding workload-specific high-IDF synonyms to a
//! (template-stripped) CUAD query lifts ≥0.8 retention past the 88%
//! template-strip baseline established in CUAD_RECALL_GAP.md.
//!
//! The hypothesis: after template stripping, the query is something like
//! `"Change of Control" The right of either party to terminate`. The
//! discriminating clause name (`Change of Control`) is in the query, but
//! the gold span often uses related-but-not-identical terms (`merger`,
//! `successor`, `acquisition`). Adding those as static synonyms via
//! `redhop::expand_query_terms` should raise the BM25 score of the
//! gold-bearing chunk.
//!
//! The mechanism direction (additive query expansion of high-IDF terms)
//! is the OPPOSITE of unweighted PRF, which failed because it added
//! corpus boilerplate (low IDF) — see CUAD_PRF_NULL.md. The synonyms
//! here are hand-curated, high-IDF by construction (`merger` is rare in
//! a generic contract corpus and specific to M&A clauses), so the
//! mechanism prediction is favorable. The probe MEASURES whether the
//! prediction holds.
//!
//! Four arms:
//!   - arm A: raw 24-word template                       (~81% baseline)
//!   - arm B: template stripped                          (~88% baseline)
//!   - arm C: template stripped + clause-name expanded   (this experiment)
//!   - arm D: raw template + clause-name expanded — control. Does
//!     expansion help even on the unstripped query, or only when paired
//!     with strip?
//!
//! Same setup as bench/compare.py and the other CUAD harnesses: n=300,
//! BM25, budget=2000, candidate_k=40, RawTopK, set-based span_recall.
//!
//! Discipline note: the CUAD clause-name dictionary lives in this file
//! and ONLY in this file. RedHop ships `expand_query_terms` (the
//! mechanism); the dict is workload-specific user data and must not leak
//! into the library. The dict is hand-crafted from inspection of CUAD's
//! known 41 clause types and what kinds of terms typically appear in
//! their gold spans.
//!
//! Run: cargo run -p redhop-examples --example cuad_clause_expansion --release

use std::collections::HashSet;

use redhop::{Vocabulary, QueryRewrite};
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

/// Hand-curated CUAD clause-name → synonyms dictionary.
///
/// Covers ~20 of the 41 CUAD clause types — the ones with clearly
/// identifiable high-IDF synonym terms. Synonyms are chosen based on
/// what tends to appear in the gold answer span text. Terms that are
/// already in every contract (`agreement`, `party`, `shall`) are
/// deliberately excluded — those are exactly the dilution failure mode
/// from CUAD_PRF_NULL.
///
/// This dict lives in the example, NOT in the library. Production users
/// of `expand_query_terms` would supply their own workload-specific dict.
fn cuad_clause_synonyms() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        ("change of control", &["merger", "successor", "acquisition", "consolidation", "stockholders"][..]),
        ("anti-assignment", &["assign", "transfer", "successors", "delegate"]),
        ("non-compete", &["restraint", "compete", "competitive", "compete", "competing"]),
        ("non-disparagement", &["disparage", "criticize", "negative", "statement"]),
        ("exclusivity", &["exclusive", "sole", "exclusively"]),
        ("most favored nation", &["mfn", "favored", "comparable", "better"]),
        ("no-solicit", &["solicit", "solicitation", "recruit", "hire"]),
        ("right of first refusal", &["rofr", "refusal", "first option", "preemptive"]),
        ("right of first offer", &["rofo", "first offer", "preemptive"]),
        ("termination for convenience", &["convenience", "without cause", "any reason"]),
        ("renewal term", &["renew", "extend", "extension", "renewable"]),
        ("notice period to terminate renewal", &["notice", "days notice", "written notice"]),
        ("governing law", &["governed", "construed", "jurisdiction", "venue", "law of"]),
        ("ip ownership assignment", &["assign", "ownership", "title", "intellectual property"]),
        ("joint ip ownership", &["jointly", "co-own", "joint ownership"]),
        ("license grant", &["grants", "license", "licensee", "licensor"]),
        ("uncapped liability", &["unlimited", "uncapped", "no limit"]),
        ("cap on liability", &["capped", "limited", "maximum", "shall not exceed"]),
        ("liquidated damages", &["liquidated", "damages", "penalty"]),
        ("warranty duration", &["warrants", "warranty", "warranted"]),
        ("insurance", &["insure", "insured", "coverage", "policy"]),
        ("audit rights", &["audit", "inspect", "books and records"]),
        ("source code escrow", &["escrow", "deposit", "source code"]),
        ("third party beneficiary", &["beneficiary", "third party", "intended"]),
        ("covenant not to sue", &["release", "waiver", "covenant", "sue"]),
        ("revenue sharing", &["royalty", "percentage", "revenue", "share"]),
        ("price restrictions", &["pricing", "price", "rate", "fees"]),
        ("minimum commitment", &["minimum", "commit", "guarantee"]),
        ("volume restriction", &["volume", "quantity", "cap"]),
        ("document name", &["title", "this agreement", "this contract"]),
        ("agreement date", &["dated", "as of", "executed", "effective date"]),
        ("effective date", &["effective", "commence", "begin"]),
        ("expiration date", &["expire", "terminate", "ends"]),
        ("parties", &["between", "and"]),
    ]
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

// CUAD template boilerplate, used by the `Stripped` arm so it can drop the
// 24-word wrapper. Matches what the analyzer surfaces for templated CUAD.
const CUAD_BOILERPLATE: &[&str] = &[
    "highlight", "the", "parts", "if", "any", "of", "this", "contract",
    "related", "to", "that", "should", "be", "reviewed", "by", "a", "lawyer",
    "details",
];

#[derive(Copy, Clone, PartialEq)]
enum Arm {
    RawTemplate,
    Stripped,
    StrippedExpanded,
    /// Control: expand on the RAW template, not the stripped one. Tests
    /// whether expansion helps independently of strip, or only in
    /// combination with it.
    RawExpanded,
}

fn run(cuad: &Cuad, arm: Arm, vocabulary: &Vocabulary) -> anyhow::Result<Cell> {
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
                let query_text = match arm {
                    Arm::RawTemplate => qa.question.clone(),
                    Arm::Stripped => extract_cuad_signal(&qa.question),
                    Arm::StrippedExpanded => {
                        let stripped = extract_cuad_signal(&qa.question);
                        vocabulary.apply(&stripped).query
                    }
                    Arm::RawExpanded => vocabulary.apply(&qa.question).query,
                };
                // (`extract_cuad_signal` is the canonical CUAD template
                // strip — matches the prior CUAD harnesses' definitions.
                // For non-CUAD workloads, use `redhop::Stripper` to compile
                // a boilerplate list once instead of writing a per-call
                // signal extractor.)
                let _ = CUAD_BOILERPLATE;
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

    println!(
        "CUAD clause-name expansion probe — does adding high-IDF synonyms push past 88%?"
    );
    println!(
        "  config: n={LIMIT_Q}, BM25, budget={BUDGET}, candidate_k={CANDIDATE_K}, RawTopK, set-based span_recall"
    );
    let dict = cuad_clause_synonyms();
    let total_syns: usize = dict.iter().map(|(_, s)| s.len()).sum();
    println!(
        "  dict: {} clause-name keys, {} total synonyms",
        dict.len(),
        total_syns
    );
    println!();

    // Compile the dictionary once — token-level matching via the
    // document's default analyzer. With the new `Vocabulary` API, lookup
    // happens at retrieval-call rate, not per-construction.
    let vocabulary = Vocabulary::new(&dict);

    // Sample expansion on the first query — also exercise the audit
    // trail so the worked example shows what the Decision Report will
    // record.
    let sample_q = &cuad.data[0].paragraphs[0].qas[0].question;
    let sample_stripped = extract_cuad_signal(sample_q);
    let sample_result = vocabulary.apply(&sample_stripped);
    println!("sample query:");
    println!("  raw:      {sample_q}");
    println!("  stripped: {sample_stripped}");
    println!("  expanded: {}", sample_result.query);
    println!(
        "  trail:    matched={:?} added={:?}",
        sample_result.record.matched, sample_result.record.added
    );
    println!();

    let arm_a = run(&cuad, Arm::RawTemplate, &vocabulary)?;
    print_arm("arm A: raw 24-word template", &arm_a);
    println!();

    let arm_b = run(&cuad, Arm::Stripped, &vocabulary)?;
    print_arm("arm B: template stripped", &arm_b);
    println!();

    let arm_c = run(&cuad, Arm::StrippedExpanded, &vocabulary)?;
    print_arm("arm C: stripped + clause-name expanded", &arm_c);
    println!();

    let arm_d = run(&cuad, Arm::RawExpanded, &vocabulary)?;
    print_arm("arm D: raw template + clause-name expanded (control)", &arm_d);
    println!();

    let a80 = pct(arm_a.retained_80, arm_a.n);
    let b80 = pct(arm_b.retained_80, arm_b.n);
    let c80 = pct(arm_c.retained_80, arm_c.n);
    let d80 = pct(arm_d.retained_80, arm_d.n);
    let delta_cb = c80 - b80;
    let delta_da = d80 - a80;
    println!("══ verdict ══");
    println!("  ≥0.8 retention:");
    println!("    A (raw template):              {a80:>5.1}%");
    println!("    B (stripped):                  {b80:>5.1}%");
    println!("    C (stripped + expanded):       {c80:>5.1}%   ΔC−B = {delta_cb:+.1}");
    println!("    D (raw + expanded, control):   {d80:>5.1}%   ΔD−A = {delta_da:+.1}");
    println!();
    if delta_cb >= 1.5 {
        println!("  ✓ Clause-name expansion gives a meaningful lift on top of template stripping (+{delta_cb:.1} pts).");
        println!("    Mechanism confirmed: additive high-IDF synonyms close additional gap.");
        if (delta_cb - delta_da).abs() < 1.0 {
            println!("    Note: control arm (D) shows similar lift, so expansion helps");
            println!("    independently of stripping. Either alone is useful.");
        } else if delta_cb > delta_da + 1.0 {
            println!("    Strip + expand combines productively (strip removes noise,");
            println!("    expand adds signal — orthogonal mechanisms).");
        }
    } else if delta_cb >= 0.5 {
        println!("  ~ Small lift (+{delta_cb:.1} pts) — within sample noise without CIs.");
        println!("    Worth a bootstrap CI before claiming the effect on this workload.");
    } else if delta_cb > -0.5 {
        println!("  ✗ Flat ({delta_cb:+.1} pts). Mechanism didn't help on this workload at this dict size.");
    } else {
        println!("  ✗ Regressed ({delta_cb:+.1} pts). The added synonyms hurt more than they helped.");
        println!("    Likely the dict synonyms are not actually high-IDF in CUAD's contract corpus.");
    }
    Ok(())
}
