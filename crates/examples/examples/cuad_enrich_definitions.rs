//! CUAD chunk-side enrichment probe — does `Vocabulary.enrich()` on
//! auto-extracted Definitions sections lift ≥0.8 retention past the
//! 90.7% strip + query-side vocabulary baseline?
//!
//! ## Prior + mechanism prediction
//!
//! Honest read going in: I expect this to be **null on CUAD.** The
//! [`VOCABULARY_ENRICH`](../../../docs/findings/VOCABULARY_ENRICH.md)
//! regime rule is `value ∝ shortness × opacity × dictionary-exists`:
//! enrich earns its keep when the retrieval unit is *short and
//! opaque* (schema columns, error codes, API symbols). CUAD chunks
//! are full prose paragraphs — neither short nor opaque. The
//! mechanism likely overlaps query-side `apply` (which already brings
//! the workload to 90.7%) without adding orthogonal signal.
//!
//! The probe is a **falsification check**, not a confirmation hunt.
//! If C ≤ B, that's the finding: enrich's regime is the chunk shape,
//! not the workload's vocabulary structure. A null here strengthens
//! the regime rule by giving it an empirically-tested boundary.
//!
//! ## Definitions extraction
//!
//! Each CUAD contract typically has a Definitions section with
//! patterns like:
//!
//! ```text
//! "Change of Control" means a merger or sale of substantially all assets.
//! "Affiliate" shall mean any entity controlling, controlled by, or under common control.
//! ```
//!
//! Regex matches:
//! - `"<term>"\s+(means|shall mean|is defined as|refers to)\s+<definition>`
//!
//! Definitions are workload-specific data the library does not ship
//! — same discipline as the query-side dict in
//! [`cuad_clause_expansion`]. We extract them per-contract here as a
//! *worked example*; production users would supply their own.
//!
//! ## Three arms
//!
//! - **A:** stripped (CUAD_RECALL_GAP baseline, ~87.7%).
//! - **B:** stripped + query-side vocabulary (the shipped CUAD
//!   workflow, 90.7% in CUAD_CLAUSE_EXPANSION).
//! - **C:** stripped + query-side vocabulary + chunk-side enrich on
//!   the per-contract Definitions vocabulary.
//!
//! ΔC − B is the chunk-side mechanism's marginal contribution
//! *on top of* the query-side workflow. That's the right comparison
//! to draw the regime line, not C vs A.
//!
//! ## Configuration matches the existing CUAD harnesses
//!
//! n=300, BM25, budget=2000, candidate_k=40, RawTopK, set-based
//! `span_recall` (matches `bench/compare.py` and
//! `cuad_clause_expansion.rs`).
//!
//! Run: cargo run -p redhop-examples --example cuad_enrich_definitions --release
//!
//! [`cuad_clause_expansion`]: ./cuad_clause_expansion.rs

use std::collections::{HashMap, HashSet};

use redhop::context::{ContextConfig, ContextStrategy};
use redhop::core::{Chunk, ChunkId, TokenCount};
use redhop::document::{Document, DocumentConfig};
use redhop::{QueryRewrite, Vocabulary};
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

/// Set-based, matches `bench/compare.py`.
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

// ─── Definitions extraction ───────────────────────────────────────────────

/// Extract `"term" means …` style definitions from a contract's full
/// text. Returns a `term → definition_body_text` map.
///
/// Heuristics:
/// - Term is in straight double-quotes or curly double-quotes.
/// - Followed by "means" | "shall mean" | "is defined as" | "refers to"
///   (case-insensitive, optional intervening words like "the"/"any").
/// - Definition body runs until the next sentence-ending period that
///   is followed by whitespace + a capital letter, or until the next
///   quoted term, whichever comes first. Capped at 400 chars to
///   prevent runaway captures on malformed text.
fn extract_definitions(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let bytes = text.as_bytes();
    let lower = text.to_lowercase();

    // Find all quoted spans `"X"` (straight quotes only — CUAD's PDF
    // → text extraction normalizes curly quotes).
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let term_start = i + 1;
        // Find the matching close quote within 80 chars (defined terms
        // are short; capping prevents matching across paragraphs).
        let term_end = match bytes[term_start..(term_start + 80).min(bytes.len())]
            .iter()
            .position(|&b| b == b'"')
        {
            Some(rel) => term_start + rel,
            None => {
                i = term_start;
                continue;
            }
        };
        let term = text[term_start..term_end].trim();
        // Skip if not a plausible defined term: too long, all
        // lowercase (defined terms are capitalized), or contains
        // newlines.
        if term.is_empty()
            || term.len() > 60
            || term.contains('\n')
            || !term.chars().any(|c| c.is_uppercase())
        {
            i = term_end + 1;
            continue;
        }

        // Look at the 30 chars after the closing quote for a "means"
        // / "shall mean" / "is defined as" / "refers to" marker.
        let after = term_end + 1;
        let window_end = (after + 30).min(text.len());
        let window = &lower[after..window_end];
        let marker_offset = ["means", "shall mean", "is defined as", "refers to"]
            .iter()
            .filter_map(|m| window.find(m).map(|p| p + m.len()))
            .min();
        let body_start = match marker_offset {
            Some(off) => after + off,
            None => {
                i = term_end + 1;
                continue;
            }
        };

        // Definition body: up to 400 chars or next `". "`+capital, or
        // next `"`, whichever comes first.
        let body_max = (body_start + 400).min(text.len());
        let body_window = &text[body_start..body_max];
        let body_end_rel = body_window
            .char_indices()
            .find_map(|(idx, c)| {
                if c == '"' {
                    return Some(idx);
                }
                if c == '.'
                    && body_window.as_bytes().get(idx + 1) == Some(&b' ')
                    && body_window
                        .as_bytes()
                        .get(idx + 2)
                        .is_some_and(u8::is_ascii_uppercase)
                {
                    return Some(idx);
                }
                None
            })
            .unwrap_or(body_window.len());
        let body = body_window[..body_end_rel].trim();
        if body.len() > 8 {
            out.entry(term.to_string())
                .or_insert_with(|| body.to_string());
        }
        i = term_end + 1;
    }
    out
}

/// Build a per-contract [`Vocabulary`] from extracted definitions.
/// Each defined term maps to the words in its definition body
/// (filtered: ≥4 chars, alphabetic, no stopwords).
fn definitions_to_vocabulary(defs: &HashMap<String, String>) -> Option<Vocabulary> {
    if defs.is_empty() {
        return None;
    }
    // Filter the definition body to the high-IDF content words. We
    // drop stopwords + very short tokens to avoid bolting the same
    // low-IDF surface onto every clause chunk (that would re-create
    // the CUAD_PRF_NULL failure mode on the chunk side).
    let stop: HashSet<&str> = [
        "the", "and", "or", "of", "to", "in", "any", "an", "a", "is", "for", "with", "by", "from",
        "as", "be", "are", "this", "that", "such", "shall", "means", "mean", "all",
    ]
    .into_iter()
    .collect();
    // Owned vec so the dictionary slices outlive the Vocabulary::new call.
    let mut owned: Vec<(String, Vec<String>)> = Vec::new();
    for (term, body) in defs {
        let body_words: Vec<String> = body
            .to_lowercase()
            .split(|c: char| !c.is_alphabetic())
            .filter(|w| w.len() >= 4 && !stop.contains(w))
            .map(|w| w.to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        if body_words.is_empty() {
            continue;
        }
        owned.push((term.clone(), body_words));
    }
    if owned.is_empty() {
        return None;
    }
    // Borrow into the `&[(&str, &[&str])]` Vocabulary::new expects.
    let borrowed: Vec<(&str, Vec<&str>)> = owned
        .iter()
        .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
        .collect();
    let refs: Vec<(&str, &[&str])> = borrowed.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    Some(Vocabulary::new(&refs))
}

/// CUAD clause-name synonyms dictionary — copy-pasted from
/// [`cuad_clause_expansion`] so the probe is reproducible standalone.
/// The probe's arm B reproduces the 90.7% baseline; arm C adds the
/// chunk-side enrichment on top.
fn cuad_clause_synonyms() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            "change of control",
            &[
                "merger",
                "successor",
                "acquisition",
                "consolidation",
                "stockholders",
            ][..],
        ),
        (
            "anti-assignment",
            &["assign", "transfer", "successors", "delegate"],
        ),
        (
            "non-compete",
            &["restraint", "compete", "competitive", "competing"],
        ),
        (
            "non-disparagement",
            &["disparage", "criticize", "negative", "statement"],
        ),
        ("exclusivity", &["exclusive", "sole", "exclusively"]),
        (
            "most favored nation",
            &["mfn", "favored", "comparable", "better"],
        ),
        (
            "no-solicit",
            &["solicit", "solicitation", "recruit", "hire"],
        ),
        (
            "right of first refusal",
            &["rofr", "refusal", "first option", "preemptive"],
        ),
        (
            "right of first offer",
            &["rofo", "first offer", "preemptive"],
        ),
        (
            "termination for convenience",
            &["convenience", "without cause", "any reason"],
        ),
        (
            "renewal term",
            &["renew", "extend", "extension", "renewable"],
        ),
        (
            "notice period to terminate renewal",
            &["notice", "days notice", "written notice"],
        ),
        (
            "governing law",
            &["governed", "construed", "jurisdiction", "venue", "law of"],
        ),
        (
            "ip ownership assignment",
            &["assign", "ownership", "title", "intellectual property"],
        ),
        (
            "joint ip ownership",
            &["jointly", "co-own", "joint ownership"],
        ),
        (
            "license grant",
            &["grants", "license", "licensee", "licensor"],
        ),
        ("uncapped liability", &["unlimited", "uncapped", "no limit"]),
        (
            "cap on liability",
            &["capped", "limited", "maximum", "shall not exceed"],
        ),
        ("liquidated damages", &["liquidated", "damages", "penalty"]),
        ("warranty duration", &["warrants", "warranty", "warranted"]),
        ("insurance", &["insure", "insured", "coverage", "policy"]),
        ("audit rights", &["audit", "inspect", "books and records"]),
        ("source code escrow", &["escrow", "deposit", "source code"]),
        (
            "third party beneficiary",
            &["beneficiary", "third party", "intended"],
        ),
        (
            "covenant not to sue",
            &["release", "waiver", "covenant", "sue"],
        ),
        (
            "revenue sharing",
            &["royalty", "percentage", "revenue", "share"],
        ),
        ("price restrictions", &["pricing", "price", "rate", "fees"]),
        ("minimum commitment", &["minimum", "commit", "guarantee"]),
        ("volume restriction", &["volume", "quantity", "cap"]),
        (
            "document name",
            &["title", "this agreement", "this contract"],
        ),
        (
            "agreement date",
            &["dated", "as of", "executed", "effective date"],
        ),
        ("effective date", &["effective", "commence", "begin"]),
        ("expiration date", &["expire", "terminate", "ends"]),
        ("parties", &["between", "and"]),
    ]
}

// ─── Harness ──────────────────────────────────────────────────────────────

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

#[derive(Copy, Clone, PartialEq)]
enum Arm {
    Stripped,
    StrippedExpanded,
    StrippedExpandedEnriched,
}

/// Build a Document for one contract. Arm C reuses RedHop's chunker
/// (from_text_with) to get baseline chunks, then re-builds the
/// Document via from_chunks_with after running `vocab.enrich(chunk)`
/// on each chunk's text. The re-build preserves source/id/metadata
/// so citations stay intact.
fn build_doc(
    title: &str,
    context: &str,
    cfg: DocumentConfig,
    enrich_vocab: Option<&Vocabulary>,
) -> anyhow::Result<Document> {
    let doc = Document::from_text_with(title, context, cfg.clone())?;
    let Some(vocab) = enrich_vocab else {
        return Ok(doc);
    };
    let enriched: Vec<Chunk> = doc
        .chunks()
        .iter()
        .map(|c| {
            let new_text = vocab.enrich(&c.text).query;
            let new_tok = new_text.split_whitespace().count().max(1);
            let mut nc = Chunk::new(
                ChunkId::new(c.id.0.clone()),
                new_text,
                c.source.clone(),
                TokenCount(new_tok),
            );
            nc.metadata = c.metadata.clone();
            nc
        })
        .collect();
    Ok(Document::from_chunks_with(enriched, cfg)?)
}

fn run(cuad: &Cuad, arm: Arm, query_vocab: &Vocabulary) -> anyhow::Result<Cell> {
    let mut acc = Cell::default();
    let mut q_count = 0usize;
    let mut contracts_with_defs = 0usize;
    let mut total_def_terms = 0usize;

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

            let enrich_vocab = if matches!(arm, Arm::StrippedExpandedEnriched) {
                let defs = extract_definitions(&para.context);
                if !defs.is_empty() {
                    contracts_with_defs += 1;
                    total_def_terms += defs.len();
                }
                definitions_to_vocabulary(&defs)
            } else {
                None
            };

            let mut doc = match build_doc(&c.title, &para.context, cfg, enrich_vocab.as_ref()) {
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
                    Arm::Stripped => extract_cuad_signal(&qa.question),
                    Arm::StrippedExpanded | Arm::StrippedExpandedEnriched => {
                        let stripped = extract_cuad_signal(&qa.question);
                        query_vocab.apply(&stripped).query
                    }
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

    if matches!(arm, Arm::StrippedExpandedEnriched) {
        eprintln!(
            "    (enrichment fired on {} of {} contracts; total {} defined terms)",
            contracts_with_defs,
            cuad.data.len(),
            total_def_terms,
        );
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
        "CUAD chunk-side enrichment probe — does Definitions-section enrich add lift on top of 90.7%?"
    );
    println!(
        "  config: n={LIMIT_Q}, BM25, budget={BUDGET}, candidate_k={CANDIDATE_K}, RawTopK, set-based span_recall"
    );

    // Sample: show the first contract's extracted definitions so we can
    // sanity-check the extractor without staring at JSON.
    if let Some(first) = cuad.data.first().and_then(|c| c.paragraphs.first()) {
        let defs = extract_definitions(&first.context);
        println!(
            "\nSample definitions extracted from contract {:?} ({} terms found):",
            cuad.data[0]
                .title
                .split('_')
                .next()
                .unwrap_or(&cuad.data[0].title),
            defs.len(),
        );
        for (i, (term, body)) in defs.iter().take(4).enumerate() {
            let snippet: String = body.chars().take(120).collect();
            println!("  [{i}] {term:?} → {snippet:?}…");
        }
    }
    println!();

    // Compile the query-side vocabulary once. Same dict as
    // cuad_clause_expansion, which produces the 90.7% baseline.
    let query_vocab = Vocabulary::new(&cuad_clause_synonyms());

    let arm_a = run(&cuad, Arm::Stripped, &query_vocab)?;
    print_arm("arm A: stripped (CUAD_RECALL_GAP baseline)", &arm_a);
    println!();

    let arm_b = run(&cuad, Arm::StrippedExpanded, &query_vocab)?;
    print_arm(
        "arm B: stripped + query-side vocabulary (shipped workflow)",
        &arm_b,
    );
    println!();

    let arm_c = run(&cuad, Arm::StrippedExpandedEnriched, &query_vocab)?;
    print_arm(
        "arm C: stripped + query-vocab + chunk-side enrich (Definitions)",
        &arm_c,
    );
    println!();

    let a80 = pct(arm_a.retained_80, arm_a.n);
    let b80 = pct(arm_b.retained_80, arm_b.n);
    let c80 = pct(arm_c.retained_80, arm_c.n);
    let delta_cb = c80 - b80;
    let delta_ba = b80 - a80;
    println!("══ verdict ══");
    println!("  ≥0.8 retention:");
    println!("    A (stripped):                            {a80:>5.1}%");
    println!("    B (stripped + query-vocab):              {b80:>5.1}%   ΔB−A = {delta_ba:+.1}");
    println!("    C (stripped + query-vocab + enrich):     {c80:>5.1}%   ΔC−B = {delta_cb:+.1}");
    println!();
    if delta_cb >= 1.5 {
        println!("  ✓ Chunk-side enrich adds meaningful lift on top of the query-side workflow.");
        println!("    Surprising — the regime rule predicted null on CUAD's prose chunks.");
        println!("    Investigate: which queries benefit? Does it survive a bootstrap CI?");
    } else if delta_cb >= 0.5 {
        println!("  ~ Small lift ({delta_cb:+.1} pts) — within sample noise without CIs.");
        println!("    Borderline; worth a bootstrap CI before claiming on CUAD specifically.");
    } else if delta_cb > -0.5 {
        println!("  ✗ Flat ({delta_cb:+.1} pts). Falsification: enrich does NOT lift on CUAD.");
        println!("    Strengthens the regime rule — chunk-side helps only when chunks are");
        println!("    short and opaque. CUAD chunks are prose paragraphs; query-side vocab");
        println!("    already captures the synonym gain.");
    } else {
        println!("  ✗ Regressed ({delta_cb:+.1} pts). Enrichment is HURTING.");
        println!("    Likely the definition bodies introduce low-IDF noise — the chunk-side");
        println!("    parallel to CUAD_PRF_NULL. Check whether the stopword filter is");
        println!("    catching the dominant noise terms.");
    }
    Ok(())
}
