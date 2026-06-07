//! Spider schema-enrich probe — the positive-side validation for
//! `Vocabulary::enrich(...)`.
//!
//! ## Prior + mechanism prediction
//!
//! The [`VOCABULARY_ENRICH`](../../../docs/findings/VOCABULARY_ENRICH.md)
//! regime rule says enrich earns its keep when
//! `value ∝ shortness × opacity × dictionary-exists`. The negative side
//! was measured directly on
//! [`CUAD_ENRICH_DEFINITIONS_NULL`](../../../docs/findings/CUAD_ENRICH_DEFINITIONS_NULL.md)
//! (long prose chunks → −2.0pt regression). The positive side has been
//! mechanism-predicted but unmeasured: schema-style retrieval (short
//! opaque column names + a decoding dictionary) is the cleanest fit.
//!
//! This probe tests that prediction on a Spider-format sample. If
//! enrichment lifts retrieval on schema chunks, the regime rule has
//! evidence on its positive side. If null, the rule is wrong even
//! where it predicts strongest.
//!
//! ## Setup
//!
//! Loads `data/spider/spider_sample.json` by default (5 databases × 6
//! questions = 30 hand-labeled examples). Set `REDHOP_SPIDER_PATH` to
//! point at the full Spider tables.json + dev.json for a larger
//! measurement.
//!
//! ## Two arms
//!
//! - **A: bare column-name chunks.** Each column becomes one
//!   `Chunk(name)`. BM25 sees only the analyzer-tokenized name.
//! - **B: enriched column-name chunks.** Each column is enriched with
//!   its cleaned name (snake_case + camelCase → spaces), type, and
//!   parent-table name. Mirrors what a workload would set up from its
//!   data dictionary.
//!
//! ΔB − A is the chunk-side mechanism's lift on schema retrieval.
//!
//! ## Metric
//!
//! Per-question column recall: of the gold-labeled columns the SQL
//! answer references, how many appear in the retrieved context (any
//! position, within `candidate_k`)? Aggregated as mean recall + the
//! fraction of questions where recall ≥0.5 / ≥0.8.
//!
//! Run: cargo run -p redhop-examples --example spider_enrich_probe --release

use std::collections::HashSet;
use std::fs;

use anyhow::Context as _;
use redhop::core::{Chunk, ChunkId, TokenCount};
use redhop::{Document, Vocabulary};
use serde::Deserialize;

#[derive(Deserialize)]
struct Spider {
    databases: Vec<Database>,
    examples: Vec<Example>,
}

#[derive(Deserialize, Clone)]
struct Database {
    db_id: String,
    tables: Vec<Table>,
}

#[derive(Deserialize, Clone)]
struct Table {
    name: String,
    columns: Vec<Column>,
}

#[derive(Deserialize, Clone)]
struct Column {
    name: String,
    #[serde(rename = "type")]
    ty: String,
    /// Hand-curated natural-language synonyms — the high-IDF terms a
    /// workload's data-dictionary would attach to this column. Empty
    /// for columns whose bare name already covers their query terms.
    #[serde(default)]
    synonyms: Vec<String>,
}

#[derive(Deserialize)]
struct Example {
    question: String,
    db_id: String,
    gold_columns: Vec<String>,
}

/// Split a column name on `_` and camelCase boundaries, lowercase
/// each token. `"Singer_ID"` → `["singer", "id"]`. `"IsOfficial"` →
/// `["is", "official"]`. `"city_code"` → `["city", "code"]`.
fn explode_name(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut prev_upper = false;
    for c in s.chars() {
        if c == '_' || c == '-' || c.is_whitespace() {
            if !current.is_empty() {
                out.push(current.to_lowercase());
                current.clear();
            }
            prev_upper = false;
            continue;
        }
        if c.is_uppercase() && !current.is_empty() && !prev_upper {
            out.push(current.to_lowercase());
            current.clear();
        }
        current.push(c);
        prev_upper = c.is_uppercase();
    }
    if !current.is_empty() {
        out.push(current.to_lowercase());
    }
    out
}

/// Build the auto-derived enrichment vocabulary — what you'd get
/// without doing any data-dictionary curation: cleaned column name +
/// type + parent-table name.
fn build_auto_vocab(db: &Database) -> Vocabulary {
    let owned: Vec<(String, Vec<String>)> = db
        .tables
        .iter()
        .flat_map(|t| {
            let table_tokens = explode_name(&t.name);
            t.columns.iter().map(move |c| {
                let mut syns: Vec<String> = explode_name(&c.name);
                syns.push(c.ty.clone());
                syns.extend(table_tokens.iter().cloned());
                let mut seen: HashSet<String> = HashSet::new();
                syns.retain(|s| seen.insert(s.clone()));
                (c.name.clone(), syns)
            })
        })
        .collect();
    let borrowed: Vec<(&str, Vec<&str>)> = owned
        .iter()
        .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
        .collect();
    let refs: Vec<(&str, &[&str])> = borrowed.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    Vocabulary::new(&refs)
}

/// Build the hand-curated enrichment vocabulary — the auto layer plus
/// the workload-specific natural-language synonyms attached to each
/// column. Mirrors the worked CUAD clause-name dictionary in
/// [CUAD_CLAUSE_EXPANSION].
fn build_curated_vocab(db: &Database) -> Vocabulary {
    let owned: Vec<(String, Vec<String>)> = db
        .tables
        .iter()
        .flat_map(|t| {
            let table_tokens = explode_name(&t.name);
            t.columns.iter().map(move |c| {
                let mut syns: Vec<String> = explode_name(&c.name);
                syns.push(c.ty.clone());
                syns.extend(table_tokens.iter().cloned());
                syns.extend(c.synonyms.iter().cloned());
                let mut seen: HashSet<String> = HashSet::new();
                syns.retain(|s| seen.insert(s.clone()));
                (c.name.clone(), syns)
            })
        })
        .collect();
    let borrowed: Vec<(&str, Vec<&str>)> = owned
        .iter()
        .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
        .collect();
    let refs: Vec<(&str, &[&str])> = borrowed.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    Vocabulary::new(&refs)
}

/// Build the schema Document for one database.
///
/// Each column becomes one chunk: id `"<table>.<column>"`, source
/// `<db_id>`, text either the bare column name (Arm A) or the
/// enriched form (Arm B).
fn build_doc(db: &Database, enrich_vocab: Option<&Vocabulary>) -> anyhow::Result<Document> {
    let chunks: Vec<Chunk> = db
        .tables
        .iter()
        .flat_map(|t| {
            t.columns.iter().map(move |c| {
                let id = format!("{}.{}", t.name, c.name);
                let text = match enrich_vocab {
                    None => c.name.clone(),
                    Some(v) => v.enrich(&c.name).text,
                };
                let tokens = text.split_whitespace().count().max(1);
                Chunk::new(ChunkId::new(id), text, db.db_id.clone(), TokenCount(tokens))
            })
        })
        .collect();
    Ok(Document::from_chunks(chunks)?)
}

/// Look up the database for an example by db_id.
fn find_db<'a>(spider: &'a Spider, db_id: &str) -> Option<&'a Database> {
    spider.databases.iter().find(|d| d.db_id == db_id)
}

#[derive(Default, Clone)]
struct Cell {
    n: usize,
    sum_recall: f64,
    retained_50: usize,
    retained_80: usize,
}

impl Cell {
    fn add(&mut self, r: f32) {
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

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

#[derive(Copy, Clone, PartialEq)]
enum Arm {
    Bare,
    AutoEnriched,
    CuratedEnriched,
}

fn run(spider: &Spider, arm: Arm) -> anyhow::Result<Cell> {
    let mut acc = Cell::default();

    for ex in &spider.examples {
        let db = match find_db(spider, &ex.db_id) {
            Some(d) => d,
            None => continue,
        };
        let vocab = match arm {
            Arm::Bare => None,
            Arm::AutoEnriched => Some(build_auto_vocab(db)),
            Arm::CuratedEnriched => Some(build_curated_vocab(db)),
        };
        let mut doc = build_doc(db, vocab.as_ref())?;

        // candidate_k=10 — non-trivial pool, still much smaller than
        // each db's column count (15-30), so the gold (1-3 columns)
        // has to actually rank. Configurable via REDHOP_SPIDER_K.
        let k: usize = std::env::var("REDHOP_SPIDER_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        let ctx = match doc.context_with(&ex.question, Some(8192), Some(k)) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let retrieved_ids: HashSet<String> = ctx.chunks.iter().map(|c| c.id.0.clone()).collect();
        let gold_set: HashSet<&String> = ex.gold_columns.iter().collect();
        let hits = gold_set
            .iter()
            .filter(|g| retrieved_ids.contains(g.as_str()))
            .count();
        let recall = if gold_set.is_empty() {
            1.0
        } else {
            hits as f32 / gold_set.len() as f32
        };
        acc.add(recall);
    }

    Ok(acc)
}

fn print_arm(label: &str, c: &Cell) {
    println!("── {label} ──");
    println!(
        "  n={}, mean recall={:.3}, ≥0.5={:.0}%, ≥0.8={:.0}%",
        c.n,
        c.sum_recall / c.n.max(1) as f64,
        pct(c.retained_50, c.n),
        pct(c.retained_80, c.n),
    );
}

fn main() -> anyhow::Result<()> {
    let path = std::env::var("REDHOP_SPIDER_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("spider/spider_sample.json"));
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("read spider data {}", path.display()))?;
    let spider: Spider = serde_json::from_str(&raw).context("parse spider JSON")?;

    let k_for_log: usize = std::env::var("REDHOP_SPIDER_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    println!("Spider schema-enrich probe — positive-side validation for Vocabulary.enrich");
    println!(
        "  config: {} databases, n={} questions, BM25, candidate_k={}, set-based column recall",
        spider.databases.len(),
        spider.examples.len(),
        k_for_log,
    );

    // Sample: show what each enrichment does to one ambiguous column.
    if let Some(db) = spider
        .databases
        .iter()
        .find(|d| d.db_id == "concert_singer")
    {
        if let Some(t) = db.tables.iter().find(|t| t.name == "singer") {
            if let Some(c) = t.columns.iter().find(|c| c.name == "Age") {
                let auto = build_auto_vocab(db);
                let curated = build_curated_vocab(db);
                println!();
                println!("Sample enrichment on 'singer.Age':");
                println!("  bare           : {:?}", c.name);
                println!("  auto-enriched  : {:?}", auto.enrich(&c.name).text);
                println!("  curated-enrich : {:?}", curated.enrich(&c.name).text);
            }
        }
    }
    println!();

    let arm_a = run(&spider, Arm::Bare)?;
    print_arm("arm A: bare column-name chunks", &arm_a);
    println!();

    let arm_b = run(&spider, Arm::AutoEnriched)?;
    print_arm("arm B: auto-enriched (cleaned name + type + table)", &arm_b);
    println!();

    let arm_c = run(&spider, Arm::CuratedEnriched)?;
    print_arm("arm C: curated-enriched (auto + workload synonyms)", &arm_c);
    println!();

    let a_mean = arm_a.sum_recall / arm_a.n.max(1) as f64;
    let b_mean = arm_b.sum_recall / arm_b.n.max(1) as f64;
    let c_mean = arm_c.sum_recall / arm_c.n.max(1) as f64;
    let a80 = pct(arm_a.retained_80, arm_a.n);
    let b80 = pct(arm_b.retained_80, arm_b.n);
    let c80 = pct(arm_c.retained_80, arm_c.n);
    let delta_ba = b_mean - a_mean;
    let delta_ca = c_mean - a_mean;
    let delta_cb = c_mean - b_mean;
    println!("══ verdict ══");
    println!(
        "  mean recall  : A={:.3}  B={:.3}  C={:.3}",
        a_mean, b_mean, c_mean
    );
    println!(
        "                 ΔB−A={:+.3}  ΔC−A={:+.3}  ΔC−B={:+.3}",
        delta_ba, delta_ca, delta_cb,
    );
    println!(
        "  ≥0.8        : A={:>5.1}% B={:>5.1}% C={:>5.1}%",
        a80, b80, c80,
    );
    println!();
    let curated_wins = delta_ca >= 0.05;
    let auto_helps = delta_ba >= 0.05;
    if curated_wins {
        println!(
            "  ✓ Curated enrich lifts retrieval ({:+.2} mean recall vs bare).",
            delta_ca,
        );
        println!("    Mechanism prediction confirmed for the schema regime when the");
        println!("    user supplies workload-curated synonyms — the same discipline as");
        println!("    CUAD_CLAUSE_EXPANSION on the query side.");
        if auto_helps {
            println!("    Auto-enrich alone also helps; curated lifts further.");
        } else {
            println!("    Auto-enrich alone (cleaned name + type + table) is flat —");
            println!("    expected: those tokens mostly already tokenize from the bare name.");
            println!("    The lift comes from the user-supplied synonyms (e.g. Age → 'old',");
            println!("    Population → 'people'), high-IDF natural-language terms that");
            println!("    bare BM25 misses.");
        }
    } else if delta_ca > -0.02 {
        println!("  ✗ Curated enrich is flat ({:+.2}).", delta_ca);
        println!("    The regime rule's positive side remains unvalidated. Either the");
        println!(
            "    sample is too small (n={}), the curation isn't sharp enough, or the",
            arm_a.n
        );
        println!("    regime rule itself is wrong. Try full Spider via REDHOP_SPIDER_PATH.");
    } else {
        println!("  ✗ Curated enrich regressed ({:+.2}).", delta_ca);
        println!("    The synonyms may be too generic / low-IDF, re-creating the");
        println!("    CUAD_PRF_NULL dilution failure mode.");
    }
    Ok(())
}
