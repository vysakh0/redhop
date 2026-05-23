//! Method-pair regret analysis from imported NeoTrace data.
//!
//! Loads `hotpot_full.neotrace.jsonl` (the haiku run) and pairs every
//! retrieval method against `cosine` (the standard dense baseline),
//! reporting the empirical answer to:
//!
//!   "If we'd switched the production retriever from cosine to method
//!   X on this workload, how often would gold-chunk recall have moved
//!   up, how often down, and what would the mean change have been?"
//!
//! This is the strongest direct use of Python-lab data: every
//! `(item_id, method)` pair already has `retrieval_recall` measured
//! against gold paragraphs. No Rust-side retrieval; we project the
//! measurements straight into `regret_summary` and
//! `bootstrap_stability`.
//!
//! Run with:
//!     cargo run -p redhop-examples --example method_pair_regret

use std::collections::BTreeMap;

use redhop_calibration::{
    analysis::regret_summary,
    loaders::neotrace::{parse_path, NeoTraceRecord},
    runner::{ActionTraceEntry, QueryOutcome},
};
use redhop_core::{RerankerLevel, RetrievalRegime};

const NEOTRACE_PATH: &str =
    "/Users/vysakh/projects/neorag/exports/neotrace/hotpot_full.neotrace.jsonl";

/// Bracket method codes seen across HotpotQA. Order matters in the
/// printed table.
const METHODS: &[&str] = &[
    "cosine",
    "bm25",
    "rrf",
    "answerability",
    "learned",
    "cross_encoder",
    "trajectory",
];

const STATIC_METHOD: &str = "cosine";

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Method-pair regret: NeoTrace traces, paired vs cosine baseline ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let records = parse_path(NEOTRACE_PATH)?;
    println!(
        "loaded {} NeoTrace records from {}",
        records.len(),
        NEOTRACE_PATH
    );

    // Group records by item_id.
    let mut by_item: BTreeMap<String, Vec<&NeoTraceRecord>> = BTreeMap::new();
    for r in &records {
        if let Some(id) = &r.item_id {
            by_item.entry(id.clone()).or_default().push(r);
        }
    }
    println!("  {} unique items", by_item.len());
    println!();

    // For each adaptive method, pair against cosine on identical
    // item_ids and emit a QueryOutcome.
    let mut table_rows: Vec<(&'static str, RegretSummary, usize, usize, usize)> = Vec::new();
    for adaptive_method in METHODS {
        if *adaptive_method == STATIC_METHOD {
            continue;
        }
        let mut outcomes: Vec<QueryOutcome> = Vec::new();
        let mut n_better = 0usize;
        let mut n_worse = 0usize;
        let mut n_same = 0usize;
        for (item_id, items) in &by_item {
            let static_r = items
                .iter()
                .find(|r| r.method.as_deref() == Some(STATIC_METHOD));
            let adapt_r = items
                .iter()
                .find(|r| r.method.as_deref() == Some(adaptive_method));
            let (Some(s), Some(a)) = (static_r, adapt_r) else {
                continue;
            };
            let rs = s.retrieval_recall.unwrap_or(0.0);
            let ra = a.retrieval_recall.unwrap_or(0.0);
            let lift = ra - rs;
            if lift > 1e-6 {
                n_better += 1;
            } else if lift < -1e-6 {
                n_worse += 1;
            } else {
                n_same += 1;
            }
            outcomes.push(QueryOutcome {
                query_id: item_id.clone(),
                true_regime: parse_regime_from_str(a.true_regime.as_deref())
                    .unwrap_or(RetrievalRegime::Easy),
                predicted_regime: None,
                predicted_regime_p: None,
                true_regime_p: None,
                gold_recall_static: rs,
                gold_recall_adaptive: ra,
                recall_lift: lift,
                intervened: true,
                abstained: false,
                escalations: 0,
                expansions: 0,
                latency_ms_adaptive: 0,
                retrieval_calls_adaptive: 1,
                rerank_calls_adaptive: 0,
                sum_actual_gain: 0.0,
                final_reranker_level: RerankerLevel::None,
                action_trace: Vec::<ActionTraceEntry>::new(),
            });
        }
        let r = regret_summary(&outcomes);
        let mean_lift: f32 = if outcomes.is_empty() {
            0.0
        } else {
            outcomes.iter().map(|o| o.recall_lift).sum::<f32>() / outcomes.len() as f32
        };
        table_rows.push((
            *adaptive_method,
            RegretSummary {
                n: outcomes.len(),
                mean_lift,
                mean_useful_lift: r.mean_useful_lift,
                mean_harmful_lift: r.mean_harmful_lift,
                useful_count: n_better,
                harmful_count: n_worse,
                neutral_count: n_same,
            },
            n_better,
            n_worse,
            n_same,
        ));
    }

    // ── Table 1: pair-wise lift summary ────────────────────────────
    println!("─── pair-wise gold-recall lift vs `{STATIC_METHOD}` ───");
    println!(
        "{:<16} {:>5} {:>+10} {:>+10} {:>+10} {:>8} {:>8} {:>8}",
        "method", "n", "mean_lift", "useful_avg", "harmful_avg", "n>cos", "n<cos", "n=cos"
    );
    println!("{}", "─".repeat(85));
    for (m, r, b, w, s) in &table_rows {
        println!(
            "{:<16} {:>5} {:>+10.3} {:>+10.3} {:>+10.3} {:>8} {:>8} {:>8}",
            m, r.n, r.mean_lift, r.mean_useful_lift, r.mean_harmful_lift, b, w, s
        );
    }

    // ── Table 2: regime-conditioned lift ───────────────────────────
    println!();
    println!("─── method × true_regime: mean recall lift vs `{STATIC_METHOD}` ───");
    print!("{:<16}", "method");
    for r in RetrievalRegime::all() {
        print!(" {:>14}", r.code());
    }
    println!();
    println!("{}", "─".repeat(110));
    for adaptive_method in METHODS {
        if *adaptive_method == STATIC_METHOD {
            continue;
        }
        // accumulate per-regime totals
        let mut totals: BTreeMap<RetrievalRegime, (f32, usize)> = BTreeMap::new();
        for items in by_item.values() {
            let static_r = items
                .iter()
                .find(|r| r.method.as_deref() == Some(STATIC_METHOD));
            let adapt_r = items
                .iter()
                .find(|r| r.method.as_deref() == Some(adaptive_method));
            let (Some(s), Some(a)) = (static_r, adapt_r) else {
                continue;
            };
            let regime =
                parse_regime_from_str(a.true_regime.as_deref()).unwrap_or(RetrievalRegime::Easy);
            let lift = a.retrieval_recall.unwrap_or(0.0) - s.retrieval_recall.unwrap_or(0.0);
            let e = totals.entry(regime).or_insert((0.0, 0));
            e.0 += lift;
            e.1 += 1;
        }
        print!("{:<16}", adaptive_method);
        for r in RetrievalRegime::all() {
            let (sum, n) = totals.get(r).copied().unwrap_or((0.0, 0));
            if n == 0 {
                print!(" {:>14}", "-");
            } else {
                print!(" {:>+13.3}({:>2})", sum / n as f32, n);
            }
        }
        println!();
    }

    // ── Headline ───────────────────────────────────────────────────
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("HEADLINE — HotpotQA, haiku run, vs cosine baseline");
    let best = table_rows.iter().max_by(|a, b| {
        a.1.mean_lift
            .partial_cmp(&b.1.mean_lift)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some((m, r, _, _, _)) = best {
        println!(
            "  best mean-lift method:  {} → {:+.3} on {} items",
            m, r.mean_lift, r.n
        );
        let useful_rate = if r.n > 0 {
            r.useful_count as f32 / r.n as f32
        } else {
            0.0
        };
        println!(
            "  fraction that improved: {:.0}% ({} / {})",
            useful_rate * 100.0,
            r.useful_count,
            r.n
        );
        println!(
            "  fraction unchanged:     {:.0}% ({} / {})",
            r.neutral_count as f32 / r.n as f32 * 100.0,
            r.neutral_count,
            r.n
        );
        println!(
            "  fraction made worse:    {:.0}% ({} / {})",
            r.harmful_count as f32 / r.n as f32 * 100.0,
            r.harmful_count,
            r.n
        );
    }
    println!();
    println!("Interpretation: this is the recall-lift ceiling — what the Python lab");
    println!("ALREADY measured when running these methods statically. A well-tuned");
    println!("Rust adaptive controller should approach this ceiling by firing the");
    println!("right method on the right queries, NOT by firing the best method on");
    println!("every query. The regime-conditioned table above shows where there is");
    println!("regime-localized signal the adaptive controller could exploit.");
    println!("════════════════════════════════════════════════════════════════════════");

    Ok(())
}

#[derive(Debug, Clone)]
struct RegretSummary {
    n: usize,
    mean_lift: f32,
    mean_useful_lift: f32,
    mean_harmful_lift: f32,
    useful_count: usize,
    harmful_count: usize,
    neutral_count: usize,
}

fn parse_regime_from_str(s: Option<&str>) -> Option<RetrievalRegime> {
    let s = s?;
    for r in RetrievalRegime::all() {
        if r.code() == s {
            return Some(*r);
        }
    }
    None
}
