//! Self-contained HTML "moat report" over a set of [`QueryOutcome`]s.
//!
//! This is the artifact a buyer opens to *see* why retrieval behaved
//! the way it did across a whole workload. It renders:
//!
//! - the regime distribution,
//! - useful vs wasted vs harmful interventions (the headline),
//! - the cost economics (selective vs uniform),
//! - the reliability/calibration diagram,
//! - a per-true-regime intervention breakdown.
//!
//! It is a single self-contained HTML file: inline CSS, no JavaScript,
//! no external assets. It emails cleanly, archives cleanly, and diffs
//! cleanly. Numbers come from the same analysis functions the CLI
//! reports use ([`crate::analysis`], [`crate::economics`],
//! [`crate::reliability`]) — there is no view-specific recomputation.

use std::collections::BTreeMap;

use redhop_core::RetrievalRegime;

use crate::analysis::{confusion_matrix, regret_summary};
use crate::economics::{economics, selective_escalation_roi, CostModel};
use crate::reliability::reliability_diagram;
use crate::runner::QueryOutcome;

/// Options controlling the report.
#[derive(Debug, Clone)]
pub struct ReportOptions {
    /// Title shown in the report header.
    pub title: String,
    /// Workload name (e.g. "HotpotQA dev, 200 items").
    pub workload: String,
    /// Cost model for the economics section.
    pub cost: CostModel,
    /// Uniform-rerank lift baseline (measured by method-pair analysis),
    /// used for the ROI multiple. `None` omits the ROI line.
    pub uniform_rerank_lift: Option<f32>,
}

impl Default for ReportOptions {
    fn default() -> Self {
        Self {
            title: "RedHop Retrieval Report".to_string(),
            workload: "workload".to_string(),
            cost: CostModel::default(),
            uniform_rerank_lift: None,
        }
    }
}

/// Render the full report as a self-contained HTML string.
pub fn render_html(outcomes: &[QueryOutcome], opts: &ReportOptions) -> String {
    let n = outcomes.len();
    let econ = economics(outcomes, &opts.cost);
    let regret = regret_summary(outcomes);
    let cm = confusion_matrix(outcomes);
    let diag = reliability_diagram(outcomes, 10);

    // Regime counts (true).
    let mut regime_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for o in outcomes {
        *regime_counts.entry(o.true_regime.code()).or_insert(0) += 1;
    }

    let n_intervened = outcomes.iter().filter(|o| o.intervened).count();
    let n_useful = outcomes
        .iter()
        .filter(|o| o.intervened && o.recall_lift > 1e-6)
        .count();
    let n_harmful = outcomes
        .iter()
        .filter(|o| o.intervened && o.recall_lift < -1e-6)
        .count();
    let n_wasted = n_intervened - n_useful - n_harmful;
    let _n_abstained = outcomes.iter().filter(|o| o.abstained).count();

    let roi = opts
        .uniform_rerank_lift
        .and_then(|u| selective_escalation_roi(&econ, u, &opts.cost));

    let mut h = String::new();
    h.push_str(&format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --fg:#1a1a2e; --muted:#6b7280; --good:#16a34a; --bad:#dc2626;
         --neutral:#9ca3af; --bar:#4f46e5; --bg:#fafafe; --card:#ffffff;
         --border:#e5e7eb; }}
* {{ box-sizing: border-box; }}
body {{ font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
        color: var(--fg); background: var(--bg); margin: 0; padding: 2rem;
        line-height: 1.45; }}
.wrap {{ max-width: 920px; margin: 0 auto; }}
h1 {{ font-size: 1.6rem; margin: 0 0 .25rem; }}
h2 {{ font-size: 1.15rem; margin: 2rem 0 .75rem; border-bottom: 2px solid var(--border);
      padding-bottom: .35rem; }}
.sub {{ color: var(--muted); margin: 0 0 1.5rem; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
          gap: .75rem; }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px;
         padding: 1rem; }}
.card .v {{ font-size: 1.5rem; font-weight: 700; }}
.card .k {{ color: var(--muted); font-size: .8rem; text-transform: uppercase;
            letter-spacing: .03em; }}
.card .note {{ color: var(--muted); font-size: .8rem; margin-top: .25rem; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: .4rem .6rem; border-bottom: 1px solid var(--border); }}
th:first-child, td:first-child {{ text-align: left; }}
.bar {{ background: var(--border); border-radius: 4px; height: 18px; position: relative;
        overflow: hidden; }}
.bar > span {{ display: block; height: 100%; background: var(--bar); }}
.stack {{ display: flex; height: 28px; border-radius: 6px; overflow: hidden;
          border: 1px solid var(--border); }}
.stack > div {{ display:flex; align-items:center; justify-content:center; color:#fff;
                font-size:.8rem; font-weight:600; }}
.good {{ background: var(--good); }} .bad {{ background: var(--bad); }}
.neutral {{ background: var(--neutral); }}
.legend {{ display:flex; gap:1rem; margin-top:.5rem; font-size:.85rem; color:var(--muted); }}
.legend i {{ display:inline-block; width:12px; height:12px; border-radius:3px;
             margin-right:.3rem; vertical-align:middle; }}
.headline {{ background: linear-gradient(135deg,#eef2ff,#faf5ff); border:1px solid #ddd6fe;
             border-radius: 12px; padding: 1.25rem; margin: 1.5rem 0; }}
.headline .big {{ font-size: 1.35rem; font-weight: 700; }}
code {{ background:#f3f4f6; padding:.1rem .35rem; border-radius:4px; font-size:.85em; }}
.foot {{ color: var(--muted); font-size: .8rem; margin-top: 2.5rem; }}
</style>
</head>
<body><div class="wrap">
<h1>{title}</h1>
<p class="sub">{workload} &middot; {n} queries</p>
"#,
        title = esc(&opts.title),
        workload = esc(&opts.workload),
        n = n,
    ));

    // ── Headline ───────────────────────────────────────────────────
    h.push_str(r#"<div class="headline">"#);
    h.push_str(&format!(
        r#"<div class="big">Adaptive recall lift: {:+.3}</div>"#,
        econ.mean_recall_lift
    ));
    h.push_str(&format!(
        r#"<p>The controller intervened on <b>{}/{}</b> queries ({:.0}%), avoiding <b>{:.0}%</b> of uniform reranking compute. "#,
        n_intervened,
        n,
        if n > 0 { n_intervened as f32 / n as f32 * 100.0 } else { 0.0 },
        econ.rerank_compute_reduction * 100.0,
    ));
    if let Some(roi) = roi {
        h.push_str(&format!(
            r#"Selective-escalation efficiency is <b>{:.1}×</b> uniform reranking.</p>"#,
            roi
        ));
    } else {
        h.push_str("</p>");
    }
    h.push_str("</div>");

    // ── Stat cards ─────────────────────────────────────────────────
    h.push_str(r#"<div class="cards">"#);
    h.push_str(&card(
        "mean recall lift",
        &format!("{:+.3}", econ.mean_recall_lift),
        "",
    ));
    h.push_str(&card(
        "intervention rate",
        &format!(
            "{:.0}%",
            if n > 0 {
                n_intervened as f32 / n as f32 * 100.0
            } else {
                0.0
            }
        ),
        &format!("{n_intervened} / {n}"),
    ));
    h.push_str(&card(
        "useful interventions",
        &format!("{:.0}%", pct(n_useful, n_intervened)),
        &format!("{n_useful} helped"),
    ));
    h.push_str(&card(
        "harmful interventions",
        &format!("{:.0}%", pct(n_harmful, n_intervened)),
        &format!("mean lift {:+.3}", regret.mean_harmful_lift),
    ));
    h.push_str(&card(
        "compute reduction",
        &format!("{:.0}%", econ.rerank_compute_reduction * 100.0),
        "vs uniform rerank",
    ));
    h.push_str(&card(
        "calibration (ECE)",
        &format!("{:.3}", diag.ece),
        "0 = perfect",
    ));
    h.push_str("</div>");

    // ── Useful vs wasted vs harmful stacked bar ───────────────────
    h.push_str("<h2>Intervention outcome</h2>");
    h.push_str(r#"<p class="sub">Of the queries the controller chose to intervene on, how many actually changed gold-chunk recall?</p>"#);
    if n_intervened > 0 {
        let u = pct(n_useful, n_intervened);
        let w = pct(n_wasted, n_intervened);
        let b = pct(n_harmful, n_intervened);
        h.push_str(r#"<div class="stack">"#);
        if u > 0.0 {
            h.push_str(&format!(
                r#"<div class="good" style="width:{u}%">{u:.0}%</div>"#
            ));
        }
        if w > 0.0 {
            h.push_str(&format!(
                r#"<div class="neutral" style="width:{w}%">{w:.0}%</div>"#
            ));
        }
        if b > 0.0 {
            h.push_str(&format!(
                r#"<div class="bad" style="width:{b}%">{b:.0}%</div>"#
            ));
        }
        h.push_str("</div>");
        h.push_str(&format!(
            r#"<div class="legend"><span><i class="good"></i>useful ({n_useful})</span><span><i class="neutral"></i>no change ({n_wasted})</span><span><i class="bad"></i>harmful ({n_harmful})</span></div>"#
        ));
    } else {
        h.push_str("<p>No interventions on this workload.</p>");
    }

    // ── Regime distribution ───────────────────────────────────────
    h.push_str("<h2>True regime distribution</h2>");
    h.push_str("<table><tr><th>regime</th><th>count</th><th>share</th><th></th></tr>");
    for r in RetrievalRegime::all() {
        let c = regime_counts.get(r.code()).copied().unwrap_or(0);
        if c == 0 {
            continue;
        }
        let frac = if n > 0 { c as f32 / n as f32 } else { 0.0 };
        h.push_str(&format!(
            r#"<tr><td>{}</td><td>{}</td><td>{:.0}%</td><td><div class="bar"><span style="width:{:.0}%"></span></div></td></tr>"#,
            r.code(),
            c,
            frac * 100.0,
            frac * 100.0
        ));
    }
    h.push_str("</table>");

    // ── Economics table ───────────────────────────────────────────
    h.push_str("<h2>Cost economics</h2>");
    h.push_str("<table>");
    h.push_str(&row2(
        "mean adaptive cost / query",
        &format!("{:.2}", econ.mean_adaptive_cost),
    ));
    h.push_str(&row2(
        "uniform-rerank cost / query",
        &format!("{:.2}", econ.uniform_cost),
    ));
    h.push_str(&row2(
        "cost fraction vs uniform",
        &format!("{:.0}%", econ.cost_fraction_vs_uniform * 100.0),
    ));
    h.push_str(&row2(
        "rerank compute avoided",
        &format!("{:.0}%", econ.rerank_compute_reduction * 100.0),
    ));
    if let Some(cpl) = econ.cost_per_unit_lift {
        h.push_str(&row2("cost per unit recall-lift", &format!("{:.1}", cpl)));
    }
    h.push_str(&row2(
        "mean rerank calls / query",
        &format!("{:.2}", econ.mean_rerank_calls),
    ));
    h.push_str("</table>");

    // ── Reliability / calibration ─────────────────────────────────
    h.push_str("<h2>Classifier calibration</h2>");
    h.push_str(&format!(
        r#"<p class="sub">Expected Calibration Error = {:.3}. Each bin: predicted confidence vs empirical correctness.</p>"#,
        diag.ece
    ));
    h.push_str("<table><tr><th>confidence bin</th><th>n</th><th>predicted</th><th>empirical</th><th></th></tr>");
    for b in &diag.bins {
        if b.count == 0 {
            continue;
        }
        h.push_str(&format!(
            r#"<tr><td>[{:.1}, {:.1})</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td><div class="bar"><span style="width:{:.0}%"></span></div></td></tr>"#,
            b.lo,
            b.hi,
            b.count,
            b.mean_predicted_p,
            b.empirical_correct,
            b.empirical_correct * 100.0
        ));
    }
    h.push_str("</table>");

    // ── Per-regime classifier metrics ─────────────────────────────
    if cm.n_predicted > 0 {
        h.push_str("<h2>Per-regime classifier metrics</h2>");
        h.push_str("<table><tr><th>regime</th><th>precision</th><th>recall</th><th>f1</th><th>support</th></tr>");
        for r in RetrievalRegime::all() {
            let m = cm.per_regime.get(r).cloned().unwrap_or_default();
            if m.support == 0 {
                continue;
            }
            h.push_str(&format!(
                r#"<tr><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{}</td></tr>"#,
                r.code(),
                m.precision,
                m.recall,
                m.f1,
                m.support
            ));
        }
        h.push_str("</table>");
    }

    h.push_str(&format!(
        r#"<p class="foot">Generated by redhop-calibration. All numbers measured from {n} query outcomes; no view-specific recomputation. Cost units are abstract — set <code>CostModel</code> to your deployment's real figures for a $/latency readout.</p>"#
    ));
    h.push_str("</div></body></html>");
    h
}

fn card(k: &str, v: &str, note: &str) -> String {
    let note_html = if note.is_empty() {
        String::new()
    } else {
        format!(r#"<div class="note">{}</div>"#, esc(note))
    };
    format!(
        r#"<div class="card"><div class="k">{}</div><div class="v">{}</div>{}</div>"#,
        esc(k),
        esc(v),
        note_html
    )
}

fn row2(k: &str, v: &str) -> String {
    format!("<tr><td>{}</td><td>{}</td></tr>", esc(k), esc(v))
}

fn pct(num: usize, denom: usize) -> f32 {
    if denom == 0 {
        0.0
    } else {
        num as f32 / denom as f32 * 100.0
    }
}

/// Minimal HTML-escaping for the small set of text we inject.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop_core::{RerankerLevel, RetrievalRegime};

    fn outcome(regime: RetrievalRegime, rerank: u32, lift: f32) -> QueryOutcome {
        QueryOutcome {
            query_id: "q".into(),
            true_regime: regime,
            predicted_regime: Some(regime),
            predicted_regime_p: Some(0.6),
            true_regime_p: None,
            gold_recall_static: 0.5,
            gold_recall_adaptive: (0.5 + lift).clamp(0.0, 1.0),
            recall_lift: lift,
            intervened: rerank > 0,
            abstained: false,
            escalations: rerank,
            expansions: 0,
            latency_ms_adaptive: 0,
            retrieval_calls_adaptive: 1,
            rerank_calls_adaptive: rerank,
            sum_actual_gain: 0.0,
            final_reranker_level: RerankerLevel::None,
            action_trace: vec![],
        }
    }

    #[test]
    fn renders_valid_self_contained_html() {
        let outs = vec![
            outcome(RetrievalRegime::DistractorHeavy, 1, 0.5),
            outcome(RetrievalRegime::DistractorHeavy, 1, 0.0),
            outcome(RetrievalRegime::Easy, 0, 0.0),
            outcome(RetrievalRegime::Ambiguous, 1, -0.25),
        ];
        let opts = ReportOptions {
            title: "Test Report".into(),
            workload: "fixture, 4 items".into(),
            cost: CostModel::default(),
            uniform_rerank_lift: Some(0.046),
        };
        let html = render_html(&outs, &opts);
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("</html>"));
        assert!(html.contains("Test Report"));
        assert!(html.contains("Adaptive recall lift"));
        // No external resource references.
        assert!(!html.contains("http://"));
        assert!(!html.contains("https://"));
        assert!(!html.contains("<script"));
    }

    #[test]
    fn html_escapes_injected_text() {
        let outs = vec![outcome(RetrievalRegime::Easy, 0, 0.0)];
        let opts = ReportOptions {
            title: "<script>alert(1)</script>".into(),
            workload: "a & b".into(),
            ..Default::default()
        };
        let html = render_html(&outs, &opts);
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;"));
        assert!(html.contains("a &amp; b"));
    }
}
