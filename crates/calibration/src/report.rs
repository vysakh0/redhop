//! ASCII reports for sweep results and reliability diagrams.
//!
//! No external plotting dependencies — everything prints to the terminal
//! as a fixed-width table or an inline bar chart. Keeps the calibration
//! loop hermetic and friendly to CI logs.

use crate::reliability::ReliabilityDiagram;
use crate::sweep::SweepReport;

/// Render a sweep report as an ASCII table.
pub fn render_sweep_table(report: &SweepReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{:<14} {:<14} {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>8} {:>7}\n",
        "min_p_distr.",
        "min_p_amb.",
        "n",
        "interv_rate",
        "mean_lift",
        "lift_when",
        "useful",
        "harmful",
        "lat_ms",
        "rerank/q",
        "argmax%",
    ));
    s.push_str(&"─".repeat(120));
    s.push('\n');
    for r in &report.rows {
        s.push_str(&format!(
            "{:<14} {:<14} {:>4} {:>10.3} {:>+10.3} {:>+10.3} {:>10.3} {:>10.3} {:>9.1} {:>8.2} {:>6.1}%\n",
            format!("{:.2}", r.min_p_distractor),
            format!("{:.2}", r.min_p_ambiguous),
            r.n,
            r.intervention_rate,
            r.mean_recall_lift,
            r.mean_recall_lift_when_intervened,
            r.fraction_useful_interventions,
            r.fraction_harmful_interventions,
            r.mean_latency_ms,
            r.mean_rerank_calls,
            r.regime_argmax_accuracy * 100.0,
        ));
    }
    s
}

/// Render a reliability diagram as an inline ASCII bar chart. Each bin
/// gets one line showing `[lo, hi)`, the bin count, and a comparison of
/// `mean_predicted_p` vs `empirical_correct` as two side-by-side bars.
pub fn render_reliability(diagram: &ReliabilityDiagram) -> String {
    let mut s = String::new();
    let title = match diagram.regime {
        Some(r) => format!("reliability for regime = {}", r.code()),
        None => "reliability for predicted argmax".to_string(),
    };
    s.push_str(&format!("─── {title} (ECE = {:.3}) ───\n", diagram.ece));
    s.push_str(&format!(
        "{:<14} {:>5} {:>10} {:>10}  {}\n",
        "bin", "n", "pred_p", "empirical", "pred / empirical"
    ));
    s.push_str(&"─".repeat(60));
    s.push('\n');
    const BAR_W: usize = 20;
    for b in &diagram.bins {
        let pred_bar = bar(b.mean_predicted_p, BAR_W);
        let emp_bar = bar(b.empirical_correct, BAR_W);
        s.push_str(&format!(
            "[{:.2},{:.2}) {:>5} {:>10.3} {:>10.3}  {} / {}\n",
            b.lo, b.hi, b.count, b.mean_predicted_p, b.empirical_correct, pred_bar, emp_bar
        ));
    }
    s
}

fn bar(value: f32, width: usize) -> String {
    let v = value.clamp(0.0, 1.0);
    let filled = (v * width as f32).round() as usize;
    let mut out = String::with_capacity(width + 2);
    out.push('|');
    for i in 0..width {
        out.push(if i < filled { '█' } else { ' ' });
    }
    out.push('|');
    out
}

/// Render a Pareto comparison: a small ASCII scatter of
/// (mean_latency_ms, mean_recall_lift) per setting. Lower-right means
/// "more lift for less latency"; the Pareto frontier traces along the
/// upper-right corner. Settings dominated by another (more lift AND
/// less latency exists somewhere else in the grid) are flagged with
/// `(dominated)`.
pub fn render_pareto(report: &SweepReport) -> String {
    let mut s = String::new();
    s.push_str("─── adaptive Pareto (latency_ms vs mean_recall_lift) ───\n");
    s.push_str(&format!(
        "{:<14} {:<14} {:>10} {:>+10} {}\n",
        "min_p_distr.", "min_p_amb.", "lat_ms", "lift", "dominated?"
    ));
    s.push_str(&"─".repeat(70));
    s.push('\n');
    for (i, r) in report.rows.iter().enumerate() {
        let dominated = report.rows.iter().enumerate().any(|(j, other)| {
            j != i
                && other.mean_latency_ms <= r.mean_latency_ms
                && other.mean_recall_lift >= r.mean_recall_lift
                && (other.mean_latency_ms < r.mean_latency_ms
                    || other.mean_recall_lift > r.mean_recall_lift)
        });
        s.push_str(&format!(
            "{:<14} {:<14} {:>10.1} {:>+10.3}  {}\n",
            format!("{:.2}", r.min_p_distractor),
            format!("{:.2}", r.min_p_ambiguous),
            r.mean_latency_ms,
            r.mean_recall_lift,
            if dominated { "(dominated)" } else { "" }
        ));
    }
    s
}
