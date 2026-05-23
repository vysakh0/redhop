//! `redhop report` — render a benchmark/compare JSON artifact to markdown/HTML.
//! Reads artifacts produced by `redhop compare --json` or the
//! `bench_context_strategies` runner (benchmarks/context/results.json).

use anyhow::Context as _;
use clap::Args as ClapArgs;
use serde_json::Value;

#[derive(ClapArgs)]
pub struct Args {
    /// Path to a results JSON artifact.
    input: String,
    /// Write a markdown report here (default: stdout).
    #[arg(long)]
    markdown: Option<String>,
    /// Also write a self-contained HTML report here.
    #[arg(long)]
    html: Option<String>,
}

pub fn run(a: Args) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(&a.input).with_context(|| format!("reading {}", a.input))?;
    let v: Value = serde_json::from_str(&raw).with_context(|| format!("parsing {}", a.input))?;

    let (title, headers, rows, meta) = flatten(&v)?;
    let md = render_markdown(&title, &headers, &rows, &meta);

    match &a.markdown {
        Some(p) => {
            std::fs::write(p, &md)?;
            println!("wrote {p}");
        }
        None if a.html.is_none() => println!("{md}"),
        None => {}
    }
    if let Some(p) = &a.html {
        std::fs::write(p, render_html(&title, &headers, &rows, &meta))?;
        println!("wrote {p}");
    }
    Ok(())
}

/// Reduce a known artifact shape to (title, column headers, rows, metadata lines).
#[allow(clippy::type_complexity)]
fn flatten(v: &Value) -> anyhow::Result<(String, Vec<String>, Vec<Vec<String>>, Vec<String>)> {
    let results = v
        .get("results")
        .and_then(|r| r.as_array())
        .context("artifact has no \"results\" array")?;

    // Collect a stable, readable column set across the two known shapes.
    let cols: Vec<(&str, fn(&Value) -> Option<String>)> = vec![
        ("population", |r| {
            r.get("population")
                .and_then(Value::as_str)
                .map(str::to_string)
        }),
        ("strategy", |r| {
            r.get("strategy")
                .and_then(Value::as_str)
                .map(str::to_string)
        }),
        ("budget", |r| r.get("budget").map(num)),
        ("chunks", |r| {
            r.get("report").and_then(|x| x.get("n_selected")).map(num)
        }),
        ("tokens", |r| {
            r.get("report")
                .and_then(|x| x.get("total_tokens"))
                .map(num)
                .or_else(|| r.get("mean_tokens").map(num))
        }),
        ("rescued", |r| {
            r.get("report")
                .and_then(|x| x.get("second_hop_rescue_count"))
                .map(num)
                .or_else(|| r.get("mean_second_hop_rescue").map(num))
        }),
        ("density", |r| {
            r.get("report")
                .and_then(|x| x.get("economics"))
                .and_then(|x| x.get("evidence_density"))
                .map(num)
                .or_else(|| r.get("evidence_density").map(num))
        }),
        ("gold_ret", |r| {
            r.get("gold_retention")
                .map(num)
                .or_else(|| r.get("gold_ret").map(num))
        }),
        ("second_hop_ret", |r| {
            r.get("second_hop_ret").map(num).or_else(|| {
                r.get("second_hop_retained").map(|b| {
                    b.as_bool()
                        .map(|x| if x { "yes".into() } else { "no".into() })
                        .unwrap_or_else(|| num(b))
                })
            })
        }),
    ];

    // Keep only columns that have at least one value.
    let mut headers = Vec::new();
    let mut keepers: Vec<fn(&Value) -> Option<String>> = Vec::new();
    for (name, f) in &cols {
        if results.iter().any(|r| f(r).is_some()) {
            headers.push(name.to_string());
            keepers.push(*f);
        }
    }
    let rows: Vec<Vec<String>> = results
        .iter()
        .map(|r| {
            keepers
                .iter()
                .map(|f| f(r).unwrap_or_else(|| "-".into()))
                .collect()
        })
        .collect();

    let title = v
        .get("benchmark")
        .and_then(Value::as_str)
        .map(|b| format!("Benchmark: {b}"))
        .or_else(|| {
            v.get("command")
                .and_then(Value::as_str)
                .map(|c| format!("Report: {c}"))
        })
        .unwrap_or_else(|| "RedHop report".into());

    let mut meta = Vec::new();
    if let Some(q) = v.get("query").and_then(Value::as_str) {
        meta.push(format!("query: {q}"));
    }
    if let Some(m) = v.get("metadata").and_then(Value::as_object) {
        for (k, val) in m {
            meta.push(format!("{k}: {val}"));
        }
    }
    Ok((title, headers, rows, meta))
}

fn num(v: &Value) -> String {
    if let Some(i) = v.as_i64() {
        i.to_string()
    } else if let Some(f) = v.as_f64() {
        format!("{f:.3}")
    } else {
        v.as_str().unwrap_or("").to_string()
    }
}

fn render_markdown(
    title: &str,
    headers: &[String],
    rows: &[Vec<String>],
    meta: &[String],
) -> String {
    let mut s = format!("# {title}\n\n");
    for m in meta {
        s.push_str(&format!("- {m}\n"));
    }
    s.push('\n');
    s.push_str(&format!("| {} |\n", headers.join(" | ")));
    s.push_str(&format!(
        "| {} |\n",
        headers
            .iter()
            .map(|_| "---")
            .collect::<Vec<_>>()
            .join(" | ")
    ));
    for row in rows {
        s.push_str(&format!("| {} |\n", row.join(" | ")));
    }
    s.push_str("\n_Generated by `redhop report`. Evidence: docs/findings/._\n");
    s
}

fn render_html(title: &str, headers: &[String], rows: &[Vec<String>], meta: &[String]) -> String {
    let th: String = headers.iter().map(|h| format!("<th>{h}</th>")).collect();
    let trs: String = rows
        .iter()
        .map(|r| {
            format!(
                "<tr>{}</tr>",
                r.iter()
                    .map(|c| format!("<td>{c}</td>"))
                    .collect::<String>()
            )
        })
        .collect();
    let meta_html: String = meta.iter().map(|m| format!("<li>{m}</li>")).collect();
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{title}</title>\
<style>body{{font-family:system-ui,sans-serif;margin:40px;max-width:1000px;color:#222}}\
table{{border-collapse:collapse;width:100%;font-size:14px;margin-top:12px}}\
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #eee}}\
th{{color:#666;text-transform:uppercase;font-size:12px}}ul{{color:#555;font-size:13px}}</style></head>\
<body><h1>{title}</h1><ul>{meta_html}</ul><table><tr>{th}</tr>{trs}</table>\
<p style=\"color:#666;font-size:13px\">Generated by <code>redhop report</code>. \
Evidence: docs/findings/.</p></body></html>\n"
    )
}
