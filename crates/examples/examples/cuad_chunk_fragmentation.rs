//! CUAD chunk-fragmentation diagnostic.
//!
//! Tests the open hypothesis from `docs/findings/CUAD_RECALL_GAP.md`:
//! "LlamaIndex's 4-point CUAD edge is likely sentence-aware chunking that
//! hits clause boundaries CUAD's gold spans happen to align with."
//!
//! Reframed to a question that's actually measurable against our own
//! chunker — which is already sentence-aware (Unicode UAX #29 via
//! `SentenceChunker`): **are CUAD gold answer spans landing inside
//! individual chunks, or are they fragmented across chunk boundaries?**
//!
//! Three outcomes possible:
//!
//!   - **Fragmentation is the gap.** If many gold spans span 2+ chunks,
//!     the chunker is the lever; a paragraph-aware or clause-aware
//!     variant could plausibly close the LlamaIndex gap.
//!   - **Spans are already inside single chunks.** Falsifies the
//!     chunker hypothesis. The gap must live somewhere else (retrieval
//!     ranking, budget cutoff, etc.). Move to next Phase 3 candidate.
//!   - **Mixed.** Document the fraction, decide if the fragmentable
//!     subset is worth a chunker variant.
//!
//! For each gold answer span, we:
//!   1. Tokenize the span (set of unique content words, same metric as
//!      `bench/compare.py` and the other CUAD harnesses).
//!   2. For every chunk in the document, count how many of the span's
//!      words it contains.
//!   3. Sort chunks by recall; the "primary" chunk is the highest.
//!      Span coverage = primary chunk recall; remaining coverage tells
//!      us how spread the span is.
//!
//! Headline metrics:
//!   - % of gold spans where the **single best chunk** covers ≥ 0.8 of
//!     the span words (i.e., the span fits inside one chunk for the
//!     metric that matters).
//!   - % of spans where **the top 2 chunks** together cover ≥ 0.8.
//!   - Histogram of (top-1 coverage) bands.
//!
//! Same dataset slice as the other CUAD harnesses (n=300, cuad_sample.json).
//! No retrieval, no LLM — pure span-vs-chunk geometry.
//!
//! Run: cargo run -p redhop-examples --example cuad_chunk_fragmentation --release

use std::collections::HashSet;

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

const LIMIT_Q: usize = 300;

#[derive(Default, Clone, Copy)]
struct Coverage {
    /// Highest single-chunk gold-word recall.
    top1: f32,
    /// Top-1 + top-2 combined gold-word recall.
    top2: f32,
    /// Top-1 + top-2 + top-3 combined gold-word recall.
    top3: f32,
    /// Number of *distinct* chunks that contain at least one gold word.
    n_chunks_touched: usize,
    /// Token count of the primary (top-1) chunk.
    primary_chunk_tokens: usize,
    /// Token count of the gold span (set-based, after tokenization).
    gold_span_tokens: usize,
}

fn span_coverage(span: &str, chunk_texts: &[(String, usize)]) -> Coverage {
    let gold_words: HashSet<String> = words(span).into_iter().collect();
    if gold_words.is_empty() {
        return Coverage::default();
    }

    // For each chunk: how many gold words does it contain, what's its token count.
    let mut per_chunk: Vec<(HashSet<String>, usize)> = Vec::with_capacity(chunk_texts.len());
    for (ct, tokens) in chunk_texts {
        let cw: HashSet<String> = words(ct).into_iter().collect();
        let hit: HashSet<String> = gold_words.intersection(&cw).cloned().collect();
        per_chunk.push((hit, *tokens));
    }

    // Sort by recall descending.
    per_chunk.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let g = gold_words.len() as f32;
    let top1_hit = per_chunk
        .first()
        .map(|(h, _)| h.clone())
        .unwrap_or_default();
    let mut top2_hit = top1_hit.clone();
    if let Some((h, _)) = per_chunk.get(1) {
        top2_hit.extend(h.iter().cloned());
    }
    let mut top3_hit = top2_hit.clone();
    if let Some((h, _)) = per_chunk.get(2) {
        top3_hit.extend(h.iter().cloned());
    }

    let n_chunks_touched = per_chunk.iter().filter(|(h, _)| !h.is_empty()).count();
    let primary_chunk_tokens = per_chunk.first().map(|(_, t)| *t).unwrap_or(0);

    Coverage {
        top1: top1_hit.len() as f32 / g,
        top2: top2_hit.len() as f32 / g,
        top3: top3_hit.len() as f32 / g,
        n_chunks_touched,
        primary_chunk_tokens,
        gold_span_tokens: gold_words.len(),
    }
}

fn band(v: f32) -> &'static str {
    if v >= 0.95 {
        ">=0.95"
    } else if v >= 0.80 {
        "0.80-0.95"
    } else if v >= 0.50 {
        "0.50-0.80"
    } else if v >= 0.20 {
        "0.20-0.50"
    } else {
        "<0.20"
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::var("REDHOP_CUAD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| redhop_examples::data_path("cuad/cuad_sample.json"));
    let raw = std::fs::read_to_string(&path)?;
    let cuad: Cuad = serde_json::from_str(&raw)?;

    println!("CUAD chunk-fragmentation diagnostic — does the default sentence chunker");
    println!("fragment gold answer spans across chunk boundaries?");
    println!("  config: default chunker (target=128, max=256, overlap_sentences=0)");
    println!("  n queries: up to {LIMIT_Q}, from cuad_sample.json");
    println!();

    let mut q_count = 0usize;
    let mut bucket_top1: std::collections::BTreeMap<&str, usize> =
        std::collections::BTreeMap::new();
    let mut bucket_top2: std::collections::BTreeMap<&str, usize> =
        std::collections::BTreeMap::new();

    let mut sum_top1 = 0.0f64;
    let mut sum_top2 = 0.0f64;
    let mut sum_top3 = 0.0f64;
    let mut span_in_one_chunk_count = 0usize;
    let mut span_in_top2_count = 0usize;
    let mut sum_chunks_touched = 0usize;
    let mut sum_primary_tokens = 0usize;
    let mut sum_gold_tokens = 0usize;

    // Worst-fragmented examples to print.
    let mut worst: Vec<(f32, String, usize)> = Vec::new();

    'outer: for c in &cuad.data {
        for para in &c.paragraphs {
            let mut doc = match Document::from_text_with(
                &c.title,
                &para.context,
                DocumentConfig::default(),
            ) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let chunk_texts: Vec<(String, usize)> = doc
                .chunks()
                .iter()
                .map(|ch| (ch.text.clone(), ch.token_count.value()))
                .collect();
            if chunk_texts.is_empty() {
                continue;
            }
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
                let cov = span_coverage(gold, &chunk_texts);
                if cov.gold_span_tokens == 0 {
                    continue;
                }

                q_count += 1;
                sum_top1 += cov.top1 as f64;
                sum_top2 += cov.top2 as f64;
                sum_top3 += cov.top3 as f64;
                if cov.top1 >= 0.8 {
                    span_in_one_chunk_count += 1;
                }
                if cov.top2 >= 0.8 {
                    span_in_top2_count += 1;
                }
                sum_chunks_touched += cov.n_chunks_touched;
                sum_primary_tokens += cov.primary_chunk_tokens;
                sum_gold_tokens += cov.gold_span_tokens;
                *bucket_top1.entry(band(cov.top1)).or_insert(0) += 1;
                *bucket_top2.entry(band(cov.top2)).or_insert(0) += 1;

                if cov.top1 < 0.5 && worst.len() < 5 {
                    worst.push((cov.top1, gold.to_string(), cov.gold_span_tokens));
                }
            }
            let _ = doc.embedded_chunks(); // touch so the compiler keeps doc alive
        }
    }

    let n = q_count.max(1) as f64;
    println!("══ headline ══");
    println!("  n analyzed:                                       {q_count}");
    println!(
        "  mean top-1 chunk coverage of gold span:           {:.3}",
        sum_top1 / n
    );
    println!(
        "  mean top-2 chunk coverage of gold span:           {:.3}",
        sum_top2 / n
    );
    println!(
        "  mean top-3 chunk coverage of gold span:           {:.3}",
        sum_top3 / n
    );
    println!();
    println!(
        "  % of gold spans where TOP-1 alone covers >= 0.8:  {:.1}%",
        100.0 * span_in_one_chunk_count as f64 / n
    );
    println!(
        "  % of gold spans where TOP-2 together cover >= 0.8:{:.1}%",
        100.0 * span_in_top2_count as f64 / n
    );
    println!();
    println!(
        "  mean chunks touched by a gold span:               {:.2}",
        sum_chunks_touched as f64 / n
    );
    println!(
        "  mean primary-chunk token count:                   {:.0}",
        sum_primary_tokens as f64 / n
    );
    println!(
        "  mean gold-span token count (set):                 {:.1}",
        sum_gold_tokens as f64 / n
    );
    println!();
    println!("── top-1 chunk coverage distribution ──");
    for (b, c) in &bucket_top1 {
        println!("  {b:>10}: {c:>4}  ({:.1}%)", 100.0 * (*c as f64) / n);
    }
    println!();
    println!("── top-2 chunk coverage distribution ──");
    for (b, c) in &bucket_top2 {
        println!("  {b:>10}: {c:>4}  ({:.1}%)", 100.0 * (*c as f64) / n);
    }
    println!();
    if !worst.is_empty() {
        println!("── worst-fragmented examples (top-1 cov < 0.5) ──");
        for (cov, gold, ntok) in &worst {
            let snip: String = gold.chars().take(120).collect();
            println!(
                "  cov={cov:.2}, |span|={ntok}:  \"{snip}{}\"",
                if gold.len() > snip.len() { "…" } else { "" }
            );
        }
        println!();
    }

    println!("══ verdict ══");
    let top1_80 = 100.0 * span_in_one_chunk_count as f64 / n;
    let top2_80 = 100.0 * span_in_top2_count as f64 / n;
    if top1_80 >= 80.0 {
        println!("  ✓ Chunker hypothesis FALSIFIED for this workload.");
        println!(
            "    {top1_80:.1}% of gold spans are already contained in a single chunk at ≥0.8."
        );
        println!("    The CUAD gap to LlamaIndex is NOT chunk-boundary fragmentation;");
        println!("    look elsewhere (retrieval ranking, budget pressure, chunk-token");
        println!("    overlap with non-gold context).");
    } else if top2_80 - top1_80 >= 15.0 {
        println!("  ~ Chunker hypothesis PARTIALLY confirmed.");
        println!("    Only {top1_80:.1}% of spans fit in 1 chunk, but {top2_80:.1}% fit");
        println!(
            "    in 2 chunks combined. A {:.1}-point lift potential if a",
            top2_80 - top1_80
        );
        println!("    paragraph-aware or larger-window chunker keeps both halves together.");
    } else {
        println!("  ✗ Spans don't even fit cleanly in the top-2 chunks ({top2_80:.1}%).");
        println!("    Either gold spans are wider than reasonable chunk sizes, OR the");
        println!("    splitter is breaking spans in unusual places. Inspect the worst-");
        println!("    fragmented examples above.");
    }
    Ok(())
}
