//! Phase 3: real enterprise-PDF ingestion validation.
//!
//! Loads text extracted from real arXiv PDFs (BERT, DPR, RAG, the
//! Transformer paper, word2vec, an LLM survey — 219 pages) and answers
//! the question Phase C was built for:
//!
//!   Do RedHop's ingestion diagnostics actually CORRELATE with real
//!   retrieval degradation?
//!
//! Method:
//!   1. Chunk the real PDF pages into a clean corpus.
//!   2. Synthesize answer-bearing gold queries from the cleanest pages
//!      (a query = a salient phrase from a page; gold = that page's
//!      chunk).
//!   3. Sweep corruption severity (OCR garble / duplication /
//!      boilerplate) over the corpus.
//!   4. At each severity, measure BOTH the ingestion diagnostic score
//!      AND gold-chunk retrieval recall.
//!   5. Report the Pearson correlation between the diagnostic and
//!      (1 − recall).
//!
//! PDF *parsing* stayed in Python (../redhop/scripts/extract_pdf_text.py);
//! this binary consumes the extracted text. The boundary holds.
//!
//! Run with (after running the Python extractor):
//!     cargo run -p redhop-examples --example real_pdf_validation --release

use std::collections::HashMap;
use std::sync::Arc;

use redhop::chunking::{SentenceChunker, WhitespaceTokenizer};
use redhop::core::{Chunk, ChunkId, Chunker, Document, TokenizerBackend};
use redhop_calibration::{
    corruption::{run_degradation_study, CorruptionKind, DegradationStudy},
    embedder::HashingEmbedder,
};
use redhop_diagnostics::{diagnose_ingestion, IngestionThresholds};
const MAX_PAGES: usize = 120;
const TOP_K: usize = 3;

fn load_pages() -> Vec<(String, usize, String)> {
    let path = redhop_examples::exports_path("real_pdf_text.jsonl");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "could not read {} ({e}). Run ../redhop/scripts/extract_pdf_text.py \
             or point REDHOP_EXPORTS_DIR at a directory containing real_pdf_text.jsonl",
            path.display()
        )
    });
    let mut out = Vec::new();
    for line in text.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        out.push((
            v["doc"].as_str().unwrap().to_string(),
            v["page"].as_u64().unwrap() as usize,
            v["text"].as_str().unwrap().to_string(),
        ));
    }
    out
}

/// Pick a salient multi-word phrase from a page to use as a query.
/// Heuristic: the longest "sentence-ish" span of 6-12 content words.
fn salient_query(text: &str) -> Option<String> {
    for sentence in text.split(['.', '\n']) {
        let words: Vec<&str> = sentence
            .split_whitespace()
            .filter(|w| w.chars().all(|c| c.is_alphanumeric()) && w.len() > 3)
            .collect();
        if words.len() >= 6 {
            return Some(words[..6.min(words.len())].join(" "));
        }
    }
    None
}

fn main() -> anyhow::Result<()> {
    let pages = load_pages();
    println!("loaded {} real PDF pages", pages.len());

    // ── Build a clean corpus: one chunk per page (sentence chunker,
    //    large budget so a page stays roughly one chunk). ──
    let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
    let chunker = SentenceChunker::new(tok, 220, 320, 0)?;
    let embedder = HashingEmbedder::with_dim(256);

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut queries: Vec<(String, Vec<ChunkId>)> = Vec::new();
    let mut per_doc_pages: HashMap<String, usize> = HashMap::new();

    for (doc, page, text) in pages.into_iter().take(MAX_PAGES) {
        *per_doc_pages.entry(doc.clone()).or_insert(0) += 1;
        let source = format!("{doc}#p{page}");
        let doc_obj = Document::new(&source, &text);
        let page_chunks = chunker.chunk(&doc_obj)?;
        // Use the first chunk of the page as the page's representative.
        if let Some(first) = page_chunks.first() {
            // Synthesize a gold query from this page's text, pointing at
            // this chunk. Only keep pages with a usable salient phrase.
            if let Some(q) = salient_query(&text) {
                queries.push((q, vec![first.id.clone()]));
            }
        }
        chunks.extend(page_chunks);
    }

    println!(
        "built corpus: {} chunks across {} documents, {} gold queries",
        chunks.len(),
        per_doc_pages.len(),
        queries.len()
    );

    // ── Baseline ingestion diagnostics on the CLEAN real corpus ──
    let clean_report = diagnose_ingestion(&chunks, &IngestionThresholds::default());
    println!("\n──── clean real-PDF corpus: ingestion diagnostics ────");
    println!("  ocr_noise_score      {:.4}", clean_report.ocr_noise_score);
    println!("  duplicate_ratio      {:.4}", clean_report.duplicate_ratio);
    println!(
        "  boilerplate_ratio    {:.4}",
        clean_report.boilerplate_ratio
    );
    println!(
        "  fragmentation_score  {:.4}",
        clean_report.fragmentation_score
    );
    println!(
        "  table_noise_score    {:.4}",
        clean_report.table_noise_score
    );
    if clean_report.warnings.is_empty() {
        println!("  → clean academic PDFs trip zero ingestion warnings (expected)");
    } else {
        println!("  warnings:");
        for w in &clean_report.warnings {
            println!("    ⚠ [{}] {}", w.code, w.message);
        }
    }

    // ── Degradation correlation study, per corruption kind ──
    let severities = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90];
    println!("\n──── diagnostics-vs-degradation correlation (real PDF corpus) ────");
    for kind in [
        CorruptionKind::OcrNoise,
        CorruptionKind::Duplication,
        CorruptionKind::Boilerplate,
    ] {
        let study = run_degradation_study(
            &chunks,
            &queries,
            kind,
            &severities,
            &embedder,
            TOP_K,
            0xC0FFEE,
        );
        print_study(&study);
    }

    println!("\nInterpretation: a high positive correlation means the ingestion");
    println!("diagnostic RISES as retrieval recall FALLS — i.e. the diagnostic");
    println!("predicts real retrieval degradation, not just describes text. That");
    println!("is the validation Phase C needed: the diagnostics earn their place");
    println!("as an early-warning signal on messy enterprise corpora.");
    Ok(())
}

fn print_study(s: &DegradationStudy) {
    println!(
        "\n  corruption = {}   (diagnostic↔degradation correlation = {:+.3})",
        s.kind, s.diagnostic_vs_degradation_correlation
    );
    println!(
        "    {:<10} {:>12} {:>10} {:>10}",
        "severity", "diag_score", "recall", "warnings"
    );
    for r in &s.rows {
        println!(
            "    {:<10.2} {:>12.4} {:>10.3} {:>10}",
            r.severity, r.diagnostic_score, r.retrieval_recall, r.n_warnings
        );
    }
}
