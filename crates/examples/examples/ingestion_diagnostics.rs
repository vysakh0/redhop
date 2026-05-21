//! Ingestion diagnostics demo (Phase C).
//!
//! Runs `diagnose_ingestion` over several deliberately-messy synthetic
//! corpora — clean, OCR-garbled, boilerplate-heavy, fragmented, and
//! table-flattened — to show each detector firing in isolation, then a
//! realistic "mixed enterprise PDF" corpus where several problems
//! co-occur.
//!
//! These corpora are synthetic and hermetic on purpose. Real PDF text
//! comes from the Python ingestion layer (NeoRAG does not parse PDFs);
//! whatever chunker produced the text, these diagnostics work on it.
//!
//! Run with:
//!     cargo run -p neorag-examples --example ingestion_diagnostics

use neorag_core::{Chunk, TokenCount};
use neorag_diagnostics::{diagnose_ingestion, IngestionReport, IngestionThresholds};

fn chunk(id: &str, text: &str) -> Chunk {
    Chunk::new(id, text, "doc", TokenCount(text.split_whitespace().count()))
}

fn print_report(label: &str, r: &IngestionReport) {
    println!("──── {label} ({} chunks) ────", r.n_chunks);
    println!("  ocr_noise_score      {:.3}", r.ocr_noise_score);
    println!("  duplicate_ratio      {:.3}", r.duplicate_ratio);
    println!("  boilerplate_ratio    {:.3}", r.boilerplate_ratio);
    println!("  fragmentation_score  {:.3}", r.fragmentation_score);
    println!("  table_noise_score    {:.3}", r.table_noise_score);
    if r.warnings.is_empty() {
        println!("  warnings: none — corpus looks clean");
    } else {
        println!("  warnings:");
        for w in &r.warnings {
            println!("    ⚠ [{}] {}", w.code, w.message);
        }
    }
    if !r.flagged_chunks.is_empty() {
        println!("  flagged chunks (worst offenders):");
        for f in r.flagged_chunks.iter().take(4) {
            println!("    {} [{}] score={:.3}", f.chunk_id, f.metric, f.score);
        }
    }
    println!();
}

fn main() {
    let cfg = IngestionThresholds::default();
    println!("NeoRAG ingestion diagnostics — detecting retrieval-corrupting corpora\n");

    // 1. Clean prose.
    let clean = vec![
        chunk("clean-1", "The transformer architecture introduced self-attention. It replaced recurrence with parallel attention over the input sequence, enabling far greater training throughput."),
        chunk("clean-2", "Retrieval-augmented generation grounds a language model in external evidence. A retriever selects relevant passages, which are concatenated into the prompt context."),
        chunk("clean-3", "Dense retrieval encodes the query and each document into a shared vector space. Cosine similarity then ranks the candidate passages for relevance."),
    ];
    print_report("CLEAN PROSE", &diagnose_ingestion(&clean, &cfg));

    // 2. OCR garbage (vowelless tokens + spaced-out letters).
    let ocr = vec![
        chunk("ocr-1", "rn th wrd xj qz mn bk th rn nw wht hppnd hr nd thr cn b sn"),
        chunk("ocr-2", "T h e q u i c k b r o w n f o x j u m p s o v e r t h e"),
        chunk("ocr-3", "scnned dcmnt wth brkn chrs nd mssng vwls thrght th pg"),
    ];
    print_report("OCR-GARBLED SCAN", &diagnose_ingestion(&ocr, &cfg));

    // 3. Duplicated sections.
    let body = "Our return policy allows refunds within thirty days of purchase provided the item is unused and in its original packaging with proof of purchase.";
    let dup = vec![
        chunk("dup-1", body),
        chunk("dup-2", body),
        chunk("dup-3", "Shipping is calculated at checkout based on destination and selected delivery speed; expedited options are available for most regions."),
        chunk("dup-4", body),
        chunk("dup-5", body),
    ];
    print_report("DUPLICATED SECTIONS", &diagnose_ingestion(&dup, &cfg));

    // 4. Header/footer boilerplate.
    let mk_bp = |n: usize, body: &str| {
        chunk(
            &format!("bp-{n}"),
            &format!("ACME CORPORATION — INTERNAL USE ONLY\nQuarterly Report\nPage {n} of 12\n{body}"),
        )
    };
    let boiler = vec![
        mk_bp(1, "Revenue grew twelve percent year over year driven by enterprise subscriptions."),
        mk_bp(2, "Operating margin expanded as cloud infrastructure costs declined per unit."),
        mk_bp(3, "Headcount increased modestly with most hiring in engineering and support."),
        mk_bp(4, "The board approved a share repurchase program of up to two hundred million."),
    ];
    print_report("HEADER/FOOTER BOILERPLATE", &diagnose_ingestion(&boiler, &cfg));

    // 5. Mid-sentence fragmentation.
    let frag = vec![
        chunk("frag-1", "and the committee therefore concluded that the proposed amendment would require"),
        chunk("frag-2", "additional review by the legal department before it could be submitted for a vote"),
        chunk("frag-3", "which the chair agreed to schedule for the following quarter pending budget"),
    ];
    print_report("MID-SENTENCE FRAGMENTATION", &diagnose_ingestion(&frag, &cfg));

    // 6. Flattened tables.
    let tables = vec![
        chunk("tbl-1", "2019 | 1,240 | 3.5% | $12,400 | 88 | 2020 | 1,560 | 4.1% | $15,600 | 92 | 2021 | 1,890 | 5.0% | $18,900 | 95 | 2022 | 2,210 | 5.4% | $22,100 | 97"),
        chunk("tbl-2", "Q1 | 45 | Q2 | 52 | Q3 | 61 | Q4 | 70 | total | 228 | avg | 57 | min | 45 | max | 70 | stdev | 11 | count | 4"),
    ];
    print_report("FLATTENED TABLES", &diagnose_ingestion(&tables, &cfg));

    // 7. Realistic mixed enterprise corpus — several problems at once.
    let mixed = vec![
        chunk("mix-1", "ACME CORP CONFIDENTIAL\nPage 1\nThe annual report summarizes financial performance for the fiscal year."),
        chunk("mix-2", "ACME CORP CONFIDENTIAL\nPage 2\nand continues the discussion of revenue without a clean sentence boundary so"),
        chunk("mix-3", "ACME CORP CONFIDENTIAL\nPage 3\n2019 | 1,240 | 2020 | 1,560 | 2021 | 1,890 | 2022 | 2,210 | 2023 | 2,540"),
        chunk("mix-4", "ACME CORP CONFIDENTIAL\nPage 4\nThe annual report summarizes financial performance for the fiscal year."),
    ];
    print_report("MIXED ENTERPRISE PDF (multiple problems)", &diagnose_ingestion(&mixed, &cfg));

    println!("Interpretation: these are DIAGNOSTICS, not a controller. They surface");
    println!("corruption so a deployment can react (re-OCR, dedup, table-aware");
    println!("extraction, sentence-aware chunking) BEFORE retrieval serves the noise.");
    println!("Real PDF text arrives via the Python ingestion layer; NeoRAG analyzes");
    println!("whatever text the chunker produced, regardless of source.");
}
