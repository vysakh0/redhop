//! `Document.context(query)` on a prose chunk with `metadata["heading"]`
//! auto-attaches the section's opening (heading-bearing) chunk so the cited
//! context arrives with its section title for the LLM. Mirrors the
//! code-neighbor default but for prose with hierarchical structure.

use redhop::core::{Chunk, ChunkId, TokenCount};
use redhop::{Document, DocumentConfig};

fn prose_with_heading(id: &str, text: &str, heading: &str, source: &str) -> Chunk {
    let mut c = Chunk::new(
        ChunkId::new(id),
        text,
        source,
        TokenCount(text.split_whitespace().count()),
    );
    c.metadata
        .insert("kind".into(), serde_json::json!("prose"));
    c.metadata
        .insert("heading".into(), serde_json::json!(heading));
    c
}

#[test]
fn prose_chunk_pulls_its_section_opener_by_default() {
    // Two markdown-shaped sections; the query lands on the deep chunk of
    // the "Refunds" section. The auto-expansion should attach the section's
    // earliest chunk (id=0) so the cited context carries the heading.
    let chunks = vec![
        prose_with_heading("0", "refund eligibility overview paragraph", "Refunds", "policy.md"),
        prose_with_heading("1", "fine print: thirty day window from purchase date", "Refunds", "policy.md"),
        prose_with_heading("2", "shipping carrier coordination details", "Shipping", "policy.md"),
    ];
    let mut doc = Document::from_chunks_with(chunks, DocumentConfig::default()).unwrap();
    let ctx = doc.context("thirty day window").unwrap();

    let ids: Vec<&str> = ctx.chunks.iter().map(|c| c.id.as_str()).collect();
    assert!(
        ids.contains(&"1"),
        "the chunk that matches the query must be cited; got {ids:?}",
    );
    assert!(
        ids.contains(&"0"),
        "auto heading expansion must include the section's opener (id=0); \
         got {ids:?}",
    );
    assert!(
        ctx.report.n_expanded >= 1,
        "report should record at least 1 expansion; got n_expanded={}",
        ctx.report.n_expanded,
    );
}

#[test]
fn opt_out_via_false_disables_prose_heading_expansion() {
    let chunks = vec![
        prose_with_heading("0", "section A opener", "A", "doc.md"),
        prose_with_heading("1", "section A deep content with target keyword", "A", "doc.md"),
    ];
    let cfg = DocumentConfig {
        prose_heading_default: false,
        ..Default::default()
    };
    let mut doc = Document::from_chunks_with(chunks, cfg).unwrap();
    let ctx = doc.context("target keyword").unwrap();
    assert_eq!(
        ctx.report.n_expanded, 0,
        "prose_heading_default=false must disable the auto-expansion"
    );
}

#[test]
fn heading_less_prose_is_unaffected_by_the_default() {
    // No chunks carry a heading → the auto-expansion shouldn't fire.
    let chunks = vec![
        {
            let mut c = Chunk::new(ChunkId::new("a"), "refund window thirty days", "notes.txt", TokenCount(4));
            c.metadata.insert("kind".into(), serde_json::json!("prose"));
            c
        },
        {
            let mut c = Chunk::new(ChunkId::new("b"), "shipping takes two days", "notes.txt", TokenCount(4));
            c.metadata.insert("kind".into(), serde_json::json!("prose"));
            c
        },
    ];
    let mut doc = Document::from_chunks_with(chunks, DocumentConfig::default()).unwrap();
    let ctx = doc.context("refund window").unwrap();
    assert_eq!(
        ctx.report.n_expanded, 0,
        "heading-less prose must not trigger the heading auto-expansion"
    );
}
