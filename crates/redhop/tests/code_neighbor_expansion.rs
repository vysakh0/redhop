//! `Document.context(query)` on code-classified chunks auto-attaches ±1
//! neighbor chunks by default (`DocumentConfig::code_neighbors_default`), so
//! citations on a hit at a function's `def` line include the next chunk
//! (the implementation body) rather than only the signature.
//!
//! Prose chunks are untouched — the auto-expansion fires only on chunks
//! tagged `metadata["kind"] == "code"`.

use redhop::core::{Chunk, ChunkId, TokenCount};
use redhop::{Document, DocumentConfig};

fn code(id: &str, text: &str, source: &str) -> Chunk {
    let mut c = Chunk::new(
        ChunkId::new(id),
        text,
        source,
        TokenCount(text.split_whitespace().count()),
    );
    c.metadata.insert("kind".into(), serde_json::json!("code"));
    c
}

fn prose(id: &str, text: &str) -> Chunk {
    let mut c = Chunk::new(
        ChunkId::new(id),
        text,
        "notes.md",
        TokenCount(text.split_whitespace().count()),
    );
    c.metadata.insert("kind".into(), serde_json::json!("prose"));
    c
}

#[test]
fn code_chunks_pull_a_neighbor_by_default() {
    // Three code chunks from the same file — the second one literally says
    // "compress_video" but the third is the implementation body that a
    // citation should include along with the def chunk.
    let chunks = vec![
        code(
            "0",
            "use crate::services::video::compress_video as service_compress;",
            "video.rs",
        ),
        code(
            "1",
            "pub async fn compress_video(file_path: &str, quality: &str)",
            "video.rs",
        ),
        code(
            "2",
            "let result = service_compress(file_path, quality).await?; Ok(result)",
            "video.rs",
        ),
        prose(
            "notes-0",
            "unrelated changelog entry about the build pipeline",
        ),
    ];

    let mut doc = Document::from_chunks_with(chunks, DocumentConfig::default()).unwrap();
    let ctx = doc.context("compress_video").unwrap();

    let ids: Vec<&str> = ctx.chunks.iter().map(|c| c.id.as_str()).collect();
    assert!(
        ids.contains(&"1"),
        "the chunk that matches the query must be cited; got {ids:?}",
    );
    assert!(
        ids.contains(&"2"),
        "auto neighbor expansion must include the implementation body chunk; \
         got {ids:?}",
    );
    assert!(
        ctx.report.n_expanded >= 1,
        "report should record at least 1 expansion; got n_expanded={}",
        ctx.report.n_expanded,
    );
}

#[test]
fn opt_out_via_zero_disables_auto_expansion() {
    let chunks = vec![
        code("0", "fn helper_a() { ... }", "lib.rs"),
        code("1", "fn target() { compress() }", "lib.rs"),
        code("2", "fn helper_b() { ... }", "lib.rs"),
    ];
    let cfg = DocumentConfig {
        code_neighbors_default: 0,
        ..Default::default()
    };
    let mut doc = Document::from_chunks_with(chunks, cfg).unwrap();
    let ctx = doc.context("target compress").unwrap();
    assert_eq!(
        ctx.report.n_expanded, 0,
        "code_neighbors_default=0 must disable the auto-expansion"
    );
}

#[test]
fn prose_only_corpora_are_not_affected_by_the_default() {
    // A document with no code-classified chunks should behave identically to
    // pre-0.1.4: just the retrieved seeds, no neighbor expansion.
    let chunks = vec![
        prose("a", "the refund window is thirty days from purchase"),
        prose("b", "shipping takes two business days from order"),
        prose("c", "warranty extends for one year after delivery"),
    ];
    let mut doc = Document::from_chunks_with(chunks, DocumentConfig::default()).unwrap();
    let ctx = doc.context("refund window").unwrap();
    assert_eq!(
        ctx.report.n_expanded, 0,
        "prose-only corpora must not trigger the code auto-expansion default"
    );
}
