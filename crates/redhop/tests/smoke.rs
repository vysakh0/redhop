//! The façade re-exports and loaders work end to end (lexical — no model needed).

#[test]
fn from_text_to_context() {
    let mut doc = redhop::Document::from_text("doc", "the refund window is thirty days").unwrap();
    let ctx = doc.context("refund window").unwrap();
    assert!(!ctx.text().is_empty());
    assert!(ctx.text().to_lowercase().contains("refund"));
}

#[cfg(feature = "files")]
#[test]
fn read_bytes_parses_and_cites() {
    // Markdown bytes → parsed, chunked, with a heading citation.
    let mut doc =
        redhop::read_bytes(b"# Policy\n\n## Refunds\nrefund within thirty days\n", "policy.md")
            .unwrap();
    let ctx = doc.context("refund within thirty days").unwrap();
    let hit = ctx.chunks.iter().find(|c| c.text.to_lowercase().contains("refund"));
    assert!(hit.is_some());
    assert_eq!(hit.unwrap().source, "policy.md");
}
