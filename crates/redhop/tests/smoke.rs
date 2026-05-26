//! The façade re-exports and loaders work end to end (lexical — no model needed).

#[test]
fn from_text_to_context() {
    let mut doc = redhop::Document::from_text("doc", "the refund window is thirty days").unwrap();
    let ctx = doc.context("refund window").unwrap();
    assert!(!ctx.text().is_empty());
    assert!(ctx.text().to_lowercase().contains("refund"));
}

#[test]
fn citations_accessor() {
    let mut doc = redhop::Document::from_text("doc", "the refund window is thirty days").unwrap();
    let ctx = doc.context("refund window").unwrap();
    let cites = redhop::citations(&ctx);
    assert!(!cites.is_empty());
    assert_eq!(cites[0].source, "doc");
    assert!(cites[0].page.is_none());
}

#[cfg(feature = "files")]
#[test]
fn read_bytes_parses_and_cites() {
    // Markdown bytes → parsed, chunked, with a heading citation.
    let mut doc =
        redhop::read_bytes(b"# Policy\n\n## Refunds\nrefund within thirty days\n", "policy.md")
            .unwrap();
    let ctx = doc.context("refund within thirty days").unwrap();
    let cites = redhop::citations(&ctx);
    let hit = cites.iter().find(|c| c.text.to_lowercase().contains("refund")).unwrap();
    assert_eq!(hit.source, "policy.md");
    assert_eq!(hit.heading.as_deref(), Some("Refunds"));
}

#[cfg(feature = "files")]
#[test]
fn read_folder_and_persist() {
    use redhop::FolderOptions;
    let dir = std::env::temp_dir().join(format!("rh-facade-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("a.txt"), "the refund window is thirty days").unwrap();
    std::fs::write(dir.join("sub/b.md"), "# Shipping\norders ship in two days").unwrap();

    // in-memory
    let doc = redhop::read_folder(&dir).unwrap();
    assert!(doc.len() >= 2);

    // ignore globs
    let doc = redhop::read_folder_with(
        &dir,
        &FolderOptions { ignore: vec!["**/*.md".into()], ..Default::default() },
    )
    .unwrap();
    assert_eq!(doc.len(), 1); // the .md is excluded

    // persist: writes an index, reload reuses it
    let opts = FolderOptions { persist: true, ..Default::default() };
    let _ = redhop::read_folder_with(&dir, &opts).unwrap();
    assert!(dir.join(".redhop/index.json").exists());
    let mut doc = redhop::read_folder_with(&dir, &opts).unwrap();
    assert!(redhop::citations(&doc.context("refund thirty days").unwrap())
        .iter()
        .any(|c| c.source.ends_with("a.txt")));

    let _ = std::fs::remove_dir_all(&dir);
}
