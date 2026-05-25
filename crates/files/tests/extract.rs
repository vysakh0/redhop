use redhop_files::extract;

#[test]
fn docx_text_and_heading() {
    let doc = extract("tests/fixtures/sample.docx").expect("extract docx");
    let text = doc.plain_text();
    assert!(text.contains("refund within 30 days"), "body text missing: {text}");
    assert!(text.contains("Refund Policy"), "heading missing");
    assert!(text.contains("A-100"), "table cell missing: {text}");
    // heading metadata tracked on later sections
    assert!(
        doc.sections.iter().any(|s| s.heading.as_deref() == Some("Refund Policy")),
        "heading metadata not attached"
    );
}

#[test]
fn plain_text_file() {
    let p = std::env::temp_dir().join("redhop_files_note.txt");
    std::fs::write(&p, "hello world").unwrap();
    let doc = extract(&p).expect("extract txt");
    assert_eq!(doc.plain_text(), "hello world");
    assert_eq!(doc.source, p.to_string_lossy());
}
