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
fn pptx_slides() {
    let doc = extract("tests/fixtures/sample.pptx").expect("extract pptx");
    let text = doc.plain_text();
    assert!(text.contains("Quarterly Review"), "slide 1 title missing: {text}");
    assert!(text.contains("twelve percent"), "slide 1 body missing");
    assert!(text.contains("Supply chain"), "slide 2 missing");
    assert!(doc.sections.iter().any(|s| s.page == Some(2)), "slide page numbers missing");
}

#[test]
fn xlsx_sheets() {
    let doc = extract("tests/fixtures/sample.xlsx").expect("extract xlsx");
    let text = doc.plain_text();
    assert!(text.contains("A-100"), "cell missing: {text}");
    assert!(text.contains("backordered"), "cell missing");
    assert!(doc.sections.iter().any(|s| s.heading.as_deref() == Some("Pricing")), "sheet name");
}

#[test]
fn pdf_text() {
    let doc = extract("tests/fixtures/sample.pdf").expect("extract pdf");
    let text = doc.plain_text();
    assert!(text.contains("Delaware"), "pdf text missing: {text}");
    assert!(text.contains("terminate"), "pdf page 2 text missing");
}

#[test]
fn plain_text_file() {
    let p = std::env::temp_dir().join("redhop_files_note.txt");
    std::fs::write(&p, "hello world").unwrap();
    let doc = extract(&p).expect("extract txt");
    assert_eq!(doc.plain_text(), "hello world");
    assert_eq!(doc.source, p.to_string_lossy());
}
