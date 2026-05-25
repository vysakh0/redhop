use redhop_files::{extract, extract_bytes};

#[test]
fn extract_bytes_matches_path_for_pdf() {
    let data = std::fs::read("tests/fixtures/sample.pdf").unwrap();
    let doc = extract_bytes(&data, "contract.pdf").expect("extract pdf bytes");
    assert_eq!(doc.source, "contract.pdf");
    assert!(doc.plain_text().contains("Delaware"));
    assert!(doc.sections.iter().any(|s| s.page == Some(2)));
}

#[test]
fn extract_bytes_docx_and_markdown() {
    let docx = std::fs::read("tests/fixtures/sample.docx").unwrap();
    let d = extract_bytes(&docx, "policy.docx").expect("docx bytes");
    assert!(d.sections.iter().any(|s| s.heading.as_deref() == Some("Refund Policy")));

    let md = b"# Title\nintro\n\n## Refunds\nrefund within 30 days\n";
    let m = extract_bytes(md, "policy.md").expect("md bytes");
    assert!(m.sections.iter().any(|s| s.heading.as_deref() == Some("Refunds")));
}

#[test]
fn extract_bytes_unsupported_extension_errors() {
    assert!(extract_bytes(b"\x89PNG", "image.png").is_err());
}

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
fn pdf_text_with_page_numbers() {
    let doc = extract("tests/fixtures/sample.pdf").expect("extract pdf");
    let text = doc.plain_text();
    assert!(text.contains("Delaware"), "pdf text missing: {text}");
    assert!(text.contains("terminate"), "pdf page 2 text missing");
    // One section per page, tagged with its page number.
    assert!(doc.sections.iter().any(|s| s.page == Some(1)), "page 1 missing");
    assert!(doc.sections.iter().any(|s| s.page == Some(2)), "page 2 missing");
    // The page-2 term sits in a section tagged page 2, not page 1.
    let p2 = doc.sections.iter().find(|s| s.text.contains("terminate")).unwrap();
    assert_eq!(p2.page, Some(2), "page-2 text mis-attributed");
}

#[test]
fn plain_text_file() {
    let p = std::env::temp_dir().join("redhop_files_note.txt");
    std::fs::write(&p, "hello world").unwrap();
    let doc = extract(&p).expect("extract txt");
    assert_eq!(doc.plain_text(), "hello world");
    assert_eq!(doc.source, p.to_string_lossy());
}

#[test]
fn markdown_sections_by_heading() {
    let p = std::env::temp_dir().join("redhop_files_doc.md");
    std::fs::write(&p, "# Intro\nwelcome\n\n## Refunds\nrefund within 30 days\n").unwrap();
    let doc = extract(&p).expect("extract md");
    let refunds = doc
        .sections
        .iter()
        .find(|s| s.heading.as_deref() == Some("Refunds"))
        .expect("Refunds heading section");
    assert!(refunds.text.contains("30 days"));
    assert_eq!(refunds.line, Some(4)); // "## Refunds" is line 4
}

#[test]
fn code_file_blocks_track_lines() {
    let p = std::env::temp_dir().join("redhop_files_mod.py");
    std::fs::write(&p, "def a():\n    return 1\n\n\ndef b():\n    return 2\n").unwrap();
    let doc = extract(&p).expect("extract py");
    assert_eq!(doc.sections.len(), 2);
    assert_eq!(doc.sections[0].line, Some(1));
    assert_eq!(doc.sections[1].line, Some(5));
}
