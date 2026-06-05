use redhop::files::{extract, extract_bytes, ExtractError, MAX_FILE_BYTES};

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
    assert!(d
        .sections
        .iter()
        .any(|s| s.heading.as_deref() == Some("Refund Policy")));

    let md = b"# Title\nintro\n\n## Refunds\nrefund within 30 days\n";
    let m = extract_bytes(md, "policy.md").expect("md bytes");
    assert!(m
        .sections
        .iter()
        .any(|s| s.heading.as_deref() == Some("Refunds")));
}

#[test]
fn extract_bytes_unsupported_extension_errors() {
    assert!(extract_bytes(b"\x89PNG", "image.png").is_err());
}

#[test]
fn docx_text_and_heading() {
    let doc = extract("tests/fixtures/sample.docx").expect("extract docx");
    let text = doc.plain_text();
    assert!(
        text.contains("refund within 30 days"),
        "body text missing: {text}"
    );
    assert!(text.contains("Refund Policy"), "heading missing");
    assert!(text.contains("A-100"), "table cell missing: {text}");
    // heading metadata tracked on later sections
    assert!(
        doc.sections
            .iter()
            .any(|s| s.heading.as_deref() == Some("Refund Policy")),
        "heading metadata not attached"
    );
}

#[test]
fn pptx_slides() {
    let doc = extract("tests/fixtures/sample.pptx").expect("extract pptx");
    let text = doc.plain_text();
    assert!(
        text.contains("Quarterly Review"),
        "slide 1 title missing: {text}"
    );
    assert!(text.contains("twelve percent"), "slide 1 body missing");
    assert!(text.contains("Supply chain"), "slide 2 missing");
    assert!(
        doc.sections.iter().any(|s| s.page == Some(2)),
        "slide page numbers missing"
    );
}

#[test]
fn xlsx_sheets() {
    let doc = extract("tests/fixtures/sample.xlsx").expect("extract xlsx");
    let text = doc.plain_text();
    assert!(text.contains("A-100"), "cell missing: {text}");
    assert!(text.contains("backordered"), "cell missing");
    assert!(
        doc.sections
            .iter()
            .any(|s| s.heading.as_deref() == Some("Pricing")),
        "sheet name"
    );
}

#[test]
fn pdf_text_with_page_numbers() {
    let doc = extract("tests/fixtures/sample.pdf").expect("extract pdf");
    let text = doc.plain_text();
    assert!(text.contains("Delaware"), "pdf text missing: {text}");
    assert!(text.contains("terminate"), "pdf page 2 text missing");
    // One section per page, tagged with its page number.
    assert!(
        doc.sections.iter().any(|s| s.page == Some(1)),
        "page 1 missing"
    );
    assert!(
        doc.sections.iter().any(|s| s.page == Some(2)),
        "page 2 missing"
    );
    // The page-2 term sits in a section tagged page 2, not page 1.
    let p2 = doc
        .sections
        .iter()
        .find(|s| s.text.contains("terminate"))
        .unwrap();
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
    std::fs::write(
        &p,
        "# Intro\nwelcome\n\n## Refunds\nrefund within 30 days\n",
    )
    .unwrap();
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

// ---- failure scenarios -------------------------------------------------------

#[test]
fn empty_extraction_is_no_text() {
    // An empty file — and, by the same central guard, a scanned/image-only PDF
    // whose pages carry no text layer — becomes an actionable NoText error rather
    // than a silent empty document.
    let err = extract_bytes(b"", "blank.txt").unwrap_err();
    assert!(
        matches!(err, ExtractError::NoText(_)),
        "empty → NoText, got {err:?}"
    );
    let err = extract_bytes(b"   \n\t  \n", "ws.txt").unwrap_err();
    assert!(
        matches!(err, ExtractError::NoText(_)),
        "whitespace → NoText"
    );
    // The message names OCR so the user knows what to do with a scan.
    assert!(err.to_string().to_lowercase().contains("ocr"));
}

#[test]
fn binary_as_text_is_rejected() {
    // A binary file masquerading as text (NUL byte in the head) is rejected
    // rather than indexed as replacement-character garbage.
    let err = extract_bytes(b"PK\x03\x04\x00\x00binary\x00stuff", "data.txt").unwrap_err();
    assert!(
        matches!(err, ExtractError::Parse(_)),
        "binary → Parse, got {err:?}"
    );
    assert!(err.to_string().to_lowercase().contains("binary"));
}

#[test]
fn oversize_input_is_rejected() {
    let big = vec![b'a'; (MAX_FILE_BYTES + 1) as usize];
    let err = extract_bytes(&big, "huge.txt").unwrap_err();
    assert!(
        matches!(err, ExtractError::TooLarge { .. }),
        "oversize → TooLarge, got {err:?}"
    );
}

#[test]
fn unsupported_extension_is_reported() {
    let err = extract_bytes(b"whatever", "archive.zip").unwrap_err();
    assert!(matches!(err, ExtractError::Unsupported(_)), "got {err:?}");
}

#[test]
fn corrupt_office_file_is_parse_error() {
    // A .docx (zip-based) that isn't actually a zip → a clean Parse error, not a panic.
    let err = extract_bytes(b"this is not a zip archive", "broken.docx").unwrap_err();
    assert!(
        matches!(err, ExtractError::Parse(_)),
        "corrupt docx → Parse, got {err:?}"
    );
}

#[test]
fn missing_file_is_io_error() {
    let err = extract("tests/fixtures/does_not_exist.pdf").unwrap_err();
    assert!(
        matches!(err, ExtractError::Io(_)),
        "missing → Io, got {err:?}"
    );
}

// ── ADVERSARIAL LOADER INPUTS ──────────────────────────────────────────────
//
// Real-world inputs the older tests don't cover. Each test asserts a
// clean error or clean skip — never a panic. The failure mode we're
// pinning here is "feeding a corrupt PDF/DOCX panics inside a parser
// dependency" or "a symlink loop infinite-recurses the folder walker".

use std::fs;
use tempfile::tempdir;

#[test]
fn extract_bytes_zero_byte_input_errors_cleanly() {
    // 0-byte buffer to a known-supported extension. Each parser should
    // emit `NoText` (or the moral equivalent) rather than panic on the
    // empty slice.
    for name in &[
        "empty.pdf",
        "empty.docx",
        "empty.pptx",
        "empty.xlsx",
        "empty.md",
    ] {
        let r = extract_bytes(b"", name);
        assert!(
            r.is_err(),
            "0-byte {name} should error, not silently succeed: got {r:?}"
        );
    }
}

#[test]
fn extract_bytes_one_byte_input_errors_cleanly() {
    // 1-byte buffer — also too small to be a valid container, must error
    // cleanly across every supported extension.
    for (name, byte) in &[
        ("tiny.pdf", b"%"),
        ("tiny.docx", b"P"),
        ("tiny.pptx", b"P"),
        ("tiny.xlsx", b"P"),
    ] {
        let r = extract_bytes(byte.as_slice(), name);
        assert!(
            r.is_err(),
            "1-byte {name} should error, not silently succeed: got {r:?}"
        );
    }
}

#[test]
fn truncated_pdf_header_only_errors_cleanly() {
    // Just the PDF magic header `%PDF-1.4` — looks like a PDF to a
    // sniffer but has no body. pdf-extract should bail without
    // panicking.
    let truncated = b"%PDF-1.4\n";
    let r = extract_bytes(truncated, "truncated.pdf");
    assert!(
        r.is_err(),
        "truncated PDF (header only) must error, not panic"
    );
}

#[test]
fn docx_missing_document_xml_errors_cleanly() {
    // A valid zip container with NO `word/document.xml` entry — should
    // produce an actionable error, not panic on `unwrap()` inside the
    // DOCX parser.
    use std::io::Write;
    let mut buf = Vec::new();
    {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
        let opts: zip::write::FileOptions<'_, ()> =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        zip.start_file("not_document.xml", opts).unwrap();
        zip.write_all(b"<not the docx body>").unwrap();
        zip.finish().unwrap();
    }
    let r = extract_bytes(&buf, "weird.docx");
    assert!(
        r.is_err(),
        "DOCX missing document.xml must error, not panic: got {r:?}"
    );
}

#[test]
fn very_long_filename_handled_cleanly() {
    // 300-char filename — most filesystems cap at 255, but extract_bytes
    // takes the name as a string and shouldn't care about path length.
    // It just dispatches by extension.
    let long_name = format!("{}.md", "a".repeat(300));
    let r = extract_bytes(b"# Hi\nbody", long_name.as_str());
    assert!(r.is_ok(), "long name should still parse markdown: {r:?}");
}

#[test]
fn read_folder_handles_symlink_loop_without_recursing_forever() {
    // Create a directory that contains a symlink pointing back to itself
    // (or to a parent in the walk path). The folder walker must not
    // infinite-recurse — the `ignore` crate honors the filesystem's
    // canonical-path tracking, but we pin the behavior so a future
    // switch can't regress it.
    #[cfg(unix)]
    {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("real.md"),
            "# Real\nthe refund window is thirty days",
        )
        .unwrap();
        std::os::unix::fs::symlink(dir.path(), dir.path().join("loop"))
            .expect("create symlink loop");

        // Read with a generous timeout via std::thread::scope; if it
        // truly infinite-loops the test process hangs and CI catches it.
        let doc = redhop::read_folder(dir.path()).expect("folder read should not panic");
        // The real file should be indexed; the symlink loop should be
        // ignored (the `ignore` crate's walk doesn't follow into a
        // recursive structure).
        assert!(doc.n_files() >= 1, "the real file should be indexed");
    }
}

#[test]
fn read_folder_handles_deep_recursion() {
    // Build a directory tree 50 levels deep with one .md file at the
    // bottom. Stack-safe walking is the bar here — the walker should
    // not blow the stack on deep recursion.
    let dir = tempdir().unwrap();
    let mut p = dir.path().to_path_buf();
    for i in 0..50 {
        p = p.join(format!("d{i}"));
        fs::create_dir(&p).unwrap();
    }
    fs::write(p.join("deep.md"), "# Deep\nrefund window thirty days").unwrap();

    let doc = redhop::read_folder(dir.path()).expect("deep recursion should not panic");
    assert_eq!(doc.n_files(), 1, "should reach the one file at depth 50");
}

#[test]
fn read_folder_on_empty_directory_errors_cleanly() {
    // A directory that exists but has no readable files at all. The
    // current behavior is to error with "no readable files under ...";
    // pin that contract so a future refactor doesn't silently return an
    // empty Document instead.
    let dir = tempdir().unwrap();
    // `Document` doesn't impl Debug (deliberately — it holds a Tokio
    // runtime), so unwrap_err() doesn't compile. Use an explicit match.
    let err = match redhop::read_folder(dir.path()) {
        Err(e) => e,
        Ok(_) => panic!("empty directory should error, not succeed silently"),
    };
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("no readable files") || msg.contains("empty"),
        "empty directory should error mentioning 'no readable files'; got: {msg}"
    );
}

#[test]
fn read_folder_with_many_files_handles_them() {
    // 200 tiny .md files in one directory — exercise the walker + index
    // build for "lots of small inputs". No assertion on retrieval
    // quality; this is a "doesn't blow up" smoke.
    let dir = tempdir().unwrap();
    for i in 0..200 {
        fs::write(
            dir.path().join(format!("doc{i}.md")),
            format!("# Doc {i}\nrefund window content section {i}"),
        )
        .unwrap();
    }
    let doc = redhop::read_folder(dir.path()).expect("200 files should index cleanly");
    assert_eq!(doc.n_files(), 200);
    assert!(doc.skipped_files().is_empty(), "no files should be skipped");
}
