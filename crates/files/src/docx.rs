//! DOCX text extraction via `docx-rs` — paragraphs (with heading tracking) and
//! flattened table cells. Text only; styling/images are ignored.

use std::path::Path;

use docx_rs::{
    DocumentChild, Paragraph, ParagraphChild, RunChild, Table, TableChild, TableCellContent,
    TableRowChild,
};

use crate::{ExtractError, ExtractedDoc, Section};

pub(crate) fn extract(path: &Path, source: String) -> Result<ExtractedDoc, ExtractError> {
    let buf = std::fs::read(path).map_err(ExtractError::Io)?;
    let docx = docx_rs::read_docx(&buf).map_err(|e| ExtractError::Parse(format!("docx: {e:?}")))?;

    let mut sections = Vec::new();
    let mut heading: Option<String> = None;

    for child in &docx.document.children {
        match child {
            DocumentChild::Paragraph(p) => {
                let text = paragraph_text(p);
                if text.trim().is_empty() {
                    continue;
                }
                if is_heading(p) {
                    heading = Some(text.clone());
                }
                sections.push(Section { text, page: None, heading: heading.clone() });
            }
            DocumentChild::Table(t) => {
                let text = table_text(t);
                if !text.trim().is_empty() {
                    sections.push(Section { text, page: None, heading: heading.clone() });
                }
            }
            _ => {}
        }
    }

    Ok(ExtractedDoc { source, sections })
}

fn is_heading(p: &Paragraph) -> bool {
    p.property
        .style
        .as_ref()
        .map(|s| s.val.to_ascii_lowercase().contains("heading"))
        .unwrap_or(false)
}

fn paragraph_text(p: &Paragraph) -> String {
    let mut s = String::new();
    for c in &p.children {
        if let ParagraphChild::Run(run) = c {
            for rc in &run.children {
                match rc {
                    RunChild::Text(t) => s.push_str(&t.text),
                    RunChild::Tab(_) => s.push('\t'),
                    _ => {}
                }
            }
        }
    }
    s
}

fn table_text(t: &Table) -> String {
    let mut rows = Vec::new();
    for TableChild::TableRow(r) in &t.rows {
        let mut cells = Vec::new();
        for TableRowChild::TableCell(cell) in &r.cells {
            let mut cell_text = String::new();
            for content in &cell.children {
                if let TableCellContent::Paragraph(p) = content {
                    if !cell_text.is_empty() {
                        cell_text.push(' ');
                    }
                    cell_text.push_str(&paragraph_text(p));
                }
            }
            cells.push(cell_text);
        }
        rows.push(cells.join(" | "));
    }
    rows.join("\n")
}
