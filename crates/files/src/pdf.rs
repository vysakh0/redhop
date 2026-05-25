//! PDF text extraction via `pdf-extract` (pure-Rust, lopdf-backed). One section
//! per page, each tagged with its 1-based page number, so retrieved chunks can
//! be cited ("contract.pdf, p.3").

use std::path::Path;

use crate::{ExtractError, ExtractedDoc, Section};

pub(crate) fn extract(path: &Path, source: String) -> Result<ExtractedDoc, ExtractError> {
    let pages = pdf_extract::extract_text_by_pages(path)
        .map_err(|e| ExtractError::Parse(format!("pdf: {e}")))?;
    let sections: Vec<Section> = pages
        .into_iter()
        .enumerate()
        .filter(|(_, text)| !text.trim().is_empty())
        .map(|(i, text)| Section {
            text,
            page: Some(i + 1),
            heading: None,
            line: None,
        })
        .collect();
    // Fall back to a single whole-document section if paging yielded nothing
    // (e.g. an unusual PDF) rather than producing an empty document.
    let sections = if sections.is_empty() {
        vec![Section::text(
            pdf_extract::extract_text(path)
                .map_err(|e| ExtractError::Parse(format!("pdf: {e}")))?,
        )]
    } else {
        sections
    };
    Ok(ExtractedDoc { source, sections })
}
