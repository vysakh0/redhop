//! PDF text extraction via `pdf-extract` (pure-Rust, lopdf-backed). Whole-document
//! text for now; per-page sections (for page citations) are a follow-up.

use std::path::Path;

use crate::{ExtractError, ExtractedDoc, Section};

pub(crate) fn extract(path: &Path, source: String) -> Result<ExtractedDoc, ExtractError> {
    let text = pdf_extract::extract_text(path)
        .map_err(|e| ExtractError::Parse(format!("pdf: {e}")))?;
    Ok(ExtractedDoc { source, sections: vec![Section::text(text)] })
}
