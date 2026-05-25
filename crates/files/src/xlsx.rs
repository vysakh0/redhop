//! XLSX / spreadsheet text extraction via `calamine`. Each sheet becomes a
//! section (heading = sheet name); rows are pipe-joined cells.

use std::path::Path;

use calamine::{open_workbook_auto, Reader};

use crate::{ExtractError, ExtractedDoc, Section};

pub(crate) fn extract(path: &Path, source: String) -> Result<ExtractedDoc, ExtractError> {
    let mut wb =
        open_workbook_auto(path).map_err(|e| ExtractError::Parse(format!("xlsx: {e}")))?;
    let mut sections = Vec::new();
    for name in wb.sheet_names().to_owned() {
        let Ok(range) = wb.worksheet_range(&name) else {
            continue;
        };
        let mut lines = Vec::new();
        for row in range.rows() {
            let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
            let line = cells.join(" | ");
            if !line.trim().is_empty() {
                lines.push(line);
            }
        }
        if !lines.is_empty() {
            sections.push(Section { text: lines.join("\n"), page: None, heading: Some(name) });
        }
    }
    Ok(ExtractedDoc { source, sections })
}
