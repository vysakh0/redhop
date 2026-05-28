//! XLSX / spreadsheet text extraction via `calamine`. Each sheet becomes a
//! section (heading = sheet name); rows are pipe-joined cells.

use calamine::{open_workbook_auto_from_rs, Reader};

use super::{ExtractError, ExtractedDoc, Section};

pub(crate) fn extract_bytes(data: &[u8], source: String) -> Result<ExtractedDoc, ExtractError> {
    let wb = open_workbook_auto_from_rs(std::io::Cursor::new(data.to_vec()))
        .map_err(|e| ExtractError::Parse(format!("xlsx: {e}")))?;
    sheets_to_sections(wb, source)
}

fn sheets_to_sections<RS: std::io::Read + std::io::Seek>(
    mut wb: calamine::Sheets<RS>,
    source: String,
) -> Result<ExtractedDoc, ExtractError> {
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
            sections.push(Section {
                text: lines.join("\n"),
                page: None,
                heading: Some(name),
                line: None,
            });
        }
    }
    Ok(ExtractedDoc { source, sections })
}
