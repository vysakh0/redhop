//! PDF text extraction via `pdf-extract` (pure-Rust, lopdf-backed). One section
//! per page, each tagged with its 1-based page number, so retrieved chunks can
//! be cited ("contract.pdf, p.3"). The page's first short, heading-shaped line
//! is best-effort lifted into `Section::heading` so the BM25 heading-field
//! search (added in 0.1.3) reaches pages by topic.

use super::{ExtractError, ExtractedDoc, Section};

pub(crate) fn extract_bytes(data: &[u8], source: String) -> Result<ExtractedDoc, ExtractError> {
    let pages = pdf_extract::extract_text_from_mem_by_pages(data)
        .map_err(|e| ExtractError::Parse(format!("pdf: {e}")))?;
    let sections: Vec<Section> = pages
        .into_iter()
        .enumerate()
        .filter(|(_, text)| !text.trim().is_empty())
        .map(|(i, text)| Section {
            heading: page_heading(&text),
            text,
            page: Some(i + 1),
            line: None,
        })
        .collect();
    // Fall back to a single whole-document section if paging yielded nothing
    // (e.g. an unusual PDF) rather than producing an empty document.
    let sections = if sections.is_empty() {
        vec![Section::text(
            pdf_extract::extract_text_from_mem(data)
                .map_err(|e| ExtractError::Parse(format!("pdf: {e}")))?,
        )]
    } else {
        sections
    };
    Ok(ExtractedDoc { source, sections })
}

/// Best-effort heading extraction for a PDF page: take the first non-empty
/// line if it looks like a heading (short, has text, doesn't end with a
/// sentence terminator or a digit, isn't an obvious page-number marker).
/// Returns `None` rather than guess when the first line is body-shaped — a
/// missing heading is better than a garbage one in the BM25 heading field.
fn page_heading(text: &str) -> Option<String> {
    let line = text.lines().find(|l| !l.trim().is_empty())?.trim();
    if !(4..=80).contains(&line.chars().count()) {
        return None;
    }
    if !line.chars().any(|c| c.is_alphabetic()) {
        return None;
    }
    // Body lines typically end with sentence punctuation; headings rarely do.
    if matches!(
        line.chars().last(),
        Some('.') | Some(',') | Some(';') | Some(':') | Some('?') | Some('!')
    ) {
        return None;
    }
    // Trailing digit catches "Section 1 of 25" and many footer-style page
    // numbers; real headings end with a word.
    if line.chars().last().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    let lower = line.to_lowercase();
    if lower.starts_with("page ") || lower.starts_with("p. ") {
        return None;
    }
    // Citations on academic-style PDFs typically open with `[N]` or `[N, M]`.
    // They're body content (the bibliography), not headings, even though the
    // surface shape (short, no terminating period) matches our other rules.
    if line.starts_with('[') && line.contains(']') {
        return None;
    }
    // Bullet / list markers — same shape as headings but they're items in a
    // body list, not section titles. Cover the common ASCII + unicode
    // markers (•, ·, *, -, –, —, →, ►, ▪, ▸).
    if matches!(line.chars().next(), Some(c) if "•·*-–—→►▪▸".contains(c))
        && line.chars().nth(1).is_some_and(|c| c.is_whitespace())
    {
        return None;
    }
    Some(line.to_string())
}

#[cfg(test)]
mod tests {
    use super::page_heading;

    #[test]
    fn page_heading_accepts_obvious_headings() {
        assert_eq!(
            page_heading("MASTER SERVICES AGREEMENT\n\nBody text follows here."),
            Some("MASTER SERVICES AGREEMENT".into())
        );
        assert_eq!(
            page_heading("Section 4 — Limitation of Liability\nThe parties agree…"),
            Some("Section 4 — Limitation of Liability".into())
        );
    }

    #[test]
    fn page_heading_rejects_body_shaped_lines() {
        // Sentence-terminated body lines (a paragraph that starts the page).
        assert_eq!(
            page_heading("The parties hereby agree to the following."),
            None
        );
        // Page number footers / running headers.
        assert_eq!(page_heading("Page 3 of 24"), None);
        assert_eq!(page_heading("Acme Corp · Q3 2025"), None); // ends with digit
                                                               // Too short / no letters / empty.
        assert_eq!(page_heading("§ 3"), None); // ends with digit + too short
        assert_eq!(page_heading("---"), None); // no letters
        assert_eq!(page_heading(""), None);
        assert_eq!(page_heading("\n\n"), None);
    }

    #[test]
    fn page_heading_rejects_citations_and_bullets() {
        // Academic/bibliography citation lines: short, no terminating period,
        // alpha — slipped through the original rules. Now explicitly rejected.
        assert_eq!(page_heading("[1] Smith et al, A Survey of RAG"), None);
        assert_eq!(page_heading("[12, 13] Compare with prior work"), None);
        // Bullet / list markers — bodies, not titles. ASCII + unicode covered.
        assert_eq!(page_heading("• Implementation timeline"), None);
        assert_eq!(page_heading("* Phase one — discovery"), None);
        assert_eq!(page_heading("- Deliverables"), None);
        assert_eq!(page_heading("– Em-dash bullets too"), None);
        assert_eq!(page_heading("→ Next steps"), None);
        // A real numbered heading is still accepted (no bullet marker).
        assert_eq!(
            page_heading("1. Introduction"),
            Some("1. Introduction".into())
        );
    }
}
