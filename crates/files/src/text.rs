//! Section extraction for plain-text formats — markdown (by heading) and any
//! other text/code file (by blank-line block). Both tag each section with the
//! 1-based line it starts at, so retrieved chunks can be cited ("notes.md →
//! Setup", "main.py:42").

use crate::Section;

/// Split markdown into sections at ATX headings (`#`, `##`, …). Each section
/// runs from one heading to the next and carries that heading's text plus the
/// 1-based line it starts on. Content before the first heading becomes a
/// heading-less section.
pub fn markdown_sections(raw: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut cur = String::new();
    let mut cur_heading: Option<String> = None;
    let mut cur_line = 1usize;

    let flush = |sections: &mut Vec<Section>,
                 text: &str,
                 heading: &Option<String>,
                 line: usize| {
        if !text.trim().is_empty() {
            sections.push(Section {
                text: text.trim_end().to_string(),
                page: None,
                heading: heading.clone(),
                line: Some(line),
            });
        }
    };

    for (i, line) in raw.lines().enumerate() {
        if let Some(title) = atx_heading(line) {
            flush(&mut sections, &cur, &cur_heading, cur_line);
            cur.clear();
            cur_heading = Some(title);
            cur_line = i + 1;
        }
        cur.push_str(line);
        cur.push('\n');
    }
    flush(&mut sections, &cur, &cur_heading, cur_line);

    // A file with no headings at all → fall back to line-block splitting so we
    // still get useful per-block line citations.
    let has_heading = sections.iter().any(|s| s.heading.is_some());
    if !has_heading {
        return line_blocks(raw);
    }
    sections
}

/// The heading text of an ATX markdown heading line (`## Title` → `Title`), or
/// `None` if the line isn't a heading. Setext (`===`) headings are ignored.
fn atx_heading(line: &str) -> Option<String> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let hashes = t.chars().take_while(|c| *c == '#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &t[hashes..];
    // Must be `#` followed by a space (or be empty) — `#foo` is not a heading.
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim();
    if title.is_empty() {
        None
    } else {
        Some(title.to_string())
    }
}

/// Split any text or code file into blank-line-separated blocks (paragraphs,
/// stanzas, functions), each tagged with the 1-based line it starts on. Files
/// with no blank lines become a single section starting at line 1.
pub fn line_blocks(raw: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut block = String::new();
    let mut block_start = 1usize;
    let mut in_block = false;

    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            if in_block {
                sections.push(Section {
                    text: block.trim_end().to_string(),
                    page: None,
                    heading: None,
                    line: Some(block_start),
                });
                block.clear();
                in_block = false;
            }
            continue;
        }
        if !in_block {
            block_start = i + 1;
            in_block = true;
        }
        block.push_str(line);
        block.push('\n');
    }
    if in_block && !block.trim().is_empty() {
        sections.push(Section {
            text: block.trim_end().to_string(),
            page: None,
            heading: None,
            line: Some(block_start),
        });
    }
    if sections.is_empty() {
        sections.push(Section::text(raw.trim_end().to_string()));
    }
    sections
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_splits_by_heading_with_lines() {
        let md = "# Title\nintro line\n\n## Setup\ninstall it\nrun it\n\n## Usage\ncall it\n";
        let secs = markdown_sections(md);
        let headings: Vec<_> = secs.iter().map(|s| s.heading.as_deref()).collect();
        assert_eq!(headings, vec![Some("Title"), Some("Setup"), Some("Usage")]);
        // "## Setup" is on line 4.
        let setup = secs.iter().find(|s| s.heading.as_deref() == Some("Setup")).unwrap();
        assert_eq!(setup.line, Some(4));
        assert!(setup.text.contains("install it"));
    }

    #[test]
    fn non_heading_markdown_falls_back_to_blocks() {
        let md = "just a paragraph\nwith two lines\n\nand another block\n";
        let secs = markdown_sections(md);
        assert_eq!(secs.len(), 2);
        assert_eq!(secs[0].line, Some(1));
        assert_eq!(secs[1].line, Some(4));
        assert!(secs.iter().all(|s| s.heading.is_none()));
    }

    #[test]
    fn line_blocks_track_start_line() {
        let code = "def a():\n    return 1\n\n\ndef b():\n    return 2\n";
        let secs = line_blocks(code);
        assert_eq!(secs.len(), 2);
        assert_eq!(secs[0].line, Some(1));
        assert_eq!(secs[1].line, Some(5));
    }

    #[test]
    fn no_blank_lines_is_one_section() {
        let secs = line_blocks("a\nb\nc\n");
        assert_eq!(secs.len(), 1);
        assert_eq!(secs[0].line, Some(1));
    }
}
