//! Section extraction for plain-text formats — markdown (by heading) and any
//! other text/code file (by blank-line block). Both tag each section with the
//! 1-based line it starts at, so retrieved chunks can be cited ("notes.md →
//! Setup", "main.py:42").

use super::Section;

/// Split markdown into sections at ATX (`#`, `##`, …) **or** setext
/// (`Title\n=====` / `Title\n-----`) headings. Each section runs from one
/// heading to the next and carries that heading's text plus the 1-based line
/// it starts on. Content before the first heading becomes a heading-less
/// section.
pub fn markdown_sections(raw: &str) -> Vec<Section> {
    let lines: Vec<&str> = raw.lines().collect();
    let mut sections: Vec<Section> = Vec::new();
    let mut cur = String::new();
    let mut cur_heading: Option<String> = None;
    let mut cur_line = 1usize;

    let flush = |sections: &mut Vec<Section>, text: &str, heading: &Option<String>, line: usize| {
        if !text.trim().is_empty() {
            sections.push(Section {
                text: text.trim_end().to_string(),
                page: None,
                heading: heading.clone(),
                line: Some(line),
            });
        }
    };

    // YAML frontmatter (`---` ... `---` at file start) shouldn't trigger
    // setext detection — the closing `---` would otherwise look like an H2
    // underline under the last YAML key. We still keep the frontmatter lines
    // in the indexed body (some users search by `author:` etc.); just skip
    // heading extraction inside the block.
    let frontmatter_end: Option<usize> = if lines.first().map(|l| l.trim()) == Some("---") {
        (1..lines.len()).find(|&j| lines[j].trim() == "---")
    } else {
        None
    };
    let in_frontmatter = |i: usize| frontmatter_end.is_some_and(|end| i <= end);

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];

        if !in_frontmatter(i) {
            if let Some(title) = atx_heading(line) {
                flush(&mut sections, &cur, &cur_heading, cur_line);
                cur.clear();
                cur_heading = Some(title);
                cur_line = i + 1;
                cur.push_str(line);
                cur.push('\n');
                i += 1;
                continue;
            }

            // Setext: a non-empty line whose NEXT line is all `=` or `-`
            // (≥3 chars, possibly with trailing whitespace). The title line
            // is included in the section body (parallel to ATX); the
            // underline is skipped so it doesn't pollute the body text.
            if i + 1 < lines.len()
                && !in_frontmatter(i + 1)
                && setext_underline_level(lines[i + 1]).is_some()
                && !line.trim().is_empty()
                && setext_underline_level(line).is_none()
            {
                let title = line.trim();
                flush(&mut sections, &cur, &cur_heading, cur_line);
                cur.clear();
                cur_heading = Some(title.to_string());
                cur_line = i + 1;
                cur.push_str(line);
                cur.push('\n');
                i += 2;
                continue;
            }
        }

        cur.push_str(line);
        cur.push('\n');
        i += 1;
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

/// `Some(1)` for `===…` (H1), `Some(2)` for `---…` (H2), `None` otherwise.
/// Requires at least 3 consecutive chars to match CommonMark.
fn setext_underline_level(line: &str) -> Option<usize> {
    let t = line.trim_end();
    if t.len() < 3 {
        return None;
    }
    if t.chars().all(|c| c == '=') {
        return Some(1);
    }
    if t.chars().all(|c| c == '-') {
        return Some(2);
    }
    None
}

/// The heading text of an ATX markdown heading line (`## Title` → `Title`), or
/// `None` if the line isn't a heading.
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

/// Split source code into blank-line blocks, each labeled with the **nearest
/// preceding definition** (function/class/…) as its heading — so citations read
/// `auth.py → def login`, not just a line number. Heuristic and language-agnostic
/// (no parser); falls back to plain line-blocks when no definitions are found.
pub fn code_sections(raw: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut block = String::new();
    let mut block_start = 1usize;
    let mut in_block = false;
    let mut current_symbol: Option<String> = None;
    let mut block_symbol: Option<String> = None;

    for (i, line) in raw.lines().enumerate() {
        if let Some(sym) = symbol_signature(line) {
            current_symbol = Some(sym);
        }
        if line.trim().is_empty() {
            if in_block {
                sections.push(Section {
                    text: block.trim_end().to_string(),
                    page: None,
                    heading: block_symbol.clone(),
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
            block_symbol = current_symbol.clone();
        }
        block.push_str(line);
        block.push('\n');
    }
    if in_block && !block.trim().is_empty() {
        sections.push(Section {
            text: block.trim_end().to_string(),
            page: None,
            heading: block_symbol,
            line: Some(block_start),
        });
    }
    if sections.is_empty() {
        sections.push(Section::text(raw.trim_end().to_string()));
    }
    sections
}

/// If `line` declares a function/class/etc., return a readable signature for it
/// (used as a citation heading). Strips leading visibility/async modifiers and
/// matches a small set of cross-language keywords.
fn symbol_signature(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let mut rest = trimmed;
    const MODIFIERS: &[&str] = &[
        "pub ",
        "export ",
        "default ",
        "public ",
        "private ",
        "protected ",
        "static ",
        "final ",
        "async ",
        "open ",
        "override ",
        "abstract ",
    ];
    loop {
        let mut stripped = false;
        for m in MODIFIERS {
            if let Some(s) = rest.strip_prefix(m) {
                rest = s.trim_start();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    const KW: &[&str] = &[
        "def ",
        "class ",
        "fn ",
        "func ",
        "function ",
        "impl ",
        "struct ",
        "enum ",
        "trait ",
        "interface ",
        "module ",
        "package ",
        "sub ",
    ];
    if KW.iter().any(|k| rest.starts_with(k)) {
        let sig = trimmed.trim_end_matches('{').trim_end();
        let sig = sig.trim_end_matches(':').trim_end();
        let sig: String = sig.chars().take(80).collect();
        Some(sig)
    } else {
        None
    }
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
        let setup = secs
            .iter()
            .find(|s| s.heading.as_deref() == Some("Setup"))
            .unwrap();
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
    fn code_sections_label_blocks_by_symbol() {
        let code = "import os\n\ndef login(user):\n    return user.token\n\nclass Account:\n    def close(self):\n        pass\n";
        let secs = code_sections(code);
        // import preamble has no symbol; the def/class blocks are labeled.
        let login = secs
            .iter()
            .find(|s| s.text.contains("return user.token"))
            .unwrap();
        assert_eq!(login.heading.as_deref(), Some("def login(user)"));
        let acct = secs
            .iter()
            .find(|s| s.text.contains("class Account"))
            .unwrap();
        assert_eq!(acct.heading.as_deref(), Some("class Account"));
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

    #[test]
    fn markdown_recognizes_setext_headings() {
        // Pre-fix only ATX (`#`) split sections; setext-style markdown (still
        // common in pandoc output / older docs / man pages) was silently
        // ignored — no heading metadata, no section break.
        let md = "\
Top Title
=========

intro paragraph

Sub Section
-----------

more text
";
        let secs = markdown_sections(md);
        let headings: Vec<_> = secs.iter().map(|s| s.heading.as_deref()).collect();
        assert_eq!(headings, vec![Some("Top Title"), Some("Sub Section")]);
        // The underline line itself is NOT in the body — body starts with
        // the title, then content (parallel to ATX behavior).
        let sub = secs
            .iter()
            .find(|s| s.heading == Some("Sub Section".into()))
            .unwrap();
        assert!(
            !sub.text.contains("---"),
            "underline must not leak into body: {sub:?}"
        );
        assert!(sub.text.contains("more text"));
    }

    #[test]
    fn setext_does_not_misfire_on_yaml_frontmatter() {
        let md = "\
---
title: a doc
date: 2026
---
# Real Heading

body
";
        let secs = markdown_sections(md);
        let headings: Vec<_> = secs.iter().map(|s| s.heading.as_deref()).collect();
        assert_eq!(headings, vec![None, Some("Real Heading")]);
    }

    #[test]
    fn setext_mixed_with_atx() {
        let md = "\
ATX-Style
=========

para A

## Mid Atx

para B

Tail Setext
-----------

para C
";
        let secs = markdown_sections(md);
        let headings: Vec<_> = secs.iter().map(|s| s.heading.as_deref()).collect();
        assert_eq!(
            headings,
            vec![Some("ATX-Style"), Some("Mid Atx"), Some("Tail Setext")]
        );
    }
}
