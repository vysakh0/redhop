//! Lean text extraction for RedHop's `from_file` / `from_folder` ingestion.
//!
//! Turns a file on disk into plain text + light structural metadata (page /
//! heading) for chunking and citations. Text-only by design — no image, layout,
//! or position extraction (that's a rendering concern, not retrieval). Each
//! format is added behind the same [`extract`] entry point.
//!
//! Supported: UTF-8 text & markdown/code, **DOCX**, **PPTX**, **XLSX**, **PDF**.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::path::Path;

mod docx;
mod pdf;
mod pptx;
mod xlsx;

/// One extracted unit of a document — a paragraph/section/slide/page of text,
/// with optional provenance used for citations.
#[derive(Debug, Clone)]
pub struct Section {
    /// The text of this section.
    pub text: String,
    /// 1-based page or slide number, when the format has them (PDF, PPTX).
    pub page: Option<usize>,
    /// Nearest heading/title this section sits under, when known.
    pub heading: Option<String>,
}

impl Section {
    /// A bare text section (no page/heading).
    pub fn text(text: impl Into<String>) -> Self {
        Self { text: text.into(), page: None, heading: None }
    }
}

/// The result of extracting a file: its source path + ordered text sections.
#[derive(Debug, Clone)]
pub struct ExtractedDoc {
    /// The file path this came from (becomes each chunk's `source`).
    pub source: String,
    /// Ordered text sections.
    pub sections: Vec<Section>,
}

impl ExtractedDoc {
    /// Join all sections into one plain-text string.
    pub fn plain_text(&self) -> String {
        self.sections
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// What can go wrong extracting a file.
#[derive(Debug)]
pub enum ExtractError {
    /// The file couldn't be read.
    Io(std::io::Error),
    /// The format isn't supported yet (the extension is listed).
    Unsupported(String),
    /// The file's bytes couldn't be parsed.
    Parse(String),
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractError::Io(e) => write!(f, "could not read file: {e}"),
            ExtractError::Unsupported(ext) => write!(
                f,
                "extracting '{ext}' files isn't supported (supported: text/markdown/code, DOCX, \
                 PPTX, XLSX, PDF) — extract the text yourself and use from_text()"
            ),
            ExtractError::Parse(m) => write!(f, "could not parse file: {m}"),
        }
    }
}
impl std::error::Error for ExtractError {}

/// Extensions we treat as plain UTF-8 text (read directly).
fn is_text_ext(ext: &str) -> bool {
    matches!(
        ext,
        "txt" | "md" | "markdown" | "mdx" | "rst" | "text" | "log" | "csv" | "tsv"
            | "json" | "jsonl" | "yaml" | "yml" | "toml" | "xml" | "html" | "htm"
            | "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "java" | "c" | "h"
            | "cpp" | "hpp" | "rb" | "php" | "sh" | "sql" | "tex"
    )
}

/// Extract a file to text + metadata, dispatching on its extension.
pub fn extract(path: impl AsRef<Path>) -> Result<ExtractedDoc, ExtractError> {
    let path = path.as_ref();
    let source = path.to_string_lossy().into_owned();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "docx" => docx::extract(path, source),
        "pptx" => pptx::extract(path, source),
        "xlsx" | "xlsm" | "xls" | "ods" => xlsx::extract(path, source),
        "pdf" => pdf::extract(path, source),
        e if is_text_ext(e) || e.is_empty() => {
            let text = std::fs::read_to_string(path).map_err(ExtractError::Io)?;
            Ok(ExtractedDoc { source, sections: vec![Section::text(text)] })
        }
        other => Err(ExtractError::Unsupported(other.to_string())),
    }
}
