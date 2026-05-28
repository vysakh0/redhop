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
mod text;
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
    /// 1-based line number this section starts at, for text & code files.
    pub line: Option<usize>,
}

impl Section {
    /// A bare text section (no page/heading/line).
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            page: None,
            heading: None,
            line: None,
        }
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

/// Largest input we'll read into memory for in-process extraction (100 MiB).
/// A bound on raw input also limits the blast radius of decompression bombs.
pub const MAX_FILE_BYTES: u64 = 100 * 1024 * 1024;

/// What can go wrong extracting a file.
#[derive(Debug)]
pub enum ExtractError {
    /// The file couldn't be read.
    Io(std::io::Error),
    /// The format isn't supported yet (the extension is listed).
    Unsupported(String),
    /// The file's bytes couldn't be parsed.
    Parse(String),
    /// The file parsed but yielded no usable text — empty, or an image-only /
    /// scanned document (carries the source path).
    NoText(String),
    /// The input is larger than [`MAX_FILE_BYTES`].
    TooLarge {
        /// Source path / name.
        source: String,
        /// Actual size in bytes.
        bytes: u64,
    },
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
            ExtractError::NoText(src) => write!(
                f,
                "no text could be extracted from '{src}'. The file may be empty, or an image-only \
                 / scanned document (e.g. a scanned PDF) — run OCR to produce a text layer, then \
                 index the result"
            ),
            ExtractError::TooLarge { source, bytes } => write!(
                f,
                "'{source}' is {bytes} bytes, over the {MAX_FILE_BYTES}-byte in-process extraction \
                 limit — split it, or extract the text yourself and use from_text()"
            ),
        }
    }
}
impl std::error::Error for ExtractError {}

/// Decode raw bytes as UTF-8 text, leniently — but reject binary data (a NUL byte
/// in the head is a reliable "this isn't text" signal) with a clear message rather
/// than indexing replacement-character garbage.
fn decode_text(data: &[u8]) -> Result<String, ExtractError> {
    if data.iter().take(8192).any(|&b| b == 0) {
        return Err(ExtractError::Parse(
            "appears to be binary data, not UTF-8 text".into(),
        ));
    }
    Ok(String::from_utf8_lossy(data).into_owned())
}

/// Extensions we treat as plain UTF-8 text (read directly).
fn is_text_ext(ext: &str) -> bool {
    matches!(
        ext,
        "txt"
            | "md"
            | "markdown"
            | "mdx"
            | "rst"
            | "text"
            | "log"
            | "csv"
            | "tsv"
            | "json"
            | "jsonl"
            | "yaml"
            | "yml"
            | "toml"
            | "xml"
            | "html"
            | "htm"
            | "rs"
            | "py"
            | "js"
            | "ts"
            | "tsx"
            | "jsx"
            | "go"
            | "java"
            | "c"
            | "h"
            | "cpp"
            | "hpp"
            | "rb"
            | "php"
            | "sh"
            | "sql"
            | "tex"
    )
}

/// Source-code extensions — sectioned by definition (symbol-aware), unlike prose.
fn is_code_ext(ext: &str) -> bool {
    matches!(
        ext,
        "rs" | "py"
            | "js"
            | "ts"
            | "tsx"
            | "jsx"
            | "mjs"
            | "cjs"
            | "go"
            | "java"
            | "kt"
            | "c"
            | "h"
            | "cpp"
            | "hpp"
            | "cc"
            | "hh"
            | "cs"
            | "rb"
            | "php"
            | "sh"
            | "bash"
            | "swift"
            | "scala"
            | "lua"
            | "pl"
            | "ml"
            | "ex"
            | "exs"
    )
}

/// Extract a file to text + metadata, dispatching on its extension. Enforces the
/// size limit up front (via the file's metadata, before reading it), then routes
/// through [`extract_bytes`] so path and in-memory ingestion behave identically.
pub fn extract(path: impl AsRef<Path>) -> Result<ExtractedDoc, ExtractError> {
    let path = path.as_ref();
    let source = path.to_string_lossy().into_owned();
    let meta = std::fs::metadata(path).map_err(ExtractError::Io)?;
    if meta.len() > MAX_FILE_BYTES {
        return Err(ExtractError::TooLarge {
            source,
            bytes: meta.len(),
        });
    }
    let data = std::fs::read(path).map_err(ExtractError::Io)?;
    extract_bytes(&data, &source)
}

/// Extract **in-memory bytes** (already-downloaded content) to text + metadata,
/// dispatching on `name`'s extension. `name` is the document's name/key (e.g.
/// `"contract.pdf"`) — it both selects the parser and becomes the chunk source
/// for citations. Use this for cloud object storage (S3 / R2 / Azure Blob / GCS),
/// HTTP downloads, or DB blobs — fetch with your own client, parse here.
pub fn extract_bytes(data: &[u8], name: &str) -> Result<ExtractedDoc, ExtractError> {
    let source = name.to_string();
    if data.len() as u64 > MAX_FILE_BYTES {
        return Err(ExtractError::TooLarge {
            source,
            bytes: data.len() as u64,
        });
    }
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let doc = match ext.as_str() {
        "docx" => docx::extract_bytes(data, source)?,
        "pptx" => pptx::extract_bytes(data, source)?,
        "xlsx" | "xlsm" | "xls" | "ods" => xlsx::extract_bytes(data, source)?,
        "pdf" => pdf::extract_bytes(data, source)?,
        "md" | "markdown" | "mdx" => ExtractedDoc {
            source,
            sections: text::markdown_sections(&decode_text(data)?),
        },
        e if is_code_ext(e) => ExtractedDoc {
            source,
            sections: text::code_sections(&decode_text(data)?),
        },
        e if is_text_ext(e) || e.is_empty() => ExtractedDoc {
            source,
            sections: text::line_blocks(&decode_text(data)?),
        },
        other => return Err(ExtractError::Unsupported(other.to_string())),
    };

    // Central empty guard: a parse that "succeeded" but produced no usable text
    // (scanned/image PDF, blank doc) becomes an actionable NoText error here,
    // rather than an empty document that fails later as a generic "no chunks".
    if doc.plain_text().trim().is_empty() {
        return Err(ExtractError::NoText(doc.source));
    }
    Ok(doc)
}
