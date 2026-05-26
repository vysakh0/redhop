//! napi bindings for RedHop — the Node.js surface over the `redhop` Rust crate.
//!
//! Mirrors the Python API: `Document.fromText/fromChunks/fromFile/fromBytes/
//! fromFolder` → `doc.context(query)` → `{ text, chunks, citations, report }`.

#![deny(clippy::all)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::bindgen_prelude::Buffer;
use napi_derive::napi;

use redhop::{ContextConfig, Document as RhDocument, DocumentConfig, RetrievalMode, Section};

fn err(e: impl std::fmt::Display) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}

// ── Options ─────────────────────────────────────────────────────────────────

/// Chunking / assembly / retrieval options for the `Document.from*` constructors.
/// Every field is optional; omit for evidence-backed defaults.
#[napi(object)]
#[derive(Default)]
pub struct Options {
    /// Source label for `fromText` (default `"document"`).
    pub source: Option<String>,
    /// Target tokens per chunk (index-time). Default 128.
    pub chunk_size: Option<u32>,
    /// Sentences of overlap between adjacent chunks. Default 1.
    pub chunk_overlap: Option<u32>,
    /// Default assembly token budget (override per call). Default 8192.
    pub token_budget: Option<u32>,
    /// Candidates to retrieve before assembly. Default 20.
    pub candidate_k: Option<u32>,
    /// Assembly strategy: `auto` (default) · `reasoning_preserving` ·
    /// `distractor_filtered` · `redundancy_pruned` · `max_density` · `raw_topk`.
    pub strategy: Option<String>,
    /// `"lexical"` (default, BM25), `"hybrid"`, or `"semantic"`.
    pub retrieval: Option<String>,
    /// Built-in embedding model to download (`"bge-small"` / `"bge-base"`).
    pub model: Option<String>,
    /// Advanced: path to a local ONNX embedding model.
    pub embedder_model: Option<String>,
    /// Advanced: path to its tokenizer.
    pub embedder_tokenizer: Option<String>,
    /// Advanced: the model's output dimension. Default 384.
    pub embedder_dim: Option<u32>,
    /// Advanced: `"cls"` (default) or `"mean"`.
    pub embedder_pooling: Option<String>,
    /// Advanced: query prefix for asymmetric models (E5: `"query: "`).
    pub embedder_query_prefix: Option<String>,
    /// Advanced: passage prefix for asymmetric models (E5: `"passage: "`).
    pub embedder_passage_prefix: Option<String>,
    /// Hybrid BM25 prune depth. Default 50.
    pub candidate_pool: Option<u32>,
}

/// Extra options for `Document.fromFolder` (in addition to the chunking/retrieval
/// `Options`, which are read from the same object).
#[napi(object)]
#[derive(Default)]
pub struct FolderOptions {
    /// Recurse into subdirectories. Default true.
    pub recursive: Option<bool>,
    /// Honor `.gitignore`. Default true.
    pub gitignore: Option<bool>,
    /// Extra gitignore-style globs to exclude, e.g. `["*.lock", "tests/**"]`.
    pub ignore: Option<Vec<String>>,
    /// Chunking/retrieval options (same fields as the other constructors).
    pub options: Option<Options>,
}

fn strategy_from_str(s: &str) -> napi::Result<redhop::ContextStrategy> {
    use redhop::ContextStrategy as S;
    Ok(match s {
        "raw_topk" => S::RawTopK,
        "distractor_filtered" => S::DistractorFiltered,
        "redundancy_pruned" => S::RedundancyPruned,
        "max_density" => S::MaxDensity,
        "reasoning_preserving" => S::ReasoningPreserving,
        "auto" => S::Auto,
        other => return Err(err(format!(
            "unknown strategy '{other}' (expected: raw_topk, distractor_filtered, \
             redundancy_pruned, max_density, reasoning_preserving, auto)"
        ))),
    })
}

fn retrieval_from_str(retrieval: Option<&str>, candidate_pool: usize) -> napi::Result<RetrievalMode> {
    Ok(match retrieval {
        None | Some("lexical") => RetrievalMode::Lexical,
        Some("hybrid") => RetrievalMode::Hybrid {
            candidate_pool: candidate_pool.max(1),
        },
        Some("semantic") => RetrievalMode::Dense,
        Some(other) => return Err(err(format!(
            "unknown retrieval mode '{other}'; use 'lexical', 'hybrid', or 'semantic'"
        ))),
    })
}

fn doc_config(o: &Options, mode: RetrievalMode) -> napi::Result<DocumentConfig> {
    let base = DocumentConfig::default();
    let strategy = match &o.strategy {
        Some(s) => strategy_from_str(s)?,
        None => base.context.strategy,
    };
    let chunk_size = o.chunk_size.unwrap_or(128) as usize;
    let context = ContextConfig {
        token_budget: o.token_budget.unwrap_or(8192) as usize,
        strategy,
        ..base.context
    };
    Ok(DocumentConfig {
        target_tokens: chunk_size,
        max_tokens: chunk_size * 2,
        overlap_sentences: o.chunk_overlap.unwrap_or(1) as usize,
        candidate_k: o.candidate_k.unwrap_or(20) as usize,
        retrieval_mode: mode,
        context,
    })
}

fn apply_dense_embedder(doc: RhDocument, o: &Options) -> napi::Result<RhDocument> {
    use redhop::embeddings::{EmbedderConfig, OnnxEmbedder, Pooling};

    // Path A — explicit local model files (advanced); `model` is ignored.
    if let (Some(m), Some(t)) = (o.embedder_model.as_ref(), o.embedder_tokenizer.as_ref()) {
        let pooling = match o.embedder_pooling.as_deref() {
            None | Some("cls") => Pooling::Cls,
            Some("mean") => Pooling::Mean,
            Some(other) => return Err(err(format!(
                "unknown embedderPooling '{other}'; use 'cls' or 'mean'"
            ))),
        };
        let dim = o.embedder_dim.unwrap_or(384) as usize;
        let load = |prefix: &str| -> napi::Result<OnnxEmbedder> {
            let mut config = EmbedderConfig::bge(dim);
            config.pooling = pooling;
            config.prefix = prefix.to_string();
            OnnxEmbedder::load(m, t, config).map_err(err)
        };
        return match (&o.embedder_query_prefix, &o.embedder_passage_prefix) {
            (q, p) if q.is_some() || p.is_some() => {
                let passage = load(p.as_deref().unwrap_or(""))?;
                let query = load(q.as_deref().unwrap_or(""))?;
                Ok(doc
                    .with_embedder(Arc::new(passage))
                    .with_query_embedder(Arc::new(query)))
            }
            _ => Ok(doc.with_embedder(Arc::new(load("")?))),
        };
    }

    // Path B — model-by-name (auto-downloaded, cached), or the default.
    let name = o.model.as_deref().unwrap_or(redhop::embeddings::DEFAULT_MODEL);
    let resolved = redhop::embeddings::resolve_model(name).map_err(err)?;
    let load = |prefix: &str| -> napi::Result<OnnxEmbedder> {
        OnnxEmbedder::load(&resolved.model_path, &resolved.tokenizer_path, resolved.config(prefix))
            .map_err(err)
    };
    if resolved.is_asymmetric() {
        let passage = load(&resolved.passage_prefix)?;
        let query = load(&resolved.query_prefix)?;
        Ok(doc
            .with_embedder(Arc::new(passage))
            .with_query_embedder(Arc::new(query)))
    } else {
        Ok(doc.with_embedder(Arc::new(load(&resolved.passage_prefix)?)))
    }
}

/// Build a document from `(source, sections)` files, applying the embedder if the
/// chosen tier needs one.
fn build(files: Vec<(String, Vec<Section>)>, o: &Options) -> napi::Result<RhDocument> {
    let mode = retrieval_from_str(o.retrieval.as_deref(), o.candidate_pool.unwrap_or(50) as usize)?;
    let needs_embedder = matches!(mode, RetrievalMode::Hybrid { .. } | RetrievalMode::Dense);
    let cfg = doc_config(o, mode)?;
    let mut doc = RhDocument::from_sources_with(files, cfg).map_err(err)?;
    if needs_embedder {
        doc = apply_dense_embedder(doc, o)?;
    }
    Ok(doc)
}

/// Parse a file/bytes into `(source, sections)` via the built-in parsers.
fn extract_path(path: &str) -> napi::Result<(String, Vec<Section>)> {
    let d = redhop::files::extract(path).map_err(err)?;
    Ok((d.source, to_sections(d.sections)))
}

fn extract_data(data: &[u8], name: &str) -> napi::Result<(String, Vec<Section>)> {
    let d = redhop::files::extract_bytes(data, name).map_err(err)?;
    Ok((d.source, to_sections(d.sections)))
}

fn to_sections(sections: Vec<redhop::files::Section>) -> Vec<Section> {
    sections
        .into_iter()
        .map(|s| Section {
            text: s.text,
            page: s.page,
            heading: s.heading,
            line: s.line,
        })
        .collect()
}

const DEFAULT_IGNORES: &[&str] = &["node_modules", "target", "__pycache__", "venv", "dist", "build"];

/// Walk a folder honoring .gitignore + custom globs (ripgrep's walker).
fn collect_files(
    root: &Path,
    recursive: bool,
    gitignore: bool,
    ignore_globs: &[String],
) -> napi::Result<Vec<PathBuf>> {
    let mut ob = ignore::overrides::OverrideBuilder::new(root);
    for d in DEFAULT_IGNORES {
        let _ = ob.add(&format!("!{d}"));
    }
    for g in ignore_globs {
        ob.add(&format!("!{g}"))
            .map_err(|e| err(format!("invalid ignore pattern '{g}': {e}")))?;
    }
    let overrides = ob.build().map_err(err)?;
    let mut wb = ignore::WalkBuilder::new(root);
    wb.hidden(true)
        .git_ignore(gitignore)
        .git_global(gitignore)
        .git_exclude(gitignore)
        .ignore(gitignore)
        .parents(gitignore)
        .require_git(false)
        .overrides(overrides);
    if !recursive {
        wb.max_depth(Some(1));
    }
    let mut out = Vec::new();
    for entry in wb.build() {
        let Ok(entry) = entry else { continue };
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            out.push(entry.into_path());
        }
    }
    out.sort();
    Ok(out)
}

// ── Result types ─────────────────────────────────────────────────────────────

/// Where one selected chunk came from — for citing the evidence.
#[napi(object)]
pub struct Citation {
    pub source: String,
    pub page: Option<u32>,
    pub heading: Option<String>,
    pub line: Option<u32>,
    pub text: String,
}

/// The Decision Report: what the assembly did, and why.
#[napi(object)]
pub struct Report {
    /// `"passthrough"` | `"prune"` | `"not_auto"`.
    pub auto_decision: String,
    pub total_tokens: u32,
    pub retained_evidence_ratio: f64,
    pub second_hop_rescues: u32,
    /// Structural-expansion chunks added (neighbors / headings).
    pub n_expanded: u32,
    /// The human-readable Decision Report.
    pub rendered: String,
}

/// The assembled context: prompt string, selected chunks, citations, report.
#[napi(object)]
pub struct BuiltContext {
    pub text: String,
    pub chunks: Vec<String>,
    pub citations: Vec<Citation>,
    pub report: Report,
}

fn to_built(ctx: redhop::BuiltContext) -> BuiltContext {
    let citations = ctx
        .chunks
        .iter()
        .map(|c| Citation {
            source: c.source.clone(),
            page: c.metadata.get("page").and_then(|v| v.as_u64()).map(|n| n as u32),
            heading: c.metadata.get("heading").and_then(|v| v.as_str()).map(String::from),
            line: c.metadata.get("line").and_then(|v| v.as_u64()).map(|n| n as u32),
            text: c.text.clone(),
        })
        .collect();
    let r = &ctx.report;
    let auto_decision = match r.auto_decision() {
        redhop::AutoDecision::Passthrough => "passthrough",
        redhop::AutoDecision::Prune => "prune",
        redhop::AutoDecision::NotAuto => "not_auto",
    }
    .to_string();
    let report = Report {
        auto_decision,
        total_tokens: r.total_tokens as u32,
        retained_evidence_ratio: r.retained_evidence_ratio as f64,
        second_hop_rescues: r.second_hop_rescue_count as u32,
        n_expanded: r.n_expanded as u32,
        rendered: r.render(None),
    };
    BuiltContext {
        text: ctx.text(),
        chunks: ctx.chunks.iter().map(|c| c.text.clone()).collect(),
        citations,
        report,
    }
}

// ── Document ───────────────────────────────────────────────────────────────

/// A document you reason over. RedHop owns chunking, internal retrieval, and
/// reasoning-aware context allocation; you think in documents and queries.
#[napi]
pub struct Document {
    inner: RhDocument,
}

#[napi]
impl Document {
    /// Build from raw text you already have.
    #[napi(factory)]
    pub fn from_text(text: String, options: Option<Options>) -> napi::Result<Document> {
        let o = options.unwrap_or_default();
        let source = o.source.clone().unwrap_or_else(|| "document".to_string());
        let inner = build(vec![(source, vec![Section { text, ..Default::default() }])], &o)?;
        Ok(Document { inner })
    }

    /// Build from chunks you already produced (array of strings).
    #[napi(factory)]
    pub fn from_chunks(chunks: Vec<String>, options: Option<Options>) -> napi::Result<Document> {
        let o = options.unwrap_or_default();
        // Each chunk is one section under a shared source.
        let sections = chunks
            .into_iter()
            .map(|text| Section { text, ..Default::default() })
            .collect();
        let inner = build(vec![("chunks".to_string(), sections)], &o)?;
        Ok(Document { inner })
    }

    /// Build straight from a file on disk — PDF, DOCX, PPTX, XLSX, or text/code.
    #[napi(factory)]
    pub fn from_file(path: String, options: Option<Options>) -> napi::Result<Document> {
        let o = options.unwrap_or_default();
        let file = extract_path(&path)?;
        let inner = build(vec![file], &o)?;
        Ok(Document { inner })
    }

    /// Build from in-memory bytes you fetched (S3 / GCS / Azure Blob / HTTP / DB).
    /// `source` (e.g. `"contract.pdf"`) picks the parser and is the citation source.
    #[napi(factory)]
    pub fn from_bytes(data: Buffer, source: String, options: Option<Options>) -> napi::Result<Document> {
        let o = options.unwrap_or_default();
        let file = extract_data(&data[..], &source)?;
        let inner = build(vec![file], &o)?;
        Ok(Document { inner })
    }

    /// Build one index from every readable file in a folder. Honors .gitignore and
    /// `ignore` globs; skips hidden + build/cache dirs. In-memory (rebuilt each run).
    #[napi(factory)]
    pub fn from_folder(path: String, options: Option<FolderOptions>) -> napi::Result<Document> {
        let fo = options.unwrap_or_default();
        let o = fo.options.unwrap_or_default();
        let root = Path::new(&path);
        if !root.is_dir() {
            return Err(err(format!("from_folder: '{path}' is not a directory")));
        }
        let paths = collect_files(
            root,
            fo.recursive.unwrap_or(true),
            fo.gitignore.unwrap_or(true),
            &fo.ignore.unwrap_or_default(),
        )?;
        let mut files: Vec<(String, Vec<Section>)> = Vec::new();
        for p in &paths {
            if let Ok(file) = extract_path(&p.to_string_lossy()) {
                files.push(file);
            }
        }
        if files.is_empty() {
            return Err(err(format!("from_folder: no readable files under '{path}'")));
        }
        let inner = build(files, &o)?;
        Ok(Document { inner })
    }

    /// Number of chunks the document holds.
    #[napi(getter)]
    pub fn chunk_count(&self) -> u32 {
        self.inner.len() as u32
    }

    /// Assemble the reasoning context for a query (retrieve → allocate).
    /// `neighbors` / `includeHeading` add structural context expansion.
    #[napi]
    pub fn context(
        &mut self,
        query: String,
        budget: Option<u32>,
        neighbors: Option<u32>,
        include_heading: Option<bool>,
    ) -> napi::Result<BuiltContext> {
        let budget = budget.map(|b| b as usize);
        let n = neighbors.unwrap_or(0) as usize;
        let heading = include_heading.unwrap_or(false);
        let ctx = if n == 0 && !heading {
            self.inner.context_with(&query, budget, None)
        } else {
            self.inner.context_expanded(&query, budget, None, n, heading)
        }
        .map_err(err)?;
        Ok(to_built(ctx))
    }
}
