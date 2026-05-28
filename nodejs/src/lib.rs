//! napi bindings for RedHop — the Node.js surface over the `redhop` Rust crate.
//!
//! Thin wrapper: the loader orchestration (config, embedder, folder persistence,
//! citations) lives in the `redhop` facade; this maps JS options to it and back.

#![deny(clippy::all)]

use napi::bindgen_prelude::Buffer;
use napi_derive::napi;

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
    /// Optional second-stage cross-encoder reranker by name (`"cross-encoder"`,
    /// auto-downloaded). Reorders the candidate pool by jointly scoring each
    /// `(query, passage)` pair. Works under any retrieval tier.
    pub rerank: Option<String>,
}

impl Options {
    fn into_load(self) -> redhop::LoadOptions {
        let u = |n: Option<u32>| n.map(|x| x as usize);
        redhop::LoadOptions {
            source: self.source,
            chunk_size: u(self.chunk_size),
            chunk_overlap: u(self.chunk_overlap),
            token_budget: u(self.token_budget),
            candidate_k: u(self.candidate_k),
            strategy: self.strategy,
            retrieval: self.retrieval,
            model: self.model,
            embedder_model: self.embedder_model,
            embedder_tokenizer: self.embedder_tokenizer,
            embedder_dim: u(self.embedder_dim),
            embedder_pooling: self.embedder_pooling,
            embedder_query_prefix: self.embedder_query_prefix,
            embedder_passage_prefix: self.embedder_passage_prefix,
            candidate_pool: u(self.candidate_pool),
            rerank: self.rerank,
        }
    }
}

/// Extra options for `Document.fromFolder` (plus the chunking/retrieval `options`).
#[napi(object)]
#[derive(Default)]
pub struct FolderOptions {
    /// Recurse into subdirectories. Default true.
    pub recursive: Option<bool>,
    /// Honor `.gitignore`. Default true.
    pub gitignore: Option<bool>,
    /// Extra gitignore-style globs to exclude, e.g. `["*.lock", "tests/**"]`.
    pub ignore: Option<Vec<String>>,
    /// Persist the index to disk and reload it incrementally on the next run.
    pub persist: Option<bool>,
    /// Where the persisted index lives (default `<folder>/.redhop`).
    pub index_dir: Option<String>,
    /// Chunking / retrieval options (same fields as the other constructors).
    pub options: Option<Options>,
}

impl FolderOptions {
    fn into_folder(self) -> redhop::FolderOptions {
        redhop::FolderOptions {
            recursive: self.recursive,
            gitignore: self.gitignore,
            ignore: self.ignore.unwrap_or_default(),
            persist: self.persist.unwrap_or(false),
            index_dir: self.index_dir,
            load: self.options.unwrap_or_default().into_load(),
        }
    }
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
    let citations = redhop::citations(&ctx)
        .into_iter()
        .map(|c| Citation {
            source: c.source,
            page: c.page.map(|n| n as u32),
            heading: c.heading,
            line: c.line.map(|n| n as u32),
            text: c.text,
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
/// reasoning-preserving context allocation; you think in documents and queries.
#[napi]
pub struct Document {
    inner: redhop::Document,
}

#[napi]
impl Document {
    /// Build from raw text you already have.
    #[napi(factory)]
    pub fn from_text(text: String, options: Option<Options>) -> napi::Result<Document> {
        let inner = redhop::text(text, &options.unwrap_or_default().into_load()).map_err(err)?;
        Ok(Document { inner })
    }

    /// Build from chunks you already produced (array of strings).
    #[napi(factory)]
    pub fn from_chunks(chunks: Vec<String>, options: Option<Options>) -> napi::Result<Document> {
        let inner = redhop::chunks(chunks, &options.unwrap_or_default().into_load()).map_err(err)?;
        Ok(Document { inner })
    }

    /// Build straight from a file on disk — PDF, DOCX, PPTX, XLSX, or text/code.
    #[napi(factory)]
    pub fn from_file(path: String, options: Option<Options>) -> napi::Result<Document> {
        let inner =
            redhop::read_file_with(&path, &options.unwrap_or_default().into_load()).map_err(err)?;
        Ok(Document { inner })
    }

    /// Build from in-memory bytes you fetched (S3 / GCS / Azure Blob / HTTP / DB).
    /// `source` (e.g. `"contract.pdf"`) picks the parser and is the citation source.
    #[napi(factory)]
    pub fn from_bytes(
        data: Buffer,
        source: String,
        options: Option<Options>,
    ) -> napi::Result<Document> {
        let inner = redhop::read_bytes_with(&data[..], &source, &options.unwrap_or_default().into_load())
            .map_err(err)?;
        Ok(Document { inner })
    }

    /// Build one index from every readable file in a folder. Honors `.gitignore` +
    /// `ignore` globs; `persist: true` saves an incremental on-disk index.
    #[napi(factory)]
    pub fn from_folder(path: String, options: Option<FolderOptions>) -> napi::Result<Document> {
        let inner =
            redhop::read_folder_with(&path, &options.unwrap_or_default().into_folder()).map_err(err)?;
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
