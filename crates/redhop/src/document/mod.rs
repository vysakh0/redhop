//! # redhop-document
//!
//! The **reasoning-preserving document context runtime** — RedHop's high-level
//! surface. You have documents and need reasoning; you should not have to
//! think about retrievers, vector stores, query engines, or ANN infrastructure.
//!
//! ```no_run
//! # fn demo() -> redhop::core::Result<()> {
//! use redhop::document::Document;
//!
//! let mut doc = Document::from_text("report.txt", "…long document text…")?;
//! let ctx = doc.context("Why did the proposed method fail?")?;
//! // feed `ctx.text()` to any LLM provider yourself — no lock-in.
//! let _prompt = ctx.text();
//! let _report = &ctx.report;  // what was retrieved/pruned and why
//! # Ok(()) }
//! ```
//!
//! ## What this layer owns (and what it does not)
//!
//! It **owns**: chunking, internal indexing/retrieval, context allocation,
//! reasoning-safe optimization, observability, and token economics. It
//! **does not own** document parsing/OCR (bring your own text — PyMuPDF,
//! Marker, Unstructured, …) and it is **not** an orchestration / agent /
//! workflow runtime.
//!
//! Internally it retrieves, but retrieval is an implementation detail: the user
//! mental model is *documents + reasoning*, not *retrieval infrastructure*. The
//! default is BM25 ([`RetrievalMode::Lexical`]) — zero dependencies, best for
//! lexical workloads. For semantic-heavy queries, opt into
//! [`RetrievalMode::Dense`] and inject an embedder via
//! [`Document::with_embedder`] (exact cosine over every chunk, no ANN). A
//! zero-model semantic tier lives in the external `semantic-bm25` crate and is
//! deliberately not wired here. This module is pure layering over
//! [`crate::chunking`], [`crate::retrieval`], and [`crate::context`] — no
//! new logic, no new architecture tower. (The three were separate crates
//! pre-0.2; they're now sibling modules in the consolidated workspace.)
//!
//! The default context policy is [`ContextStrategy::Auto`]: do nothing under
//! headroom, prune under dilution, preserve bridge evidence, and report every
//! decision. See `docs/findings/CONTEXT_DILUTION.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::Arc;

use crate::chunking::{SentenceChunker, WhitespaceTokenizer};
use crate::context::{
    analyze_context, build_context, build_context_expanded, BuiltContext, ContextConfig,
    ContextReport, ContextStrategy, ExpansionPlan,
};
use crate::core::{
    Chunk, Chunker, Document as SourceDoc, EmbeddingProvider, Query, Reranker, Result,
    RetrievalResult, Retriever, TokenizerBackend,
};
use crate::retrieval::{Bm25Retriever, LocalRerankRetriever};
use tokio::runtime::{Builder, Runtime};

/// How a [`Document`] retrieves candidates internally.
///
/// The default ([`RetrievalMode::Lexical`]) is BM25 — zero dependencies, fast,
/// and the right tool for lexical/keyword workloads. For semantic-heavy queries
/// (paraphrase, low query↔passage overlap), opt into [`RetrievalMode::Dense`]
/// and supply an embedder via [`Document::with_embedder`]: the dense model
/// cosines the query against *every* chunk — exact brute force, no ANN, for
/// bounded corpora (`docs/findings/GLOBAL_DENSE.md`). A zero-model semantic tier
/// (corpus-graph second-order rerank) lives in the external `semantic-bm25` crate
/// and is deliberately *not* wired here (`docs/findings/SEMANTIC_ZERO_DEP.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    /// BM25 lexical retrieval. The default; needs no model or embedder.
    Lexical,
    /// Hybrid: BM25 prunes the corpus to a candidate pool, then a dense model
    /// reorders **only that pool**. Requires an embedder. Embeds only the
    /// ~`candidate_pool` candidates per query (not the whole corpus), so it scales
    /// to **large local corpora without a vector DB** — the agent/folder case.
    Hybrid {
        /// BM25 candidate-pool depth the dense stage reorders (e.g. 50).
        candidate_pool: usize,
    },
    /// Global dense: cosine the query against **every** chunk embedding (exact
    /// brute force, no ANN). Requires an embedder set via
    /// [`Document::with_embedder`]. Best recall on **bounded** corpora; embeds the
    /// whole corpus up front, so it doesn't scale to large/persistent collections —
    /// use `Hybrid` (no DB) or a real vector store there.
    Dense,
}

/// Tuning for a [`Document`]'s internal chunking, retrieval, and context
/// policy. Every field has an evidence-backed default; users reason about
/// documents, not knobs.
#[derive(Debug, Clone)]
pub struct DocumentConfig {
    /// Target tokens per chunk (sentence-budgeted chunking).
    pub target_tokens: usize,
    /// Hard cap on tokens per chunk.
    pub max_tokens: usize,
    /// Sentences of overlap between adjacent chunks.
    pub overlap_sentences: usize,
    /// How many candidate chunks to retrieve before context assembly.
    pub candidate_k: usize,
    /// Retrieval mode. Defaults to [`RetrievalMode::Lexical`] (BM25).
    pub retrieval_mode: RetrievalMode,
    /// When a reranker is supplied (see [`Document::with_reranker`]), how many
    /// first-stage candidates it reorders before truncating to the requested
    /// `k`. Larger gives the reranker more to work with, at more model calls.
    /// Ignored when no reranker is set.
    pub rerank_pool: usize,
    /// Context-assembly policy. Defaults to [`ContextStrategy::Auto`].
    pub context: ContextConfig,
    /// Floor on the number of candidates returned to the context assembler.
    /// When the primary retriever (under [`RetrievalMode::Hybrid`] or
    /// [`RetrievalMode::Dense`]) returns fewer than this many results, a
    /// lexical (BM25) fallback over the same chunks tops it up until the
    /// floor is met. **Default `0`** — the floor is opt-in. Has no effect
    /// under [`RetrievalMode::Lexical`] (the primary already is BM25). Pair
    /// with [`ContextReport::low_confidence_retrieval`] (issue #1) to detect
    /// when the fallback fired with weak chunks.
    pub min_candidates: usize,
    /// Neighbors to attach automatically when the retrieved set includes a
    /// code-classified chunk. The default `1` makes [`Document::context`]
    /// behave like [`Document::context_expanded`] with `neighbors=1` for
    /// code-shaped corpora — so a citation that lands on a function's `def`
    /// line also brings the implementation chunk along, instead of the user
    /// having to opt into expansion explicitly. Set to `0` to disable. Has
    /// no effect on result sets where no chunk has `metadata["kind"]=="code"`.
    pub code_neighbors_default: usize,
    /// When `true` (default), [`Document::context`] attaches the section's
    /// opening chunk (the heading) to every cited chunk that carries a
    /// `metadata["heading"]` — so a citation deep inside `## Refunds → ###
    /// Eligibility` arrives at the LLM with the section title attached for
    /// context. Mirrors the code-neighbors default but for prose with
    /// hierarchical structure (markdown, DOCX, PPTX, XLSX, and now PDF).
    /// Set to `false` to keep citations strictly to the retrieved chunks.
    pub prose_heading_default: bool,
}

impl Default for DocumentConfig {
    fn default() -> Self {
        Self {
            // 128-token chunks: a sweep across budgets/datasets showed finer
            // chunks pack better under tight budgets (multi-hop ≥0.8 retention
            // 54%→77%) and tie at large budgets — so 128 is the robust default
            // over the previous 256. See docs/findings/CHUNK_GRANULARITY.md.
            target_tokens: 128,
            max_tokens: 256,
            overlap_sentences: 1,
            candidate_k: 20,
            retrieval_mode: RetrievalMode::Lexical,
            rerank_pool: 50,
            // The runtime's philosophy: size-gated, conservative, observable.
            context: ContextConfig {
                strategy: ContextStrategy::Auto,
                ..Default::default()
            },
            // Opt-in: the strict-superset contract restored in 0.1.3 (issue
            // #1) is already the right behavior for almost every caller. Set
            // this when an LLM downstream refuses to answer on empty
            // contexts and a known-weak chunk is better than nothing.
            min_candidates: 0,
            // Code chunks are fixed-token windows, so a single function often
            // spans 2-3 chunks. Default `context()` on a code hit would cite
            // only the chunk that matched (typically the `def` line), losing
            // the body. Pulling ±1 neighbors as part of the default
            // pull-the-implementation-too behavior. Set to 0 to disable.
            code_neighbors_default: 1,
            // For sectioned prose, attach each cited chunk's section heading
            // chunk so the LLM has the topic context. Cheap (one extra chunk
            // per cited section), bounded by the token budget.
            prose_heading_default: true,
        }
    }
}

/// A unit of an ingested document (e.g. a parsed file): text plus optional
/// provenance. Provenance becomes per-chunk citation metadata (`page`/`heading`).
#[derive(Debug, Clone, Default)]
pub struct Section {
    /// The section's text.
    pub text: String,
    /// 1-based page or slide number, when the source has them.
    pub page: Option<usize>,
    /// Nearest heading/title/sheet name, when known.
    pub heading: Option<String>,
    /// 1-based line the section starts at (text & code files).
    pub line: Option<usize>,
}

/// A document you reason over. Holds its chunks and a lazily-built internal
/// index; `context()` and `analyze()` retrieve candidates and hand them to the
/// context runtime. Retrieval is an internal detail — never surfaced.
pub struct Document {
    chunks: Vec<Chunk>,
    cfg: DocumentConfig,
    rt: Runtime,
    // Optional embedder for `RetrievalMode::Dense`. None for the default
    // lexical path, so the ONNX/model dependency only exists when a caller opts
    // in by supplying one via `with_embedder`. Embeds passages (and the query,
    // unless a separate `query_embedder` is set).
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    // Optional separate query-side embedder for asymmetric models (e.g. E5).
    query_embedder: Option<Arc<dyn EmbeddingProvider>>,
    // Lexical analyzer driving the BM25 retriever (and the
    // ContextConfig.analyzer inside `cfg.context`, kept in lockstep). Defaults
    // to English Snowball Porter2; override via `with_analyzer`. Always
    // populated — there is no "no analyzer" state.
    analyzer: Arc<dyn crate::analyzer::Analyzer>,
    // Lazily built on first query so construction is cheap (goal: lazy
    // chunk/index init).
    retriever: Option<Box<dyn Retriever>>,
    // Optional second-stage reranker (e.g. a cross-encoder). When set, the
    // first stage fetches `cfg.rerank_pool` candidates and the reranker reorders
    // them down to the requested `k`. None ⇒ the first-stage ranking stands.
    reranker: Option<Arc<dyn Reranker>>,
    // Lazy BM25 index used to top up the primary retriever when it returns
    // fewer than `cfg.min_candidates`. Initialized on first fallback need so
    // documents that never trigger the floor never pay for a second index.
    fallback_bm25: Option<Bm25Retriever>,
    // Number of source files indexed into this Document. 1 for the
    // single-source constructors (`from_text`, `from_chunks`, `read_file`,
    // `read_bytes`); the readable file count for `read_folder` /
    // `read_folder_with`.
    n_files: usize,
    // Files that `read_folder` / `read_folder_with` skipped, as
    // `(source_path, reason)` pairs — unsupported formats, unreadable bytes,
    // or no extractable text. Empty for single-source constructors.
    skipped_files: Vec<(String, String)>,
}

/// Renumber chunk ids to `0..n` so a merged set (e.g. from several files) has
/// unique ids. Embeddings/metadata ride on the chunk, so renumbering is safe.
fn reassign_ids(chunks: &mut [Chunk]) {
    for (i, c) in chunks.iter_mut().enumerate() {
        c.id = crate::core::ChunkId::new(format!("{i}"));
    }
}

/// True iff the retrieval result is a code-classified chunk. Drives the
/// `code_neighbors_default` auto-expansion in [`Document::context_with`].
fn is_code_chunk(r: &RetrievalResult) -> bool {
    r.chunk.metadata.get("kind").and_then(|v| v.as_str()) == Some("code")
}

/// True iff the retrieval result is a prose chunk carrying a non-empty
/// section heading (markdown / DOCX / PPTX / XLSX / PDF headings). Drives
/// the `prose_heading_default` auto-expansion in
/// [`Document::context_with`]. Code chunks are excluded — they get their
/// own neighbor expansion path.
fn has_prose_heading(r: &RetrievalResult) -> bool {
    if is_code_chunk(r) {
        return false;
    }
    r.chunk
        .metadata
        .get("heading")
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty())
}

/// Classify a source by extension. `"code"` and `"data"` are chunked **verbatim**
/// (formatting preserved) rather than sentence-reflowed, and `"code"` is routed to
/// lexical retrieval (BM25) under the hybrid tier. Stamped as each chunk's `kind`.
fn chunk_kind(source: &str) -> &'static str {
    let ext = std::path::Path::new(source)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    const CODE: &[&str] = &[
        "py", "ts", "tsx", "js", "jsx", "mjs", "cjs", "rs", "go", "java", "kt", "kts", "c", "h",
        "cpp", "hpp", "cc", "hh", "cs", "rb", "php", "sh", "bash", "zsh", "sql", "swift", "scala",
        "lua", "r", "pl", "ml", "ex", "exs",
    ];
    const DATA: &[&str] = &[
        "json", "jsonl", "ndjson", "yaml", "yml", "toml", "csv", "tsv", "xml",
    ];
    let e = ext.as_str();
    if CODE.contains(&e) {
        "code"
    } else if DATA.contains(&e) {
        "data"
    } else {
        "prose"
    }
}

/// Split text into chunks **verbatim** — preserving lines/formatting — packing
/// whole lines up to `max_tokens` (whitespace-token count). Used for code/data,
/// where sentence reflow would mangle the content. Ids are placeholders (callers
/// re-id); metadata is attached by [`Document::chunk_sections`].
fn verbatim_chunks(source: &str, text: &str, max_tokens: usize) -> Vec<Chunk> {
    let max = max_tokens.max(1);
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut buf_tok = 0usize;
    let push = |buf: &mut Vec<&str>, buf_tok: &mut usize, chunks: &mut Vec<Chunk>| {
        if buf.is_empty() {
            return;
        }
        let id = crate::core::ChunkId::new(chunks.len().to_string());
        chunks.push(Chunk::new(
            id,
            buf.join("\n"),
            source,
            crate::core::TokenCount(*buf_tok),
        ));
        buf.clear();
        *buf_tok = 0;
    };
    for line in text.lines() {
        let lt = line.split_whitespace().count().max(1);
        if buf_tok + lt > max && !buf.is_empty() {
            push(&mut buf, &mut buf_tok, &mut chunks);
        }
        buf.push(line);
        buf_tok += lt;
    }
    push(&mut buf, &mut buf_tok, &mut chunks);
    chunks
}

impl Document {
    /// Build a document from raw text, chunking it with the default policy.
    /// Bring your own parser/OCR — this layer takes text, not PDFs.
    pub fn from_text(source: impl Into<String>, text: impl Into<String>) -> Result<Self> {
        Self::from_text_with(source, text, DocumentConfig::default())
    }

    /// Build from raw text with an explicit [`DocumentConfig`].
    pub fn from_text_with(
        source: impl Into<String>,
        text: impl Into<String>,
        cfg: DocumentConfig,
    ) -> Result<Self> {
        let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
        let chunker = SentenceChunker::new(
            tok,
            cfg.target_tokens,
            cfg.max_tokens,
            cfg.overlap_sentences,
        )?;
        let chunks = chunker.chunk(&SourceDoc::new(source, text))?;
        Self::from_chunks_with(chunks, cfg)
    }

    /// Build from sections (e.g. a parsed file). Each section is chunked with the
    /// configured policy, and **every resulting chunk carries `source` plus the
    /// section's `page`/`heading` as metadata** — so retrieved chunks can be cited
    /// ("from contract.pdf, p.3"). Chunk ids are made unique across sections.
    pub fn from_sections_with(
        source: impl Into<String>,
        sections: Vec<Section>,
        cfg: DocumentConfig,
    ) -> Result<Self> {
        Self::from_sources_with(vec![(source.into(), sections)], cfg)
    }

    /// Build one document from **many sources** (e.g. every file in a folder).
    /// Each `(source, sections)` pair is chunked with the configured policy; every
    /// chunk keeps **its own** `source` plus the section's `page`/`heading`/`line`
    /// metadata, so retrieval over the combined index still cites the right file.
    /// Chunk ids are made unique across all sources.
    pub fn from_sources_with(
        files: Vec<(String, Vec<Section>)>,
        cfg: DocumentConfig,
    ) -> Result<Self> {
        let mut all: Vec<Chunk> = Vec::new();
        for (source, sections) in &files {
            all.extend(Self::chunk_sections(source, sections, &cfg)?);
        }
        reassign_ids(&mut all);
        Self::from_chunks_with(all, cfg)
    }

    /// Chunk **one** source's sections into chunks (carrying `page`/`heading`/
    /// `line` metadata), *without* building a `Document`. Ids start at 0 — callers
    /// that merge chunks from several sources must re-id (see [`Document::from_chunks_with`]).
    /// This is the building block for incremental / persisted indexing: re-chunk
    /// only the files that changed, keep the cached chunks for the rest.
    pub fn chunk_sections(
        source: &str,
        sections: &[Section],
        cfg: &DocumentConfig,
    ) -> Result<Vec<Chunk>> {
        // Code & structured data are chunked verbatim (formatting preserved);
        // prose is sentence-packed. The kind also routes retrieval (code → lexical).
        let kind = chunk_kind(source);
        let sentence_chunker = if kind == "prose" {
            let tok: Arc<dyn TokenizerBackend> = Arc::new(WhitespaceTokenizer::new());
            Some(SentenceChunker::new(
                tok,
                cfg.target_tokens,
                cfg.max_tokens,
                cfg.overlap_sentences,
            )?)
        } else {
            None
        };

        let mut out: Vec<Chunk> = Vec::new();
        for sec in sections {
            if sec.text.trim().is_empty() {
                continue;
            }
            let mut chunks = match &sentence_chunker {
                Some(ch) => ch.chunk(&SourceDoc::new(source.to_string(), sec.text.clone()))?,
                None => verbatim_chunks(source, &sec.text, cfg.max_tokens),
            };
            for c in &mut chunks {
                if let Some(p) = sec.page {
                    c.metadata
                        .insert("page".to_string(), serde_json::Value::from(p as u64));
                }
                if let Some(h) = &sec.heading {
                    c.metadata
                        .insert("heading".to_string(), serde_json::Value::String(h.clone()));
                }
                if let Some(l) = sec.line {
                    c.metadata
                        .insert("line".to_string(), serde_json::Value::from(l as u64));
                }
                c.metadata.insert(
                    "kind".to_string(),
                    serde_json::Value::String(kind.to_string()),
                );
            }
            out.extend(chunks);
        }
        reassign_ids(&mut out);
        Ok(out)
    }

    /// Build from chunks you already produced (your own chunker/parser).
    pub fn from_chunks(chunks: Vec<Chunk>) -> Result<Self> {
        Self::from_chunks_with(chunks, DocumentConfig::default())
    }

    /// Build from chunks with an explicit [`DocumentConfig`].
    pub fn from_chunks_with(mut chunks: Vec<Chunk>, cfg: DocumentConfig) -> Result<Self> {
        if chunks.is_empty() {
            return Err(crate::core::Error::InvalidConfig(
                "cannot build a Document with no chunks — the text was empty or produced no \
                 chunks. Pass non-empty text to `from_text`, or chunks to `from_chunks`."
                    .into(),
            ));
        }
        // Stamp a stable per-source `chunk_index` so the input order is
        // preserved through retrieval. Caller-supplied chunks via
        // `from_chunks*` may not carry the chunker's `sentence_range`, so
        // this metadata key is what `ContextConfig::preserve_order` reads
        // to reconstruct source-document order downstream.
        let mut per_source: std::collections::HashMap<String, i64> =
            std::collections::HashMap::new();
        for chunk in chunks.iter_mut() {
            // Only stamp if the chunk doesn't already carry `chunk_index`
            // (callers — and future chunkers — may stamp it themselves).
            if !chunk.metadata.contains_key("chunk_index") {
                let idx = per_source.entry(chunk.source.clone()).or_insert(0);
                let val = *idx;
                *idx += 1;
                chunk
                    .metadata
                    .insert("chunk_index".to_string(), serde_json::json!(val));
            }
        }
        // A current-thread runtime is enough: the internal retriever's work is
        // CPU-bound (Tantivy on a blocking worker); we only block_on it.
        let rt = Builder::new_current_thread().build()?;
        // The Document-level analyzer mirrors `cfg.context.analyzer`. The
        // ContextConfig is the source of truth for analyzer choice when a
        // loader (`LoadOptions::language`) sets it; we lift it onto the
        // Document so retrievers can read it cheaply without traversing cfg.
        let analyzer = cfg.context.analyzer.clone();
        Ok(Self {
            chunks,
            cfg,
            rt,
            embedder: None,
            query_embedder: None,
            analyzer,
            retriever: None,
            reranker: None,
            fallback_bm25: None,
            n_files: 1,
            skipped_files: Vec::new(),
        })
    }

    /// Number of source files indexed into this Document.
    ///
    /// - `1` for the single-source constructors ([`Document::from_text`],
    ///   [`Document::from_chunks`], `read_file`, `read_bytes`).
    /// - The readable file count for `read_folder` / `read_folder_with`
    ///   (excludes the ones in [`Document::skipped_files`]).
    pub fn n_files(&self) -> usize {
        self.n_files
    }

    /// Files that `read_folder` / `read_folder_with` skipped, as
    /// `(source_path, reason)` pairs — unsupported formats, unreadable
    /// bytes, no extractable text (e.g. scanned PDFs without OCR), etc.
    /// Empty for single-source constructors.
    pub fn skipped_files(&self) -> &[(String, String)] {
        &self.skipped_files
    }

    /// Internal setter used by the folder loaders (`read_folder_with`)
    /// to record how many sources actually contributed chunks and which
    /// were skipped along the way. Only ever called from the
    /// `files`-feature-gated loader paths.
    #[cfg(feature = "files")]
    pub(crate) fn set_folder_provenance(&mut self, n_files: usize, skipped: Vec<(String, String)>) {
        self.n_files = n_files;
        self.skipped_files = skipped;
    }

    /// Supply the embedder used by [`RetrievalMode::Dense`]. The library
    /// stays neutral about model choice — build any [`EmbeddingProvider`]
    /// yourself (e.g. an ONNX BGE embedder behind your own `onnx` feature) and
    /// inject it here. No effect under [`RetrievalMode::Lexical`].
    ///
    /// Call before the first `context`/`analyze` query (it resets the lazily
    /// built index).
    pub fn with_embedder(mut self, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedder = Some(embedder);
        self.retriever = None;
        self
    }

    /// Supply the [`Analyzer`](crate::analyzer::Analyzer) that drives the
    /// BM25 retriever AND the grounding scorer in [`crate::context`].
    /// Defaults to English Snowball Porter2; swap for another language with
    /// e.g. `Document::from_text("d", "Bücher")?.with_analyzer(Arc::new(SnowballAnalyzer::german()))`.
    ///
    /// Sets both `self.analyzer` (drives the retrievers) AND
    /// `self.cfg.context.analyzer` (drives the grounding scorer) so the two
    /// layers stay in lockstep — that's the whole point of the trait. Resets
    /// any lazily-built BM25 index / fallback (analyzer is fixed at index
    /// time per Tantivy's constraints).
    ///
    /// Call before the first `context`/`analyze` query.
    pub fn with_analyzer(mut self, analyzer: Arc<dyn crate::analyzer::Analyzer>) -> Self {
        self.analyzer = analyzer.clone();
        self.cfg.context.analyzer = analyzer;
        self.retriever = None;
        self.fallback_bm25 = None;
        self
    }

    /// Supply a **separate query-side embedder** for asymmetric models — e.g. E5,
    /// which needs a `passage:` prefix on documents (the [`Document::with_embedder`]
    /// one) and a `query:` prefix on queries (this one). For symmetric models
    /// (BGE, MiniLM) you don't need this — `with_embedder` alone handles both.
    /// Has no effect under [`RetrievalMode::Lexical`].
    pub fn with_query_embedder(mut self, query_embedder: Arc<dyn EmbeddingProvider>) -> Self {
        self.query_embedder = Some(query_embedder);
        self.retriever = None;
        self
    }

    /// Supply a second-stage [`Reranker`] (e.g. a cross-encoder). The first stage
    /// retrieves `cfg.rerank_pool` candidates and the reranker reorders them down
    /// to the requested `k` — jointly scoring each `(query, passage)` pair, which
    /// is more accurate than the first-stage ranking but costs a model call per
    /// candidate. The library stays model-neutral: build any [`Reranker`] and
    /// inject it here. Works under any retrieval mode (it reorders whatever the
    /// first stage surfaced). No-op until the next `context`/`analyze` query.
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Number of chunks the document holds.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// The document's chunks (text, source, citation metadata, and embedding when
    /// one has been attached). Read-only view for inspection or persistence.
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// The chunks **with embeddings filled in** — indexes the document if needed
    /// (so dense/hybrid embeddings are computed), then returns a clone of the
    /// chunks with each `embedding` populated from the retriever's cache. For the
    /// lexical tier (no embeddings) this is just the chunks. Use this to persist a
    /// folder index: saved chunks carry their vectors, so reloading skips
    /// re-embedding everything.
    pub fn embedded_chunks(&mut self) -> Result<Vec<Chunk>> {
        self.ensure_indexed()?;
        let mut out = self.chunks.clone();
        if let Some(map) = self.retriever.as_ref().and_then(|r| r.embeddings()) {
            for c in &mut out {
                if c.embedding.is_none() {
                    if let Some(e) = map.get(c.id.as_str()) {
                        c.embedding = Some(e.clone());
                    }
                }
            }
        }
        Ok(out)
    }

    /// Whether the document has no chunks.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Total tokens across all of the document's chunks — the full size before
    /// any retrieval or context allocation.
    pub fn total_tokens(&self) -> usize {
        self.chunks.iter().map(|c| c.token_count.value()).sum()
    }

    /// Assemble the reasoning context for a query: retrieve candidates from
    /// the internal index, then allocate them under the context policy.
    /// Returns the prompt context plus a [`ContextReport`] of what it did.
    ///
    /// Uses the document's default token budget. Budget is a *query-time*
    /// concern (it doesn't touch the index) — vary it per call with
    /// [`Document::context_with`].
    pub fn context(&mut self, query: &str) -> Result<BuiltContext> {
        self.context_with(query, None, None)
    }

    /// [`Document::context`] with optional per-query overrides. `budget` and
    /// `candidate_k` are query-time and require **no** re-indexing (unlike
    /// chunk size, which is fixed at construction). `None` keeps the
    /// document's default.
    pub fn context_with(
        &mut self,
        query: &str,
        budget: Option<usize>,
        candidate_k: Option<usize>,
    ) -> Result<BuiltContext> {
        self.context_inner(query, budget, candidate_k, &[])
    }

    /// [`Document::context`] with a chain of query-side rewrites applied
    /// before retrieval. Each rewrite ([`crate::rewrite::Stripper`],
    /// [`crate::rewrite::Vocabulary`], or anything implementing
    /// [`crate::rewrite::QueryRewrite`]) runs in order; the rewritten
    /// query is the one BM25 sees; the per-stage audit trail lands in
    /// `ctx.report.query_rewrites` so every change is auditable in the
    /// Decision Report.
    ///
    /// ```no_run
    /// # use redhop::{Document, rewrite::{Stripper, Vocabulary}};
    /// # fn main() -> redhop::Result<()> {
    /// let stripper = Stripper::new(&[
    ///     "highlight", "the", "parts", "of", "this", "contract",
    ///     "related", "to",
    /// ]);
    /// let vocabulary = Vocabulary::new(&[
    ///     ("change of control", &["merger", "successor", "acquisition"][..]),
    /// ]);
    /// let mut doc = Document::from_text("contract.pdf", "…")?;
    /// let ctx = doc.context_with_rewrites(
    ///     "Highlight the parts of this contract related to \"Change of Control\".",
    ///     &[&stripper, &vocabulary],
    /// )?;
    /// for r in &ctx.report.query_rewrites {
    ///     println!("{}: {:?} → added {:?}", r.stage, r.matched, r.added);
    /// }
    /// # Ok(()) }
    /// ```
    pub fn context_with_rewrites(
        &mut self,
        query: &str,
        rewrites: &[&dyn crate::rewrite::QueryRewrite],
    ) -> Result<BuiltContext> {
        self.context_inner(query, None, None, rewrites)
    }

    /// Internal shared body for [`Self::context`], [`Self::context_with`],
    /// and [`Self::context_with_rewrites`]. Threading the rewrite chain
    /// through one path keeps retrieval and assembly in lockstep — BM25
    /// sees the rewritten query iff the report records the corresponding
    /// trail.
    fn context_inner(
        &mut self,
        query: &str,
        budget: Option<usize>,
        candidate_k: Option<usize>,
        rewrites: &[&dyn crate::rewrite::QueryRewrite],
    ) -> Result<BuiltContext> {
        let (rewritten, trail) = crate::rewrite::apply_chain(query, rewrites);
        let k = candidate_k.unwrap_or(self.cfg.candidate_k);
        let results = self.retrieve(&rewritten, k)?;
        let mut cfg = self.cfg.context.clone();
        if let Some(b) = budget {
            cfg.token_budget = b;
        }
        let q = Query::new(&rewritten);

        // Auto-expand based on what the retrieved set looks like:
        //
        // - **code** chunks → pull ±`code_neighbors_default` adjacent chunks
        //   so a citation on a `def` line includes the implementation body.
        // - **prose** chunks with section heading metadata → attach the
        //   section's opening chunk via `include_heading` so a deep-section
        //   citation arrives with its parent heading for context.
        //
        // Either default can be turned off (`code_neighbors_default = 0` or
        // `prose_heading_default = false`); when both fire on a mixed corpus
        // (some chunks code, some prose-with-headings) both apply.
        let neighbors = if self.cfg.code_neighbors_default > 0 && results.iter().any(is_code_chunk)
        {
            self.cfg.code_neighbors_default
        } else {
            0
        };
        let include_heading =
            self.cfg.prose_heading_default && results.iter().any(has_prose_heading);
        let mut ctx = if neighbors > 0 || include_heading {
            let plan = self.expansion_plan(&results, neighbors, include_heading);
            build_context_expanded(&q, &results, &cfg, &plan)
        } else {
            build_context(&q, &results, &cfg)
        };
        // Attach the rewrite trail to the report so the chain is auditable.
        // Empty trail when no rewrites were supplied (the report's
        // `query_rewrites` is `Vec::new()` by default).
        crate::context::attach_rewrite_trail(&mut ctx, trail);
        Ok(ctx)
    }

    /// [`Document::context_with`] plus **structural context expansion**: after the
    /// normal relevance-and-budget selection, attach to each selected chunk its
    /// `neighbors` adjacent chunks (i±1, i±2, … in the same file) and — when
    /// `include_heading` — its section's heading chunk. Companions are
    /// deterministic (from document order and headings, no model), exempt from the
    /// distractor filter, bounded by the token budget, and emitted in document
    /// order so each hit reads as a contiguous window. `neighbors=0` +
    /// `include_heading=false` is exactly [`Document::context_with`].
    pub fn context_expanded(
        &mut self,
        query: &str,
        budget: Option<usize>,
        candidate_k: Option<usize>,
        neighbors: usize,
        include_heading: bool,
    ) -> Result<BuiltContext> {
        let k = candidate_k.unwrap_or(self.cfg.candidate_k);
        let results = self.retrieve(query, k)?;
        let mut cfg = self.cfg.context.clone();
        if let Some(b) = budget {
            cfg.token_budget = b;
        }
        let q = Query::new(query);
        if neighbors == 0 && !include_heading {
            return Ok(build_context(&q, &results, &cfg));
        }
        let plan = self.expansion_plan(&results, neighbors, include_heading);
        Ok(build_context_expanded(&q, &results, &cfg, &plan))
    }

    /// Build the deterministic [`ExpansionPlan`] for a retrieved set: each seed's
    /// adjacent same-file neighbors (nearest first) and section-heading chunk,
    /// plus the document position of every chunk involved (for reading order).
    fn expansion_plan(
        &self,
        results: &[RetrievalResult],
        neighbors: usize,
        include_heading: bool,
    ) -> ExpansionPlan {
        use std::collections::HashMap;
        let pos: HashMap<&str, usize> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.as_str(), i))
            .collect();
        let heading_of = |c: &Chunk| -> Option<String> {
            c.metadata
                .get("heading")
                .and_then(|v| v.as_str())
                .map(String::from)
        };

        let mut plan = ExpansionPlan::default();
        for r in results {
            let id = r.chunk.id.as_str();
            let Some(&idx) = pos.get(id) else { continue };
            plan.position.insert(id.to_string(), idx);
            let seed = &self.chunks[idx];
            let mut comps: Vec<Chunk> = Vec::new();

            // Adjacent neighbors in the SAME file, nearest first: i-1, i+1, i-2, …
            for d in 1..=neighbors {
                for cand in [idx.checked_sub(d), idx.checked_add(d)]
                    .into_iter()
                    .flatten()
                {
                    if cand < self.chunks.len() && self.chunks[cand].source == seed.source {
                        comps.push(self.chunks[cand].clone());
                    }
                }
            }

            // The section heading: earliest chunk in the same (source, heading).
            if include_heading {
                if let Some(h) = heading_of(seed) {
                    if let Some(hc) = self.chunks.iter().find(|c| {
                        c.source == seed.source && heading_of(c).as_deref() == Some(h.as_str())
                    }) {
                        if hc.id != seed.id {
                            comps.push(hc.clone());
                        }
                    }
                }
            }

            for c in &comps {
                if let Some(&j) = pos.get(c.id.as_str()) {
                    plan.position.insert(c.id.as_str().to_string(), j);
                }
            }
            plan.companions.insert(id.to_string(), comps);
        }
        plan
    }

    /// Diagnose the retrieval for a query **without** modifying anything:
    /// distractor load, evidence density, second-hop candidates, and (for the
    /// Auto policy) whether it would prune. Pure observability.
    pub fn analyze(&mut self, query: &str) -> Result<ContextReport> {
        let results = self.retrieve(query, self.cfg.candidate_k)?;
        Ok(analyze_context(
            &Query::new(query),
            &results,
            &self.cfg.context,
        ))
    }

    fn ensure_indexed(&mut self) -> Result<()> {
        if self.retriever.is_none() {
            let mut r: Box<dyn Retriever> = match self.cfg.retrieval_mode {
                RetrievalMode::Lexical => {
                    Box::new(Bm25Retriever::with_analyzer(self.analyzer.clone())?)
                }
                RetrievalMode::Hybrid { candidate_pool } => {
                    let embedder = self.embedder.clone().ok_or_else(|| {
                        crate::core::Error::InvalidConfig(
                            "RetrievalMode::Hybrid requires an embedder — supply one with \
                             `Document::with_embedder(...)`, or use the default \
                             RetrievalMode::Lexical."
                                .into(),
                        )
                    })?;
                    let r = match self.query_embedder.clone() {
                        Some(q) => LocalRerankRetriever::new_with_query_embedder(
                            embedder,
                            q,
                            candidate_pool,
                        )?,
                        None => LocalRerankRetriever::new(embedder, candidate_pool)?,
                    };
                    Box::new(r.with_analyzer(self.analyzer.clone())?)
                }
                RetrievalMode::Dense => {
                    let embedder = self.embedder.clone().ok_or_else(|| {
                        crate::core::Error::InvalidConfig(
                            "RetrievalMode::Dense requires an embedder — supply one with \
                             `Document::with_embedder(...)`, or use the default \
                             RetrievalMode::Lexical."
                                .into(),
                        )
                    })?;
                    // candidate_pool is unused for global; pass a sane value.
                    let r = match self.query_embedder.clone() {
                        Some(q) => LocalRerankRetriever::new_with_query_embedder(embedder, q, 1)?,
                        None => LocalRerankRetriever::new(embedder, 1)?,
                    };
                    // Analyzer threaded through so the optional fallback /
                    // hybrid-degradation path agrees with the rest of the
                    // stack on what a term is. Global Dense doesn't use it
                    // for ranking but the internal BM25 might still be
                    // queried.
                    Box::new(r.with_analyzer(self.analyzer.clone())?.global())
                }
            };
            self.rt.block_on(r.index(&self.chunks))?;
            self.retriever = Some(r);
        }
        Ok(())
    }

    fn retrieve(&mut self, query: &str, k: usize) -> Result<Vec<RetrievalResult>> {
        self.ensure_indexed()?;
        let q = Query::new(query);
        // Primary retrieval scoped so its &self borrow drops before the
        // fallback path may need &mut self for lazy BM25 init.
        let mut results = {
            let retriever = self.retriever.as_ref().expect("indexed above");
            match self.reranker.as_ref() {
                Some(reranker) => {
                    let pool = self.cfg.rerank_pool.max(k);
                    let cand = self.rt.block_on(retriever.retrieve(&q, pool))?;
                    self.rt.block_on(reranker.rerank(&q, cand, k))?
                }
                None => self.rt.block_on(retriever.retrieve(&q, k))?,
            }
        };

        // Top-up fallback (issue #1): if the primary retriever returned
        // fewer than `cfg.min_candidates`, pad from BM25 over the same
        // chunks until the floor is met. No-op under Lexical (the primary
        // already *is* BM25) and when the floor is 0 (the default).
        if results.len() < self.cfg.min_candidates
            && !matches!(self.cfg.retrieval_mode, RetrievalMode::Lexical)
        {
            self.ensure_fallback_indexed()?;
            let need = self.cfg.min_candidates;
            let existing: std::collections::HashSet<crate::core::ChunkId> =
                results.iter().map(|r| r.chunk.id.clone()).collect();
            let fallback = self
                .fallback_bm25
                .as_ref()
                .expect("ensure_fallback_indexed populated it");
            let lex = self.rt.block_on(fallback.retrieve(&q, need.max(k)))?;
            for r in lex {
                if results.len() >= need {
                    break;
                }
                if !existing.contains(&r.chunk.id) {
                    results.push(r);
                }
            }
        }

        // Retrievers (e.g. the Tantivy-backed BM25 index) only round-trip
        // id/text/source, so per-chunk `metadata` (page/heading for citations) is
        // lost. Re-attach it from the source chunks, keyed by id.
        if self.chunks.iter().any(|c| !c.metadata.is_empty()) {
            for r in &mut results {
                if let Some(orig) = self.chunks.iter().find(|c| c.id == r.chunk.id) {
                    r.chunk.metadata = orig.metadata.clone();
                }
            }
        }
        Ok(results)
    }

    /// Lazily build a BM25 index over the document's chunks for the
    /// `min_candidates` fallback path. Indexing happens at most once per
    /// document; documents that never trigger the floor never pay.
    fn ensure_fallback_indexed(&mut self) -> Result<()> {
        if self.fallback_bm25.is_some() {
            return Ok(());
        }
        // Same analyzer as the primary retriever — otherwise the fallback's
        // notion of "matches the query" would diverge from the primary's,
        // which is exactly the silent-search-miss class we're architecturally
        // avoiding.
        let mut bm25 = Bm25Retriever::with_analyzer(self.analyzer.clone())?;
        self.rt.block_on(bm25.index(&self.chunks))?;
        self.fallback_bm25 = Some(bm25);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "The safety lamp was invented by Humphry Davy. \
        Humphry Davy was born in Penzance, Cornwall, England, and was a chemist. \
        Photosynthesis converts sunlight into chemical energy in green plants. \
        The Eiffel Tower is located in Paris, France. \
        Rust is a systems programming language focused on memory safety.";

    #[test]
    fn from_text_chunks_and_retrieves() {
        let mut doc = Document::from_text_with(
            "doc",
            TEXT,
            DocumentConfig {
                target_tokens: 8,
                max_tokens: 16,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(doc.len() >= 2, "text should produce multiple chunks");
        let ctx = doc
            .context("Where was the safety lamp inventor born?")
            .unwrap();
        assert!(!ctx.text().is_empty());
        // The retrieved+assembled context should reach the answer evidence.
        assert!(ctx.text().to_lowercase().contains("penzance"));
    }

    #[test]
    fn reranker_controls_final_selection() {
        use crate::core::{RetrievalMethod, Score};
        use async_trait::async_trait;

        // A model-free stand-in for a cross-encoder: keep only candidates whose
        // text mentions "photosynthesis". This proves the reranker, not the BM25
        // first stage, decides what reaches the assembled context.
        struct KeepPhotosynthesis;
        #[async_trait]
        impl Reranker for KeepPhotosynthesis {
            async fn rerank(
                &self,
                _q: &Query,
                cands: Vec<RetrievalResult>,
                top_k: usize,
            ) -> Result<Vec<RetrievalResult>> {
                let mut kept: Vec<RetrievalResult> = cands
                    .into_iter()
                    .filter(|r| r.chunk.text.to_lowercase().contains("photosynthesis"))
                    .map(|mut r| {
                        r.score = Score {
                            value: 1.0,
                            method: RetrievalMethod::Rerank,
                        };
                        r
                    })
                    .collect();
                kept.truncate(top_k);
                Ok(kept)
            }
            fn name(&self) -> &'static str {
                "keep_photosynthesis"
            }
        }

        // A query whose terms span several chunks (incl. "photosynthesis", so the
        // BM25 pool surfaces that chunk for the reranker to act on).
        let query = "Davy photosynthesis Eiffel Rust";
        let mut doc = Document::from_text_with(
            "doc",
            TEXT,
            DocumentConfig {
                target_tokens: 8,
                max_tokens: 16,
                ..Default::default()
            },
        )
        .unwrap();
        let baseline = doc.context(query).unwrap().text().to_lowercase();

        let mut doc = doc.with_reranker(Arc::new(KeepPhotosynthesis));
        let reranked = doc.context(query).unwrap().text().to_lowercase();

        // The reranker dictates the final set: only the photosynthesis chunk
        // survives, the others are gone, and the output differs from the baseline.
        assert!(
            reranked.contains("photosynthesis"),
            "reranker should keep it"
        );
        assert!(
            !reranked.contains("eiffel"),
            "reranker should drop the rest"
        );
        assert_ne!(baseline, reranked, "reranker should change the selection");
    }

    #[test]
    fn analyze_is_non_destructive_and_reports_a_decision() {
        let mut doc = Document::from_text_with(
            "doc",
            TEXT,
            DocumentConfig {
                target_tokens: 8,
                max_tokens: 16,
                ..Default::default()
            },
        )
        .unwrap();
        let report = doc.analyze("safety lamp inventor nationality").unwrap();
        assert!(report.n_input_chunks > 0);
        // Auto on a small retrieval → passthrough (no pruning).
        assert_eq!(report.strategy, ContextStrategy::RawTopK);
    }

    #[test]
    fn sections_carry_citation_metadata_through_retrieval() {
        let sections = vec![
            Section {
                text: "Customers may request a refund within 30 days of purchase.".into(),
                page: Some(3),
                heading: Some("Refund Policy".into()),
                line: Some(10),
            },
            Section {
                text: "Orders ship within two business days.".into(),
                page: Some(4),
                heading: Some("Shipping".into()),
                line: Some(20),
            },
        ];
        let mut doc =
            Document::from_sections_with("contract.pdf", sections, DocumentConfig::default())
                .unwrap();
        let ctx = doc.context("refund window").unwrap();
        let cited = ctx
            .chunks
            .iter()
            .find(|c| c.text.contains("refund"))
            .expect("refund chunk retrieved");
        // Metadata survives the Tantivy round-trip (re-attached by id).
        assert_eq!(cited.source, "contract.pdf");
        assert_eq!(cited.metadata.get("page").and_then(|v| v.as_u64()), Some(3));
        assert_eq!(
            cited.metadata.get("heading").and_then(|v| v.as_str()),
            Some("Refund Policy")
        );
        assert_eq!(
            cited.metadata.get("line").and_then(|v| v.as_u64()),
            Some(10)
        );
    }

    #[test]
    fn from_sources_keeps_per_file_source_and_metadata() {
        let files = vec![
            (
                "refunds.md".to_string(),
                vec![Section {
                    text: "Customers may request a refund within 30 days.".into(),
                    page: None,
                    heading: Some("Refund Policy".into()),
                    line: Some(5),
                }],
            ),
            (
                "shipping.txt".to_string(),
                vec![Section {
                    text: "Orders ship within two business days.".into(),
                    page: None,
                    heading: None,
                    line: Some(1),
                }],
            ),
        ];
        let mut doc = Document::from_sources_with(files, DocumentConfig::default()).unwrap();
        assert_eq!(doc.len(), 2, "one chunk per file");
        let ctx = doc.context("refund within days").unwrap();
        let cited = ctx
            .chunks
            .iter()
            .find(|c| c.text.contains("refund"))
            .expect("refund chunk retrieved");
        // The retrieved chunk carries ITS file's source + metadata, not the other's.
        assert_eq!(cited.source, "refunds.md");
        assert_eq!(
            cited.metadata.get("heading").and_then(|v| v.as_str()),
            Some("Refund Policy")
        );
        assert_eq!(cited.metadata.get("line").and_then(|v| v.as_u64()), Some(5));
    }

    #[test]
    fn empty_document_errors_clearly() {
        let err = Document::from_chunks(vec![])
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no chunks"), "unhelpful error: {err}");
        // Whitespace-only text produces no chunks → same clear error.
        assert!(Document::from_text("doc", "   \n  ").is_err());
    }

    #[test]
    fn from_chunks_skips_chunking() {
        let chunks = vec![
            Chunk::new(
                crate::core::ChunkId::new("a"),
                "Humphry Davy was British.",
                "doc",
                crate::core::TokenCount(4),
            ),
            Chunk::new(
                crate::core::ChunkId::new("b"),
                "The safety lamp was invented by Humphry Davy.",
                "doc",
                crate::core::TokenCount(8),
            ),
        ];
        let mut doc = Document::from_chunks(chunks).unwrap();
        assert_eq!(doc.len(), 2);
        let ctx = doc.context("who invented the safety lamp").unwrap();
        assert!(!ctx.text().is_empty());
    }

    // Deterministic stub embedder (no model) for the Dense path.
    struct StubEmbedder;

    #[async_trait::async_trait]
    impl EmbeddingProvider for StubEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<crate::core::Embedding>> {
            Ok(texts
                .iter()
                .map(|t| {
                    crate::core::Embedding::from(vec![
                        t.matches("alpha").count() as f32,
                        t.matches("beta").count() as f32,
                    ])
                })
                .collect())
        }
        fn dim(&self) -> usize {
            2
        }
        fn name(&self) -> &'static str {
            "stub"
        }
    }

    fn rerank_cfg() -> DocumentConfig {
        DocumentConfig {
            retrieval_mode: RetrievalMode::Dense,
            ..Default::default()
        }
    }

    #[test]
    fn dense_rerank_without_embedder_errors_clearly() {
        let chunks = vec![Chunk::new(
            "a",
            "alpha text",
            "doc",
            crate::core::TokenCount(2),
        )];
        let mut doc = Document::from_chunks_with(chunks, rerank_cfg()).unwrap();
        let err = doc.context("alpha").unwrap_err().to_string();
        assert!(err.contains("embedder"), "unhelpful error: {err}");
    }

    #[test]
    fn dense_rerank_reorders_with_injected_embedder() {
        let chunks = vec![
            Chunk::new("a", "alpha alpha alpha", "doc", crate::core::TokenCount(3)),
            Chunk::new("b", "beta beta beta", "doc", crate::core::TokenCount(3)),
        ];
        let mut doc = Document::from_chunks_with(chunks, rerank_cfg())
            .unwrap()
            .with_embedder(Arc::new(StubEmbedder));
        // Query lexically hits both; the embedding leans to "beta".
        let ctx = doc.context("alpha beta beta").unwrap();
        assert!(ctx.text().contains("beta"));
    }
}
