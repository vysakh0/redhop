//! napi bindings for RedHop — the Node.js surface over the `redhop` Rust crate.
//!
//! Thin wrapper: the loader orchestration (config, embedder, folder persistence,
//! citations) lives in the `redhop` facade; this maps JS options to it and back.

#![deny(clippy::all)]

use std::collections::HashMap;

use napi::bindgen_prelude::{Buffer, Either};
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
    /// Floor on the number of candidates delivered to the assembler. Under
    /// `"hybrid"` / `"semantic"` retrieval, if the primary tier returns
    /// fewer, a BM25 fallback tops the result up to this number. Default
    /// `0` (off). No effect under `"lexical"`. Pair with
    /// `report.lowConfidenceRetrieval` to detect a weak-fallback case.
    pub min_candidates: Option<u32>,
    /// Lexical analyzer language. Drives both BM25 retrieval and the
    /// grounding scorer's term extraction so the two layers agree on what
    /// counts as "the same term" (English `compression` finds `compress`,
    /// German `Bücher` finds `Buch`, etc.). Default `"english"`.
    ///
    /// Supported: `"arabic"`, `"danish"`, `"dutch"`, `"english"`,
    /// `"finnish"`, `"french"`, `"german"`, `"greek"`, `"hungarian"`,
    /// `"italian"`, `"norwegian"`, `"portuguese"`, `"romanian"`,
    /// `"russian"`, `"spanish"`, `"swedish"`, `"tamil"`, `"turkish"` —
    /// the 18 Snowball Porter2 languages. Unknown strings ERROR (no
    /// silent fallback to English; a typo'd `"germann"` surfaces).
    pub language: Option<String>,
    /// Preserve source-document order of selected chunks instead of the
    /// strategy's relevance order. Useful for chat histories / transcripts
    /// where chronology matters. Default `false`. See
    /// `crates/examples/examples/chat_rag.rs` for the worked pattern.
    pub preserve_order: Option<bool>,
    /// When `Document.context(query)` retrieves a chunk classified as code
    /// (`metadata.kind === "code"`), auto-pull this many adjacent chunks
    /// (i±1, … in the same file) so the implementation body arrives with
    /// the signature. Default `1`; pass `0` to disable. See
    /// `docs/findings/CODE_NEIGHBORS_DEFAULT.md` for the measured tradeoff.
    pub code_neighbors_default: Option<u32>,
    /// When `Document.context(query)` retrieves a prose chunk with a
    /// `metadata.heading`, auto-attach the section's heading chunk for
    /// context. Default `true`. See
    /// `docs/findings/PROSE_HEADING_DEFAULT.md` for the measurement.
    pub prose_heading_default: Option<bool>,
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
            min_candidates: u(self.min_candidates),
            language: self.language,
            preserve_order: self.preserve_order,
            code_neighbors_default: u(self.code_neighbors_default),
            prose_heading_default: self.prose_heading_default,
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
#[derive(Clone)]
pub struct Citation {
    pub source: String,
    pub page: Option<u32>,
    pub heading: Option<String>,
    pub line: Option<u32>,
    pub text: String,
}

/// The Decision Report: what the assembly did, and why.
#[napi(object)]
#[derive(Clone)]
pub struct Report {
    /// The strategy actually used — the resolved concrete strategy after
    /// `auto` (if requested) was decided. One of `"raw_topk"`,
    /// `"distractor_filtered"`, `"redundancy_pruned"`, `"max_density"`,
    /// `"reasoning_preserving"`. Never `"auto"` (Auto is always resolved
    /// before assembly).
    pub strategy: String,
    /// What the caller requested — may be `"auto"`. Differs from `strategy`
    /// when the Auto policy resolved to a concrete action.
    pub requested_strategy: String,
    /// `"passthrough"` | `"prune"` | `"not_auto"`.
    pub auto_decision: String,
    /// Total tokens in the input (retrieved) set, before assembly.
    pub input_tokens: u32,
    /// The token-budget cap applied during assembly.
    pub token_budget: u32,
    pub total_tokens: u32,
    /// `total_tokens / token_budget` — how much of the budget was used.
    pub token_utilization: f64,
    /// How many chunks were in the input (retrieved) set.
    pub n_input_chunks: u32,
    /// How many chunks survived assembly into the final context.
    pub n_selected: u32,
    /// Fraction of input chunks below the grounding bar (distractors).
    pub input_distractor_ratio: f64,
    pub retained_evidence_ratio: f64,
    /// Number of below-bar chunks that were RESCUED as linked second hops
    /// by `reasoning_preserving` (rather than dropped as distractors).
    pub second_hop_rescues: u32,
    /// Permanent alias for `secondHopRescues`. Both names will always be
    /// present and equal — keeps parity with Python's
    /// `report.second_hop_rescue_count` getter while preserving the
    /// shorter `secondHopRescues` that 0.2.0 shipped.
    pub second_hop_rescue_count: u32,
    /// Compared with a relevance-only baseline (DistractorFiltered),
    /// how many MORE chunks did the chosen strategy retain? Positive
    /// values mean rescued reasoning evidence beyond what a relevance
    /// filter would have kept.
    pub reasoning_preservation_delta: u32,
    /// Structural-expansion chunks added (neighbors / headings).
    pub n_expanded: u32,
    /// Chunks dropped because their grounding was below the distractor
    /// bar (a subset of `removedTotal`).
    pub distractors_pruned: u32,
    /// Total chunks dropped during assembly (distractors + redundant +
    /// over-budget).
    pub removed_total: u32,
    /// Fraction of context tokens that are query-relevant (answer-bearing
    /// density proxy).
    pub evidence_density: f64,
    /// Fraction of selected chunks below the distractor grounding cutoff.
    pub distractor_ratio: f64,
    /// Estimated tokens spent on chunks below the grounding bar (waste,
    /// not evidence).
    pub estimated_waste_tokens: u32,
    /// `true` when nothing in the assembled context was above the grounding
    /// floor — the query may share little vocabulary with the corpus.
    pub low_confidence_retrieval: bool,
    /// The grounding ceiling that `low_confidence_retrieval` applied.
    pub low_confidence_threshold: f64,
    /// Audit trail of query rewrites applied (one entry per stage, in
    /// chain order). Empty when no rewrite chain was used. See
    /// `Document.contextWithRewrites`.
    pub query_rewrites: Vec<RewriteRecord>,
    /// Query-level diagnosis: term/corpus match facts and a small set
    /// of bounded hints with evidence citations. See the design doc
    /// `docs/design/REPORT_DIAGNOSIS.md`.
    pub diagnosis: Diagnosis,
    /// The human-readable Decision Report.
    pub rendered: String,
}

/// Query-level facts and bounded hints. See the design doc
/// `docs/design/REPORT_DIAGNOSIS.md`.
#[napi(object)]
#[derive(Clone)]
pub struct Diagnosis {
    /// Analyzed query terms, in first-occurrence order.
    pub query_terms: Vec<String>,
    /// `true` when corpus-level stats were computed (the call went
    /// through a `Document`); `false` for direct `buildContext` callers.
    pub corpus_stats_available: bool,
    /// Query terms that appear in zero chunks of the corpus.
    pub zero_match_terms: Vec<String>,
    /// Per-term corpus stats for terms that do appear (`df > 0`).
    pub term_stats: Vec<TermStat>,
    /// Query terms that appear in no retrieved candidate chunk.
    pub terms_unmatched_in_candidates: Vec<String>,
    /// Number of candidates handed to assembly.
    pub n_candidates: u32,
    /// Relative score spread across the top candidates.
    /// `null` when fewer than 2 candidates or the top score is `<= 0`.
    pub score_spread: Option<f64>,
    /// `true` when assembly selected zero chunks.
    pub empty_context: bool,
    /// Hints that fired, from the closed registry.
    pub hints: Vec<DiagnosisHint>,
}

/// Per-term corpus statistics for one query term that appears in the
/// corpus.
#[napi(object)]
#[derive(Clone)]
pub struct TermStat {
    /// The analyzed term (matches how the corpus was indexed).
    pub term: String,
    /// Number of corpus chunks containing the term.
    pub df: u32,
    /// `df / total corpus chunks`, in `[0, 1]`.
    pub df_ratio: f64,
}

/// One bounded hint from the closed registry. Carries the observation,
/// a code clients can branch on, and a path to the doc or finding that
/// justifies it.
#[napi(object)]
#[derive(Clone)]
pub struct DiagnosisHint {
    /// Stable identifier (snake_case): `"empty_context"`,
    /// `"vocab_mismatch"`, `"low_confidence"`,
    /// `"low_discrimination_query"`, `"underdetermined_query"`.
    pub code: String,
    /// One or two sentences. Observation only.
    pub message: String,
    /// Repo-relative path of the doc or finding grounding this hint.
    pub evidence: String,
}

/// One step in the rewrite chain — what the stage matched / added /
/// removed. Surfaced on `Report.queryRewrites` so the rewrite chain is
/// auditable.
#[napi(object)]
#[derive(Clone)]
pub struct RewriteRecord {
    /// The `QueryRewrite::name` of the stage that emitted this record
    /// (`"strip"` or `"vocabulary"` for the built-in implementations).
    pub stage: String,
    /// Input query handed to this stage.
    pub from_query: String,
    /// Output query produced by this stage.
    pub to_query: String,
    /// Surface forms in the input that this stage matched.
    pub matched: Vec<String>,
    /// Surface forms this stage *added* to the query (Vocabulary).
    pub added: Vec<String>,
    /// Surface forms this stage *removed* from the query (Stripper).
    pub removed: Vec<String>,
}

fn to_record(r: &redhop::RewriteRecord) -> RewriteRecord {
    RewriteRecord {
        stage: r.stage.clone(),
        from_query: r.from.clone(),
        to_query: r.to.clone(),
        matched: r.matched.clone(),
        added: r.added.clone(),
        removed: r.removed.clone(),
    }
}

fn strategy_to_str(s: redhop::ContextStrategy) -> &'static str {
    match s {
        redhop::ContextStrategy::RawTopK => "raw_topk",
        redhop::ContextStrategy::DistractorFiltered => "distractor_filtered",
        redhop::ContextStrategy::RedundancyPruned => "redundancy_pruned",
        redhop::ContextStrategy::MaxDensity => "max_density",
        redhop::ContextStrategy::ReasoningPreserving => "reasoning_preserving",
        redhop::ContextStrategy::Auto => "auto",
    }
}

/// A file that `Document.fromFolder` skipped, with the reason why.
#[napi(object)]
pub struct SkippedFile {
    /// Source path of the file that was skipped.
    pub source: String,
    /// Human-readable reason: unsupported format, unreadable bytes, no
    /// extractable text, etc.
    pub reason: String,
}

/// The assembled context: prompt string, selected chunks, citations, report.
///
/// Class (not plain object) so it can hold the underlying Rust `BuiltContext`
/// for in-process operations like [`evaluate`]. All four existing fields
/// (`text`, `chunks`, `citations`, `report`) remain accessible as JS
/// properties via getters.
#[napi]
pub struct BuiltContext {
    text_: String,
    chunks_: Vec<String>,
    citations_: Vec<Citation>,
    report_: Report,
    // Hidden: carries the full Rust struct so `redhop.evaluate(...)` can
    // read chunk IDs and the complete report shape.
    inner: redhop::context::BuiltContext,
}

#[napi]
impl BuiltContext {
    /// The assembled context as a single prompt string (drop-in for `llm.generate`).
    #[napi(getter)]
    pub fn text(&self) -> String {
        self.text_.clone()
    }
    /// Selected chunks in presentation order, as plain strings.
    #[napi(getter)]
    pub fn chunks(&self) -> Vec<String> {
        self.chunks_.clone()
    }
    /// Per-chunk provenance — source, page, heading, line, text.
    #[napi(getter)]
    pub fn citations(&self) -> Vec<Citation> {
        self.citations_.clone()
    }
    /// The Decision Report: what was kept, what was dropped, why.
    #[napi(getter)]
    pub fn report(&self) -> Report {
        self.report_.clone()
    }
}

fn to_report(r: &redhop::ContextReport) -> Report {
    let auto_decision = match r.auto_decision() {
        redhop::AutoDecision::Passthrough => "passthrough",
        redhop::AutoDecision::Prune => "prune",
        redhop::AutoDecision::NotAuto => "not_auto",
    }
    .to_string();
    Report {
        strategy: strategy_to_str(r.strategy).to_string(),
        requested_strategy: strategy_to_str(r.requested_strategy).to_string(),
        auto_decision,
        input_tokens: r.input_tokens as u32,
        token_budget: r.token_budget as u32,
        total_tokens: r.total_tokens as u32,
        token_utilization: r.token_utilization as f64,
        n_input_chunks: r.n_input_chunks as u32,
        n_selected: r.n_selected as u32,
        input_distractor_ratio: r.input_distractor_ratio as f64,
        retained_evidence_ratio: r.retained_evidence_ratio as f64,
        second_hop_rescues: r.second_hop_rescue_count as u32,
        // Permanent alias — same value, longer name (matches Python's
        // `report.second_hop_rescue_count`). See struct doc comment.
        second_hop_rescue_count: r.second_hop_rescue_count as u32,
        reasoning_preservation_delta: r.reasoning_preservation_delta as u32,
        n_expanded: r.n_expanded as u32,
        distractors_pruned: r.removed.distractor as u32,
        removed_total: r.removed.total as u32,
        evidence_density: r.economics.evidence_density as f64,
        distractor_ratio: r.economics.distractor_ratio as f64,
        estimated_waste_tokens: r.economics.estimated_waste_tokens as u32,
        low_confidence_retrieval: r.low_confidence_retrieval,
        low_confidence_threshold: r.low_confidence_threshold as f64,
        query_rewrites: r.query_rewrites.iter().map(to_record).collect(),
        diagnosis: to_diagnosis(&r.diagnosis),
        rendered: r.render(None),
    }
}

fn hint_code_to_str(c: redhop::HintCode) -> &'static str {
    match c {
        redhop::HintCode::EmptyContext => "empty_context",
        redhop::HintCode::VocabMismatch => "vocab_mismatch",
        redhop::HintCode::LowConfidence => "low_confidence",
        redhop::HintCode::LowDiscriminationQuery => "low_discrimination_query",
        redhop::HintCode::UnderdeterminedQuery => "underdetermined_query",
        // `HintCode` is `#[non_exhaustive]` — future variants degrade
        // to a stable sentinel rather than crash the binding.
        _ => "unknown",
    }
}

fn to_diagnosis(d: &redhop::Diagnosis) -> Diagnosis {
    Diagnosis {
        query_terms: d.query_terms.clone(),
        corpus_stats_available: d.corpus_stats_available,
        zero_match_terms: d.zero_match_terms.clone(),
        term_stats: d
            .term_stats
            .iter()
            .map(|t| TermStat {
                term: t.term.clone(),
                df: t.df,
                df_ratio: t.df_ratio as f64,
            })
            .collect(),
        terms_unmatched_in_candidates: d.terms_unmatched_in_candidates.clone(),
        n_candidates: d.n_candidates as u32,
        score_spread: d.score_spread.map(|s| s as f64),
        empty_context: d.empty_context,
        hints: d
            .hints
            .iter()
            .map(|h| DiagnosisHint {
                code: hint_code_to_str(h.code).to_string(),
                message: h.message.clone(),
                evidence: h.evidence.clone(),
            })
            .collect(),
    }
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
    let report = to_report(&ctx.report);
    BuiltContext {
        text_: ctx.text(),
        chunks_: ctx.chunks.iter().map(|c| c.text.clone()).collect(),
        citations_: citations,
        report_: report,
        inner: ctx,
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

    /// Build from typed `Chunk` instances you already produced.
    /// Bypasses the chunker — the caller's chunks are preserved 1-to-1
    /// (no resplitting) with their `source` / `id` / `metadata` all
    /// surviving into the index.
    #[napi(factory)]
    pub fn from_chunks(chunks: Vec<&Chunk>, options: Option<Options>) -> napi::Result<Document> {
        let core_chunks: Vec<redhop::core::Chunk> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| c.to_core(i))
            .collect();
        let inner = redhop::chunks_typed(core_chunks, &options.unwrap_or_default().into_load())
            .map_err(err)?;
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
        let inner =
            redhop::read_bytes_with(&data[..], &source, &options.unwrap_or_default().into_load())
                .map_err(err)?;
        Ok(Document { inner })
    }

    /// Build one index from every readable file in a folder. Honors `.gitignore` +
    /// `ignore` globs; `persist: true` saves an incremental on-disk index.
    #[napi(factory)]
    pub fn from_folder(path: String, options: Option<FolderOptions>) -> napi::Result<Document> {
        let inner = redhop::read_folder_with(&path, &options.unwrap_or_default().into_folder())
            .map_err(err)?;
        Ok(Document { inner })
    }

    /// Number of chunks the document holds.
    #[napi(getter)]
    pub fn chunk_count(&self) -> u32 {
        self.inner.len() as u32
    }

    /// Number of source files indexed into this Document. `1` for the
    /// single-source constructors (`fromText`, `fromFile`, `fromBytes`,
    /// `fromChunks`); the readable file count for `fromFolder` (excludes
    /// `skippedFiles`).
    #[napi(getter)]
    pub fn n_files(&self) -> u32 {
        self.inner.n_files() as u32
    }

    /// Files that `fromFolder` skipped, as `{ source, reason }` objects —
    /// unsupported formats, unreadable bytes, no extractable text (e.g.
    /// scanned PDFs without OCR), etc. Empty array for single-source
    /// constructors.
    #[napi(getter)]
    pub fn skipped_files(&self) -> Vec<SkippedFile> {
        self.inner
            .skipped_files()
            .iter()
            .map(|(source, reason)| SkippedFile {
                source: source.clone(),
                reason: reason.clone(),
            })
            .collect()
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
            self.inner
                .context_expanded(&query, budget, None, n, heading)
        }
        .map_err(err)?;
        Ok(to_built(ctx))
    }

    /// Run a query through a chain of `Stripper` / `Vocabulary` rewrites
    /// before retrieval, then assemble context as `.context(...)` does.
    ///
    /// `rewrites` is an ordered array — each stage sees the previous
    /// stage's output. The per-stage audit trail lands on
    /// `ctx.report.queryRewrites` as an array of `RewriteRecord`.
    ///
    /// Mirrors `redhop::Document::context_with_rewrites` in the Rust core.
    #[napi]
    pub fn context_with_rewrites(
        &mut self,
        query: String,
        rewrites: Vec<Either<&Stripper, &Vocabulary>>,
    ) -> napi::Result<BuiltContext> {
        let owned: Vec<OwnedRewrite> = rewrites
            .into_iter()
            .map(|e| match e {
                Either::A(s) => OwnedRewrite::Stripper(s.inner.clone()),
                Either::B(v) => OwnedRewrite::Vocabulary(v.inner.clone()),
            })
            .collect();
        let refs = rewrite_refs(&owned);
        let ctx = self
            .inner
            .context_with_rewrites(&query, &refs)
            .map_err(err)?;
        Ok(to_built(ctx))
    }

    /// Pure diagnostics: retrieve + score for the query but DON'T assemble
    /// the prompt. Returns the same `Report` shape as `context().report` so
    /// callers can audit what would happen without paying assembly cost or
    /// stringifying the chunks.
    #[napi]
    pub fn analyze(&mut self, query: String) -> napi::Result<Report> {
        let r = self.inner.analyze(&query).map_err(err)?;
        Ok(to_report(&r))
    }
}

/// Query grounding of a chunk's text: stopword-removed, Snowball-stemmed
/// query-term overlap, in `[0, 1]`. The same relevance signal the strategies
/// use internally; exposed so external code reuses redhop's exact notion of
/// relevance instead of drifting from it. Uses the English Snowball analyzer
/// — fixed here so the signal stays comparable across calls, independent of
/// any `Document`'s configured analyzer. (Since 0.3.2 the `Document` default
/// is a minimal raw pipeline; for analyzer-aware grounding go through
/// `Document.context().report`.)
#[napi]
pub fn grounding_score(query: String, text: String) -> f64 {
    redhop::context::grounding_score(&query, &text) as f64
}

/// Chunk↔chunk linkage strength: term-set Jaccard over the same normalized
/// terms — the bridge signal `reasoning_preserving` uses to decide whether
/// a low-relevance chunk is a rescuable second hop. In `[0, 1]`. Uses the
/// English Snowball analyzer (same fixed choice as `groundingScore`).
#[napi]
pub fn link_strength(a: String, b: String) -> f64 {
    redhop::context::link_strength(&a, &b) as f64
}

// ── Query-side diagnostics (templated workloads) ────────────────────────────
//
// Backed by `redhop::analyze_query_set`; pair with `Stripper` /
// `Vocabulary` (above) for the detect → strip → expand workflow. See
// docs/findings/QUERY_SET_ANALYZER.md for the cross-workload probe.

/// Diagnostic report over a representative sample of a workload's queries.
///
/// Returned by `analyzeQuerySet`. Fields:
/// - `nQueries` — how many queries were analyzed
/// - `isTemplated` — `true` if share ≥ 0.50 AND ≥ 2 boilerplate terms
/// - `templateWordShare` — mean fraction of each query that's shared (0..1)
/// - `boilerplateTerms` — words in ≥ 80% of queries, sorted by frequency desc
/// - `estimatedDilutionCost` — `"high"` | `"medium"` | `"low"` | `"none"`
/// - `suggestedAction` — workload-shape recommendation
#[napi(object)]
pub struct QuerySetReport {
    pub n_queries: u32,
    pub is_templated: bool,
    pub template_word_share: f64,
    pub boilerplate_terms: Vec<String>,
    pub estimated_dilution_cost: String,
    pub suggested_action: String,
}

fn cost_str(c: redhop::DilutionCost) -> String {
    match c {
        redhop::DilutionCost::High => "high",
        redhop::DilutionCost::Medium => "medium",
        redhop::DilutionCost::Low => "low",
        redhop::DilutionCost::None => "none",
    }
    .to_string()
}

/// Compiled boilerplate-removal rewrite. Token-level matching using the
/// document's analyzer, so a single-token stripper key cannot accidentally
/// erase a substring inside a longer word (an `"of"` stripper does **not**
/// erase the `"of"` inside `"office"`).
///
/// ```js
/// const stripper = new redhop.Stripper([
///   "highlight", "the", "parts", "of", "this",
///   "contract", "related", "to",
/// ]);
/// ```
///
/// Pass into the rewrite chain on `Document.contextWithRewrites(...)`.
#[napi]
pub struct Stripper {
    inner: redhop::Stripper,
}

#[napi]
impl Stripper {
    #[napi(constructor)]
    pub fn new(boilerplate: Vec<String>) -> Self {
        Self {
            inner: redhop::Stripper::new(&boilerplate),
        }
    }
    /// Apply the stripper to a query string outside the rewrite chain
    /// (ad-hoc / pipeline-outside cases). Returns the rewritten string;
    /// the audit record is discarded — use the chain via
    /// `Document.contextWithRewrites(...)` if you want the trail in the
    /// Decision Report.
    #[napi]
    pub fn apply(&self, query: String) -> String {
        use redhop::QueryRewrite;
        self.inner.apply(&query).text
    }
    /// Diagnose how this Stripper would act on `query`. Returns an object
    /// with the analyzer's view of the input and the configured boilerplate
    /// terms that did / did not fire / are silent no-ops.
    ///
    /// Read `probableSilentNoOp` first: a non-empty list means you
    /// configured a term, the raw query contains it, but the analyzer
    /// stemmed/normalized it away from the query's tokens (the actual
    /// bug). Empty list means every configured term that should have
    /// fired, did. The long-tail `unusedBoilerplate` is informational
    /// (terms genuinely absent from this input).
    ///
    /// ```js
    /// const s = new redhop.Stripper(["highlight", "the", "of", "antidisestablishmentarianism"]);
    /// const effect = s.isEffectiveOn("highlight the parts of office hours");
    /// console.log(effect.removedTerms);           // [ 'highlight', 'the', 'of' ]
    /// console.log(effect.unusedBoilerplate);      // [ 'antidisestablishmentarianism' ]
    /// console.log(effect.probableSilentNoOp);     // [] — every present term fired
    /// console.log(effect.stripped);               // 'parts office hours'
    /// ```
    #[napi]
    pub fn is_effective_on(&self, query: String) -> StripperEffect {
        let e = self.inner.is_effective_on(&query);
        StripperEffect {
            original: e.original,
            stripped: e.stripped,
            original_tokens: e.original_tokens,
            stripped_tokens: e.stripped_tokens,
            removed_terms: e.removed_terms,
            unused_boilerplate: e.unused_boilerplate,
            probable_silent_no_op: e.probable_silent_no_op,
        }
    }
    /// Number of distinct boilerplate surface forms compiled.
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        self.inner.len() as u32
    }
}

/// Diagnostic output of [`Stripper.isEffectiveOn`].
///
/// Every configured boilerplate term lands in exactly one of `removedTerms`
/// (fired on this query) or `unusedBoilerplate` (did not fire). The token
/// streams expose the analyzer's view so users can debug
/// "my Stripper compiled but seems to do nothing" without reading source.
#[napi(object)]
pub struct StripperEffect {
    /// The query passed in, verbatim.
    pub original: String,
    /// What `apply(query)` would return.
    pub stripped: String,
    /// Analyzer tokens of the original query.
    pub original_tokens: Vec<String>,
    /// Analyzer tokens of the stripped output.
    pub stripped_tokens: Vec<String>,
    /// Configured surface forms that fired on this query.
    pub removed_terms: Vec<String>,
    /// Configured surface forms that did NOT fire on this query. Includes
    /// both "absent from input" and "present but mis-analyzed"; see
    /// `probableSilentNoOp` for the actual-bug subset.
    pub unused_boilerplate: Vec<String>,
    /// Subset of `unusedBoilerplate` where the term's lowercased raw
    /// substring DOES appear in the query. These are the surprising
    /// ones — terms the user expected to fire (the input contains them)
    /// that nonetheless didn't strip. Usually a Snowball-stemming or
    /// analyzer-token boundary mismatch. Empty list = no silent no-ops.
    pub probable_silent_no_op: Vec<String>,
}

/// Compiled workload-curated equivalence classes (term → synonyms). Each
/// entry's key, when found in the query at token granularity, triggers
/// appending its synonyms to the rewritten query. With
/// `Vocabulary.bidirectional({...})`, any class member can be the trigger
/// and the others get appended (so PTO ↔ "paid time off" works in both
/// directions).
///
/// ```js
/// const vocab = new redhop.Vocabulary({
///   "change of control": ["merger", "successor", "acquisition"],
///   "non-compete":       ["restraint", "non-competition"],
/// });
/// // symmetric:
/// const pto = redhop.Vocabulary.bidirectional({
///   pto: ["paid time off", "vacation"],
/// });
/// ```
#[napi]
pub struct Vocabulary {
    inner: redhop::Vocabulary,
}

fn build_vocab_refs(
    entries: &std::collections::HashMap<String, Vec<String>>,
) -> (Vec<(String, Vec<String>)>, ()) {
    let pairs: Vec<(String, Vec<String>)> = entries
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    (pairs, ())
}

#[napi]
impl Vocabulary {
    /// Asymmetric vocabulary: the first form of each entry is the only
    /// trigger; the rest are appended on match.
    #[napi(constructor)]
    pub fn new(entries: std::collections::HashMap<String, Vec<String>>) -> Self {
        let (pairs, _) = build_vocab_refs(&entries);
        let borrowed: Vec<(&str, Vec<&str>)> = pairs
            .iter()
            .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
            .collect();
        let final_refs: Vec<(&str, &[&str])> =
            borrowed.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        Self {
            inner: redhop::Vocabulary::new(&final_refs),
        }
    }
    /// Symmetric (bidirectional) vocabulary: any class member can trigger
    /// expansion; the other members get appended.
    #[napi(factory)]
    pub fn bidirectional(entries: std::collections::HashMap<String, Vec<String>>) -> Self {
        let (pairs, _) = build_vocab_refs(&entries);
        let borrowed: Vec<(&str, Vec<&str>)> = pairs
            .iter()
            .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
            .collect();
        let final_refs: Vec<(&str, &[&str])> =
            borrowed.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        Self {
            inner: redhop::Vocabulary::bidirectional(&final_refs),
        }
    }
    /// Apply the vocabulary to a query string outside the rewrite chain.
    /// Returns the rewritten string; the audit record is discarded — use
    /// the chain via `Document.contextWithRewrites(...)` if you want the
    /// trail in the Decision Report.
    #[napi]
    pub fn apply(&self, query: String) -> String {
        use redhop::QueryRewrite;
        self.inner.apply(&query).text
    }
    /// Chunk-side enrichment — the symmetric to `apply`. Same compiled
    /// vocabulary, applied at ingest time to a chunk's text so opaque
    /// coded units (column names, error codes, API symbols, defined
    /// terms) become matchable for natural-language queries that don't
    /// share surface forms with them.
    ///
    /// Returns `{ text, record }` so the caller can collect the per-chunk
    /// audit trail. Use pattern:
    ///
    /// ```js
    /// const vocab = new redhop.Vocabulary({
    ///   usrSvc:  ["user service", "signup", "account creation"],
    ///   calcAmt: ["calculate amount", "billing total"],
    /// });
    /// const enrichedChunks = [];
    /// for (const chunk of rawChunks) {
    ///   const { text, record } = vocab.enrich(chunk);
    ///   enrichedChunks.push(text);
    ///   // if (record.matched.length) audit.push(record);
    /// }
    /// const doc = redhop.Document.fromChunks(enrichedChunks);
    /// ```
    ///
    /// **When this earns its keep.** `value ∝ shortness × opacity ×
    /// dictionary-exists`. Schema columns, API symbols, error codes,
    /// defined contract terms, clinical abbreviations — all extreme
    /// cases. Long descriptive prose is redundant. Bolting the same
    /// boilerplate onto every chunk re-creates the low-IDF dilution
    /// from CUAD_PRF_NULL. See `docs/findings/VOCABULARY_ENRICH.md`.
    #[napi]
    pub fn enrich(&self, chunk: String) -> EnrichResult {
        let r = self.inner.enrich(&chunk);
        EnrichResult {
            text: r.text,
            record: to_record(&r.record),
        }
    }
    /// Number of equivalence classes compiled.
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        self.inner.len() as u32
    }
}

/// Result of [`Vocabulary.enrich`] — the enriched chunk text plus the
/// per-chunk audit record. The record's `stage` is `"enrich"` (vs
/// `"vocabulary"` for the query-side `apply`).
#[napi(object)]
pub struct EnrichResult {
    /// The chunk text with any matched vocabulary's synonyms appended.
    pub text: String,
    /// The audit record for this enrichment: what was matched, what
    /// surface forms were added, and the before/after text.
    pub record: RewriteRecord,
}

/// Internal: a heterogeneous rewrite stage holding an owned Rust copy of
/// the underlying compiled rewrite. Stripper and Vocabulary are cheap to
/// clone (Arc<Analyzer> + Vec<String>), so we copy rather than juggle
/// napi borrow guards.
enum OwnedRewrite {
    Stripper(redhop::Stripper),
    Vocabulary(redhop::Vocabulary),
}

fn rewrite_refs(held: &[OwnedRewrite]) -> Vec<&dyn redhop::QueryRewrite> {
    held.iter()
        .map(|r| -> &dyn redhop::QueryRewrite {
            match r {
                OwnedRewrite::Stripper(s) => s,
                OwnedRewrite::Vocabulary(v) => v,
            }
        })
        .collect()
}

/// Diagnostic over a representative sample of queries — detects
/// templated-workload dilution and reports which terms are doing it.
///
/// Returns a [`QuerySetReport`]. Read `.isTemplated`, `.boilerplateTerms`,
/// `.suggestedAction`. See `docs/findings/QUERY_SET_ANALYZER.md` for the
/// cross-workload probe that validated the heuristic.
///
/// ```js
/// const { analyzeQuerySet, Stripper } = require("redhop");
/// const r = analyzeQuerySet(myQueries);
/// if (r.isTemplated) {
///   const stripper = new Stripper(r.boilerplateTerms);
///   const ctx = doc.contextWithRewrites(query, [stripper]);
/// }
/// ```
#[napi]
pub fn analyze_query_set(queries: Vec<String>) -> QuerySetReport {
    let r = redhop::analyze_query_set(&queries);
    QuerySetReport {
        n_queries: r.n_queries as u32,
        is_templated: r.is_templated,
        template_word_share: r.template_word_share as f64,
        boilerplate_terms: r.boilerplate_terms,
        estimated_dilution_cost: cost_str(r.estimated_dilution_cost),
        suggested_action: r.suggested_action,
    }
}

// ── In-process evaluation (no LLM judge) ────────────────────────────────────
// Backed by `redhop::evaluate`. See `docs/findings/EVALUATE_API.md`.

/// Optional inputs for [`evaluate`]. Any combination of fields is
/// supported — pass `answer` to unlock the lexical answer-quality
/// proxies (`faithfulnessLexical`, `relevancyLexical`); pass `goldChunks`
/// to unlock `contextRecall` / `contextPrecision`; pass `goldAnswer` to
/// unlock `answerTokenRecall`; pass `answer` AND `goldAnswer` to unlock
/// `correctnessLexical`. Omit all for self-eval only.
#[napi(object)]
pub struct EvaluateOptions {
    /// The LLM's answer text (what the model produced from the context).
    /// Unlocks the lexical answer-quality proxies. Distinct from
    /// `goldAnswer`, which is the ground truth.
    pub answer: Option<String>,
    /// IDs of chunks that should appear in the assembled context.
    pub gold_chunks: Option<Vec<String>>,
    /// Ground-truth answer text.
    pub gold_answer: Option<String>,
    /// **Opt-in**: compute `faithfulness_judged` via two-pass claim
    /// decomposition — extract atomic claims, verify them in a single
    /// batched LLM call, return the mean score. More accurate than
    /// the single-prompt default but costs +1 LLM call per evaluate.
    /// Only meaningful when a `judge` is also supplied (via
    /// `evaluateWithJudge`); ignored by the sync `evaluate(...)`.
    pub decompose_faithfulness: Option<bool>,
    /// **Opt-in**: compute `correctness_judged` via three-pass
    /// classification — extract claims from the answer, extract from
    /// the gold, classify each as TP / FP / FN, return F1. Also
    /// populates the diagnostic counters `nCorrectnessTp`,
    /// `nCorrectnessFp`, `nCorrectnessFn` on the report so callers
    /// can debug which facts were missed vs hallucinated.
    pub decompose_correctness: Option<bool>,
}

/// In-process evaluation report for one (query, BuiltContext) pair.
///
/// Self-eval fields are always populated; gold-relative fields are
/// `null`/`undefined` unless the corresponding option was supplied. The
/// composite `overall` blends whichever fields are present.
#[napi(object)]
pub struct EvalReport {
    /// `selected ∩ gold / |gold|`. `null` unless `goldChunks` was supplied.
    pub context_recall: Option<f64>,
    /// `selected ∩ gold / |selected|`. `null` unless `goldChunks` was supplied.
    pub context_precision: Option<f64>,
    /// Fraction of stemmed content terms in the gold answer that appear in
    /// the assembled context. `null` unless `goldAnswer` was supplied.
    pub answer_token_recall: Option<f64>,
    /// Sentence-level token-overlap proxy for faithfulness: fraction of
    /// the answer's sentences with at least half their content terms in
    /// the assembled context. **Lexical proxy, NOT real faithfulness** —
    /// an LLM judge is the right tool. `null` unless `answer` was supplied.
    pub faithfulness_lexical: Option<f64>,
    /// Token-overlap between the query and the answer (Snowball-stemmed,
    /// stopword-filtered). Proxy for "did the answer address the question".
    /// `null` unless `answer` was supplied.
    pub relevancy_lexical: Option<f64>,
    /// Token-overlap between the LLM's answer and the gold answer. Proxy
    /// for "did the LLM produce the right tokens". `null` unless BOTH
    /// `answer` and `goldAnswer` were supplied.
    pub correctness_lexical: Option<f64>,
    /// LLM-judged faithfulness. Populated by the async `evaluateWithJudge`
    /// path; always `null` from the sync `evaluate` (JS callbacks can't
    /// fire during a sync napi call). See `Judge.fromCallable(...)` for
    /// how to wire a JS LLM client.
    pub faithfulness_judged: Option<f64>,
    /// LLM-judged relevancy. See `faithfulness_judged` for the sync vs
    /// async note.
    pub relevancy_judged: Option<f64>,
    /// LLM-judged correctness. See `faithfulness_judged` for the sync
    /// vs async note.
    pub correctness_judged: Option<f64>,
    /// Number of atomic claims the judge extracted when
    /// `decomposeFaithfulness: true` was passed and a judge was
    /// supplied. `null` otherwise.
    pub n_faithfulness_claims_extracted: Option<u32>,
    /// Number of those claims the judge scored ≥ 0.5 against the
    /// context. `null` under the same conditions as
    /// `nFaithfulnessClaimsExtracted`.
    pub n_faithfulness_claims_supported: Option<u32>,
    /// Number of answer-claims supported by a reference claim.
    /// Populated only when `decomposeCorrectness: true` was passed
    /// and the classification pass succeeded.
    pub n_correctness_tp: Option<u32>,
    /// Number of answer-claims NOT supported by any reference claim.
    /// Same population conditions as `nCorrectnessTp`.
    pub n_correctness_fp: Option<u32>,
    /// Number of reference-claims NOT covered by the answer. Same
    /// population conditions as `nCorrectnessTp`.
    pub n_correctness_fn: Option<u32>,
    /// Mean grounding over selected chunks, in `[0, 1]`.
    pub mean_grounding: f64,
    /// Fraction of context tokens that are query-relevant.
    pub evidence_density: f64,
    /// Fraction of input evidence that made it through assembly.
    pub retained_evidence_ratio: f64,
    /// Bridge passages saved by the reasoning-preserving rescue.
    pub second_hop_rescues: u32,
    /// `true` when every selected chunk is at-or-below the grounding ceiling.
    pub low_confidence: bool,
    /// Tokens spent on below-bar chunks.
    pub estimated_waste_tokens: u32,
    /// Composite score in `[0, 1]` — the headline, blended.
    pub overall: f64,
}

/// Evaluate an assembled `BuiltContext` against optional ground truth.
///
/// Self-eval (meanGrounding, evidenceDensity, secondHopRescues,
/// lowConfidence, …) is always populated. Pass `options.goldChunks` to
/// unlock `contextRecall` / `contextPrecision`; pass `options.goldAnswer`
/// to unlock `answerTokenRecall`. Both optional, any combination
/// supported.
///
/// **What self-eval tells you, and what it doesn't.** Without gold, the
/// returned metrics describe how *focused* the context is on the query —
/// not whether the *correct* answer-bearing chunk is in there. A dense,
/// query-focused context can still be confidently wrong. For A/B
/// comparisons of `Stripper` / `Vocabulary` chains where you care about
/// correctness, supply `goldChunks` or `goldAnswer`; that's what unlocks
/// the retrieval-correctness signal.
///
/// Zero LLM calls — every metric is computed from the same primitives the
/// runtime uses to make its Decision Report. See `EVALUATE_API.md` for
/// the "refraction not independent measurement" design choice.
///
/// ```js
/// const ctx = doc.context("refund window");
/// const report = redhop.evaluate("refund window", ctx, {
///   goldChunks: ["§3.4"],
///   goldAnswer: "thirty days",
/// });
/// console.log(report.overall, report.contextRecall);
/// ```
#[napi]
pub fn evaluate(
    query: String,
    context: &BuiltContext,
    options: Option<EvaluateOptions>,
) -> EvalReport {
    let opts = options.unwrap_or(EvaluateOptions {
        answer: None,
        gold_chunks: None,
        gold_answer: None,
        decompose_faithfulness: None,
        decompose_correctness: None,
    });
    let q = redhop::core::Query::new(&query);
    let chunk_refs: Option<Vec<&str>> = opts
        .gold_chunks
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());
    let gold = match (chunk_refs.as_deref(), opts.gold_answer.as_deref()) {
        (None, None) => redhop::EvalGold::None,
        (Some(c), None) => redhop::EvalGold::Chunks(c),
        (None, Some(a)) => redhop::EvalGold::Answer(a),
        (Some(c), Some(a)) => redhop::EvalGold::Both {
            gold_chunk_ids: c,
            gold_answer: a,
        },
    };
    // Sync evaluate intentionally passes `None` for the judge — JS
    // callbacks can't safely fire during a sync napi call. Use
    // `evaluateWithJudge` (async) for judged metrics; the
    // decomposeFaithfulness / decomposeCorrectness options are
    // meaningless without a judge.
    let r = redhop::evaluate(
        &q,
        &context.inner,
        opts.answer.as_deref(),
        gold,
        None,
        redhop::EvalConfig::default(),
    );
    eval_report_from_rust(r)
}

fn eval_report_from_rust(r: redhop::EvalReport) -> EvalReport {
    EvalReport {
        context_recall: r.context_recall.map(|v| v as f64),
        context_precision: r.context_precision.map(|v| v as f64),
        answer_token_recall: r.answer_token_recall.map(|v| v as f64),
        faithfulness_lexical: r.faithfulness_lexical.map(|v| v as f64),
        relevancy_lexical: r.relevancy_lexical.map(|v| v as f64),
        correctness_lexical: r.correctness_lexical.map(|v| v as f64),
        faithfulness_judged: r.faithfulness_judged.map(|v| v as f64),
        relevancy_judged: r.relevancy_judged.map(|v| v as f64),
        correctness_judged: r.correctness_judged.map(|v| v as f64),
        n_faithfulness_claims_extracted: r.n_faithfulness_claims_extracted.map(|v| v as u32),
        n_faithfulness_claims_supported: r.n_faithfulness_claims_supported.map(|v| v as u32),
        n_correctness_tp: r.n_correctness_tp.map(|v| v as u32),
        n_correctness_fp: r.n_correctness_fp.map(|v| v as u32),
        n_correctness_fn: r.n_correctness_fn.map(|v| v as u32),
        mean_grounding: r.mean_grounding as f64,
        evidence_density: r.evidence_density as f64,
        retained_evidence_ratio: r.retained_evidence_ratio as f64,
        second_hop_rescues: r.second_hop_rescues as u32,
        low_confidence: r.low_confidence,
        estimated_waste_tokens: r.estimated_waste_tokens as u32,
        overall: r.overall as f64,
    }
}

/// Reverse of `eval_report_from_rust`. Needed by `summarize(reports)` so we
/// can reconstruct redhop::EvalReport from the napi surface form.
fn eval_report_to_rust(r: &EvalReport) -> redhop::EvalReport {
    redhop::EvalReport {
        context_recall: r.context_recall.map(|v| v as f32),
        context_precision: r.context_precision.map(|v| v as f32),
        answer_token_recall: r.answer_token_recall.map(|v| v as f32),
        faithfulness_lexical: r.faithfulness_lexical.map(|v| v as f32),
        relevancy_lexical: r.relevancy_lexical.map(|v| v as f32),
        correctness_lexical: r.correctness_lexical.map(|v| v as f32),
        faithfulness_judged: r.faithfulness_judged.map(|v| v as f32),
        relevancy_judged: r.relevancy_judged.map(|v| v as f32),
        correctness_judged: r.correctness_judged.map(|v| v as f32),
        n_faithfulness_claims_extracted: r.n_faithfulness_claims_extracted.map(|v| v as usize),
        n_faithfulness_claims_supported: r.n_faithfulness_claims_supported.map(|v| v as usize),
        n_correctness_tp: r.n_correctness_tp.map(|v| v as usize),
        n_correctness_fp: r.n_correctness_fp.map(|v| v as usize),
        n_correctness_fn: r.n_correctness_fn.map(|v| v as usize),
        mean_grounding: r.mean_grounding as f32,
        evidence_density: r.evidence_density as f32,
        retained_evidence_ratio: r.retained_evidence_ratio as f32,
        second_hop_rescues: r.second_hop_rescues as usize,
        low_confidence: r.low_confidence,
        estimated_waste_tokens: r.estimated_waste_tokens as usize,
        overall: r.overall as f32,
    }
}

/// Test-set aggregation. Pass an array of `EvalReport`s (typically one per
/// `(query, ctx, answer, gold)` triple in your test set) and get back means,
/// medians, and conditionally-present subset counts. Always-present metrics
/// are means/medians over all reports; optional metrics (gold-relative
/// scores, judged scores) are means over only the subset where they were
/// populated, with an `nWith…` companion so callers can spot "this number is
/// computed from 3 of 200 reports" before reading too much into it.
///
/// Mirrors `redhop.summarize(reports)` in Python and `redhop::summarize(&[...])`
/// in Rust.
#[napi]
pub fn summarize(reports: Vec<EvalReport>) -> EvalSummary {
    let inner_reports: Vec<redhop::EvalReport> =
        reports.iter().map(eval_report_to_rust).collect();
    let s = redhop::summarize(&inner_reports);
    EvalSummary {
        n: s.n as u32,
        mean_overall: s.mean_overall as f64,
        median_overall: s.median_overall as f64,
        mean_grounding: s.mean_grounding as f64,
        mean_evidence_density: s.mean_evidence_density as f64,
        low_confidence_rate: s.low_confidence_rate as f64,
        mean_context_recall: s.mean_context_recall.map(|v| v as f64),
        n_with_context_recall: s.n_with_context_recall as u32,
        mean_context_precision: s.mean_context_precision.map(|v| v as f64),
        n_with_context_precision: s.n_with_context_precision as u32,
        mean_answer_token_recall: s.mean_answer_token_recall.map(|v| v as f64),
        n_with_answer_token_recall: s.n_with_answer_token_recall as u32,
        mean_faithfulness_lexical: s.mean_faithfulness_lexical.map(|v| v as f64),
        n_with_faithfulness_lexical: s.n_with_faithfulness_lexical as u32,
        mean_relevancy_lexical: s.mean_relevancy_lexical.map(|v| v as f64),
        n_with_relevancy_lexical: s.n_with_relevancy_lexical as u32,
        mean_correctness_lexical: s.mean_correctness_lexical.map(|v| v as f64),
        n_with_correctness_lexical: s.n_with_correctness_lexical as u32,
        mean_faithfulness_judged: s.mean_faithfulness_judged.map(|v| v as f64),
        n_with_faithfulness_judged: s.n_with_faithfulness_judged as u32,
        mean_relevancy_judged: s.mean_relevancy_judged.map(|v| v as f64),
        n_with_relevancy_judged: s.n_with_relevancy_judged as u32,
        mean_correctness_judged: s.mean_correctness_judged.map(|v| v as f64),
        n_with_correctness_judged: s.n_with_correctness_judged as u32,
    }
}

/// Aggregate stats for a test set. Returned by `summarize(reports)`.
/// Always-present metrics are means/medians over all `n` reports;
/// conditionally-present metrics (gold-relative + LLM-judged) are means
/// over only the subset where each was populated. `nWith…` reports the
/// subset size so a caller can spot a number computed from 3 of 200
/// reports before reading too much into it.
#[napi(object)]
pub struct EvalSummary {
    /// Total number of reports in the input.
    pub n: u32,
    /// Mean of `overall` across all reports.
    pub mean_overall: f64,
    /// Median of `overall` across all reports.
    pub median_overall: f64,
    /// Mean of `meanGrounding` across all reports.
    pub mean_grounding: f64,
    /// Mean of `evidenceDensity` across all reports.
    pub mean_evidence_density: f64,
    /// Fraction of reports where `lowConfidence` was true, in `[0, 1]`.
    pub low_confidence_rate: f64,
    pub mean_context_recall: Option<f64>,
    /// Number of reports where `contextRecall` was populated.
    pub n_with_context_recall: u32,
    pub mean_context_precision: Option<f64>,
    pub n_with_context_precision: u32,
    pub mean_answer_token_recall: Option<f64>,
    pub n_with_answer_token_recall: u32,
    pub mean_faithfulness_lexical: Option<f64>,
    pub n_with_faithfulness_lexical: u32,
    pub mean_relevancy_lexical: Option<f64>,
    pub n_with_relevancy_lexical: u32,
    pub mean_correctness_lexical: Option<f64>,
    pub n_with_correctness_lexical: u32,
    pub mean_faithfulness_judged: Option<f64>,
    pub n_with_faithfulness_judged: u32,
    pub mean_relevancy_judged: Option<f64>,
    pub n_with_relevancy_judged: u32,
    pub mean_correctness_judged: Option<f64>,
    pub n_with_correctness_judged: u32,
}

// ── judged: JS-callable Judge for `evaluateWithJudge` ───────────────────────
//
// The judged metrics need a way for Rust to call into JS while
// `redhop::evaluate` runs. JS is single-threaded — we can't call back
// into the JS engine while Rust is holding the napi main-thread call.
// Standard napi-rs pattern: wrap the JS function as a
// `ThreadsafeFunction` (which is `Send + Sync` and can be called from
// any thread) and run `redhop::evaluate` on a `spawn_blocking` worker
// so the JS event loop is free to service the callback.

use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction};

/// judged Judge: wraps a JS callable that scores a (prompt, system)
/// pair on `[0, 1]`. Construct via `Judge.fromCallable(fn, name?)` and
/// pass to `evaluateWithJudge(query, ctx, judge, opts)`. Call
/// `.cached()` to memoize identical prompts across calls — useful for
/// CI determinism and avoiding double-billing on re-runs.
///
/// **Callback signature.** The JS callable receives THREE positional
/// args: `(err, prompt, system)`. The first arg is napi-rs's error
/// channel (null on the normal path; non-null only if the napi layer
/// itself failed to deserialize the input — vanishingly rare for
/// plain strings). Most users pass it through with destructuring:
///
/// ```js
/// const judge = redhop.Judge.fromCallable(async (err, prompt, system) => {
///   if (err) throw err;
///   const resp = await openai.chat.completions.create({ ... });
///   return parseFloat(resp.choices[0].message.content.trim());
/// }, "gpt-4o-mini").cached();
///
/// const report = await redhop.evaluateWithJudge(
///   query, ctx, judge,
///   { answer: "Thirty days from purchase.", goldAnswer: "thirty days" },
/// );
/// ```
///
/// **Return type.** The callable may return either a `number` (the
/// score) or a `string` (the raw LLM text — needed for the Phase-6
/// claim-extraction call when `decomposeFaithfulness: true`). A
/// numeric string ("0.8", "80%") is parsed via the same `parse_score`
/// the Rust core uses; a non-numeric string keeps score=0 but
/// preserves raw_text for downstream parsing.
///
/// **Error isolation.** A JS-thrown exception in the callable surfaces
/// as `Err` on the underlying future. `redhop::evaluate` treats that
/// as "this metric is unavailable" and leaves the corresponding
/// `_judged` field as `null` — the process doesn't crash and the
/// lexical fields stay populated.
#[napi]
pub struct Judge {
    inner: std::sync::Arc<dyn redhop::judge::Judge>,
}

#[napi]
impl Judge {
    /// Wrap a JS callable as a Judge. The callable is invoked as
    /// `callable(prompt, system) -> number | Promise<number>`. The
    /// returned value is normalized to `[0, 1]` via the same parser
    /// the Rust core uses (`judge::parse_score` accepts floats,
    /// percentages, "yes"/"no").
    ///
    /// `name` namespaces the cache key — swap it when you swap models
    /// so cached `gpt-4o-mini` scores don't reappear under
    /// `claude-haiku`.
    #[napi(factory)]
    pub fn from_callable(
        // `ErrorStrategy::CalleeHandled` lets a JS-thrown exception
        // surface as Err on the `call_async` future instead of FATAL-ing
        // the process. The trade-off: the JS callback's first arg is an
        // `err` (null on success); the actual prompt/system args are
        // shifted to positions 1 and 2. JS-side helpers can wrap this
        // shape ergonomically — see `Judge.fromCallable` docs.
        callable: ThreadsafeFunction<
            (String, Option<String>),
            ErrorStrategy::CalleeHandled,
        >,
        name: Option<String>,
    ) -> Self {
        let name = name.unwrap_or_else(|| "callable".to_string());
        let tsfn = std::sync::Arc::new(callable);
        let cb_name = name.clone();
        let cb_tsfn = tsfn.clone();
        let judge = redhop::judge::CallableJudge::with_name(
            name,
            move |req: &redhop::judge::JudgeRequest<'_>| {
                let prompt = req.prompt.to_string();
                let system = req.system.map(|s| s.to_string());
                // Driving the JS callback from a `spawn_blocking` worker:
                // call_async returns a Future the napi-rs runtime
                // resolves on the JS main thread when the callback
                // settles. block_on the future from this blocking
                // thread is safe (we're not on the runtime's worker).
                // A JS error / non-number return surfaces as Err(napi),
                // which we convert to the Judge trait's Error.
                // Accept either a numeric return (the score-only happy
                // path) or a string return (raw LLM text — needed by the
                // claim-decomposition extraction call, where the
                // judge response is a list of claims rather than a score).
                let future = cb_tsfn.call_async::<Either<f64, String>>(
                    Ok((prompt, system)),
                );
                let handle = tokio::runtime::Handle::current();
                let value = handle.block_on(future).map_err(|e| {
                    redhop::core::Error::Other(format!(
                        "js judge future failed (thrown exception, \
                         wrong return type, or transport error): {e}"
                    ))
                })?;
                match value {
                    Either::A(score) => Ok(redhop::judge::JudgeResponse {
                        score: (score as f32).clamp(0.0, 1.0),
                        raw_text: format!("{score}"),
                        model: cb_name.clone(),
                    }),
                    Either::B(raw) => {
                        // Best-effort parse for the score (handles
                        // numeric strings, percentages, yes/no — the same
                        // parser the OpenAI/Anthropic adapter would use).
                        // Extraction calls' raw_text is what's load-bearing;
                        // a parse failure leaves score at 0 but raw_text
                        // intact, which is the right behavior since the
                        // raw text is what the decomposition path needs.
                        let score = redhop::judge::parse_score(&raw).unwrap_or(0.0);
                        Ok(redhop::judge::JudgeResponse {
                            score: score.clamp(0.0, 1.0),
                            raw_text: raw,
                            model: cb_name.clone(),
                        })
                    }
                }
            },
        );
        Self {
            inner: std::sync::Arc::new(judge),
        }
    }

    /// Return a NEW Judge wrapping this one with an in-memory cache.
    /// Identical `(prompt, system)` pairs hit the cache instead of
    /// re-calling the JS callable.
    #[napi]
    pub fn cached(&self) -> Self {
        let arc_inner = self.inner.clone();
        let forwarder = redhop::judge::CallableJudge::with_name(
            self.inner.name().to_string(),
            move |req: &redhop::judge::JudgeRequest<'_>| arc_inner.score(req),
        );
        let cached = redhop::judge::CachedJudge::new(forwarder);
        Self {
            inner: std::sync::Arc::new(cached),
        }
    }

    /// Stable identifier used in the cache key.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

/// Options for [`evaluate_with_judge`] — same as `EvaluateOptions` plus
/// a `judge` field. Distinct type because napi-rs doesn't allow object-
/// type options with class-typed fields directly; we accept the Judge
/// via the function-level signature instead and split the options into
/// the existing struct (free of class types) + a separate Judge arg.
#[napi]
pub async fn evaluate_with_judge(
    query: String,
    context: &BuiltContext,
    judge: &Judge,
    options: Option<EvaluateOptions>,
) -> napi::Result<EvalReport> {
    let opts = options.unwrap_or(EvaluateOptions {
        answer: None,
        gold_chunks: None,
        gold_answer: None,
        decompose_faithfulness: None,
        decompose_correctness: None,
    });
    let ctx_inner = context.inner.clone();
    let judge_inner = judge.inner.clone();
    let result = tokio::task::spawn_blocking(move || {
        let q = redhop::core::Query::new(&query);
        let chunk_refs: Option<Vec<&str>> = opts
            .gold_chunks
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let gold = match (chunk_refs.as_deref(), opts.gold_answer.as_deref()) {
            (None, None) => redhop::EvalGold::None,
            (Some(c), None) => redhop::EvalGold::Chunks(c),
            (None, Some(a)) => redhop::EvalGold::Answer(a),
            (Some(c), Some(a)) => redhop::EvalGold::Both {
                gold_chunk_ids: c,
                gold_answer: a,
            },
        };
        let judge_ref: Option<&dyn redhop::judge::Judge> =
            Some(judge_inner.as_ref());
        let config = redhop::EvalConfig {
            decompose_faithfulness: opts.decompose_faithfulness.unwrap_or(false),
            decompose_correctness: opts.decompose_correctness.unwrap_or(false),
        };
        redhop::evaluate(&q, &ctx_inner, opts.answer.as_deref(), gold, judge_ref, config)
    })
    .await
    .map_err(|e| napi::Error::from_reason(format!("evaluate join: {e}")))?;
    Ok(eval_report_from_rust(result))
}

// ── Aspect critique ─────────────────────────────────────────────────────────
//
// One judge call per aspect. Same Judge type used by `evaluateWithJudge`;
// cached judges memoize identical aspect prompts across runs.

/// One qualitative dimension to score the answer on. `highIsGood`
/// controls polarity — when `false` (e.g. harmfulness), the LLM's
/// raw score is INVERTED to `1 - raw` before being returned, so high
/// values still mean "good answer" across the report.
#[napi(object)]
pub struct Aspect {
    /// Short label, used as the key in the returned `CritiqueReport`.
    pub name: String,
    /// What to score, phrased as a question the LLM can answer 0–1.
    pub definition: String,
    /// `true` keeps the LLM's raw score; `false` inverts it (so high
    /// = good regardless of aspect polarity). Default `true`.
    pub high_is_good: Option<bool>,
}

/// One scored aspect — the same shape as a tuple, but napi-rs only
/// serializes plain objects so we name it.
#[napi(object)]
pub struct AspectScore {
    /// Aspect's `name`.
    pub name: String,
    /// Polarity-corrected score in `[0, 1]`, or `null` if the judge
    /// errored on this aspect.
    pub score: Option<f64>,
}

/// Result of [`critique`]. The `scores` array preserves input order;
/// `null` scores are aspects where the judge call errored.
#[napi(object)]
pub struct CritiqueReport {
    /// Number of aspects scored.
    pub n: u32,
    /// Per-aspect scores in input order. Iterate the array, or look
    /// up by name on the JS side: `report.scores.find(s => s.name === "x")?.score`.
    pub scores: Vec<AspectScore>,
}

/// Optional inputs for [`critique`] beyond the required `answer`,
/// `aspects`, and `judge` args.
#[napi(object)]
pub struct CritiqueOptions {
    /// Context to embed in each aspect's prompt — useful for aspects
    /// that need the source material to score.
    pub context: Option<String>,
    /// Query to embed in each aspect's prompt — useful for aspects
    /// like "does the answer address the question".
    pub query: Option<String>,
}

#[napi]
pub async fn critique(
    answer: String,
    aspects: Vec<Aspect>,
    judge: &Judge,
    options: Option<CritiqueOptions>,
) -> napi::Result<CritiqueReport> {
    let judge_inner = judge.inner.clone();
    let opts = options.unwrap_or(CritiqueOptions {
        context: None,
        query: None,
    });
    let answer_owned = answer;
    let result = tokio::task::spawn_blocking(move || {
        let rh_aspects: Vec<redhop::critique::Aspect<'_>> = aspects
            .iter()
            .map(|a| redhop::critique::Aspect {
                name: &a.name,
                definition: &a.definition,
                high_is_good: a.high_is_good.unwrap_or(true),
            })
            .collect();
        redhop::critique(
            redhop::critique::CritiqueInputs {
                answer: &answer_owned,
                aspects: &rh_aspects,
                context: opts.context.as_deref(),
                query: opts.query.as_deref(),
            },
            judge_inner.as_ref(),
        )
    })
    .await
    .map_err(|e| napi::Error::from_reason(format!("critique join: {e}")))?;
    let scores: Vec<AspectScore> = result
        .scores
        .into_iter()
        .map(|(name, score)| AspectScore {
            name,
            score: score.map(|s| s as f64),
        })
        .collect();
    Ok(CritiqueReport {
        n: scores.len() as u32,
        scores,
    })
}

// ── Low-level context functions (caller brings their own chunks) ────────────
//
// `Document.fromChunks(...)` is the high-level path: hand RedHop a list of
// chunk strings and let it own the index. The functions below are for users
// who do their own retrieval (vector DB, BM25 outside RedHop, hybrid stacks)
// and want RedHop just for the final assembly + diagnostics step.

/// Optional fields for the [`Chunk`] constructor's options bag.
///
/// `metadata` is an open object (JSON-compatible values). Known keys are
/// picked up by the citations machinery — currently `page` (int),
/// `heading` (string), `line` (int). Anything else is preserved but
/// not surfaced by the built-in citations contract.
#[napi(object)]
#[derive(Default)]
pub struct ChunkOptions {
    /// The chunk's stable identifier. Defaults to `c0`, `c1`, …
    /// based on position in the array passed to `Document.fromChunks`.
    pub id: Option<String>,
    /// The chunk's provenance: file path, URL, logical handle. Defaults
    /// to `"input"`. This is what `ctx.citations[*].source` displays.
    pub source: Option<String>,
    /// Open metadata object (JSON-compatible values). Known keys
    /// (`page`, `heading`, `line`) are picked up by citations.
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Token count (defaults to whitespace word count).
    pub token_count: Option<u32>,
    /// Optional precomputed dense vector for embedding-based scoring.
    pub embedding: Option<Vec<f64>>,
}

/// One unit of content in a [`Document`] — the construction primitive
/// for callers who pre-chunked their corpus elsewhere (schema rows,
/// API endpoints, code symbols, defined contract terms, pre-segmented
/// paragraphs).
///
/// Two concepts are kept distinct:
/// - **`source`** — *provenance*: where the chunk came from.
/// - **`id`** — *identity*: a stable identifier for dedup and gold-chunk
///   evaluation.
///
/// ```js
/// const chunks = [
///   new redhop.Chunk(
///     "orders.amt (decimal) — order amount / revenue / spend in USD",
///     { source: "schema.sql", id: "orders.amt",
///       metadata: { table: "orders", column: "amt", type: "decimal" } },
///   ),
///   new redhop.Chunk(
///     "9.1 Governing Law. This Agreement shall be governed by …",
///     { source: "contract.pdf", metadata: { page: 12, heading: "9.1 Governing Law" } },
///   ),
/// ];
/// const doc = redhop.Document.fromChunks(chunks);
/// ```
#[napi]
#[derive(Clone)]
pub struct Chunk {
    text_: String,
    source_: Option<String>,
    id_: Option<String>,
    metadata_: HashMap<String, serde_json::Value>,
    token_count_: Option<u32>,
    embedding_: Option<Vec<f64>>,
}

#[napi]
impl Chunk {
    /// Construct a chunk from text plus optional fields.
    #[napi(constructor)]
    pub fn new(text: String, options: Option<ChunkOptions>) -> Self {
        let opts = options.unwrap_or_default();
        Self {
            text_: text,
            source_: opts.source,
            id_: opts.id,
            metadata_: opts.metadata.unwrap_or_default(),
            token_count_: opts.token_count,
            embedding_: opts.embedding,
        }
    }

    #[napi(getter)]
    pub fn text(&self) -> String {
        self.text_.clone()
    }

    #[napi(getter)]
    pub fn source(&self) -> Option<String> {
        self.source_.clone()
    }

    #[napi(getter)]
    pub fn id(&self) -> Option<String> {
        self.id_.clone()
    }

    #[napi(getter)]
    pub fn token_count(&self) -> Option<u32> {
        self.token_count_
    }

    #[napi(getter)]
    pub fn metadata(&self) -> HashMap<String, serde_json::Value> {
        self.metadata_.clone()
    }
}

impl Chunk {
    /// Materialize this typed chunk to a Rust core `Chunk`. `idx`
    /// fills in an auto-id (`c0`, `c1`, …) when the user didn't
    /// supply one.
    fn to_core(&self, idx: usize) -> redhop::core::Chunk {
        use redhop::core::{Chunk as CoreChunk, ChunkId, Embedding, TokenCount};
        let id = self.id_.clone().unwrap_or_else(|| format!("c{idx}"));
        let source = self.source_.clone().unwrap_or_else(|| "input".into());
        let token_count = self
            .token_count_
            .map(|n| n as usize)
            .unwrap_or_else(|| self.text_.split_whitespace().count().max(1));
        let mut chunk = CoreChunk::new(
            ChunkId::new(id),
            &self.text_,
            source,
            TokenCount(token_count),
        );
        chunk.metadata = self.metadata_.clone();
        if let Some(e) = &self.embedding_ {
            let v: Vec<f32> = e.iter().map(|x| *x as f32).collect();
            chunk = chunk.with_embedding(Embedding::from(v));
        }
        chunk
    }
}

/// Optional knobs for the low-level context functions. Every field is
/// optional; defaults match RedHop's `ContextConfig::default()`.
#[napi(object)]
#[derive(Default)]
pub struct ContextOptions {
    /// Assembly strategy: `auto` (default for `Document`) / `reasoning_preserving`
    /// (default for the low-level path) / `distractor_filtered` / `redundancy_pruned`
    /// / `max_density` / `raw_topk`.
    pub strategy: Option<String>,
    /// Token budget cap on the assembled context. Default 8192.
    pub token_budget: Option<u32>,
    /// Grounding bar below which a chunk is treated as a distractor.
    /// Default 0.10.
    pub distractor_min_grounding: Option<f64>,
    /// Jaccard floor for chunk↔chunk linkage. Default 0.12.
    pub link_min_jaccard: Option<f64>,
    /// Token-count gate for the Auto strategy's passthrough decision.
    /// Default 1500.
    pub auto_passthrough_max_tokens: Option<u32>,
    /// Cosine ceiling above which a chunk is treated as redundant. Default 0.92.
    pub redundancy_max_cosine: Option<f64>,
    /// Preserve source-document order of selected chunks instead of the
    /// strategy's relevance order. Useful for chat histories / transcripts
    /// where chronology matters. Default `false` (existing relevance-first
    /// behavior). See `docs/findings/CHAT_RAG.md` (TODO) and
    /// `crates/examples/examples/chat_rag.rs` for the worked pattern.
    pub preserve_order: Option<bool>,
}

fn chunk_to_retrieval_result(c: &Chunk, idx: usize) -> redhop::core::RetrievalResult {
    use redhop::core::{RetrievalMethod, RetrievalResult, Score, ScoreBreakdown};
    RetrievalResult {
        chunk: c.to_core(idx),
        score: Score {
            value: 1.0,
            method: RetrievalMethod::Dense,
        },
        breakdown: ScoreBreakdown::default(),
    }
}

fn build_context_config(opts: Option<ContextOptions>) -> napi::Result<redhop::ContextConfig> {
    let o = opts.unwrap_or_default();
    let base = redhop::ContextConfig::default();
    let strategy = match o.strategy {
        Some(s) => redhop::strategy_from_str(&s).map_err(err)?,
        None => base.strategy,
    };
    Ok(redhop::ContextConfig {
        token_budget: o
            .token_budget
            .map(|n| n as usize)
            .unwrap_or(base.token_budget),
        strategy,
        distractor_min_grounding: o
            .distractor_min_grounding
            .map(|n| n as f32)
            .unwrap_or(base.distractor_min_grounding),
        link_min_jaccard: o
            .link_min_jaccard
            .map(|n| n as f32)
            .unwrap_or(base.link_min_jaccard),
        auto_passthrough_max_tokens: o
            .auto_passthrough_max_tokens
            .map(|n| n as usize)
            .unwrap_or(base.auto_passthrough_max_tokens),
        redundancy_max_cosine: o
            .redundancy_max_cosine
            .map(|n| n as f32)
            .unwrap_or(base.redundancy_max_cosine),
        preserve_order: o.preserve_order.unwrap_or(base.preserve_order),
        ..base
    })
}

/// Assemble the reasoning context from caller-supplied retrieved chunks
/// (skip RedHop's chunking + retrieval and use it only for the final
/// allocation step). Returns the same `BuiltContext` shape as
/// `Document.context()`.
#[napi]
pub fn build_context(
    query: String,
    retrieved_chunks: Vec<&Chunk>,
    options: Option<ContextOptions>,
) -> napi::Result<BuiltContext> {
    let cfg = build_context_config(options)?;
    let q = redhop::core::Query::new(&query);
    let retrieved: Vec<redhop::core::RetrievalResult> = retrieved_chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| chunk_to_retrieval_result(c, i))
        .collect();
    let ctx = redhop::context::build_context(&q, &retrieved, &cfg);
    Ok(to_built(ctx))
}

/// Distractor-only filter (no budget truncation): keep everything above the
/// grounding bar, drop only the off-topic chunks. Returns a `BuiltContext`
/// whose `text` / `chunks` / `citations` reflect the filtered set.
#[napi]
pub fn filter_context(
    query: String,
    retrieved_chunks: Vec<&Chunk>,
    options: Option<ContextOptions>,
) -> napi::Result<BuiltContext> {
    let mut cfg = build_context_config(options)?;
    // filter = build with no budget truncation
    cfg.token_budget = usize::MAX;
    let q = redhop::core::Query::new(&query);
    let retrieved: Vec<redhop::core::RetrievalResult> = retrieved_chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| chunk_to_retrieval_result(c, i))
        .collect();
    let ctx = redhop::context::filter_context(&q, &retrieved, &cfg);
    Ok(to_built(ctx))
}

/// Pure diagnostics over caller-supplied chunks: what would RedHop do, and
/// why, without paying the assembly cost. Returns the same `Report` shape
/// as `Document.context().report`.
///
/// Like Python's `redhop.analyze_context`, this is a "no-budget" surface —
/// the `token_budget` option, if supplied, is ignored so the report's
/// `budget_utilization` reflects pure-analysis semantics (all chunks
/// counted) and stays consistent across bindings.
#[napi]
pub fn analyze_context(
    query: String,
    retrieved_chunks: Vec<&Chunk>,
    options: Option<ContextOptions>,
) -> napi::Result<Report> {
    let mut cfg = build_context_config(options)?;
    cfg.token_budget = usize::MAX;
    let q = redhop::core::Query::new(&query);
    let retrieved: Vec<redhop::core::RetrievalResult> = retrieved_chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| chunk_to_retrieval_result(c, i))
        .collect();
    let report = redhop::context::analyze_context(&q, &retrieved, &cfg);
    Ok(to_report(&report))
}

/// Token economics over caller-supplied chunks (evidence density, distractor
/// ratio, redundancy, estimated wasted tokens). Returns a JSON string —
/// callers `JSON.parse()` for the typed shape.
///
/// Like Python's `redhop.context_economics`, this is a "no-budget" surface
/// — the `token_budget` option is ignored so `budget_utilization` is
/// computed against an unbounded budget (essentially 0), matching the
/// "pure analysis, no filtering, no truncation" intent.
#[napi]
pub fn context_economics(
    query: String,
    retrieved_chunks: Vec<&Chunk>,
    options: Option<ContextOptions>,
) -> napi::Result<String> {
    let mut cfg = build_context_config(options)?;
    cfg.token_budget = usize::MAX;
    let q = redhop::core::Query::new(&query);
    let retrieved: Vec<redhop::core::RetrievalResult> = retrieved_chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| chunk_to_retrieval_result(c, i))
        .collect();
    let econ = redhop::context::context_economics(&q, &retrieved, &cfg);
    serde_json::to_string(&econ).map_err(err)
}
