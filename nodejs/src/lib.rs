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
    /// The human-readable Decision Report.
    pub rendered: String,
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
        rendered: r.render(None),
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
            self.inner.context_expanded(&query, budget, None, n, heading)
        }
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
/// relevance instead of drifting from it. Uses the default English analyzer;
/// non-English content should reach grounding via `Document.context().report`
/// (which carries the configured analyzer end-to-end).
#[napi]
pub fn grounding_score(query: String, text: String) -> f64 {
    redhop::context::grounding_score(&query, &text) as f64
}

/// Chunk↔chunk linkage strength: term-set Jaccard over the same normalized
/// terms — the bridge signal `reasoning_preserving` uses to decide whether
/// a low-relevance chunk is a rescuable second hop. In `[0, 1]`. Uses the
/// default English analyzer.
#[napi]
pub fn link_strength(a: String, b: String) -> f64 {
    redhop::context::link_strength(&a, &b) as f64
}

// ── Query-side diagnostics (templated workloads) ────────────────────────────
//
// Backed by `redhop::analyze_query_set` and `redhop::drop_template_terms`.
// See docs/findings/QUERY_SET_ANALYZER.md for the cross-workload probe.

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

/// Drop boilerplate tokens from a query before retrieval.
///
/// Token matching is case-insensitive on alphanumeric tokens; surviving
/// tokens are rejoined with single spaces, with punctuation preserved.
/// Mechanism: docs/findings/CUAD_RECALL_GAP.md.
///
/// ```js
/// const { dropTemplateTerms } = require("redhop");
/// const stripped = dropTemplateTerms(
///   'Highlight the parts related to "Change of Control".',
///   ["highlight", "the", "parts", "related", "to"],
/// );
/// // stripped === '"Change of Control".'
/// ```
#[napi]
pub fn drop_template_terms(query: String, boilerplate: Vec<String>) -> String {
    let bp: Vec<&str> = boilerplate.iter().map(|s| s.as_str()).collect();
    redhop::drop_template_terms(&query, &bp)
}

/// Diagnostic over a representative sample of queries — detects
/// templated-workload dilution and reports which terms are doing it.
///
/// Returns a [`QuerySetReport`]. Read `.isTemplated`, `.boilerplateTerms`,
/// `.suggestedAction`. See `docs/findings/QUERY_SET_ANALYZER.md` for the
/// cross-workload probe that validated the heuristic.
///
/// ```js
/// const { analyzeQuerySet, dropTemplateTerms } = require("redhop");
/// const r = analyzeQuerySet(myQueries);
/// if (r.isTemplated) {
///   const stripped = dropTemplateTerms(query, r.boilerplateTerms);
///   const ctx = doc.context(stripped, { strategy: "raw_topk" });
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

/// Optional gold signals for [`evaluate`]. Any combination of fields is
/// supported — pass `goldChunks` to unlock `contextRecall` /
/// `contextPrecision`; pass `goldAnswer` to unlock `answerTokenRecall`;
/// pass both for all three. Omit both for self-eval only.
#[napi(object)]
pub struct EvaluateOptions {
    /// IDs of chunks that should appear in the assembled context.
    pub gold_chunks: Option<Vec<String>>,
    /// Ground-truth answer text.
    pub gold_answer: Option<String>,
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
        gold_chunks: None,
        gold_answer: None,
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
    let r = redhop::evaluate(&q, &context.inner, gold);
    EvalReport {
        context_recall: r.context_recall.map(|v| v as f64),
        context_precision: r.context_precision.map(|v| v as f64),
        answer_token_recall: r.answer_token_recall.map(|v| v as f64),
        mean_grounding: r.mean_grounding as f64,
        evidence_density: r.evidence_density as f64,
        retained_evidence_ratio: r.retained_evidence_ratio as f64,
        second_hop_rescues: r.second_hop_rescues as u32,
        low_confidence: r.low_confidence,
        estimated_waste_tokens: r.estimated_waste_tokens as u32,
        overall: r.overall as f64,
    }
}

// ── Low-level context functions (caller brings their own chunks) ────────────
//
// `Document.fromChunks(...)` is the high-level path: hand RedHop a list of
// chunk strings and let it own the index. The functions below are for users
// who do their own retrieval (vector DB, BM25 outside RedHop, hybrid stacks)
// and want RedHop just for the final assembly + diagnostics step.

/// A single retrieved chunk for the low-level `buildContext` / `filterContext`
/// / `analyzeContext` / `contextEconomics` functions.
#[napi(object)]
pub struct ChunkInput {
    /// The chunk text. Required.
    pub text: String,
    /// Stable identifier. Defaults to `c<index>`.
    pub id: Option<String>,
    /// Source path / label. Defaults to `"input"`.
    pub source: Option<String>,
    /// Token count (defaults to whitespace word count).
    pub token_count: Option<u32>,
    /// Optional dense vector (for embedding-based scoring downstream).
    pub embedding: Option<Vec<f64>>,
    /// Retrieval score from the upstream retriever. Defaults to `1.0`.
    pub score: Option<f64>,
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
}

fn build_chunk_input(c: ChunkInput, idx: usize) -> redhop::core::RetrievalResult {
    use redhop::core::{
        Chunk, ChunkId, Embedding, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
        TokenCount,
    };
    let id = c.id.unwrap_or_else(|| format!("c{idx}"));
    let source = c.source.unwrap_or_else(|| "input".to_string());
    let token_count = c
        .token_count
        .map(|n| n as usize)
        .unwrap_or_else(|| c.text.split_whitespace().count().max(1));
    let mut chunk = Chunk::new(ChunkId::new(id), &c.text, source, TokenCount(token_count));
    if let Some(e) = c.embedding {
        let v: Vec<f32> = e.into_iter().map(|x| x as f32).collect();
        chunk = chunk.with_embedding(Embedding::from(v));
    }
    RetrievalResult {
        chunk,
        score: Score {
            value: c.score.unwrap_or(1.0) as f32,
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
        token_budget: o.token_budget.map(|n| n as usize).unwrap_or(base.token_budget),
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
    retrieved_chunks: Vec<ChunkInput>,
    options: Option<ContextOptions>,
) -> napi::Result<BuiltContext> {
    let cfg = build_context_config(options)?;
    let q = redhop::core::Query::new(&query);
    let retrieved: Vec<redhop::core::RetrievalResult> = retrieved_chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| build_chunk_input(c, i))
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
    retrieved_chunks: Vec<ChunkInput>,
    options: Option<ContextOptions>,
) -> napi::Result<BuiltContext> {
    let mut cfg = build_context_config(options)?;
    // filter = build with no budget truncation
    cfg.token_budget = usize::MAX;
    let q = redhop::core::Query::new(&query);
    let retrieved: Vec<redhop::core::RetrievalResult> = retrieved_chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| build_chunk_input(c, i))
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
    retrieved_chunks: Vec<ChunkInput>,
    options: Option<ContextOptions>,
) -> napi::Result<Report> {
    let mut cfg = build_context_config(options)?;
    cfg.token_budget = usize::MAX;
    let q = redhop::core::Query::new(&query);
    let retrieved: Vec<redhop::core::RetrievalResult> = retrieved_chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| build_chunk_input(c, i))
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
    retrieved_chunks: Vec<ChunkInput>,
    options: Option<ContextOptions>,
) -> napi::Result<String> {
    let mut cfg = build_context_config(options)?;
    cfg.token_budget = usize::MAX;
    let q = redhop::core::Query::new(&query);
    let retrieved: Vec<redhop::core::RetrievalResult> = retrieved_chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| build_chunk_input(c, i))
        .collect();
    let econ = redhop::context::context_economics(&q, &retrieved, &cfg);
    serde_json::to_string(&econ).map_err(err)
}
