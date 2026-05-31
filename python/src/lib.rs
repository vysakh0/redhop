//! pyo3 bindings for RedHop context optimization.
//!
//! Thin wrapper over the stable `redhop-context` public API — no logic is
//! duplicated here. Rust remains the source of truth; this module only maps
//! Pythonic inputs (dicts/lists/strings) to the Rust types and wraps the
//! results in small Python classes.

use std::path::{Path, PathBuf};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use redhop::context::{
    analyze_context as rh_analyze, build_context as rh_build, context_economics as rh_economics,
    filter_context as rh_filter, grounding_score as rh_grounding, link_strength as rh_link,
    AutoDecision, ContextConfig, ContextReport as RhReport, ContextStrategy,
};
use redhop::core::{
    Chunk, ChunkId, Embedding, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenCount,
};
use redhop::document::{
    Document as RhDocument, DocumentConfig, RetrievalMode, Section as RhSection,
};

fn strategy_from_str(s: &str) -> PyResult<ContextStrategy> {
    Ok(match s {
        "raw_topk" => ContextStrategy::RawTopK,
        "distractor_filtered" => ContextStrategy::DistractorFiltered,
        "redundancy_pruned" => ContextStrategy::RedundancyPruned,
        "max_density" => ContextStrategy::MaxDensity,
        "reasoning_preserving" => ContextStrategy::ReasoningPreserving,
        "auto" => ContextStrategy::Auto,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown strategy '{other}' (expected: raw_topk, distractor_filtered, \
                 redundancy_pruned, max_density, reasoning_preserving, auto)"
            )))
        }
    })
}

fn strategy_to_str(s: ContextStrategy) -> &'static str {
    match s {
        ContextStrategy::RawTopK => "raw_topk",
        ContextStrategy::DistractorFiltered => "distractor_filtered",
        ContextStrategy::RedundancyPruned => "redundancy_pruned",
        ContextStrategy::MaxDensity => "max_density",
        ContextStrategy::ReasoningPreserving => "reasoning_preserving",
        ContextStrategy::Auto => "auto",
    }
}

/// Convert one Python chunk (a str, or a dict with at least `text`) into a
/// `RetrievalResult`.
fn chunk_from_py(item: &Bound<'_, PyAny>, idx: usize) -> PyResult<RetrievalResult> {
    let (id, text, source, token_count, embedding, score) = if let Ok(s) = item.extract::<String>()
    {
        (format!("c{idx}"), s, None, None, None, None)
    } else if let Ok(d) = item.downcast::<PyDict>() {
        let text: String = d
            .get_item("text")?
            .ok_or_else(|| PyValueError::new_err(format!("chunk {idx} missing 'text'")))?
            .extract()?;
        let id: String = match d.get_item("id")? {
            Some(v) => v.extract()?,
            None => format!("c{idx}"),
        };
        let source: Option<String> = match d.get_item("source")? {
            Some(v) => Some(v.extract()?),
            None => None,
        };
        let token_count: Option<usize> = match d.get_item("token_count")? {
            Some(v) => Some(v.extract()?),
            None => None,
        };
        let embedding: Option<Vec<f32>> = match d.get_item("embedding")? {
            Some(v) => Some(v.extract()?),
            None => None,
        };
        let score: Option<f32> = match d.get_item("score")? {
            Some(v) => Some(v.extract()?),
            None => None,
        };
        (id, text, source, token_count, embedding, score)
    } else {
        return Err(PyValueError::new_err(format!(
            "chunk {idx} must be a string or a dict with a 'text' field"
        )));
    };

    let tok = token_count.unwrap_or_else(|| text.split_whitespace().count().max(1));
    let mut chunk = Chunk::new(
        ChunkId::new(id),
        text,
        source.unwrap_or_else(|| "input".into()),
        TokenCount(tok),
    );
    if let Some(e) = embedding {
        chunk = chunk.with_embedding(Embedding::from(e));
    }
    Ok(RetrievalResult {
        chunk,
        score: Score {
            value: score.unwrap_or(1.0),
            method: RetrievalMethod::Dense,
        },
        breakdown: ScoreBreakdown::default(),
    })
}

fn chunks_from_py(chunks: &Bound<'_, PyAny>) -> PyResult<Vec<RetrievalResult>> {
    let list = chunks.try_iter()?;
    let mut out = Vec::new();
    for (i, item) in list.enumerate() {
        out.push(chunk_from_py(&item?, i)?);
    }
    Ok(out)
}

fn config(
    strategy: Option<String>,
    token_budget: usize,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
    auto_passthrough_max_tokens: usize,
    redundancy_max_cosine: f32,
) -> PyResult<ContextConfig> {
    let strat = match strategy {
        Some(s) => strategy_from_str(&s)?,
        None => ContextStrategy::ReasoningPreserving,
    };
    Ok(ContextConfig {
        token_budget,
        strategy: strat,
        distractor_min_grounding,
        link_min_jaccard,
        auto_passthrough_max_tokens,
        redundancy_max_cosine,
        // Inherits the Rust-side default (0.10). Not exposed as a Python kwarg
        // yet — the signal it drives is observable on the report regardless.
        ..ContextConfig::default()
    })
}

/// Observability trace for one context assembly. `str(report)` is the
/// human-readable Context Optimization Report.
#[pyclass]
#[derive(Clone)]
struct ContextReport {
    inner: RhReport,
    rendered: String,
}

#[pymethods]
impl ContextReport {
    #[getter]
    fn strategy(&self) -> &'static str {
        strategy_to_str(self.inner.strategy)
    }
    /// What the caller requested (may be "auto"). Differs from `strategy` when
    /// the Auto policy resolved to a concrete action.
    #[getter]
    fn requested_strategy(&self) -> &'static str {
        strategy_to_str(self.inner.requested_strategy)
    }
    /// The runtime's Auto decision: "passthrough", "prune", or "not_auto".
    #[getter]
    fn auto_decision(&self) -> &'static str {
        match self.inner.auto_decision() {
            AutoDecision::NotAuto => "not_auto",
            AutoDecision::Passthrough => "passthrough",
            AutoDecision::Prune => "prune",
        }
    }
    /// Total tokens in the input (retrieved) set, before assembly.
    #[getter]
    fn input_tokens(&self) -> usize {
        self.inner.input_tokens
    }
    #[getter]
    fn token_budget(&self) -> usize {
        self.inner.token_budget
    }
    #[getter]
    fn total_tokens(&self) -> usize {
        self.inner.total_tokens
    }
    #[getter]
    fn token_utilization(&self) -> f32 {
        self.inner.token_utilization
    }
    #[getter]
    fn n_input_chunks(&self) -> usize {
        self.inner.n_input_chunks
    }
    #[getter]
    fn n_selected(&self) -> usize {
        self.inner.n_selected
    }
    #[getter]
    fn input_distractor_ratio(&self) -> f32 {
        self.inner.input_distractor_ratio
    }
    #[getter]
    fn retained_evidence_ratio(&self) -> f32 {
        self.inner.retained_evidence_ratio
    }
    #[getter]
    fn second_hop_rescue_count(&self) -> usize {
        self.inner.second_hop_rescue_count
    }
    #[getter]
    fn reasoning_preservation_delta(&self) -> usize {
        self.inner.reasoning_preservation_delta
    }
    #[getter]
    fn n_expanded(&self) -> usize {
        self.inner.n_expanded
    }
    /// `True` when nothing in the assembled context was above the grounding
    /// floor — the query may share little vocabulary with the corpus.
    #[getter]
    fn low_confidence_retrieval(&self) -> bool {
        self.inner.low_confidence_retrieval
    }
    /// The grounding ceiling that `low_confidence_retrieval` applied.
    #[getter]
    fn low_confidence_threshold(&self) -> f32 {
        self.inner.low_confidence_threshold
    }
    #[getter]
    fn distractors_pruned(&self) -> usize {
        self.inner.removed.distractor
    }
    #[getter]
    fn removed_total(&self) -> usize {
        self.inner.removed.total
    }
    #[getter]
    fn evidence_density(&self) -> f32 {
        self.inner.economics.evidence_density
    }
    #[getter]
    fn distractor_ratio(&self) -> f32 {
        self.inner.economics.distractor_ratio
    }
    #[getter]
    fn estimated_waste_tokens(&self) -> usize {
        self.inner.economics.estimated_waste_tokens
    }
    /// The full report as a JSON string (Python wrapper exposes `.to_dict()`).
    fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("serialize report: {e}")))
    }
    fn __str__(&self) -> String {
        self.rendered.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "ContextReport(strategy={}, {}→{} chunks, {} tokens, rescues={})",
            self.strategy(),
            self.inner.n_input_chunks,
            self.inner.n_selected,
            self.inner.total_tokens,
            self.inner.second_hop_rescue_count
        )
    }
}

/// The assembled context: `.text()` is the prompt string; `.report` is the
/// telemetry; `.chunks` are the selected chunk texts in order.
/// One selected chunk's provenance, for citations.
struct CiteData {
    text: String,
    source: String,
    page: Option<u64>,
    heading: Option<String>,
    line: Option<u64>,
}

/// Provenance for each selected chunk (source + page/heading/line metadata).
fn cites_of(chunks: &[Chunk]) -> Vec<CiteData> {
    chunks
        .iter()
        .map(|c| CiteData {
            text: c.text.clone(),
            source: c.source.clone(),
            page: c.metadata.get("page").and_then(|v| v.as_u64()),
            heading: c
                .metadata
                .get("heading")
                .and_then(|v| v.as_str().map(String::from)),
            line: c.metadata.get("line").and_then(|v| v.as_u64()),
        })
        .collect()
}

#[pyclass]
struct BuiltContext {
    text: String,
    chunks: Vec<String>,
    cites: Vec<CiteData>,
    report: ContextReport,
}

#[pymethods]
impl BuiltContext {
    /// The assembled context as a single prompt string (drop-in for `llm.generate`).
    fn text(&self) -> &str {
        &self.text
    }
    #[getter]
    fn chunks(&self) -> Vec<String> {
        self.chunks.clone()
    }
    /// Provenance of each selected chunk, in order — a list of dicts with
    /// `source`, `page` (or None), `heading` (or None), `line` (or None), and
    /// `text`. Use these to cite where the answer's context came from
    /// (e.g. "contract.pdf, p.3" or "notes.md → Setup" or "main.py:42").
    #[getter]
    fn citations<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.cites
            .iter()
            .map(|c| {
                let d = PyDict::new(py);
                d.set_item("source", &c.source)?;
                d.set_item("page", c.page)?;
                d.set_item("heading", &c.heading)?;
                d.set_item("line", c.line)?;
                d.set_item("text", &c.text)?;
                Ok(d)
            })
            .collect()
    }
    #[getter]
    fn report(&self) -> ContextReport {
        self.report.clone()
    }
    fn __repr__(&self) -> String {
        self.report.__repr__()
    }
}

#[pyfunction]
#[pyo3(signature = (query, retrieved_chunks, strategy=None, token_budget=8192,
       distractor_min_grounding=0.10, link_min_jaccard=0.12, auto_passthrough_max_tokens=1500, redundancy_max_cosine=0.92))]
#[allow(clippy::too_many_arguments)]
fn build_context(
    query: &str,
    retrieved_chunks: &Bound<'_, PyAny>,
    strategy: Option<String>,
    token_budget: usize,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
    auto_passthrough_max_tokens: usize,
    redundancy_max_cosine: f32,
) -> PyResult<BuiltContext> {
    let cfg = config(
        strategy,
        token_budget,
        distractor_min_grounding,
        link_min_jaccard,
        auto_passthrough_max_tokens,
        redundancy_max_cosine,
    )?;
    let q = Query::new(query);
    let retrieved = chunks_from_py(retrieved_chunks)?;
    let before = rh_analyze(&q, &retrieved, &cfg);
    let ctx = rh_build(&q, &retrieved, &cfg);
    let rendered = ctx.report.render(Some(&before));
    Ok(BuiltContext {
        text: ctx.text(),
        chunks: ctx.chunks.iter().map(|c| c.text.clone()).collect(),
        cites: cites_of(&ctx.chunks),
        report: ContextReport {
            inner: ctx.report,
            rendered,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (query, retrieved_chunks, strategy=None,
       distractor_min_grounding=0.10, link_min_jaccard=0.12, auto_passthrough_max_tokens=1500, redundancy_max_cosine=0.92))]
#[allow(clippy::too_many_arguments)]
fn filter_context(
    query: &str,
    retrieved_chunks: &Bound<'_, PyAny>,
    strategy: Option<String>,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
    auto_passthrough_max_tokens: usize,
    redundancy_max_cosine: f32,
) -> PyResult<BuiltContext> {
    // filter = build with no budget truncation.
    let cfg = config(
        strategy,
        usize::MAX,
        distractor_min_grounding,
        link_min_jaccard,
        auto_passthrough_max_tokens,
        redundancy_max_cosine,
    )?;
    let q = Query::new(query);
    let retrieved = chunks_from_py(retrieved_chunks)?;
    let before = rh_analyze(&q, &retrieved, &cfg);
    let ctx = rh_filter(&q, &retrieved, &cfg);
    let rendered = ctx.report.render(Some(&before));
    Ok(BuiltContext {
        text: ctx.text(),
        chunks: ctx.chunks.iter().map(|c| c.text.clone()).collect(),
        cites: cites_of(&ctx.chunks),
        report: ContextReport {
            inner: ctx.report,
            rendered,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (query, retrieved_chunks, strategy=None, distractor_min_grounding=0.10,
       link_min_jaccard=0.12, auto_passthrough_max_tokens=1500))]
fn analyze_context(
    query: &str,
    retrieved_chunks: &Bound<'_, PyAny>,
    strategy: Option<String>,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
    auto_passthrough_max_tokens: usize,
) -> PyResult<ContextReport> {
    // strategy is recorded on the report; with "auto" the reported strategy is
    // the gate's decision for this input (passthrough vs prune) — pure diagnostics.
    let cfg = config(
        strategy,
        usize::MAX,
        distractor_min_grounding,
        link_min_jaccard,
        auto_passthrough_max_tokens,
        0.92,
    )?;
    let q = Query::new(query);
    let retrieved = chunks_from_py(retrieved_chunks)?;
    let report = rh_analyze(&q, &retrieved, &cfg);
    let rendered = report.render(None);
    Ok(ContextReport {
        inner: report,
        rendered,
    })
}

#[pyfunction]
#[pyo3(signature = (query, retrieved_chunks, distractor_min_grounding=0.10, link_min_jaccard=0.12))]
fn context_economics(
    query: &str,
    retrieved_chunks: &Bound<'_, PyAny>,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
) -> PyResult<String> {
    let cfg = config(
        None,
        usize::MAX,
        distractor_min_grounding,
        link_min_jaccard,
        8_000,
        0.92,
    )?;
    let q = Query::new(query);
    let retrieved = chunks_from_py(retrieved_chunks)?;
    let econ = rh_economics(&q, &retrieved, &cfg);
    serde_json::to_string(&econ).map_err(|e| PyValueError::new_err(format!("serialize: {e}")))
}

/// Query grounding of a chunk's text in [0,1] — the relevance signal the
/// strategies use (stopword-removed, stemmed query-term overlap). Observability.
#[pyfunction]
fn grounding_score(query: &str, text: &str) -> f32 {
    rh_grounding(query, text)
}

/// Linkage strength between two chunks' text in [0,1] — the bridge signal
/// ReasoningPreserving uses to rescue a second hop.
#[pyfunction]
fn link_strength(a: &str, b: &str) -> f32 {
    rh_link(a, b)
}

/// Build the Python `BuiltContext` wrapper from a Rust one.
fn py_built(ctx: redhop::context::BuiltContext) -> BuiltContext {
    let text = ctx.text();
    let chunks = ctx.chunks.iter().map(|c| c.text.clone()).collect();
    let cites = cites_of(&ctx.chunks);
    let rendered = ctx.report.render(None);
    BuiltContext {
        text,
        chunks,
        cites,
        report: ContextReport {
            inner: ctx.report,
            rendered,
        },
    }
}

fn to_py<T>(r: redhop::core::Result<T>) -> PyResult<T> {
    r.map_err(|e| PyValueError::new_err(e.to_string()))
}

fn retrieval_from_str(retrieval: Option<&str>, candidate_pool: usize) -> PyResult<RetrievalMode> {
    Ok(match retrieval {
        None | Some("lexical") => RetrievalMode::Lexical,
        // BM25 prune → dense rerank the pool: scales to large corpora, no vector DB.
        Some("hybrid") => RetrievalMode::Hybrid {
            candidate_pool: candidate_pool.max(1),
        },
        // Global dense over every chunk: best recall on small/bounded corpora.
        Some("semantic") => RetrievalMode::Dense,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown retrieval mode '{other}'; use 'lexical' (default, BM25), 'hybrid' \
                 (BM25 prune → dense rerank — large corpora, no vector DB), or 'semantic' \
                 (global dense over every chunk — small/bounded corpora)"
            )))
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn doc_config(
    strategy: Option<String>,
    token_budget: usize,
    candidate_k: usize,
    chunk_size: usize,
    chunk_overlap: usize,
    retrieval_mode: RetrievalMode,
) -> PyResult<DocumentConfig> {
    let base = DocumentConfig::default();
    let strategy = match strategy {
        Some(s) => strategy_from_str(&s)?,
        None => base.context.strategy,
    };
    let context = ContextConfig {
        token_budget,
        strategy,
        ..base.context
    };
    Ok(DocumentConfig {
        // chunk_size is the target tokens/chunk; cap (max) at 2x so an
        // occasional long sentence isn't split mid-thought.
        target_tokens: chunk_size,
        max_tokens: chunk_size * 2,
        overlap_sentences: chunk_overlap,
        candidate_k,
        retrieval_mode,
        rerank_pool: base.rerank_pool,
        context,
        // Inherits the Rust-side default (0 = off). Surface this as a Python
        // kwarg once a real user asks; for now the issue-#1 fix in Phase 1
        // restored the documented hybrid contract on its own.
        min_candidates: base.min_candidates,
        // Inherits the Rust-side default (1 = on for code chunks). The
        // auto-expansion only fires on code-classified chunks, so a
        // Python user on a text/prose corpus sees no change.
        code_neighbors_default: base.code_neighbors_default,
        // Inherits the Rust-side default (true). Fires only on chunks that
        // carry a section heading — attaches the section's opening chunk
        // for prose-with-headings contexts.
        prose_heading_default: base.prose_heading_default,
    })
}

/// Attach a dense embedder for `retrieval="semantic"`/`"hybrid"`. The embedding
/// engine + model live behind the crate's `semantic` feature; the lean (lexical)
/// wheel raises a clear error rather than silently degrading.
#[cfg(feature = "semantic")]
#[allow(clippy::too_many_arguments)]
fn apply_dense_embedder(
    doc: RhDocument,
    model: Option<String>,
    embedder_model: Option<String>,
    embedder_tokenizer: Option<String>,
    embedder_dim: usize,
    embedder_pooling: Option<String>,
    embedder_query_prefix: Option<String>,
    embedder_passage_prefix: Option<String>,
) -> PyResult<RhDocument> {
    // Path A — explicit model files (custom / offline / power user). These win
    // if given; `model=` is ignored.
    if let (Some(m), Some(t)) = (embedder_model.as_ref(), embedder_tokenizer.as_ref()) {
        let pooling = match embedder_pooling.as_deref() {
            None | Some("cls") => redhop::embeddings::Pooling::Cls,
            Some("mean") => redhop::embeddings::Pooling::Mean,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "unknown embedder_pooling '{other}'; use 'cls' (default, e.g. BGE) or 'mean' \
                     (e.g. MiniLM / GTE / E5)"
                )))
            }
        };
        let load = |prefix: &str| -> PyResult<redhop::embeddings::OnnxEmbedder> {
            let mut config = redhop::embeddings::EmbedderConfig::bge(embedder_dim);
            config.pooling = pooling;
            config.prefix = prefix.to_string();
            redhop::embeddings::OnnxEmbedder::load(m, t, config)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        };
        return match (embedder_query_prefix, embedder_passage_prefix) {
            (q, p) if q.is_some() || p.is_some() => {
                let passage = load(p.as_deref().unwrap_or(""))?;
                let query = load(q.as_deref().unwrap_or(""))?;
                Ok(doc
                    .with_embedder(std::sync::Arc::new(passage))
                    .with_query_embedder(std::sync::Arc::new(query)))
            }
            _ => Ok(doc.with_embedder(std::sync::Arc::new(load("")?))),
        };
    }

    // Path B — model-by-name, auto-downloaded from HuggingFace (cached). With
    // neither `model` nor explicit paths, fall back to the recommended default.
    let name = model.as_deref().unwrap_or(redhop::embeddings::DEFAULT_MODEL);
    let resolved =
        redhop::embeddings::resolve_model(name).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let load = |prefix: &str| -> PyResult<redhop::embeddings::OnnxEmbedder> {
        redhop::embeddings::OnnxEmbedder::load(
            &resolved.model_path,
            &resolved.tokenizer_path,
            resolved.config(prefix),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    };
    if resolved.is_asymmetric() {
        let passage = load(&resolved.passage_prefix)?;
        let query = load(&resolved.query_prefix)?;
        Ok(doc
            .with_embedder(std::sync::Arc::new(passage))
            .with_query_embedder(std::sync::Arc::new(query)))
    } else {
        // passage_prefix == query_prefix (usually ""): one embedder for both sides.
        Ok(doc.with_embedder(std::sync::Arc::new(load(&resolved.passage_prefix)?)))
    }
}

#[cfg(not(feature = "semantic"))]
#[allow(clippy::too_many_arguments)]
fn apply_dense_embedder(
    _doc: RhDocument,
    _model: Option<String>,
    _embedder_model: Option<String>,
    _embedder_tokenizer: Option<String>,
    _embedder_dim: usize,
    _embedder_pooling: Option<String>,
    _embedder_query_prefix: Option<String>,
    _embedder_passage_prefix: Option<String>,
) -> PyResult<RhDocument> {
    Err(PyValueError::new_err(
        "retrieval='semantic'/'hybrid' needs the semantic tier, which this build was compiled \
         without. The standard `pip install redhop` includes it — reinstall that; if you built \
         from source, add `--features semantic`.",
    ))
}

/// Attach a second-stage cross-encoder reranker named by `rerank` (no-op when
/// `None`). `"cross-encoder"` auto-downloads the MS-MARCO MiniLM model.
#[cfg(feature = "semantic")]
fn apply_reranker(doc: RhDocument, rerank: Option<String>) -> PyResult<RhDocument> {
    let Some(name) = rerank else { return Ok(doc) };
    let r = redhop::embeddings::resolve_reranker(&name)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ce =
        redhop::reranking::OnnxCrossEncoder::load(&r.model_path, &r.tokenizer_path, r.max_seq_len)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(doc.with_reranker(std::sync::Arc::new(ce)))
}

#[cfg(not(feature = "semantic"))]
fn apply_reranker(doc: RhDocument, rerank: Option<String>) -> PyResult<RhDocument> {
    match rerank {
        None => Ok(doc),
        Some(_) => Err(PyValueError::new_err(
            "rerank='cross-encoder' needs the semantic tier, which this build was compiled \
             without. The standard `pip install redhop` includes it.",
        )),
    }
}

/// Read a file to `(source, sections)` for `from_file`.
///
/// With the `files` feature, `redhop-files` parses text/markdown/code + DOCX/PPTX/
/// XLSX/PDF. Without it, only UTF-8 text files are read; binary formats return a
/// clear, helpful error.
#[cfg(feature = "files")]
fn extract_file_text(path: &str) -> PyResult<(String, Vec<RhSection>)> {
    let doc = redhop::files::extract(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((doc.source, to_rh_sections(doc.sections)))
}

/// Map `redhop-files` sections to the binding's `RhSection`.
#[cfg(feature = "files")]
fn to_rh_sections(sections: Vec<redhop::files::Section>) -> Vec<RhSection> {
    sections
        .into_iter()
        .map(|s| RhSection {
            text: s.text,
            page: s.page,
            heading: s.heading,
            line: s.line,
        })
        .collect()
}

/// Parse already-in-memory `bytes` to `(source, sections)` for `from_bytes`.
/// `name` (e.g. `"contract.pdf"`) selects the parser by extension and becomes the
/// citation source.
#[cfg(feature = "files")]
fn extract_bytes_sections(data: &[u8], name: &str) -> PyResult<(String, Vec<RhSection>)> {
    let doc = redhop::files::extract_bytes(data, name)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((doc.source, to_rh_sections(doc.sections)))
}

#[cfg(not(feature = "files"))]
fn extract_bytes_sections(data: &[u8], name: &str) -> PyResult<(String, Vec<RhSection>)> {
    let lower = name.to_lowercase();
    const NEEDS_PARSER: &[&str] = &[
        ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".odt", ".odp", ".ods", ".rtf",
    ];
    if let Some(ext) = NEEDS_PARSER.iter().find(|e| lower.ends_with(**e)) {
        return Err(PyValueError::new_err(format!(
            "from_bytes can't parse {} — this build was compiled without the document parsers. \
             The standard `pip install redhop` includes them. Or decode the text yourself and \
             use from_text().",
            ext.trim_start_matches('.')
        )));
    }
    let text = String::from_utf8_lossy(data).into_owned();
    Ok((
        name.to_string(),
        vec![RhSection {
            text,
            page: None,
            heading: None,
            line: None,
        }],
    ))
}

#[cfg(not(feature = "files"))]
fn extract_file_text(path: &str) -> PyResult<(String, Vec<RhSection>)> {
    let lower = path.to_lowercase();
    const NEEDS_PARSER: &[&str] = &[
        ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".odt", ".odp", ".ods", ".rtf",
    ];
    if let Some(ext) = NEEDS_PARSER.iter().find(|e| lower.ends_with(**e)) {
        return Err(PyValueError::new_err(format!(
            "from_file can't parse {} — this build was compiled without the document parsers. \
             The standard `pip install redhop` includes them (reinstall that); from source, add \
             `--features files`. Or extract the text yourself and use from_text().",
            ext.trim_start_matches('.')
        )));
    }
    let text = std::fs::read_to_string(path).map_err(|e| {
        PyValueError::new_err(format!(
            "could not read '{path}' as text ({e}). This build reads UTF-8 text files; \
             PDF/DOCX/PPTX/XLSX parsing is in the standard `pip install redhop`."
        ))
    })?;
    Ok((
        path.to_string(),
        vec![RhSection {
            text,
            page: None,
            heading: None,
            line: None,
        }],
    ))
}

/// Build/cache dirs excluded by default even without a `.gitignore`.
const DEFAULT_IGNORES: &[&str] = &[
    "node_modules",
    "target",
    "__pycache__",
    "venv",
    "dist",
    "build",
];

/// Collect the files to index under `root`, using the same walker ripgrep does:
/// skips hidden entries, honors `.gitignore`/`.ignore` (when `gitignore`), excludes
/// build/cache dirs and any caller `ignore` globs (gitignore-style). Sorted for
/// deterministic chunk ids. `recursive=false` stays in the top level.
fn collect_files(
    root: &Path,
    recursive: bool,
    gitignore: bool,
    ignore_globs: &[String],
) -> PyResult<Vec<PathBuf>> {
    // Override globs: a `!glob` excludes. All-exclude → everything else included.
    let mut ob = ignore::overrides::OverrideBuilder::new(root);
    for d in DEFAULT_IGNORES {
        let _ = ob.add(&format!("!{d}"));
    }
    for g in ignore_globs {
        ob.add(&format!("!{g}")).map_err(|e| {
            PyValueError::new_err(format!("from_folder: invalid ignore pattern '{g}': {e}"))
        })?;
    }
    let overrides = ob
        .build()
        .map_err(|e| PyValueError::new_err(format!("from_folder: ignore patterns: {e}")))?;

    let mut wb = ignore::WalkBuilder::new(root);
    wb.hidden(true) // skip dotfiles (.git, .venv, .redhop)
        .git_ignore(gitignore)
        .git_global(gitignore)
        .git_exclude(gitignore)
        .ignore(gitignore)
        .parents(gitignore)
        // Honor a .gitignore even when the folder isn't itself a git repo.
        .require_git(false)
        .overrides(overrides);
    if !recursive {
        wb.max_depth(Some(1));
    }

    let mut out = Vec::new();
    for entry in wb.build() {
        // Skip unreadable entries rather than failing the whole walk.
        let Ok(entry) = entry else { continue };
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            out.push(entry.into_path());
        }
    }
    out.sort();
    Ok(out)
}

// ---- Persistent folder index (opt-in via `from_folder(persist=True)`) --------

/// On-disk format version. Bump when the layout changes incompatibly.
const INDEX_VERSION: u32 = 1;
/// The index file written inside the index directory.
const INDEX_FILE: &str = "index.json";

/// One file's cached contribution to a folder index: its chunks (with
/// embeddings) plus the stat used to detect changes on the next run.
#[derive(serde::Serialize, serde::Deserialize)]
struct CachedFile {
    source: String,
    mtime: u64,
    size: u64,
    chunks: Vec<Chunk>,
}

/// The persisted folder index: a fingerprint of the indexing config (so a cache
/// built with different settings is ignored) plus the per-file caches.
#[derive(serde::Serialize, serde::Deserialize)]
struct PersistedIndex {
    version: u32,
    fingerprint: String,
    files: Vec<CachedFile>,
}

/// `(mtime_nanos, size)` for change detection, or `None` if the file vanished.
fn file_stat(path: &Path) -> Option<(u64, u64)> {
    let m = std::fs::metadata(path).ok()?;
    let mtime = m
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos() as u64;
    Some((mtime, m.len()))
}

/// A string identifying the indexing settings. A cache only counts as valid if
/// its fingerprint matches — change the chunking/retrieval/model and we rebuild.
#[allow(clippy::too_many_arguments)]
fn index_fingerprint(
    chunk_size: usize,
    chunk_overlap: usize,
    retrieval: &Option<String>,
    model: &Option<String>,
    embedder_model: &Option<String>,
    embedder_dim: usize,
) -> String {
    let s = |o: &Option<String>| o.clone().unwrap_or_default();
    format!(
        "v{INDEX_VERSION}|cs={chunk_size}|co={chunk_overlap}|r={}|m={}|em={}|ed={embedder_dim}",
        s(retrieval),
        s(model),
        s(embedder_model),
    )
}

/// Resolve where the index lives: explicit `index_dir`, else `<folder>/.redhop`.
fn resolve_index_dir(folder: &Path, index_dir: &Option<String>) -> PathBuf {
    match index_dir {
        Some(d) => PathBuf::from(d),
        None => folder.join(".redhop"),
    }
}

/// Load a persisted index if present and its fingerprint matches; otherwise None
/// (missing, unreadable, wrong version, or stale settings → full rebuild).
fn load_index(dir: &Path, fingerprint: &str) -> Option<PersistedIndex> {
    let raw = std::fs::read_to_string(dir.join(INDEX_FILE)).ok()?;
    let idx: PersistedIndex = serde_json::from_str(&raw).ok()?;
    if idx.version == INDEX_VERSION && idx.fingerprint == fingerprint {
        Some(idx)
    } else {
        None
    }
}

/// Build a document from chunks you already have (cached + freshly chunked),
/// applying the embedder when the tier needs one. Mirrors `build_text_doc` but
/// skips chunking (the chunks are ready).
#[allow(clippy::too_many_arguments)]
fn build_chunks_doc(
    chunks: Vec<Chunk>,
    strategy: Option<String>,
    chunk_size: usize,
    chunk_overlap: usize,
    token_budget: usize,
    candidate_k: usize,
    retrieval: Option<String>,
    model: Option<String>,
    embedder_model: Option<String>,
    embedder_tokenizer: Option<String>,
    embedder_dim: usize,
    embedder_pooling: Option<String>,
    embedder_query_prefix: Option<String>,
    embedder_passage_prefix: Option<String>,
    candidate_pool: usize,
    rerank: Option<String>,
) -> PyResult<RhDocument> {
    let mode = retrieval_from_str(retrieval.as_deref(), candidate_pool)?;
    let needs_embedder = matches!(mode, RetrievalMode::Hybrid { .. } | RetrievalMode::Dense);
    let cfg = doc_config(
        strategy,
        token_budget,
        candidate_k,
        chunk_size,
        chunk_overlap,
        mode,
    )?;
    let mut inner = to_py(RhDocument::from_chunks_with(chunks, cfg))?;
    if needs_embedder {
        inner = apply_dense_embedder(
            inner,
            model,
            embedder_model,
            embedder_tokenizer,
            embedder_dim,
            embedder_pooling,
            embedder_query_prefix,
            embedder_passage_prefix,
        )?;
    }
    apply_reranker(inner, rerank)
}

/// Build a folder index with on-disk persistence + incremental re-index.
/// Reuses cached chunks (with embeddings) for files whose mtime+size are
/// unchanged, re-chunks new/changed files, drops removed ones, and rewrites the
/// index only when something changed.
#[allow(clippy::too_many_arguments)]
/// `(document, skipped (path, reason), files indexed)` — the result of building a
/// folder index.
type FolderBuild = (RhDocument, Vec<(String, String)>, usize);

#[allow(clippy::too_many_arguments)]
fn build_folder_persisted(
    folder: &Path,
    recursive: bool,
    gitignore: bool,
    ignore_globs: &[String],
    index_dir: &Option<String>,
    strategy: Option<String>,
    chunk_size: usize,
    chunk_overlap: usize,
    token_budget: usize,
    candidate_k: usize,
    retrieval: Option<String>,
    model: Option<String>,
    embedder_model: Option<String>,
    embedder_tokenizer: Option<String>,
    embedder_dim: usize,
    embedder_pooling: Option<String>,
    embedder_query_prefix: Option<String>,
    embedder_passage_prefix: Option<String>,
    candidate_pool: usize,
    rerank: Option<String>,
) -> PyResult<FolderBuild> {
    use std::collections::{HashMap, HashSet};

    let dir = resolve_index_dir(folder, index_dir);
    let fingerprint = index_fingerprint(
        chunk_size,
        chunk_overlap,
        &retrieval,
        &model,
        &embedder_model,
        embedder_dim,
    );

    // Load the prior index (if any, and if its settings still match).
    let mut cache_map: HashMap<String, CachedFile> = HashMap::new();
    let had_cache = match load_index(&dir, &fingerprint) {
        Some(idx) => {
            for f in idx.files {
                cache_map.insert(f.source.clone(), f);
            }
            true
        }
        None => false,
    };

    // Walk the folder. Config for re-chunking new/changed files.
    let paths = collect_files(folder, recursive, gitignore, ignore_globs)?;
    let dir_abs = dir.canonicalize().ok();
    let mode = retrieval_from_str(retrieval.as_deref(), candidate_pool)?;
    let cfg = doc_config(
        strategy.clone(),
        token_budget,
        candidate_k,
        chunk_size,
        chunk_overlap,
        mode,
    )?;

    let mut entries: Vec<CachedFile> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut changed = !had_cache;

    for p in &paths {
        // Never ingest our own index directory if it lives inside the folder.
        if let (Some(da), Ok(pa)) = (&dir_abs, p.canonicalize()) {
            if pa.starts_with(da) {
                continue;
            }
        }
        let source = p.to_string_lossy().to_string();
        let (mtime, size) = match file_stat(p) {
            Some(s) => s,
            None => continue,
        };
        seen.insert(source.clone());
        // Unchanged → reuse the cached chunks (with their embeddings).
        if let Some(prev) = cache_map.get(&source) {
            if prev.mtime == mtime && prev.size == size {
                entries.push(CachedFile {
                    source,
                    mtime,
                    size,
                    chunks: prev.chunks.clone(),
                });
                continue;
            }
        }
        // New or changed → extract + chunk fresh; record skips (path, reason).
        let sections = match extract_file_text(&source) {
            Ok((_, s)) => s,
            Err(e) => {
                skipped.push((source.clone(), e.to_string()));
                continue;
            }
        };
        let chunks = to_py(RhDocument::chunk_sections(&source, &sections, &cfg))?;
        entries.push(CachedFile {
            source,
            mtime,
            size,
            chunks,
        });
        changed = true;
    }
    // Files present in the cache but gone from disk count as a change.
    if cache_map.keys().any(|s| !seen.contains(s)) {
        changed = true;
    }

    // Merge every file's chunks into one corpus and give them unique ids.
    let mut all: Vec<Chunk> = Vec::new();
    for f in &entries {
        all.extend(f.chunks.iter().cloned());
    }
    if all.is_empty() {
        return Err(PyValueError::new_err(format!(
            "from_folder: no readable files found under '{}'. This build reads \
             text/code files; the standard `pip install redhop` also parses PDF/DOCX/PPTX/XLSX.",
            folder.display()
        )));
    }
    for (i, c) in all.iter_mut().enumerate() {
        c.id = ChunkId::new(format!("{i}"));
    }

    let n_files = entries.len();
    let mut doc = build_chunks_doc(
        all,
        strategy,
        chunk_size,
        chunk_overlap,
        token_budget,
        candidate_k,
        retrieval,
        model,
        embedder_model,
        embedder_tokenizer,
        embedder_dim,
        embedder_pooling,
        embedder_query_prefix,
        embedder_passage_prefix,
        candidate_pool,
        rerank,
    )?;

    // Persist only when the corpus changed. Pull embeddings onto the chunks first
    // (so reloads skip re-embedding), then regroup them by file.
    if changed {
        let embedded = to_py(doc.embedded_chunks())?;
        let mut by_source: HashMap<String, Vec<Chunk>> = HashMap::new();
        for c in embedded {
            by_source.entry(c.source.clone()).or_default().push(c);
        }
        let files_out: Vec<CachedFile> = entries
            .into_iter()
            .map(|f| CachedFile {
                chunks: by_source.remove(&f.source).unwrap_or_default(),
                source: f.source,
                mtime: f.mtime,
                size: f.size,
            })
            .collect();
        let idx = PersistedIndex {
            version: INDEX_VERSION,
            fingerprint,
            files: files_out,
        };
        std::fs::create_dir_all(&dir).map_err(|e| {
            PyValueError::new_err(format!(
                "from_folder: creating index dir '{}': {e}",
                dir.display()
            ))
        })?;
        let json = serde_json::to_string(&idx)
            .map_err(|e| PyValueError::new_err(format!("from_folder: serializing index: {e}")))?;
        std::fs::write(dir.join(INDEX_FILE), json).map_err(|e| {
            PyValueError::new_err(format!(
                "from_folder: writing index to '{}': {e}",
                dir.display()
            ))
        })?;
    }

    Ok((doc, skipped, n_files))
}

/// Shared construction for text-backed documents (used by `from_text` and
/// `from_file`): resolve the tier, build the config, chunk+index, attach the
/// embedder if the tier needs one.
#[allow(clippy::too_many_arguments)]
fn build_text_doc(
    files: Vec<(String, Vec<RhSection>)>,
    strategy: Option<String>,
    chunk_size: usize,
    chunk_overlap: usize,
    token_budget: usize,
    candidate_k: usize,
    retrieval: Option<String>,
    model: Option<String>,
    embedder_model: Option<String>,
    embedder_tokenizer: Option<String>,
    embedder_dim: usize,
    embedder_pooling: Option<String>,
    embedder_query_prefix: Option<String>,
    embedder_passage_prefix: Option<String>,
    candidate_pool: usize,
    rerank: Option<String>,
) -> PyResult<RhDocument> {
    let mode = retrieval_from_str(retrieval.as_deref(), candidate_pool)?;
    let needs_embedder = matches!(mode, RetrievalMode::Hybrid { .. } | RetrievalMode::Dense);
    let cfg = doc_config(
        strategy,
        token_budget,
        candidate_k,
        chunk_size,
        chunk_overlap,
        mode,
    )?;
    let mut inner = to_py(RhDocument::from_sources_with(files, cfg))?;
    if needs_embedder {
        inner = apply_dense_embedder(
            inner,
            model,
            embedder_model,
            embedder_tokenizer,
            embedder_dim,
            embedder_pooling,
            embedder_query_prefix,
            embedder_passage_prefix,
        )?;
    }
    apply_reranker(inner, rerank)
}

/// A document you reason over. Bring your own parser/OCR; RedHop owns chunking,
/// internal retrieval, and reasoning-preserving context allocation. Retrieval is an
/// internal detail — you think in documents and queries, not retrievers.
#[pyclass]
struct Document {
    inner: RhDocument,
    // Files skipped during `from_folder` (path, reason). Empty for single-doc
    // constructors. Surfaced via the `skipped_files` getter.
    skipped: Vec<(String, String)>,
    // Number of files actually indexed (1 for single-doc constructors).
    n_files: usize,
}

impl Document {
    /// Wrap an inner document from a single-source constructor (no skips).
    fn single(inner: RhDocument) -> Self {
        Self {
            inner,
            skipped: Vec::new(),
            n_files: 1,
        }
    }
}

#[pymethods]
impl Document {
    /// Build from raw text (chunked + indexed internally).
    ///
    /// `chunk_size` (target tokens/chunk) and `chunk_overlap` are **index-time**
    /// — they fix how the document is split and cannot change per query.
    /// `token_budget` is the *default* assembly budget; override it per call on
    /// `context(query, budget=...)` (query-time, no re-indexing). `strategy`
    /// defaults to the size-gated Auto policy.
    #[staticmethod]
    #[pyo3(signature = (text, source="document", strategy=None, chunk_size=128,
                        chunk_overlap=1, token_budget=8192, candidate_k=20,
                        retrieval=None, model=None, embedder_model=None, embedder_tokenizer=None,
                        embedder_dim=384, embedder_pooling=None, embedder_query_prefix=None,
                        embedder_passage_prefix=None, candidate_pool=50, rerank=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_text(
        text: &str,
        source: &str,
        strategy: Option<String>,
        chunk_size: usize,
        chunk_overlap: usize,
        token_budget: usize,
        candidate_k: usize,
        retrieval: Option<String>,
        model: Option<String>,
        embedder_model: Option<String>,
        embedder_tokenizer: Option<String>,
        embedder_dim: usize,
        embedder_pooling: Option<String>,
        embedder_query_prefix: Option<String>,
        embedder_passage_prefix: Option<String>,
        candidate_pool: usize,
        rerank: Option<String>,
    ) -> PyResult<Self> {
        let sections = vec![RhSection {
            text: text.to_string(),
            page: None,
            heading: None,
            line: None,
        }];
        let inner = build_text_doc(
            vec![(source.to_string(), sections)],
            strategy,
            chunk_size,
            chunk_overlap,
            token_budget,
            candidate_k,
            retrieval,
            model,
            embedder_model,
            embedder_tokenizer,
            embedder_dim,
            embedder_pooling,
            embedder_query_prefix,
            embedder_passage_prefix,
            candidate_pool,
            rerank,
        )?;
        Ok(Self::single(inner))
    }

    /// Build straight from a file on disk — RedHop reads it, chunks, and indexes;
    /// the file path becomes each chunk's source, with page/heading/line tracked
    /// for citations.
    ///
    /// The base install reads **UTF-8 text & code** (`.txt`, `.md`, `.rst`, source,
    /// `.json`, `.csv`, logs, …). The standard `pip install redhop` also parses PDF, DOCX,
    /// PPTX, and XLSX.
    #[staticmethod]
    #[pyo3(signature = (path, strategy=None, chunk_size=128, chunk_overlap=1,
                        token_budget=8192, candidate_k=20, retrieval=None, model=None,
                        embedder_model=None, embedder_tokenizer=None, embedder_dim=384,
                        embedder_pooling=None, embedder_query_prefix=None,
                        embedder_passage_prefix=None, candidate_pool=50, rerank=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_file(
        path: &str,
        strategy: Option<String>,
        chunk_size: usize,
        chunk_overlap: usize,
        token_budget: usize,
        candidate_k: usize,
        retrieval: Option<String>,
        model: Option<String>,
        embedder_model: Option<String>,
        embedder_tokenizer: Option<String>,
        embedder_dim: usize,
        embedder_pooling: Option<String>,
        embedder_query_prefix: Option<String>,
        embedder_passage_prefix: Option<String>,
        candidate_pool: usize,
        rerank: Option<String>,
    ) -> PyResult<Self> {
        let (source, sections) = extract_file_text(path)?;
        let inner = build_text_doc(
            vec![(source, sections)],
            strategy,
            chunk_size,
            chunk_overlap,
            token_budget,
            candidate_k,
            retrieval,
            model,
            embedder_model,
            embedder_tokenizer,
            embedder_dim,
            embedder_pooling,
            embedder_query_prefix,
            embedder_passage_prefix,
            candidate_pool,
            rerank,
        )?;
        Ok(Self::single(inner))
    }

    /// Build from in-memory **bytes** you already fetched — RedHop parses, chunks,
    /// and indexes them. `source` is the document's name/key (e.g. `"contract.pdf"`):
    /// its extension selects the parser, and it becomes each chunk's citation source.
    ///
    /// This is the on-ramp for **cloud object storage** (S3, Cloudflare R2, Azure
    /// Blob, GCS), HTTP downloads, or database blobs — fetch the bytes with your own
    /// client, hand them here. RedHop never touches your cloud credentials. Same
    /// formats as `from_file` (PDF/DOCX/PPTX/XLSX + text/code).
    #[staticmethod]
    #[pyo3(signature = (data, source, strategy=None, chunk_size=128, chunk_overlap=1,
                        token_budget=8192, candidate_k=20, retrieval=None, model=None,
                        embedder_model=None, embedder_tokenizer=None, embedder_dim=384,
                        embedder_pooling=None, embedder_query_prefix=None,
                        embedder_passage_prefix=None, candidate_pool=50, rerank=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_bytes(
        data: Vec<u8>,
        source: &str,
        strategy: Option<String>,
        chunk_size: usize,
        chunk_overlap: usize,
        token_budget: usize,
        candidate_k: usize,
        retrieval: Option<String>,
        model: Option<String>,
        embedder_model: Option<String>,
        embedder_tokenizer: Option<String>,
        embedder_dim: usize,
        embedder_pooling: Option<String>,
        embedder_query_prefix: Option<String>,
        embedder_passage_prefix: Option<String>,
        candidate_pool: usize,
        rerank: Option<String>,
    ) -> PyResult<Self> {
        let (source, sections) = extract_bytes_sections(&data, source)?;
        let inner = build_text_doc(
            vec![(source, sections)],
            strategy,
            chunk_size,
            chunk_overlap,
            token_budget,
            candidate_k,
            retrieval,
            model,
            embedder_model,
            embedder_tokenizer,
            embedder_dim,
            embedder_pooling,
            embedder_query_prefix,
            embedder_passage_prefix,
            candidate_pool,
            rerank,
        )?;
        Ok(Self::single(inner))
    }

    /// Build one document from **every readable file in a folder** — RedHop walks
    /// the directory, reads/chunks/indexes each file it can, and keeps each chunk's
    /// own file path as its `source` (so citations point at the right file).
    ///
    /// Files it can't parse (unsupported formats, unreadable bytes) are skipped;
    /// hidden entries and build/cache dirs (`node_modules`, `target`, `__pycache__`,
    /// …) are ignored. With the base install only text/code files are read; install
    /// the standard install also parses PDF/DOCX/PPTX/XLSX. `recursive=False` stays in
    /// the top level.
    ///
    /// Set `persist=True` to save the index to disk and **incrementally reload** it:
    /// on the next run, files whose modified-time and size are unchanged are reused
    /// from the cache (no re-parsing, no re-embedding), only new/changed files are
    /// processed, and removed files are dropped. The index defaults to
    /// `<folder>/.redhop`; pass `index_dir=` to put it elsewhere. Without `persist`
    /// (the default) the index is in-memory and rebuilt each run.
    ///
    /// Hidden entries (`.git`, dotfiles) and build/cache dirs (`node_modules`,
    /// `target`, …) are always skipped. By default it also honors **`.gitignore`**
    /// (set `gitignore=False` to index ignored files too), and you can pass extra
    /// **`ignore=[...]`** gitignore-style globs (e.g. `["*.lock", "tests/**"]`).
    #[staticmethod]
    #[pyo3(signature = (path, recursive=true, persist=false, index_dir=None,
                        strategy=None, chunk_size=128, chunk_overlap=1,
                        token_budget=8192, candidate_k=20, retrieval=None, model=None,
                        embedder_model=None, embedder_tokenizer=None, embedder_dim=384,
                        embedder_pooling=None, embedder_query_prefix=None,
                        embedder_passage_prefix=None, candidate_pool=50,
                        ignore=None, gitignore=true, rerank=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_folder(
        path: &str,
        recursive: bool,
        persist: bool,
        index_dir: Option<String>,
        strategy: Option<String>,
        chunk_size: usize,
        chunk_overlap: usize,
        token_budget: usize,
        candidate_k: usize,
        retrieval: Option<String>,
        model: Option<String>,
        embedder_model: Option<String>,
        embedder_tokenizer: Option<String>,
        embedder_dim: usize,
        embedder_pooling: Option<String>,
        embedder_query_prefix: Option<String>,
        embedder_passage_prefix: Option<String>,
        candidate_pool: usize,
        ignore: Option<Vec<String>>,
        gitignore: bool,
        rerank: Option<String>,
    ) -> PyResult<Self> {
        let root = Path::new(path);
        if !root.is_dir() {
            return Err(PyValueError::new_err(format!(
                "from_folder: '{path}' is not a directory"
            )));
        }
        let ignore_globs = ignore.unwrap_or_default();

        // Persisted path: incremental on-disk index.
        if persist || index_dir.is_some() {
            let (inner, skipped, n_files) = build_folder_persisted(
                root,
                recursive,
                gitignore,
                &ignore_globs,
                &index_dir,
                strategy,
                chunk_size,
                chunk_overlap,
                token_budget,
                candidate_k,
                retrieval,
                model,
                embedder_model,
                embedder_tokenizer,
                embedder_dim,
                embedder_pooling,
                embedder_query_prefix,
                embedder_passage_prefix,
                candidate_pool,
                rerank,
            )?;
            return Ok(Self {
                inner,
                skipped,
                n_files,
            });
        }

        // In-memory path: walk, extract, build one index (rebuilt each run).
        // Files we can't read/parse are skipped but recorded (path, reason) so the
        // caller can see what was dropped via `skipped_files`.
        let paths = collect_files(root, recursive, gitignore, &ignore_globs)?;
        let mut files: Vec<(String, Vec<RhSection>)> = Vec::new();
        let mut skipped: Vec<(String, String)> = Vec::new();
        for p in &paths {
            let sp = p.to_string_lossy();
            match extract_file_text(&sp) {
                Ok((source, sections)) => files.push((source, sections)),
                Err(e) => skipped.push((sp.into_owned(), e.to_string())),
            }
        }
        if files.is_empty() {
            return Err(PyValueError::new_err(format!(
                "from_folder: no readable files found under '{path}'. The base install reads \
                 text/code files; the standard `pip install redhop` also parses PDF/DOCX/PPTX/XLSX."
            )));
        }
        let n_files = files.len();
        let inner = build_text_doc(
            files,
            strategy,
            chunk_size,
            chunk_overlap,
            token_budget,
            candidate_k,
            retrieval,
            model,
            embedder_model,
            embedder_tokenizer,
            embedder_dim,
            embedder_pooling,
            embedder_query_prefix,
            embedder_passage_prefix,
            candidate_pool,
            rerank,
        )?;
        Ok(Self {
            inner,
            skipped,
            n_files,
        })
    }

    /// Build from chunks you already produced (strings or `{"text", ...}` dicts).
    /// `chunk_size`/`chunk_overlap` don't apply here (you chunked already).
    #[staticmethod]
    #[pyo3(signature = (chunks, strategy=None, token_budget=8192, candidate_k=20,
                        retrieval=None, model=None, embedder_model=None, embedder_tokenizer=None,
                        embedder_dim=384, embedder_pooling=None, embedder_query_prefix=None,
                        embedder_passage_prefix=None, candidate_pool=50, rerank=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_chunks(
        chunks: &Bound<'_, PyAny>,
        strategy: Option<String>,
        token_budget: usize,
        candidate_k: usize,
        retrieval: Option<String>,
        model: Option<String>,
        embedder_model: Option<String>,
        embedder_tokenizer: Option<String>,
        embedder_dim: usize,
        embedder_pooling: Option<String>,
        embedder_query_prefix: Option<String>,
        embedder_passage_prefix: Option<String>,
        candidate_pool: usize,
        rerank: Option<String>,
    ) -> PyResult<Self> {
        let chunk_vec: Vec<Chunk> = chunks_from_py(chunks)?
            .into_iter()
            .map(|r| r.chunk)
            .collect();
        let mode = retrieval_from_str(retrieval.as_deref(), candidate_pool)?;
        let needs_embedder = matches!(mode, RetrievalMode::Hybrid { .. } | RetrievalMode::Dense);
        let cfg = doc_config(strategy, token_budget, candidate_k, 256, 1, mode)?;
        let mut inner = to_py(RhDocument::from_chunks_with(chunk_vec, cfg))?;
        if needs_embedder {
            inner = apply_dense_embedder(
                inner,
                model,
                embedder_model,
                embedder_tokenizer,
                embedder_dim,
                embedder_pooling,
                embedder_query_prefix,
                embedder_passage_prefix,
            )?;
        }
        let inner = apply_reranker(inner, rerank)?;
        Ok(Self::single(inner))
    }

    /// Assemble the reasoning context for a query (retrieve → allocate).
    /// `budget` overrides the document's default token budget for this call
    /// only (query-time; no re-indexing).
    ///
    /// **Structural expansion** (optional): `neighbors=N` also includes the N
    /// adjacent chunks on each side of every selected chunk (i±1, … in the same
    /// file); `include_heading=True` adds the section's heading. Companions are
    /// deterministic (document order + headings, no model), added only within the
    /// token budget, and emitted in reading order so each hit is a contiguous
    /// window. They show up in `ctx.citations` and as `report.n_expanded`.
    /// Returns a `BuiltContext`.
    #[pyo3(signature = (query, budget=None, neighbors=0, include_heading=false))]
    fn context(
        &mut self,
        query: &str,
        budget: Option<usize>,
        neighbors: usize,
        include_heading: bool,
    ) -> PyResult<BuiltContext> {
        let built = if neighbors == 0 && !include_heading {
            self.inner.context_with(query, budget, None)
        } else {
            self.inner
                .context_expanded(query, budget, None, neighbors, include_heading)
        };
        Ok(py_built(to_py(built)?))
    }

    /// Diagnose retrieval for a query **without** modifying anything (pure
    /// observability). For `strategy="auto"`, `report.strategy` is the decision.
    fn analyze(&mut self, query: &str) -> PyResult<ContextReport> {
        let report = to_py(self.inner.analyze(query))?;
        let rendered = report.render(None);
        Ok(ContextReport {
            inner: report,
            rendered,
        })
    }

    #[getter]
    fn n_chunks(&self) -> usize {
        self.inner.len()
    }
    /// Number of files actually indexed (1 for from_text/from_file/from_bytes;
    /// the readable count for from_folder).
    #[getter]
    fn n_files(&self) -> usize {
        self.n_files
    }
    /// Files `from_folder` skipped, as `(path, reason)` pairs — unsupported
    /// formats, unreadable bytes, or no extractable text (e.g. scanned PDFs).
    /// Empty for single-document constructors.
    #[getter]
    fn skipped_files(&self) -> Vec<(String, String)> {
        self.skipped.clone()
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    fn __repr__(&self) -> String {
        format!("Document({} chunks)", self.inner.len())
    }
}

#[pymodule]
fn _redhop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<BuiltContext>()?;
    m.add_class::<ContextReport>()?;
    m.add_class::<Document>()?;
    m.add_function(wrap_pyfunction!(build_context, m)?)?;
    m.add_function(wrap_pyfunction!(filter_context, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_context, m)?)?;
    m.add_function(wrap_pyfunction!(context_economics, m)?)?;
    m.add_function(wrap_pyfunction!(grounding_score, m)?)?;
    m.add_function(wrap_pyfunction!(link_strength, m)?)?;
    Ok(())
}
