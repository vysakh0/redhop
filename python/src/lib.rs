//! pyo3 bindings for RedHop context optimization.
//!
//! Thin wrapper over the stable `redhop-context` public API — no logic is
//! duplicated here. Rust remains the source of truth; this module only maps
//! Pythonic inputs (dicts/lists/strings) to the Rust types and wraps the
//! results in small Python classes.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use redhop_context::{
    analyze_context as rh_analyze, build_context as rh_build, context_economics as rh_economics,
    filter_context as rh_filter, grounding_score as rh_grounding, link_strength as rh_link,
    AutoDecision, ContextConfig, ContextReport as RhReport, ContextStrategy,
};
use redhop_core::{
    Chunk, ChunkId, Embedding, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenCount,
};
use redhop_document::{Document as RhDocument, DocumentConfig, RetrievalMode};

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
#[pyclass]
struct BuiltContext {
    text: String,
    chunks: Vec<String>,
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
fn py_built(ctx: redhop_context::BuiltContext) -> BuiltContext {
    let text = ctx.text();
    let chunks = ctx.chunks.iter().map(|c| c.text.clone()).collect();
    let rendered = ctx.report.render(None);
    BuiltContext {
        text,
        chunks,
        report: ContextReport {
            inner: ctx.report,
            rendered,
        },
    }
}

fn to_py<T>(r: redhop_core::Result<T>) -> PyResult<T> {
    r.map_err(|e| PyValueError::new_err(e.to_string()))
}

fn retrieval_from_str(retrieval: Option<&str>, candidate_pool: usize) -> PyResult<RetrievalMode> {
    Ok(match retrieval {
        None | Some("lexical") => RetrievalMode::Lexical,
        Some("rerank") => RetrievalMode::DenseRerank {
            candidate_pool: candidate_pool.max(1),
        },
        Some("dense") => RetrievalMode::Dense,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown retrieval mode '{other}'; use 'lexical' (default), 'rerank' \
                 (BM25 prune → dense), or 'dense' (global dense, bounded corpora)"
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
        context,
    })
}

/// Attach a dense embedder for `retrieval="rerank"`. The ONNX runtime + model
/// live behind the crate's `onnx` feature; the default (lexical) wheel raises a
/// clear error rather than silently degrading.
#[cfg(feature = "onnx")]
#[allow(clippy::too_many_arguments)]
fn apply_rerank(
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
            None | Some("cls") => redhop_embeddings::Pooling::Cls,
            Some("mean") => redhop_embeddings::Pooling::Mean,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "unknown embedder_pooling '{other}'; use 'cls' (default, e.g. BGE) or 'mean' \
                     (e.g. MiniLM / GTE / E5)"
                )))
            }
        };
        let load = |prefix: &str| -> PyResult<redhop_embeddings::OnnxEmbedder> {
            let mut config = redhop_embeddings::EmbedderConfig::bge(embedder_dim);
            config.pooling = pooling;
            config.prefix = prefix.to_string();
            redhop_embeddings::OnnxEmbedder::load(m, t, config)
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
    let name = model
        .as_deref()
        .unwrap_or(redhop_embeddings::DEFAULT_MODEL);
    let resolved =
        redhop_embeddings::resolve_model(name).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let load = |prefix: &str| -> PyResult<redhop_embeddings::OnnxEmbedder> {
        redhop_embeddings::OnnxEmbedder::load(
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

#[cfg(not(feature = "onnx"))]
#[allow(clippy::too_many_arguments)]
fn apply_rerank(
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
        "retrieval='rerank' needs an ONNX-enabled build of redhop — the default wheel is \
         lexical-only (no native runtime). Reinstall an onnx-enabled build (built with the \
         `onnx` cargo feature, e.g. `maturin develop --features onnx`).",
    ))
}

/// A document you reason over. Bring your own parser/OCR; RedHop owns chunking,
/// internal retrieval, and reasoning-aware context allocation. Retrieval is an
/// internal detail — you think in documents and queries, not retrievers.
#[pyclass]
struct Document {
    inner: RhDocument,
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
                        embedder_passage_prefix=None, candidate_pool=50))]
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
    ) -> PyResult<Self> {
        let mode = retrieval_from_str(retrieval.as_deref(), candidate_pool)?;
        let rerank = matches!(
            mode,
            RetrievalMode::DenseRerank { .. } | RetrievalMode::Dense
        );
        let cfg = doc_config(
            strategy,
            token_budget,
            candidate_k,
            chunk_size,
            chunk_overlap,
            mode,
        )?;
        let mut inner = to_py(RhDocument::from_text_with(source, text, cfg))?;
        if rerank {
            inner = apply_rerank(
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
        Ok(Self { inner })
    }

    /// Build from chunks you already produced (strings or `{"text", ...}` dicts).
    /// `chunk_size`/`chunk_overlap` don't apply here (you chunked already).
    #[staticmethod]
    #[pyo3(signature = (chunks, strategy=None, token_budget=8192, candidate_k=20,
                        retrieval=None, model=None, embedder_model=None, embedder_tokenizer=None,
                        embedder_dim=384, embedder_pooling=None, embedder_query_prefix=None,
                        embedder_passage_prefix=None, candidate_pool=50))]
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
    ) -> PyResult<Self> {
        let chunk_vec: Vec<Chunk> = chunks_from_py(chunks)?
            .into_iter()
            .map(|r| r.chunk)
            .collect();
        let mode = retrieval_from_str(retrieval.as_deref(), candidate_pool)?;
        let rerank = matches!(
            mode,
            RetrievalMode::DenseRerank { .. } | RetrievalMode::Dense
        );
        let cfg = doc_config(strategy, token_budget, candidate_k, 256, 1, mode)?;
        let mut inner = to_py(RhDocument::from_chunks_with(chunk_vec, cfg))?;
        if rerank {
            inner = apply_rerank(
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
        Ok(Self { inner })
    }

    /// Assemble the reasoning context for a query (retrieve → allocate).
    /// `budget` overrides the document's default token budget for this call
    /// only (query-time; no re-indexing). Returns a `BuiltContext`.
    #[pyo3(signature = (query, budget=None))]
    fn context(&mut self, query: &str, budget: Option<usize>) -> PyResult<BuiltContext> {
        Ok(py_built(to_py(
            self.inner.context_with(query, budget, None),
        )?))
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
