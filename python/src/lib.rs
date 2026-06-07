//! pyo3 bindings for RedHop context optimization.
//!
//! Thin wrapper over the stable `redhop-context` public API — no logic is
//! duplicated here. Rust remains the source of truth; this module only maps
//! Pythonic inputs (dicts/lists/strings) to the Rust types and wraps the
//! results in small Python classes.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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

/// Thin forwarder over [`redhop::strategy_from_str`] — the canonical
/// string→enum mapping lives in the Rust crate so every binding shares it
/// and the unknown-strategy error message can't drift.
fn strategy_from_str(s: &str) -> PyResult<ContextStrategy> {
    redhop::strategy_from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
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

/// Convert a Python value to a `serde_json::Value` for chunk metadata.
///
/// Accepts: `None`, `bool`, `int`, `float`, `str`, `list`, `dict` (any
/// nesting). Rejects unsupported types with a clear error so users
/// see which key carried bad data.
fn py_to_json(v: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if v.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = v.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = v.extract::<i64>() {
        Ok(serde_json::Value::Number(serde_json::Number::from(i)))
    } else if let Ok(f) = v.extract::<f64>() {
        serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .ok_or_else(|| PyValueError::new_err("non-finite float in chunk metadata"))
    } else if let Ok(s) = v.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = v.downcast::<PyList>() {
        let mut arr = Vec::with_capacity(list.len());
        for item in list.iter() {
            arr.push(py_to_json(&item)?);
        }
        Ok(serde_json::Value::Array(arr))
    } else if let Ok(d) = v.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, val) in d.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&val)?);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        Err(PyValueError::new_err(format!(
            "unsupported chunk metadata value: {} (allowed: None, bool, int, float, str, list, dict)",
            v.get_type().name()?,
        )))
    }
}

/// Inverse of [`py_to_json`]: convert a `serde_json::Value` back to a
/// Python object for the metadata getter.
fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<PyObject> {
    use pyo3::IntoPy;
    Ok(match v {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.into_py(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py(py)
            } else {
                py.None()
            }
        }
        serde_json::Value::String(s) => s.clone().into_py(py),
        serde_json::Value::Array(arr) => {
            let mut items: Vec<PyObject> = Vec::with_capacity(arr.len());
            for x in arr {
                items.push(json_to_py(py, x)?);
            }
            items.into_py(py)
        }
        serde_json::Value::Object(map) => {
            let d = PyDict::new(py);
            for (k, val) in map {
                d.set_item(k, json_to_py(py, val)?)?;
            }
            d.into()
        }
    })
}

/// One unit of content in a [`Document`] — the construction primitive
/// for callers who pre-chunked their corpus elsewhere (schema rows,
/// API endpoints, code symbols, defined contract terms, pre-segmented
/// paragraphs).
///
/// Two concepts are kept distinct:
/// - **`source`** — *provenance*: where the chunk came from (file
///   path, URL, logical handle). This is what `ctx.citations[*].source`
///   displays. Defaults to `"input"` if omitted.
/// - **`id`** — the chunk's *identity*: a stable identifier used for
///   dedup and gold-chunk evaluation. Defaults to `"c0"`, `"c1"`, …
///   based on position in the list passed to `Document.from_chunks`.
///
/// `metadata` is an open dict (`{str: Any}`, JSON-compatible values).
/// The citations machinery picks up known keys — currently `page`
/// (int), `heading` (str), `line` (int). Anything else is preserved
/// but not surfaced by the built-in citations contract.
///
/// ```python
/// chunks = [
///     redhop.Chunk(
///         "orders.amt (decimal) — order amount / revenue / spend in USD",
///         source="schema.sql",
///         id="orders.amt",
///         metadata={"table": "orders", "column": "amt", "type": "decimal"},
///     ),
///     redhop.Chunk(
///         "9.1 Governing Law. This Agreement shall be governed by …",
///         source="contract.pdf",
///         metadata={"page": 12, "heading": "9.1 Governing Law"},
///     ),
/// ]
/// doc = redhop.Document.from_chunks(chunks)
/// ```
#[pyclass(name = "Chunk", module = "redhop")]
#[derive(Clone)]
struct PyChunk {
    text: String,
    source: Option<String>,
    id: Option<String>,
    metadata: HashMap<String, serde_json::Value>,
    token_count: Option<usize>,
    embedding: Option<Vec<f32>>,
}

impl PyChunk {
    /// Materialize this Python-side chunk into a Rust core
    /// [`redhop::core::Chunk`], using `idx` to fill in an auto-id when
    /// the user didn't supply one.
    fn to_core(&self, idx: usize) -> redhop::core::Chunk {
        let id = self.id.clone().unwrap_or_else(|| format!("c{idx}"));
        let source = self.source.clone().unwrap_or_else(|| "input".into());
        let tok = self
            .token_count
            .unwrap_or_else(|| self.text.split_whitespace().count().max(1));
        let mut chunk = redhop::core::Chunk::new(
            ChunkId::new(id),
            self.text.clone(),
            source,
            TokenCount(tok),
        );
        chunk.metadata = self.metadata.clone();
        if let Some(e) = &self.embedding {
            chunk = chunk.with_embedding(Embedding::from(e.clone()));
        }
        chunk
    }
}

#[pymethods]
impl PyChunk {
    #[new]
    #[pyo3(signature = (text, *, source=None, id=None, metadata=None, token_count=None, embedding=None))]
    fn py_new(
        text: String,
        source: Option<String>,
        id: Option<String>,
        metadata: Option<&Bound<'_, PyDict>>,
        token_count: Option<usize>,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let mut meta = HashMap::new();
        if let Some(m) = metadata {
            for (k, v) in m.iter() {
                let key: String = k.extract()?;
                meta.insert(key, py_to_json(&v)?);
            }
        }
        Ok(Self {
            text,
            source,
            id,
            metadata: meta,
            token_count,
            embedding,
        })
    }

    /// The chunk's text content.
    #[getter]
    fn text(&self) -> &str {
        &self.text
    }

    /// The chunk's provenance (file path, URL, logical handle). `None`
    /// if not supplied — defaults to `"input"` when materialized.
    #[getter]
    fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// The chunk's stable identifier. `None` if not supplied — defaults
    /// to `c0`, `c1`, … based on position when materialized.
    #[getter]
    fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Token count (whitespace-counted if not supplied).
    #[getter]
    fn token_count(&self) -> Option<usize> {
        self.token_count
    }

    /// Open metadata dict. Citations pick up known keys (`page`,
    /// `heading`, `line`); anything else is preserved.
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.metadata {
            d.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let snippet: String = self.text.chars().take(40).collect();
        let ellipsis = if self.text.chars().count() > 40 {
            "…"
        } else {
            ""
        };
        let src = self
            .source
            .as_deref()
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|| "None".into());
        let id = self
            .id
            .as_deref()
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|| "None".into());
        format!(
            "Chunk(text={:?}{}, source={}, id={})",
            snippet, ellipsis, src, id,
        )
    }
}

/// Iterate a Python sequence of [`Chunk`] objects into the
/// `RetrievalResult` shape the Rust core expects. Rejects strings,
/// dicts, and anything else with a clear migration message — as of
/// 0.3.0, `Document.from_chunks` and the low-level
/// `build_context` / `filter_context` / `analyze_context` entry
/// points all require typed `redhop.Chunk` instances.
fn chunks_from_py(chunks: &Bound<'_, PyAny>) -> PyResult<Vec<RetrievalResult>> {
    let mut out = Vec::new();
    for (i, item) in chunks.try_iter()?.enumerate() {
        let item = item?;
        let chunk: PyRef<'_, PyChunk> = item.extract().map_err(|_| {
            let got = item
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "?".to_string());
            PyValueError::new_err(format!(
                "chunk {i}: expected redhop.Chunk(text, source=..., ...); got {got}. \
                 As of 0.3.0, strings and dicts are no longer accepted — wrap your input as \
                 `redhop.Chunk(text, source='myfile.txt')`."
            ))
        })?;
        let core = chunk.to_core(i);
        out.push(RetrievalResult {
            chunk: core,
            score: Score {
                value: 1.0,
                method: RetrievalMethod::Dense,
            },
            breakdown: ScoreBreakdown::default(),
        });
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn config(
    strategy: Option<String>,
    token_budget: usize,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
    auto_passthrough_max_tokens: usize,
    redundancy_max_cosine: f32,
    preserve_order: bool,
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
        preserve_order,
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
    /// Audit trail of query rewrites applied (one [`RewriteRecord`] per
    /// stage, in chain order). Empty when no rewrite chain was used.
    #[getter]
    fn query_rewrites(&self) -> Vec<RewriteRecord> {
        self.inner
            .query_rewrites
            .iter()
            .cloned()
            .map(|r| RewriteRecord { inner: r })
            .collect()
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
    // Underlying Rust BuiltContext, kept for in-process operations like
    // `redhop.evaluate(...)` that need chunk IDs, score breakdowns, or the
    // full report shape. Never exposed to Python directly.
    inner: redhop::context::BuiltContext,
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
       distractor_min_grounding=0.10, link_min_jaccard=0.12, auto_passthrough_max_tokens=1500, redundancy_max_cosine=0.92, preserve_order=false))]
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
    preserve_order: bool,
) -> PyResult<BuiltContext> {
    let cfg = config(
        strategy,
        token_budget,
        distractor_min_grounding,
        link_min_jaccard,
        auto_passthrough_max_tokens,
        redundancy_max_cosine,
        preserve_order,
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
            inner: ctx.report.clone(),
            rendered,
        },
        inner: ctx,
    })
}

#[pyfunction]
#[pyo3(signature = (query, retrieved_chunks, strategy=None,
       distractor_min_grounding=0.10, link_min_jaccard=0.12, auto_passthrough_max_tokens=1500, redundancy_max_cosine=0.92, preserve_order=false))]
#[allow(clippy::too_many_arguments)]
fn filter_context(
    query: &str,
    retrieved_chunks: &Bound<'_, PyAny>,
    strategy: Option<String>,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
    auto_passthrough_max_tokens: usize,
    redundancy_max_cosine: f32,
    preserve_order: bool,
) -> PyResult<BuiltContext> {
    // filter = build with no budget truncation.
    let cfg = config(
        strategy,
        usize::MAX,
        distractor_min_grounding,
        link_min_jaccard,
        auto_passthrough_max_tokens,
        redundancy_max_cosine,
        preserve_order,
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
            inner: ctx.report.clone(),
            rendered,
        },
        inner: ctx,
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
        false, // preserve_order doesn't apply to analyze (no chunk emission)
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
        false, // preserve_order doesn't apply to economics (no chunk emission)
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
            inner: ctx.report.clone(),
            rendered,
        },
        inner: ctx,
    }
}

fn to_py<T>(r: redhop::core::Result<T>) -> PyResult<T> {
    r.map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Thin forwarder over [`redhop::retrieval_from_str`] — the canonical
/// string→enum mapping lives in the Rust crate so every binding shares it
/// and the unknown-mode error message can't drift.
fn retrieval_from_str(retrieval: Option<&str>, candidate_pool: usize) -> PyResult<RetrievalMode> {
    redhop::retrieval_from_str(retrieval, candidate_pool)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn doc_config(
    strategy: Option<String>,
    token_budget: usize,
    candidate_k: usize,
    chunk_size: usize,
    chunk_overlap: usize,
    retrieval_mode: RetrievalMode,
    language: Option<String>,
    preserve_order: bool,
) -> PyResult<DocumentConfig> {
    let base = DocumentConfig::default();
    let strategy = match strategy {
        Some(s) => strategy_from_str(&s)?,
        None => base.context.strategy,
    };
    // Route language string → SnowballAnalyzer builtin. Errors on unknown
    // names so a typo'd `"germann"` surfaces, not silently falls back to
    // English ranking.
    let analyzer: std::sync::Arc<dyn redhop::analyzer::Analyzer> = match language {
        None => base.context.analyzer.clone(),
        Some(name) => match redhop::analyzer::SnowballAnalyzer::by_name(&name) {
            Some(a) => std::sync::Arc::new(a),
            None => {
                return Err(PyValueError::new_err(format!(
                    "unknown language '{name}'; supported: arabic, danish, dutch, \
                     english, finnish, french, german, greek, hungarian, italian, \
                     norwegian, portuguese, romanian, russian, spanish, swedish, \
                     tamil, turkish"
                )));
            }
        },
    };
    let context = ContextConfig {
        token_budget,
        strategy,
        analyzer,
        preserve_order,
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
    let name = model
        .as_deref()
        .unwrap_or(redhop::embeddings::DEFAULT_MODEL);
    let resolved = redhop::embeddings::resolve_model(name)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
    language: Option<String>,
    preserve_order: bool,
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
        language,
        preserve_order,
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
    // `n_files` and `skipped_files` live on the inner Rust `RhDocument` —
    // single-source constructors default to `n_files=1`, `skipped_files=[]`;
    // the folder loaders populate them as files get walked. Python's
    // getters below pull straight through.
    inner: RhDocument,
}

impl Document {
    /// Wrap an inner document from a single-source constructor.
    fn single(inner: RhDocument) -> Self {
        Self { inner }
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
                        embedder_passage_prefix=None, candidate_pool=50, rerank=None, language=None,
                        preserve_order=false))]
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

        language: Option<String>,
        preserve_order: bool,
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
            language,
            preserve_order,
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
                        embedder_passage_prefix=None, candidate_pool=50, rerank=None, language=None,
                        preserve_order=false))]
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

        language: Option<String>,
        preserve_order: bool,
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
            language,
            preserve_order,
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
                        embedder_passage_prefix=None, candidate_pool=50, rerank=None, language=None,
                        preserve_order=false))]
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

        language: Option<String>,
        preserve_order: bool,
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
            language,
            preserve_order,
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
                        ignore=None, gitignore=true, rerank=None, language=None,
                        preserve_order=false))]
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

        language: Option<String>,
        preserve_order: bool,
    ) -> PyResult<Self> {
        // The walk + persist + cache-format + skipped-tracking all live in
        // Rust's `redhop::read_folder_with` so Python and Node share one
        // implementation. The Document carries `n_files()` and
        // `skipped_files()` accessors, which surface as Python getters
        // further down.
        #[cfg(feature = "files")]
        {
            let fo = redhop::FolderOptions {
                recursive: Some(recursive),
                gitignore: Some(gitignore),
                ignore: ignore.unwrap_or_default(),
                persist,
                index_dir,
                load: redhop::LoadOptions {
                    source: None,
                    chunk_size: Some(chunk_size),
                    chunk_overlap: Some(chunk_overlap),
                    token_budget: Some(token_budget),
                    candidate_k: Some(candidate_k),
                    strategy,
                    retrieval,
                    model,
                    embedder_model,
                    embedder_tokenizer,
                    embedder_dim: Some(embedder_dim),
                    embedder_pooling,
                    embedder_query_prefix,
                    embedder_passage_prefix,
                    candidate_pool: Some(candidate_pool),
                    rerank,
                    min_candidates: None,
                    language,
                    preserve_order: Some(preserve_order),
                },
            };
            let inner = to_py(redhop::read_folder_with(path, &fo))?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "files"))]
        {
            // Suppress unused-kwarg warnings under the lean (no-files) build.
            let _ = (
                path,
                recursive,
                persist,
                index_dir,
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
                ignore,
                gitignore,
                rerank,
                language,
                preserve_order,
            );
            Err(PyValueError::new_err(
                "from_folder requires the file-parsing tier. The standard \
                 `pip install redhop` includes it; if you built from source, \
                 add `--features files`.",
            ))
        }
    }

    /// Build from chunks you already produced (strings or `{"text", ...}` dicts).
    /// `chunk_size`/`chunk_overlap` don't apply here (you chunked already).
    #[staticmethod]
    #[pyo3(signature = (chunks, strategy=None, token_budget=8192, candidate_k=20,
                        retrieval=None, model=None, embedder_model=None, embedder_tokenizer=None,
                        embedder_dim=384, embedder_pooling=None, embedder_query_prefix=None,
                        embedder_passage_prefix=None, candidate_pool=50, rerank=None, language=None,
                        preserve_order=false))]
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

        language: Option<String>,
        preserve_order: bool,
    ) -> PyResult<Self> {
        let chunk_vec: Vec<Chunk> = chunks_from_py(chunks)?
            .into_iter()
            .map(|r| r.chunk)
            .collect();
        let mode = retrieval_from_str(retrieval.as_deref(), candidate_pool)?;
        let needs_embedder = matches!(mode, RetrievalMode::Hybrid { .. } | RetrievalMode::Dense);
        let cfg = doc_config(strategy, token_budget, candidate_k, 256, 1, mode, language, preserve_order)?;
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

    /// Run a query through a chain of [`Stripper`]/[`Vocabulary`] rewrites
    /// before retrieval, then assemble context as `.context(...)` does.
    ///
    /// `rewrites` is an ordered list — each stage sees the previous
    /// stage's output. The per-stage audit trail lands on
    /// `ctx.report.query_rewrites` as a list of `RewriteRecord`.
    ///
    /// Mirrors `redhop::Document::context_with_rewrites` in the Rust core.
    #[pyo3(signature = (query, rewrites))]
    fn context_with_rewrites(
        &mut self,
        query: &str,
        rewrites: &Bound<'_, PyAny>,
    ) -> PyResult<BuiltContext> {
        let owned = extract_rewrites(rewrites)?;
        let refs = rewrite_refs(&owned);
        Ok(py_built(to_py(self.inner.context_with_rewrites(query, &refs))?))
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
        self.inner.n_files()
    }
    /// Files `from_folder` skipped, as `(path, reason)` pairs — unsupported
    /// formats, unreadable bytes, or no extractable text (e.g. scanned PDFs).
    /// Empty for single-document constructors.
    #[getter]
    fn skipped_files(&self) -> Vec<(String, String)> {
        self.inner.skipped_files().to_vec()
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    fn __repr__(&self) -> String {
        format!("Document({} chunks)", self.inner.len())
    }
}

// ─── Query-set analyzer (templated-workload diagnostics) ─────────────────────
// Backed by `redhop::analyze_query_set` + `redhop::drop_template_terms`. See
// docs/findings/QUERY_SET_ANALYZER.md for the cross-workload probe that
// validated the heuristic.

fn cost_to_str(c: redhop::DilutionCost) -> &'static str {
    match c {
        redhop::DilutionCost::High => "high",
        redhop::DilutionCost::Medium => "medium",
        redhop::DilutionCost::Low => "low",
        redhop::DilutionCost::None => "none",
    }
}

/// Diagnostic report over a representative sample of a workload's queries.
///
/// Returned by [`analyze_query_set`]. Fields surface as read-only Python
/// attributes so the report can be passed across pickle / IPC boundaries
/// without losing structure.
#[pyclass(module = "redhop")]
#[derive(Clone)]
struct QuerySetReport {
    inner: redhop::QuerySetReport,
}

#[pymethods]
impl QuerySetReport {
    /// How many queries were analyzed.
    #[getter]
    fn n_queries(&self) -> usize {
        self.inner.n_queries
    }
    /// `True` when the workload looks templated (share ≥ 0.50 AND ≥ 2
    /// boilerplate terms).
    #[getter]
    fn is_templated(&self) -> bool {
        self.inner.is_templated
    }
    /// Mean fraction of each query that's shared boilerplate, 0.0–1.0.
    #[getter]
    fn template_word_share(&self) -> f32 {
        self.inner.template_word_share
    }
    /// Words appearing in ≥ 80% of the query set, sorted by frequency desc.
    /// Pass to [`drop_template_terms`].
    #[getter]
    fn boilerplate_terms(&self) -> Vec<String> {
        self.inner.boilerplate_terms.clone()
    }
    /// One of: `"high"`, `"medium"`, `"low"`, `"none"`.
    #[getter]
    fn estimated_dilution_cost(&self) -> &'static str {
        cost_to_str(self.inner.estimated_dilution_cost)
    }
    /// Human-readable recommendation describing what (if anything) to do.
    #[getter]
    fn suggested_action(&self) -> String {
        self.inner.suggested_action.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "QuerySetReport(n={}, is_templated={}, share={:.3}, cost={}, terms={})",
            self.inner.n_queries,
            self.inner.is_templated,
            self.inner.template_word_share,
            cost_to_str(self.inner.estimated_dilution_cost),
            self.inner.boilerplate_terms.len(),
        )
    }
}

// ─── Query-side rewrites (Stripper + Vocabulary) ──────────────────────────
// Backed by `redhop::rewrite::{Stripper, Vocabulary, QueryRewrite}`. Both
// implementations compile their content once and emit auditable trace
// records that land in `ContextReport.query_rewrites`. See
// `docs/findings/CUAD_RECALL_GAP.md` (mechanism) and
// `docs/findings/CUAD_CLAUSE_EXPANSION.md` (worked CUAD example with
// numbers).

/// One step in the rewrite chain — what the stage matched / added /
/// removed.
///
/// Returned per-stage on `ctx.report.query_rewrites` so the rewrite
/// chain is auditable. Fields surface as read-only Python attributes.
#[pyclass(module = "redhop")]
#[derive(Clone)]
struct RewriteRecord {
    inner: redhop::RewriteRecord,
}

#[pymethods]
impl RewriteRecord {
    /// The `QueryRewrite::name` of the stage that emitted this record
    /// (`"strip"` or `"vocabulary"` for the built-in implementations).
    #[getter]
    fn stage(&self) -> String {
        self.inner.stage.clone()
    }
    /// Input query handed to this stage.
    #[getter]
    fn from_query(&self) -> String {
        self.inner.from.clone()
    }
    /// Output query this stage produced. The next stage's `from_query`
    /// matches this; the final stage's `to_query` is what BM25 saw.
    #[getter]
    fn to_query(&self) -> String {
        self.inner.to.clone()
    }
    /// Surface forms this stage *matched* in the input.
    #[getter]
    fn matched(&self) -> Vec<String> {
        self.inner.matched.clone()
    }
    /// Surface forms this stage *added* to the query (Vocabulary).
    #[getter]
    fn added(&self) -> Vec<String> {
        self.inner.added.clone()
    }
    /// Surface forms this stage *removed* from the query (Stripper).
    #[getter]
    fn removed(&self) -> Vec<String> {
        self.inner.removed.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "RewriteRecord(stage={:?}, matched={:?}, added={:?}, removed={:?})",
            self.inner.stage, self.inner.matched, self.inner.added, self.inner.removed,
        )
    }
}

/// Compiled boilerplate-removal rewrite. Token-level matching using the
/// document's analyzer, so a single-token stripper key cannot accidentally
/// erase a substring inside a longer word (an `"of"` stripper does **not**
/// erase the `"of"` inside `"office"`).
///
/// ```python
/// stripper = redhop.Stripper([
///     "highlight", "the", "parts", "of", "this",
///     "contract", "related", "to",
/// ])
/// ```
///
/// Pass into the rewrite chain on `Document.context_with_rewrites(...)`.
#[pyclass(module = "redhop")]
struct Stripper {
    inner: redhop::Stripper,
}

#[pymethods]
impl Stripper {
    #[new]
    fn new(boilerplate: Vec<String>) -> Self {
        Self {
            inner: redhop::Stripper::new(&boilerplate),
        }
    }
    /// Apply the stripper to a query string outside the rewrite chain
    /// (useful for ad-hoc / pipeline-outside cases). Returns the
    /// rewritten string; the audit record is discarded — use the
    /// chain via `Document.context_with_rewrites(...)` if you want
    /// the trail in the Decision Report.
    fn apply(&self, query: &str) -> String {
        use redhop::QueryRewrite;
        self.inner.apply(query).text
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    fn __repr__(&self) -> String {
        format!("Stripper(n={})", self.inner.len())
    }
}

/// Compiled workload-curated equivalence classes (term → synonyms). Each
/// entry's key, when found in the query at token granularity, triggers
/// appending its synonyms to the rewritten query. With
/// `Vocabulary.bidirectional({...})`, any class member can be the trigger
/// and the others get appended (so PTO ↔ "paid time off" works in both
/// directions).
///
/// ```python
/// vocab = redhop.Vocabulary({
///     "change of control": ["merger", "successor", "acquisition"],
///     "non-compete":       ["restraint", "non-competition"],
/// })
/// # symmetric (PTO ↔ paid time off ↔ vacation):
/// pto = redhop.Vocabulary.bidirectional({"pto": ["paid time off", "vacation"]})
/// ```
#[pyclass(module = "redhop")]
struct Vocabulary {
    inner: redhop::Vocabulary,
}

#[pymethods]
impl Vocabulary {
    /// Asymmetric vocabulary: the first form of each entry is the only
    /// trigger; the rest are appended on match.
    #[new]
    fn new(entries: std::collections::HashMap<String, Vec<String>>) -> Self {
        let pairs: Vec<(String, Vec<String>)> = entries.into_iter().collect();
        let borrowed: Vec<(&str, Vec<&str>)> = pairs
            .iter()
            .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
            .collect();
        let final_refs: Vec<(&str, &[&str])> = borrowed
            .iter()
            .map(|(k, v)| (*k, v.as_slice()))
            .collect();
        Self {
            inner: redhop::Vocabulary::new(&final_refs),
        }
    }

    /// Symmetric (bidirectional) vocabulary: any class member can trigger;
    /// the other members get appended.
    #[staticmethod]
    fn bidirectional(entries: std::collections::HashMap<String, Vec<String>>) -> Self {
        let pairs: Vec<(String, Vec<String>)> = entries.into_iter().collect();
        let borrowed: Vec<(&str, Vec<&str>)> = pairs
            .iter()
            .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
            .collect();
        let final_refs: Vec<(&str, &[&str])> = borrowed
            .iter()
            .map(|(k, v)| (*k, v.as_slice()))
            .collect();
        Self {
            inner: redhop::Vocabulary::bidirectional(&final_refs),
        }
    }
    /// Apply the vocabulary to a query string outside the rewrite chain
    /// (useful for ad-hoc / pipeline-outside cases). Returns the
    /// rewritten string with synonyms appended; the audit record is
    /// discarded — use the chain via `Document.context_with_rewrites(...)`
    /// if you want the trail in the Decision Report.
    fn apply(&self, query: &str) -> String {
        use redhop::QueryRewrite;
        self.inner.apply(query).text
    }
    /// Chunk-side enrichment — the symmetric to `apply`. Same compiled
    /// vocabulary, applied at ingest time to a chunk's text so opaque
    /// coded units (column names, error codes, API symbols, defined
    /// terms) become matchable for natural-language queries that don't
    /// share surface forms with them.
    ///
    /// Returns `(enriched_text, RewriteRecord)` so the caller can
    /// collect the per-chunk audit trail. Use pattern:
    ///
    /// ```python
    /// vocab = redhop.Vocabulary({
    ///     "usrSvc":  ["user service", "signup", "account creation"],
    ///     "calcAmt": ["calculate amount", "billing total"],
    /// })
    /// enriched_chunks = []
    /// audit = []
    /// for chunk in raw_chunks:
    ///     text, rec = vocab.enrich(chunk)
    ///     enriched_chunks.append(text)
    ///     if rec.matched:
    ///         audit.append(rec)
    /// doc = redhop.Document.from_chunks(enriched_chunks)
    /// ```
    ///
    /// **When this earns its keep.** `value ∝ shortness × opacity ×
    /// dictionary-exists`. Schema columns, API symbols, error codes,
    /// defined contract terms, clinical abbreviations — all extreme
    /// cases. Long descriptive prose is redundant. Bolting the same
    /// boilerplate onto every chunk re-creates the low-IDF dilution
    /// from CUAD_PRF_NULL. See `docs/findings/VOCABULARY_ENRICH.md`.
    fn enrich(&self, chunk: &str) -> (String, RewriteRecord) {
        let r = self.inner.enrich(chunk);
        (r.text, RewriteRecord { inner: r.record })
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    fn __repr__(&self) -> String {
        format!("Vocabulary(n={})", self.inner.len())
    }
}

/// Owned Rust copy of the underlying rewrite so we can hand out
/// `&dyn QueryRewrite` references without juggling pyo3 borrow guards.
/// Stripper and Vocabulary are cheap to clone (Arc<Analyzer> + Vec<String>).
enum OwnedRewrite {
    Stripper(redhop::Stripper),
    Vocabulary(redhop::Vocabulary),
}

fn extract_rewrites(rewrites: &Bound<'_, PyAny>) -> PyResult<Vec<OwnedRewrite>> {
    let mut out = Vec::new();
    for item in rewrites.try_iter()? {
        let item = item?;
        if let Ok(s) = item.extract::<PyRef<'_, Stripper>>() {
            out.push(OwnedRewrite::Stripper(s.inner.clone()));
        } else if let Ok(v) = item.extract::<PyRef<'_, Vocabulary>>() {
            out.push(OwnedRewrite::Vocabulary(v.inner.clone()));
        } else {
            return Err(PyValueError::new_err(
                "rewrites entry must be a Stripper or Vocabulary instance",
            ));
        }
    }
    Ok(out)
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
/// Returns a [`QuerySetReport`]; read `.is_templated`, `.boilerplate_terms`,
/// `.suggested_action`. See `docs/findings/QUERY_SET_ANALYZER.md`.
#[pyfunction]
fn analyze_query_set(queries: Vec<String>) -> QuerySetReport {
    QuerySetReport {
        inner: redhop::analyze_query_set(&queries),
    }
}

// ─── In-process evaluation (no LLM judge) ───────────────────────────────────
// Backed by `redhop::evaluate`. The Rust enum `EvalGold` is hidden behind
// idiomatic Python kwargs (`gold_chunks=`, `gold_answer=`) — both optional,
// any combination supported. See `docs/findings/EVALUATE_API.md` for the
// design rationale ("refraction, not independent measurement").

/// In-process evaluation report for one (query, BuiltContext) pair.
///
/// Self-eval fields are always populated; gold-relative fields are `None`
/// unless the corresponding `gold_*` kwarg was supplied to `evaluate`. The
/// composite `overall` blends whichever fields are present. Read fields off
/// the object as attributes.
#[pyclass(module = "redhop")]
#[derive(Clone)]
struct EvalReport {
    inner: redhop::EvalReport,
}

#[pymethods]
impl EvalReport {
    /// Fraction of gold chunks that survived assembly. `None` unless
    /// `gold_chunks=` was supplied.
    #[getter]
    fn context_recall(&self) -> Option<f32> {
        self.inner.context_recall
    }
    /// Fraction of selected chunks that were gold. `None` unless
    /// `gold_chunks=` was supplied.
    #[getter]
    fn context_precision(&self) -> Option<f32> {
        self.inner.context_precision
    }
    /// Fraction of stemmed content terms in the gold answer that appear in
    /// the assembled context. `None` unless `gold_answer=` was supplied.
    #[getter]
    fn answer_token_recall(&self) -> Option<f32> {
        self.inner.answer_token_recall
    }
    /// Mean grounding score over selected chunks, in `[0, 1]`. Same scorer
    /// the runtime uses for `ContextStrategy::DistractorFiltered`.
    #[getter]
    fn mean_grounding(&self) -> f32 {
        self.inner.mean_grounding
    }
    /// Fraction of context tokens that are query-relevant.
    #[getter]
    fn evidence_density(&self) -> f32 {
        self.inner.evidence_density
    }
    /// Fraction of input evidence that made it through assembly.
    #[getter]
    fn retained_evidence_ratio(&self) -> f32 {
        self.inner.retained_evidence_ratio
    }
    /// Number of bridge passages the reasoning-preserving rescue saved.
    #[getter]
    fn second_hop_rescues(&self) -> usize {
        self.inner.second_hop_rescues
    }
    /// True when every selected chunk is at-or-below the grounding ceiling
    /// — i.e. the retrieval itself was weak.
    #[getter]
    fn low_confidence(&self) -> bool {
        self.inner.low_confidence
    }
    /// Tokens spent on below-bar chunks.
    #[getter]
    fn estimated_waste_tokens(&self) -> usize {
        self.inner.estimated_waste_tokens
    }
    /// Composite score in `[0, 1]` blending whichever fields above are
    /// available. Use as the headline; use individual fields to debug.
    #[getter]
    fn overall(&self) -> f32 {
        self.inner.overall
    }
    fn __repr__(&self) -> String {
        format!(
            "EvalReport(overall={:.3}, mean_grounding={:.3}, recall={}, precision={}, answer_recall={}, low_confidence={})",
            self.inner.overall,
            self.inner.mean_grounding,
            self.inner
                .context_recall
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "None".into()),
            self.inner
                .context_precision
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "None".into()),
            self.inner
                .answer_token_recall
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "None".into()),
            self.inner.low_confidence,
        )
    }
}

/// Evaluate an assembled `BuiltContext` against optional ground truth.
///
/// Self-eval (mean_grounding, evidence_density, second_hop_rescues,
/// low_confidence, …) is always populated. Pass `gold_chunks=` to unlock
/// `context_recall` / `context_precision`; pass `gold_answer=` to unlock
/// `answer_token_recall`. Both optional, any combination supported.
///
/// Zero LLM calls — every metric is computed from the same primitives the
/// runtime uses to make its Decision Report. See `EVALUATE_API.md` for
/// the "refraction not independent measurement" design choice.
#[pyfunction]
#[pyo3(signature = (query, context, *, gold_chunks=None, gold_answer=None))]
fn evaluate(
    query: &str,
    context: &BuiltContext,
    gold_chunks: Option<Vec<String>>,
    gold_answer: Option<&str>,
) -> EvalReport {
    let q = Query::new(query);
    // Borrow gold_chunks as &[&str] so it matches the redhop::EvalGold borrowed shape.
    let chunk_refs: Option<Vec<&str>> = gold_chunks
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());
    let gold = match (chunk_refs.as_deref(), gold_answer) {
        (None, None) => redhop::EvalGold::None,
        (Some(c), None) => redhop::EvalGold::Chunks(c),
        (None, Some(a)) => redhop::EvalGold::Answer(a),
        (Some(c), Some(a)) => redhop::EvalGold::Both {
            gold_chunk_ids: c,
            gold_answer: a,
        },
    };
    EvalReport {
        inner: redhop::evaluate(&q, &context.inner, gold),
    }
}

#[pymodule]
fn _redhop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<BuiltContext>()?;
    m.add_class::<PyChunk>()?;
    m.add_class::<ContextReport>()?;
    m.add_class::<Document>()?;
    m.add_class::<QuerySetReport>()?;
    m.add_class::<EvalReport>()?;
    m.add_class::<RewriteRecord>()?;
    m.add_class::<Stripper>()?;
    m.add_class::<Vocabulary>()?;
    m.add_function(wrap_pyfunction!(build_context, m)?)?;
    m.add_function(wrap_pyfunction!(filter_context, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_context, m)?)?;
    m.add_function(wrap_pyfunction!(context_economics, m)?)?;
    m.add_function(wrap_pyfunction!(grounding_score, m)?)?;
    m.add_function(wrap_pyfunction!(link_strength, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_query_set, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    Ok(())
}
