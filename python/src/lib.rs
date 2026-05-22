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
    analyze_context as rh_analyze, build_context as rh_build, filter_context as rh_filter,
    context_economics as rh_economics, grounding_score as rh_grounding, link_strength as rh_link,
    ContextConfig, ContextReport as RhReport, ContextStrategy,
};
use redhop_core::{
    Chunk, ChunkId, Embedding, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenCount,
};

fn strategy_from_str(s: &str) -> PyResult<ContextStrategy> {
    Ok(match s {
        "raw_topk" => ContextStrategy::RawTopK,
        "distractor_filtered" => ContextStrategy::DistractorFiltered,
        "redundancy_pruned" => ContextStrategy::RedundancyPruned,
        "max_density" => ContextStrategy::MaxDensity,
        "reasoning_preserving" => ContextStrategy::ReasoningPreserving,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown strategy '{other}' (expected: raw_topk, distractor_filtered, \
                 redundancy_pruned, max_density, reasoning_preserving)"
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
    }
}

/// Convert one Python chunk (a str, or a dict with at least `text`) into a
/// `RetrievalResult`.
fn chunk_from_py(item: &Bound<'_, PyAny>, idx: usize) -> PyResult<RetrievalResult> {
    let (id, text, source, token_count, embedding, score) = if let Ok(s) =
        item.extract::<String>()
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
        score: Score { value: score.unwrap_or(1.0), method: RetrievalMethod::Dense },
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
       distractor_min_grounding=0.10, link_min_jaccard=0.12, redundancy_max_cosine=0.92))]
#[allow(clippy::too_many_arguments)]
fn build_context(
    query: &str,
    retrieved_chunks: &Bound<'_, PyAny>,
    strategy: Option<String>,
    token_budget: usize,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
    redundancy_max_cosine: f32,
) -> PyResult<BuiltContext> {
    let cfg = config(strategy, token_budget, distractor_min_grounding, link_min_jaccard, redundancy_max_cosine)?;
    let q = Query::new(query);
    let retrieved = chunks_from_py(retrieved_chunks)?;
    let before = rh_analyze(&q, &retrieved, &cfg);
    let ctx = rh_build(&q, &retrieved, &cfg);
    let rendered = ctx.report.render(Some(&before));
    Ok(BuiltContext {
        text: ctx.text(),
        chunks: ctx.chunks.iter().map(|c| c.text.clone()).collect(),
        report: ContextReport { inner: ctx.report, rendered },
    })
}

#[pyfunction]
#[pyo3(signature = (query, retrieved_chunks, strategy=None,
       distractor_min_grounding=0.10, link_min_jaccard=0.12, redundancy_max_cosine=0.92))]
fn filter_context(
    query: &str,
    retrieved_chunks: &Bound<'_, PyAny>,
    strategy: Option<String>,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
    redundancy_max_cosine: f32,
) -> PyResult<BuiltContext> {
    // filter = build with no budget truncation.
    let cfg = config(strategy, usize::MAX, distractor_min_grounding, link_min_jaccard, redundancy_max_cosine)?;
    let q = Query::new(query);
    let retrieved = chunks_from_py(retrieved_chunks)?;
    let before = rh_analyze(&q, &retrieved, &cfg);
    let ctx = rh_filter(&q, &retrieved, &cfg);
    let rendered = ctx.report.render(Some(&before));
    Ok(BuiltContext {
        text: ctx.text(),
        chunks: ctx.chunks.iter().map(|c| c.text.clone()).collect(),
        report: ContextReport { inner: ctx.report, rendered },
    })
}

#[pyfunction]
#[pyo3(signature = (query, retrieved_chunks, distractor_min_grounding=0.10, link_min_jaccard=0.12))]
fn analyze_context(
    query: &str,
    retrieved_chunks: &Bound<'_, PyAny>,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
) -> PyResult<ContextReport> {
    let cfg = config(None, usize::MAX, distractor_min_grounding, link_min_jaccard, 0.92)?;
    let q = Query::new(query);
    let retrieved = chunks_from_py(retrieved_chunks)?;
    let report = rh_analyze(&q, &retrieved, &cfg);
    let rendered = report.render(None);
    Ok(ContextReport { inner: report, rendered })
}

#[pyfunction]
#[pyo3(signature = (query, retrieved_chunks, distractor_min_grounding=0.10, link_min_jaccard=0.12))]
fn context_economics(
    query: &str,
    retrieved_chunks: &Bound<'_, PyAny>,
    distractor_min_grounding: f32,
    link_min_jaccard: f32,
) -> PyResult<String> {
    let cfg = config(None, usize::MAX, distractor_min_grounding, link_min_jaccard, 0.92)?;
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

#[pymodule]
fn _redhop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<BuiltContext>()?;
    m.add_class::<ContextReport>()?;
    m.add_function(wrap_pyfunction!(build_context, m)?)?;
    m.add_function(wrap_pyfunction!(filter_context, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_context, m)?)?;
    m.add_function(wrap_pyfunction!(context_economics, m)?)?;
    m.add_function(wrap_pyfunction!(grounding_score, m)?)?;
    m.add_function(wrap_pyfunction!(link_strength, m)?)?;
    Ok(())
}
