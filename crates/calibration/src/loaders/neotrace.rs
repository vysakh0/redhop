//! NeoTrace JSONL loader.
//!
//! NeoTrace is the canonical interchange format with the Python RedHop
//! research repo. See `docs/NEOTRACE_SCHEMA.md` for the full
//! specification.
//!
//! ## Why this loader is structured the way it is
//!
//! Other loaders in this module (`hotpotqa`, `musique`) consume raw
//! benchmark JSON, build a `LabeledCorpus`, and let the calibration
//! pipeline run retrieval against it. NeoTrace is different: each
//! record already carries the *outcome* of an external retrieval run —
//! the retrieved indices, the metrics, the answer text. There is no
//! retrieval for the Rust side to do; the loader produces
//! [`QueryOutcome`]s **directly** from the recorded fields, which the
//! sweep/analysis utilities then consume as if Rust had run them.
//!
//! Two import modes are therefore offered:
//!
//! - [`load_corpus`] — extract a [`LabeledCorpus`] only (questions +
//!   true regime + gold chunk ids). Use this when you want to re-run
//!   retrieval through the Rust pipeline against the same queries.
//! - [`load_outcomes`] — extract [`QueryOutcome`]s. Use this to plug
//!   the Python lab's measurements straight into the Rust analysis
//!   tools (confusion matrix, regret, threshold stability, reliability
//!   diagrams) without rerunning anything.
//!
//! ## Gold chunk ids
//!
//! NeoTrace stores gold passages as integer indices into a per-item
//! paragraph pool (HotpotQA/MuSiQue style). The loader synthesizes
//! string chunk ids of the form `"<item_id>::para::<idx>"` so they
//! round-trip into [`ChunkId`] without ambiguity.
//!
//! ## Schema version
//!
//! The loader accepts `schema_version: "neotrace/1"` exactly. Other
//! version strings are a hard error — there is no silent upgrade path.

use std::collections::BTreeMap;

use redhop_core::{ChunkId, Document, Error, Result, RetrievalRegime};
use serde::{Deserialize, Serialize};

use crate::dataset::{LabeledCorpus, LabeledQuery};
use crate::runner::{ActionTraceEntry, QueryOutcome};

const ACCEPTED_SCHEMA_VERSION: &str = "neotrace/1";

/// One NeoTrace record. Optional fields use `Option` so the loader
/// tolerates the natural sparsity of the source files (some come from
/// experiments that don't record `gold_answer`, some don't record
/// `retrieved`, etc.).
// Fields mirror the `neotrace/1` wire format column-for-column; the schema is
// documented in docs/NEOTRACE_SCHEMA.md rather than repeated per field.
#[allow(missing_docs)]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NeoTraceRecord {
    /// Schema version string; must equal `"neotrace/1"`.
    pub schema_version: String,
    /// Source experiment id.
    #[serde(default)]
    pub run_id: Option<String>,
    /// Stable cross-method item id.
    #[serde(default)]
    pub item_id: Option<String>,
    /// Retrieval method code (`cosine`, `bm25`, `learned`, …). See
    /// `docs/NEOTRACE_SCHEMA.md` for the canonical set.
    #[serde(default)]
    pub method: Option<String>,
    /// Generator model id.
    #[serde(default)]
    pub model: Option<String>,

    /// Question text.
    pub question: String,
    /// Gold answer string when available.
    #[serde(default)]
    pub gold_answer: Option<String>,
    /// Gold paragraph indices (HotpotQA / MuSiQue style).
    #[serde(default)]
    pub gold_para: Option<Vec<u32>>,
    /// Gold chunk ids when the source corpus uses string ids.
    #[serde(default)]
    pub gold_chunk_ids: Option<Vec<String>>,
    /// Source document id (per-doc benchmarks).
    #[serde(default)]
    pub doc: Option<String>,
    /// Question kind tag (`bridge`/`comparison`, or in-house `A`/`B`/`C`).
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    /// HotpotQA difficulty level.
    #[serde(default)]
    pub level: Option<String>,
    /// Chunks in this example's pool.
    #[serde(default)]
    pub n_chunks: Option<u32>,

    /// Retrieved indices into the per-item pool.
    #[serde(default)]
    pub retrieved: Option<Vec<u32>>,
    /// Retrieved chunk ids when applicable.
    #[serde(default)]
    pub retrieved_chunk_ids: Option<Vec<String>>,
    /// `|gold ∩ retrieved| / |gold|`.
    #[serde(default)]
    pub retrieval_recall: Option<f32>,
    /// Top-k used during retrieval.
    #[serde(default)]
    pub top_k: Option<u32>,

    // Evidence-quality metrics (the post-pivot column set).
    #[serde(default)]
    pub continuity: Option<f32>,
    #[serde(default)]
    pub answer_span_density: Option<f32>,
    #[serde(default)]
    pub answer_bearing_fraction: Option<f32>,
    #[serde(default)]
    pub distractor_ratio: Option<f32>,
    #[serde(default)]
    pub query_overlap: Option<f32>,
    #[serde(default)]
    pub entity_overlap: Option<f32>,
    #[serde(default)]
    pub purity: Option<f32>,

    // Generation outcome.
    #[serde(default)]
    pub answer: Option<String>,
    #[serde(default)]
    pub ans_similarity: Option<f32>,
    #[serde(default)]
    pub ans_kw_recall: Option<f32>,

    // Regime + judge.
    #[serde(default)]
    pub true_regime: Option<String>,
    #[serde(default)]
    pub predicted_regime: Option<String>,
    #[serde(default)]
    pub predicted_regime_p: Option<f32>,
    #[serde(default)]
    pub judge_model: Option<String>,
    #[serde(default)]
    pub judge_score: Option<f32>,
    #[serde(default)]
    pub judge_preferred: Option<String>,
    #[serde(default)]
    pub judge_reason: Option<String>,

    /// Optional action trace from a prior adaptive run.
    #[serde(default)]
    pub action_trace: Option<Vec<NeoTraceAction>>,

    /// Whether an adaptive controller intervened on this record.
    #[serde(default)]
    pub intervened: Option<bool>,

    /// End-to-end latency.
    #[serde(default)]
    pub latency_ms: Option<u64>,
}

/// One element of [`NeoTraceRecord::action_trace`].
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NeoTraceAction {
    /// Action code (`stop`, `expand_top_k`, …).
    pub action: String,
    /// Iteration the action ran on.
    #[serde(default)]
    pub iteration: u32,
    /// Policy's expected gain.
    #[serde(default)]
    pub expected_gain: f32,
    /// Measured gain.
    #[serde(default)]
    pub actual_gain: Option<f32>,
    /// Latency.
    #[serde(default)]
    pub latency_ms: u64,
    /// Retrieval calls.
    #[serde(default)]
    pub retrieval_calls: u32,
    /// Reranker calls.
    #[serde(default)]
    pub rerank_calls: u32,
    /// Rationale.
    #[serde(default)]
    pub rationale: String,
}

/// Parse a NeoTrace JSONL string into one record per non-blank line.
pub fn parse_jsonl(s: &str) -> Result<Vec<NeoTraceRecord>> {
    let mut out = Vec::new();
    for (i, line) in s.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let r: NeoTraceRecord = serde_json::from_str(line)
            .map_err(|e| Error::msg(format!("neotrace line {}: {e}", i + 1)))?;
        if r.schema_version != ACCEPTED_SCHEMA_VERSION {
            return Err(Error::msg(format!(
                "neotrace line {}: unsupported schema_version {:?} (expected {:?})",
                i + 1,
                r.schema_version,
                ACCEPTED_SCHEMA_VERSION
            )));
        }
        out.push(r);
    }
    Ok(out)
}

/// Parse a NeoTrace file.
pub fn parse_path(path: impl AsRef<std::path::Path>) -> Result<Vec<NeoTraceRecord>> {
    let s = std::fs::read_to_string(path)?;
    parse_jsonl(&s)
}

// ─────────────────────────────────────────────────────────────────────
// LabeledCorpus extraction
// ─────────────────────────────────────────────────────────────────────

/// Build a [`LabeledCorpus`] from a list of NeoTrace records.
///
/// Each unique `item_id` (or composite `(doc, question)` for files that
/// lack one) becomes one [`LabeledQuery`]. The per-item paragraph pool
/// is encoded as a single synthetic [`Document`] with `source =
/// item_id` and `text` left empty — the corpus is a *label* container,
/// not an index. Callers that want to re-run retrieval through the
/// Rust pipeline must materialize document text from elsewhere (e.g.
/// the raw HotpotQA / MuSiQue context fields). The synthetic chunk ids
/// from `gold_para` are emitted as `"<item_id>::para::<idx>"`, matching
/// `parse_synthetic_chunk_id` below.
///
/// `regime_override` lets callers replace the per-record `true_regime`
/// with a custom mapping (e.g. project-specific judge labels). Pass
/// `None` to use whatever NeoTrace recorded.
pub fn load_corpus<F>(
    records: &[NeoTraceRecord],
    mut regime_override: Option<F>,
) -> Result<LabeledCorpus>
where
    F: FnMut(&NeoTraceRecord) -> Option<RetrievalRegime>,
{
    let mut seen: BTreeMap<String, LabeledQuery> = BTreeMap::new();
    let mut seen_docs: BTreeMap<String, ()> = BTreeMap::new();
    for r in records {
        let id = resolve_item_id(r);
        let regime = regime_override
            .as_mut()
            .and_then(|f| f(r))
            .or_else(|| r.true_regime.as_deref().and_then(parse_regime))
            .unwrap_or(RetrievalRegime::Easy);
        let q = seen.entry(id.clone()).or_insert_with(|| {
            let mut lq = LabeledQuery::new(id.clone(), r.question.clone(), regime);
            lq.gold_chunk_ids = synthetic_gold(r);
            lq
        });
        // If a later record provides a better regime label (the first
        // one was `Easy` from the fallback above, but a subsequent
        // record carries an explicit `true_regime`), upgrade it.
        if q.true_regime == RetrievalRegime::Easy && r.true_regime.is_some() {
            q.true_regime = regime;
        }
        seen_docs.insert(id, ());
    }
    let docs: Vec<Document> = seen_docs
        .into_keys()
        .map(|id| Document::new(id, String::new()))
        .collect();
    Ok(LabeledCorpus {
        docs,
        queries: seen.into_values().collect(),
    })
}

fn synthetic_gold(r: &NeoTraceRecord) -> Vec<ChunkId> {
    // Prefer string ids when present; fall back to constructing them
    // from `gold_para`.
    if let Some(ids) = &r.gold_chunk_ids {
        return ids.iter().cloned().map(ChunkId::new).collect();
    }
    let item_id = resolve_item_id(r);
    if let Some(paras) = &r.gold_para {
        return paras
            .iter()
            .map(|p| ChunkId::new(format!("{item_id}::para::{p}")))
            .collect();
    }
    Vec::new()
}

fn resolve_item_id(r: &NeoTraceRecord) -> String {
    if let Some(id) = &r.item_id {
        return id.clone();
    }
    // Composite key for evidence/learned-style records.
    let doc = r.doc.clone().unwrap_or_default();
    let h = fnv64(r.question.as_bytes());
    format!("{doc}::{:08x}", h as u32)
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn parse_regime(code: &str) -> Option<RetrievalRegime> {
    for r in RetrievalRegime::all() {
        if r.code() == code {
            return Some(*r);
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────
// QueryOutcome extraction
// ─────────────────────────────────────────────────────────────────────

/// Build [`QueryOutcome`]s directly from NeoTrace records.
///
/// This is the entry point that bypasses Rust-side retrieval entirely:
/// the calibration analyses (confusion matrix, regret, threshold
/// stability, reliability diagrams) run against measurements produced
/// by the Python lab.
///
/// `static_method` is the method code (e.g. `"cosine"`) whose
/// `retrieval_recall` is treated as the "static" baseline for
/// `recall_lift` computations. Records for other methods become the
/// "adaptive" comparison. Records whose method matches neither
/// `static_method` nor `adaptive_method` are passed through with
/// `recall_lift = 0` (gold_recall_static = gold_recall_adaptive).
///
/// `adaptive_method` may be `None` in which case every non-static
/// record becomes its own adaptive comparison against the matching
/// static record (paired by `item_id`).
pub fn load_outcomes(
    records: &[NeoTraceRecord],
    static_method: &str,
    adaptive_method: Option<&str>,
) -> Result<Vec<QueryOutcome>> {
    // Group records by item_id so we can pair static vs adaptive.
    let mut grouped: BTreeMap<String, Vec<&NeoTraceRecord>> = BTreeMap::new();
    for r in records {
        grouped.entry(resolve_item_id(r)).or_default().push(r);
    }
    let mut out = Vec::new();
    for (item_id, items) in grouped {
        let static_rec = items
            .iter()
            .find(|r| r.method.as_deref() == Some(static_method))
            .copied();
        let recall_static = static_rec.and_then(|r| r.retrieval_recall).unwrap_or(0.0);
        for r in &items {
            let method_code = r.method.as_deref().unwrap_or("");
            if method_code == static_method {
                // Emit the static record itself as a no-op outcome so
                // the confusion matrix sees its predicted regime.
                out.push(to_outcome(
                    &item_id,
                    r,
                    recall_static,
                    recall_static,
                    /* intervened */ false,
                ));
                continue;
            }
            if let Some(am) = adaptive_method {
                if method_code != am {
                    continue;
                }
            }
            let recall_adaptive = r.retrieval_recall.unwrap_or(recall_static);
            out.push(to_outcome(
                &item_id,
                r,
                recall_static,
                recall_adaptive,
                /* intervened */ true,
            ));
        }
    }
    Ok(out)
}

fn to_outcome(
    item_id: &str,
    r: &NeoTraceRecord,
    recall_static: f32,
    recall_adaptive: f32,
    intervened: bool,
) -> QueryOutcome {
    let true_regime = r
        .true_regime
        .as_deref()
        .and_then(parse_regime)
        .unwrap_or(RetrievalRegime::Easy);
    let predicted_regime = r.predicted_regime.as_deref().and_then(parse_regime);
    let action_trace: Vec<ActionTraceEntry> = r
        .action_trace
        .as_ref()
        .map(|v| {
            v.iter()
                .map(|a| ActionTraceEntry {
                    action: a.action.clone(),
                    expected_gain: a.expected_gain,
                    actual_gain: a.actual_gain,
                })
                .collect()
        })
        .unwrap_or_default();
    let latency_ms = r
        .latency_ms
        .or_else(|| {
            r.action_trace
                .as_ref()
                .map(|v| v.iter().map(|a| a.latency_ms).sum())
        })
        .unwrap_or(0);
    let (escalations, expansions, rerank_calls, retrieval_calls) = r
        .action_trace
        .as_ref()
        .map(|v| {
            let mut e = 0;
            let mut x = 0;
            let mut rrk = 0;
            let mut rtv = 0;
            for a in v {
                rrk += a.rerank_calls;
                rtv += a.retrieval_calls;
                match a.action.as_str() {
                    "escalate_reranker" => e += 1,
                    "expand_top_k" => x += 1,
                    _ => {}
                }
            }
            (e, x, rrk, rtv)
        })
        .unwrap_or((0, 0, 0, 0));

    let sum_actual_gain = action_trace
        .iter()
        .filter_map(|a| a.actual_gain)
        .sum::<f32>();

    QueryOutcome {
        query_id: item_id.to_string(),
        true_regime,
        predicted_regime,
        predicted_regime_p: r.predicted_regime_p,
        true_regime_p: None,
        gold_recall_static: recall_static,
        gold_recall_adaptive: recall_adaptive,
        recall_lift: recall_adaptive - recall_static,
        intervened: intervened || r.intervened.unwrap_or(false),
        abstained: false,
        escalations,
        expansions,
        latency_ms_adaptive: latency_ms,
        retrieval_calls_adaptive: retrieval_calls.max(1),
        rerank_calls_adaptive: rerank_calls,
        sum_actual_gain,
        final_reranker_level: redhop_core::RerankerLevel::None,
        action_trace,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny in-memory fixture mirroring the exporter output shape.
    const MINI: &str = r#"{"schema_version":"neotrace/1","run_id":"hotpot_smoke","item_id":"q1","method":"cosine","model":"haiku","question":"Where was X born?","gold_answer":"London","gold_para":[2,5],"type":"bridge","level":"hard","retrieved":[2,5,9],"retrieval_recall":1.0,"top_k":4,"continuity":0.5,"answer_span_density":0.8,"distractor_ratio":0.2,"true_regime":"distractor_heavy"}
{"schema_version":"neotrace/1","run_id":"hotpot_smoke","item_id":"q1","method":"cross_encoder","model":"haiku","question":"Where was X born?","gold_answer":"London","gold_para":[2,5],"retrieved":[2,5,8],"retrieval_recall":1.0,"continuity":0.6,"answer_span_density":0.85,"distractor_ratio":0.15,"true_regime":"distractor_heavy"}
{"schema_version":"neotrace/1","run_id":"hotpot_smoke","item_id":"q2","method":"cosine","model":"haiku","question":"What color is the sky?","gold_answer":"blue","gold_para":[0],"retrieved":[0,3,7],"retrieval_recall":1.0,"true_regime":"easy"}
"#;

    #[test]
    fn parses_jsonl_with_schema_version_check() {
        let recs = parse_jsonl(MINI).unwrap();
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].schema_version, "neotrace/1");
        assert_eq!(recs[1].method.as_deref(), Some("cross_encoder"));
    }

    #[test]
    fn rejects_unsupported_schema_version() {
        let bad = r#"{"schema_version":"neotrace/99","question":"q"}"#;
        let r = parse_jsonl(bad);
        assert!(r.is_err());
    }

    #[test]
    fn load_corpus_creates_one_query_per_unique_item_id() {
        let recs = parse_jsonl(MINI).unwrap();
        let corpus =
            load_corpus::<fn(&NeoTraceRecord) -> Option<RetrievalRegime>>(&recs, None).unwrap();
        assert_eq!(corpus.queries.len(), 2);
        // q1 is hard+bridge → distractor_heavy
        let q1 = corpus.queries.iter().find(|q| q.id == "q1").unwrap();
        assert_eq!(q1.true_regime, RetrievalRegime::DistractorHeavy);
        // q1 gold = [2, 5] → ChunkId("q1::para::2"), ChunkId("q1::para::5")
        assert_eq!(q1.gold_chunk_ids.len(), 2);
        assert_eq!(q1.gold_chunk_ids[0].as_str(), "q1::para::2");
        assert_eq!(q1.gold_chunk_ids[1].as_str(), "q1::para::5");
        // q2 is easy
        let q2 = corpus.queries.iter().find(|q| q.id == "q2").unwrap();
        assert_eq!(q2.true_regime, RetrievalRegime::Easy);
    }

    #[test]
    fn load_outcomes_pairs_static_vs_adaptive_by_item_id() {
        // Construct a record where adaptive (cross_encoder) has lower
        // recall than static (cosine) — should produce negative lift.
        let recs = parse_jsonl(MINI).unwrap();
        let outcomes = load_outcomes(&recs, "cosine", Some("cross_encoder")).unwrap();
        // 3 records → 2 cosine outcomes + 1 paired cross_encoder = 3 outcomes.
        assert_eq!(outcomes.len(), 3);
        let q1_adaptive = outcomes
            .iter()
            .find(|o| o.query_id == "q1" && o.intervened)
            .unwrap();
        // Both methods reported recall = 1.0; lift = 0.
        assert!(q1_adaptive.recall_lift.abs() < 1e-5);
        assert!(q1_adaptive.intervened);
        // Static record is also emitted for q1.
        let q1_static = outcomes
            .iter()
            .find(|o| o.query_id == "q1" && !o.intervened)
            .unwrap();
        assert_eq!(q1_static.gold_recall_static, q1_static.gold_recall_adaptive);
    }

    #[test]
    fn unknown_regime_falls_back_to_easy() {
        let r = r#"{"schema_version":"neotrace/1","item_id":"x","method":"cosine","question":"?","true_regime":"unknown_label"}"#;
        let recs = parse_jsonl(r).unwrap();
        let corpus =
            load_corpus::<fn(&NeoTraceRecord) -> Option<RetrievalRegime>>(&recs, None).unwrap();
        assert_eq!(corpus.queries[0].true_regime, RetrievalRegime::Easy);
    }

    #[test]
    fn regime_override_replaces_recorded_label() {
        let recs = parse_jsonl(MINI).unwrap();
        let force_sparse: fn(&NeoTraceRecord) -> Option<RetrievalRegime> =
            |_| Some(RetrievalRegime::Sparse);
        let corpus = load_corpus(&recs, Some(force_sparse)).unwrap();
        for q in &corpus.queries {
            assert_eq!(q.true_regime, RetrievalRegime::Sparse);
        }
    }
}
