//! Generic JSONL loader for custom workloads.
//!
//! Provides an escape hatch for users whose data doesn't fit HotpotQA
//! or MuSiQue shape. Expected JSONL row shape:
//!
//! ```json
//! {
//!   "id": "...",
//!   "question": "...",
//!   "regime": "easy" | "saturated" | "distractor_heavy" | "ambiguous" | "sparse",
//!   "gold_chunk_ids": ["docA::sent::0", "docB::sent::2"],
//!   "documents": [
//!     {"source": "docA", "text": "..."},
//!     {"source": "docB", "text": "..."}
//!   ]
//! }
//! ```
//!
//! Documents in each row are merged into a single deduplicated corpus
//! pool keyed on `source`. Gold chunk ids are stored verbatim — the
//! loader does NOT chunk and re-resolve, which puts the user in
//! control of the labeling. If your gold ids reference chunks the
//! chunker doesn't actually produce, calibration will silently report
//! zero gold-recall for those queries.

use std::collections::BTreeMap;

use neorag_core::{ChunkId, Document, Error, Result, RetrievalRegime};
use serde::{Deserialize, Serialize};

use crate::dataset::{LabeledCorpus, LabeledQuery};

/// One row of a generic JSONL calibration file.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonlRow {
    /// Stable query id.
    pub id: String,
    /// Question text.
    pub question: String,
    /// Regime label, as a stable string code. See
    /// [`RetrievalRegime::code`].
    pub regime: String,
    /// Gold chunk ids — strings only, the loader wraps them in
    /// [`ChunkId`].
    #[serde(default)]
    pub gold_chunk_ids: Vec<String>,
    /// Documents associated with this query. Merged with other rows'
    /// documents (keyed by `source`).
    #[serde(default)]
    pub documents: Vec<JsonlDoc>,
}

/// One document row.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonlDoc {
    /// Stable document source identifier.
    pub source: String,
    /// Document text.
    pub text: String,
}

/// Parse JSONL from a string and assemble a [`LabeledCorpus`].
///
/// The loader is strict: unknown regime codes cause an error. Use
/// [`RetrievalRegime::code`] return values: `"easy"`, `"saturated"`,
/// `"distractor_heavy"`, `"ambiguous"`, `"sparse"`.
pub fn load_jsonl(s: &str) -> Result<LabeledCorpus> {
    let mut docs_by_source: BTreeMap<String, String> = BTreeMap::new();
    let mut queries = Vec::new();
    for (i, line) in s.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: JsonlRow = serde_json::from_str(line)
            .map_err(|e| Error::msg(format!("jsonl row {}: {e}", i + 1)))?;
        for d in row.documents {
            docs_by_source.entry(d.source).or_insert(d.text);
        }
        let regime = parse_regime(&row.regime).ok_or_else(|| {
            Error::msg(format!(
                "jsonl row {}: unknown regime code {:?}",
                i + 1,
                row.regime
            ))
        })?;
        let mut q = LabeledQuery::new(row.id, row.question, regime);
        q.gold_chunk_ids = row.gold_chunk_ids.into_iter().map(ChunkId::new).collect();
        queries.push(q);
    }
    let docs = docs_by_source
        .into_iter()
        .map(|(source, text)| Document::new(source, text))
        .collect();
    Ok(LabeledCorpus { docs, queries })
}

fn parse_regime(code: &str) -> Option<RetrievalRegime> {
    for r in RetrievalRegime::all() {
        if r.code() == code {
            return Some(*r);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI_JSONL: &str = r#"{"id": "q1", "question": "what?", "regime": "easy", "gold_chunk_ids": ["doc-a::sent::0"], "documents": [{"source": "doc-a", "text": "answer text"}]}
{"id": "q2", "question": "why?", "regime": "sparse", "documents": [{"source": "doc-a", "text": "answer text"}, {"source": "doc-b", "text": "more text"}]}
"#;

    #[test]
    fn parses_jsonl_and_dedupes_documents() {
        let c = load_jsonl(MINI_JSONL).unwrap();
        assert_eq!(c.queries.len(), 2);
        assert_eq!(c.docs.len(), 2); // doc-a appears in both rows but deduped
        assert_eq!(c.queries[0].true_regime, RetrievalRegime::Easy);
        assert_eq!(c.queries[1].true_regime, RetrievalRegime::Sparse);
    }

    #[test]
    fn unknown_regime_errors() {
        let bad = r#"{"id": "q", "question": "?", "regime": "frantic"}"#;
        assert!(load_jsonl(bad).is_err());
    }
}
