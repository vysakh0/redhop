//! JSON / JSONL rendering of traces.

use crate::trace::RetrievalTrace;

/// Render a single trace as a one-line JSON object (JSONL-ready).
pub fn render_line(trace: &RetrievalTrace) -> String {
    trace.to_json()
}

/// Render many traces as a JSONL string (one object per line).
pub fn render_jsonl<'a, I: IntoIterator<Item = &'a RetrievalTrace>>(traces: I) -> String {
    let mut s = String::new();
    for t in traces {
        s.push_str(&t.to_json());
        s.push('\n');
    }
    s
}

/// Render many traces as a pretty-printed JSON array (human inspection).
pub fn render_pretty_array(traces: &[RetrievalTrace]) -> String {
    serde_json::to_string_pretty(traces).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::RetrievalTrace;
    use std::collections::BTreeMap;

    fn trace(q: &str) -> RetrievalTrace {
        RetrievalTrace {
            query: q.into(),
            final_regime: Some("easy".into()),
            regime_probabilities: BTreeMap::new(),
            iterations: vec![],
            final_candidate_count: 1,
            final_top_k: 4,
            final_reranker_level: "none".into(),
            terminal_action: Some("stop".into()),
            abstained: false,
            intervened: false,
            total_latency_ms: 0,
            total_retrieval_calls: 1,
            total_rerank_calls: 0,
        }
    }

    #[test]
    fn jsonl_has_one_line_per_trace() {
        let traces = vec![trace("a"), trace("b"), trace("c")];
        let out = render_jsonl(&traces);
        assert_eq!(out.lines().count(), 3);
        for line in out.lines() {
            let _: RetrievalTrace = serde_json::from_str(line).unwrap();
        }
    }
}
