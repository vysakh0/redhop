//! Human-readable CLI rendering of a [`RetrievalTrace`].

use crate::trace::RetrievalTrace;

/// Render a single trace as an ASCII block.
pub fn render(trace: &RetrievalTrace) -> String {
    let mut s = String::new();
    s.push_str(&format!("query: {}\n", trace.query));
    s.push_str(&format!(
        "regime: {}  (intervened={}, abstained={})\n",
        trace.final_regime.as_deref().unwrap_or("none"),
        trace.intervened,
        trace.abstained
    ));

    // Regime distribution as a compact inline bar set.
    if !trace.regime_probabilities.is_empty() {
        s.push_str("regime distribution:\n");
        // Stable display order.
        let order = ["easy", "saturated", "distractor_heavy", "ambiguous", "sparse"];
        for code in order {
            if let Some(&p) = trace.regime_probabilities.get(code) {
                s.push_str(&format!("  {:<18} {:.3}  {}\n", code, p, bar(p, 20)));
            }
        }
    }

    s.push_str(&format!(
        "final: {} candidates, top_k={}, reranker={}\n",
        trace.final_candidate_count, trace.final_top_k, trace.final_reranker_level
    ));
    s.push_str(&format!(
        "cost: {} retrieval calls, {} rerank calls, {} ms total\n",
        trace.total_retrieval_calls, trace.total_rerank_calls, trace.total_latency_ms
    ));

    s.push_str("actions:\n");
    for it in &trace.iterations {
        s.push_str(&format!(
            "  [iter {}] {:<18} expected={:+.3} actual={}\n",
            it.iteration,
            it.action,
            it.expected_gain,
            it.actual_gain
                .map(|g| format!("{g:+.3}"))
                .unwrap_or_else(|| "n/a".to_string())
        ));
        s.push_str(&format!("       why: {}\n", it.rationale));
        // Show the diagnostics the decision was based on, when present.
        let mut diag_bits = Vec::new();
        if let Some(v) = it.lexical_grounding {
            diag_bits.push(format!("lex_g={v:.2}"));
        }
        if let Some(v) = it.semantic_grounding {
            diag_bits.push(format!("sem_g={v:.2}"));
        }
        if let Some(v) = it.distractor_ratio {
            diag_bits.push(format!("distract={v:.2}"));
        }
        if let Some(v) = it.semantic_redundancy {
            diag_bits.push(format!("sem_redund={v:.2}"));
        }
        if !diag_bits.is_empty() {
            s.push_str(&format!("       diagnostics: {}\n", diag_bits.join(" ")));
        }
    }
    s
}

fn bar(value: f32, width: usize) -> String {
    let v = value.clamp(0.0, 1.0);
    let filled = (v * width as f32).round() as usize;
    let mut out = String::with_capacity(width + 2);
    out.push('|');
    for i in 0..width {
        out.push(if i < filled { '#' } else { ' ' });
    }
    out.push('|');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::TraceIteration;
    use std::collections::BTreeMap;

    fn trace() -> RetrievalTrace {
        let mut probs = BTreeMap::new();
        probs.insert("easy".to_string(), 0.6);
        probs.insert("ambiguous".to_string(), 0.2);
        RetrievalTrace {
            query: "test query".into(),
            final_regime: Some("easy".into()),
            regime_probabilities: probs,
            iterations: vec![TraceIteration {
                iteration: 0,
                action: "stop".into(),
                rationale: "p(Easy)=0.60 ≥ 0.40".into(),
                expected_gain: 0.0,
                actual_gain: None,
                latency_ms: 0,
                retrieval_calls: 0,
                rerank_calls: 0,
                chunks_delta: 0,
                lexical_grounding: Some(0.9),
                semantic_grounding: None,
                distractor_ratio: Some(0.0),
                semantic_redundancy: None,
            }],
            final_candidate_count: 2,
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
    fn renders_query_and_action() {
        let out = render(&trace());
        assert!(out.contains("query: test query"));
        assert!(out.contains("regime: easy"));
        assert!(out.contains("stop"));
        assert!(out.contains("lex_g=0.90"));
    }
}
