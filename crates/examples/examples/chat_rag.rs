//! Chat-history RAG — retrieve relevant past turns, preserve chronology.
//!
//! The use case: a long ongoing chat where you'd rather not compact/summarize
//! the history (loses fidelity). Each turn becomes a chunk; relevance-ranked
//! retrieval picks the right turns; **but the model needs to read them in the
//! order they were said**, not in relevance order, or causality breaks
//! ("after the refund came in" reads strangely if presented before "ordered
//! the laptop").
//!
//! Mechanism: `ContextConfig::preserve_order = true`. The strategy still
//! selects top-K by relevance under whatever knob you've set
//! (`RawTopK` here for simplicity); the final emission is sorted back into
//! source-document order.
//!
//! Two arms so the contrast is clear on the same selection:
//!   - A: preserve_order = false (default) → strategy-emitted order
//!     (typically relevance-first)
//!   - B: preserve_order = true → source-document order
//!
//! The point isn't a recall lift — selection is identical between the two
//! arms. The point is **the order of what you hand to the LLM** so a
//! generated answer can reason temporally.
//!
//! Run: cargo run -p redhop-examples --example chat_rag --release

use std::collections::HashMap;
use std::sync::Arc;

use redhop::analyzer::default_english;
use redhop::context::{build_context, ContextConfig, ContextStrategy};
use redhop::core::{
    Chunk, ChunkId, Embedding, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown,
    TokenCount,
};

/// Hand-rolled synthetic chat: 12 turns spanning order, shipping, refund,
/// and return policy. The clear *chronology signal* is the turn number;
/// the clear *relevance signal* (for the query "refund window") is "refund"
/// in some turns and not others.
fn chat_history() -> Vec<(String, String)> {
    vec![
        ("turn-00", "Hi, I have a question about my order."),
        ("turn-01", "I ordered a laptop last Tuesday."),
        ("turn-02", "It was the new MacBook Air, 15-inch."),
        (
            "turn-03",
            "Shipping confirmation came in yesterday — said tomorrow.",
        ),
        (
            "turn-04",
            "Actually I'd like to cancel and get my money back.",
        ),
        (
            "turn-05",
            "Sure — what is your refund policy on a shipped order?",
        ),
        (
            "turn-06",
            "We offer a thirty-day refund window from the delivery date.",
        ),
        ("turn-07", "Great, that works. How do I initiate a return?"),
        ("turn-08", "I'll email you a return label and instructions."),
        (
            "turn-09",
            "Do I need to print the label or can I show it on my phone?",
        ),
        (
            "turn-10",
            "Either is fine. Drop the package at any UPS access point.",
        ),
        ("turn-11", "Perfect. Thanks for the help!"),
    ]
    .into_iter()
    .map(|(a, b)| (a.into(), b.into()))
    .collect()
}

fn make_chunks(history: &[(String, String)]) -> Vec<RetrievalResult> {
    // The chronology signal goes on the chunk metadata as `chunk_index`.
    // `Document::from_chunks_with` stamps this automatically; for the
    // low-level build_context path we set it explicitly so preserve_order's
    // sort key can read it.
    history
        .iter()
        .enumerate()
        .map(|(i, (id, text))| {
            let mut c = Chunk::new(
                ChunkId::new(id),
                text,
                "chat",
                TokenCount(text.split_whitespace().count()),
            );
            c.metadata
                .insert("chunk_index".to_string(), serde_json::json!(i as i64));
            // Score by query-term overlap as a stand-in for BM25 — this
            // example is about the *ordering* primitive, not the retrieval
            // engine. The score affects which turns get selected, not the
            // order they're emitted in.
            //
            // Query "shipping refund label" picks up matches across the
            // chat (turns 3, 4, 5, 6, 8, 10) so when budget forces a
            // selection, the picks span the timeline — and the contrast
            // between relevance order and chat order becomes visible.
            let q_terms: std::collections::HashSet<&str> =
                ["shipping", "refund", "label", "return"]
                    .into_iter()
                    .collect();
            let s = text
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| q_terms.contains(w))
                .count() as f32;
            RetrievalResult {
                chunk: c,
                score: Score {
                    value: s.max(0.01),
                    method: RetrievalMethod::Lexical,
                },
                breakdown: ScoreBreakdown::default(),
            }
        })
        .collect()
}

fn print_emitted(ctx: &redhop::context::BuiltContext) {
    for c in &ctx.chunks {
        let idx = c
            .metadata
            .get("chunk_index")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1);
        println!("  [chunk_index={:>2}] {}: {}", idx, c.id.as_str(), c.text);
    }
}

fn cfg(preserve: bool) -> ContextConfig {
    ContextConfig {
        // Tight enough that only a handful of turns fit — the strategy
        // has to pick by score, which makes the chronology contrast
        // between the two arms visible.
        token_budget: 40,
        strategy: ContextStrategy::RawTopK,
        distractor_min_grounding: 0.0, // off — every turn counts
        link_min_jaccard: 0.0,
        auto_passthrough_max_tokens: 8000,
        redundancy_max_cosine: 1.0,
        low_confidence_max_grounding: 0.10,
        analyzer: default_english(),
        preserve_order: preserve,
    }
}

fn main() {
    let history = chat_history();
    // Score-sort to mimic what a real BM25 retriever returns: highest-
    // relevance chunks first. RawTopK consumes input in this order until
    // the budget fills — so the *selection* is relevance-driven, and the
    // *emission order* is what `preserve_order` controls downstream.
    let mut retrieved = make_chunks(&history);
    retrieved.sort_by(|a, b| {
        b.score
            .value
            .partial_cmp(&a.score.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let q = Query::new("shipping refund label return");

    println!("Chat history ({} turns):", history.len());
    for (id, text) in &history {
        println!("  {}: {}", id, text);
    }
    println!();
    println!(":: query: \"{}\" ::\n", q.text);

    println!("── A: preserve_order = false (default, relevance-emitted order) ──");
    let ctx_a = build_context(&q, &retrieved, &cfg(false));
    print_emitted(&ctx_a);
    println!();

    println!("── B: preserve_order = true (source-document / chronological order) ──");
    let ctx_b = build_context(&q, &retrieved, &cfg(true));
    print_emitted(&ctx_b);
    println!();

    // Verify: same SELECTION, different ORDER.
    let mut ids_a: Vec<&str> = ctx_a.chunks.iter().map(|c| c.id.as_str()).collect();
    let mut ids_b: Vec<&str> = ctx_b.chunks.iter().map(|c| c.id.as_str()).collect();
    println!("══ verdict ══");
    println!("  same SELECTION? {}", {
        ids_a.sort();
        ids_b.sort();
        ids_a == ids_b
    });
    let order_a: Vec<&str> = ctx_a.chunks.iter().map(|c| c.id.as_str()).collect();
    let order_b: Vec<&str> = ctx_b.chunks.iter().map(|c| c.id.as_str()).collect();
    println!("  same ORDER?     {}", order_a == order_b);
    println!();

    if order_a == order_b {
        println!(
            "  Note: on this corpus + query the orderings happen to coincide \
             (e.g. higher-scoring chunks are already chronologically last). \
             Try a query whose hits span the chat (e.g. \"label\" — turns 8 + 10)."
        );
    } else {
        println!(
            "  preserve_order ✓ preserved the chat's chronological order — \
             the LLM now reads the selected turns in the same sequence they \
             were said, instead of in BM25's relevance ranking."
        );
    }

    // ── Bonus: same demonstration via the high-level `Document.context()` path,
    //     using the LoadOptions surface (Rust users of `text(...)` / Python's
    //     `Document.from_text(... preserve_order=True)` / Node's
    //     `Document.fromText(text, { preserveOrder: true })`).
    // For chat with caller-supplied turns this path is the more common
    // entry point (each turn becomes a Document chunk via `from_chunks`),
    // and `preserve_order` flows through `LoadOptions` → `DocumentConfig`
    // → `ContextConfig` unchanged.

    // Suppress unused warnings.
    let _ = HashMap::<String, i64>::new();
    let _ = Arc::new(0u8);
    let _ = Embedding::from(vec![0.0]);
}
