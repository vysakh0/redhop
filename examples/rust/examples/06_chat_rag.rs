//! 06 · Chat RAG with chronology preserved — `preserve_order = true`.
//!
//! Real-world scenario:
//!     A customer-support agent's chat session has been going for an
//!     hour and has 30+ turns. Rather than summarizing or compacting
//!     the history (lossy), the team retrieves the few past turns
//!     relevant to the user's *current* question and pulls those into
//!     the LLM prompt. But causality breaks if the retrieved turns are
//!     presented in relevance order — "after the refund came in" reads
//!     strangely if it's shown before "ordered the laptop." They want
//!     the same relevance-driven selection but with chronological
//!     emission.
//!
//! What this demonstrates:
//!     - `ContextConfig { preserve_order: true, .. }` set on the
//!       DocumentConfig used by `Document::from_chunks_with(...)`.
//!     - Selection stays relevance-driven; emission becomes chronological.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 06_chat_rag --release

use redhop::context::ContextConfig;
use redhop::core::{Chunk, ChunkId, TokenCount};
use redhop::{Document, DocumentConfig};

const CHAT_HISTORY: &[(&str, &str)] = &[
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
    ("turn-07", "So I just send it back after it arrives?"),
    (
        "turn-08",
        "Yes — print the return label from your Orders page and drop it off.",
    ),
    ("turn-09", "Does the refund come right away?"),
    (
        "turn-10",
        "We refund within five business days of receiving the return.",
    ),
    ("turn-11", "Got it, thanks for your help!"),
];

fn build_doc(preserve_order: bool) -> anyhow::Result<Document> {
    let chunks: Vec<Chunk> = CHAT_HISTORY
        .iter()
        .map(|(tid, text)| {
            let tok = text.split_whitespace().count();
            Chunk::new(ChunkId::new(*tid), *text, "chat", TokenCount(tok))
        })
        .collect();
    let cfg = DocumentConfig {
        context: ContextConfig {
            preserve_order,
            ..DocumentConfig::default().context
        },
        ..DocumentConfig::default()
    };
    Ok(Document::from_chunks_with(chunks, cfg)?)
}

fn show_arm(label: &str, preserve_order: bool, query: &str) -> anyhow::Result<()> {
    let mut doc = build_doc(preserve_order)?;
    let ctx = doc.context(query)?;
    println!("─── {} ─────────────────────────────────", label);
    println!("  selected turns, in emission order:");
    // Iterate ctx.chunks directly so we can show the chunk id (turn-XX)
    // — citations only surface source/page/heading/line.
    for c in &ctx.chunks {
        println!("    {}: {}", c.id.as_str(), c.text);
    }
    println!();
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let query = "remind me — when does the refund actually come back?";
    println!("Current user question: {:?}\n", query);

    // Arm A: default (preserve_order=false). Strategy-emitted order.
    show_arm("Arm A · default (relevance-emitted)", false, query)?;

    // Arm B: preserve_order=true. Same selection, chronological emission.
    show_arm("Arm B · preserve_order=true (chronological)", true, query)?;

    println!("Both arms select the same turns by relevance; only the");
    println!("emission order differs. For the LLM, that ordering controls");
    println!("whether causality reads correctly — `refund` after `return`");
    println!("after `ordered`. See");
    println!("crates/examples/examples/chat_rag.rs for a longer worked");
    println!("contrast with more nuance.");
    Ok(())
}
