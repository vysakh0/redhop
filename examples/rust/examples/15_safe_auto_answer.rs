//! 15 · Safe auto-answers — when should a chatbot answer vs ask?
//!
//! Real-world scenario:
//!     A US store's help bot answers FAQs. The expensive failure is a
//!     *confident wrong answer*, so the bot should auto-answer only when
//!     retrieval clearly matched, and otherwise ask a clarifying
//!     question (or hand off). RedHop does not ship a router or a
//!     threshold — it gives you the *signals* and a deterministic eval to
//!     measure the gate. You own the "if confident then answer, else ask"
//!     logic. This is the pattern from the safe-auto-answers guide.
//!
//! What this demonstrates:
//!     - `ctx.report.low_confidence_retrieval` — the primary gate
//!       ("nothing relevant matched").
//!     - `evaluate(query, ctx, EvalGold::None).mean_grounding` — a
//!       no-gold confidence *strength* in [0,1] (how query-relevant the
//!       assembled context is). Confidence is a measured signal, not the
//!       model's self-report. (`ctx.report.diagnosis.score_spread` is a
//!       complementary margin signal when several candidates compete; it
//!       is None when a single chunk dominates, so it is not the gate
//!       here — see the guide for when each applies.)
//!     - `evaluate(..., EvalGold::Chunks)` to MEASURE the gate on a
//!       labeled set: auto-precision (correct among auto-answered) and
//!       unsafe-auto (auto-answered when we should have asked, target 0).
//!     - The headline: a good gate "gets cautious, not wrong" — it routes
//!       weak retrievals to clarify, keeping auto-precision high and
//!       unsafe-auto at 0.
//!
//!     tau here is illustrative. In production you DERIVE it: sweep on a
//!     labeled dev set and pick the smallest tau hitting your precision
//!     target (e.g. 99%). See the guide.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 15_safe_auto_answer --release

use redhop::core::{Chunk, ChunkId, Query, TokenCount};
use redhop::{chunks_typed, evaluate, Document, EvalConfig, EvalGold, LoadOptions};

const FAQ: &[(&str, &str)] = &[
    ("faq-refund", "Refunds. Return any item within 30 days for a full refund, no questions asked."),
    ("faq-shipping", "Shipping. Standard shipping is free on orders over 35 dollars and arrives in 5 to 7 business days."),
    ("faq-hours", "Store hours. Our stores are open 9am to 9pm Monday through Saturday, and 10am to 6pm on Sunday."),
    ("faq-giftcard", "Gift cards. Gift cards never expire and can be used online or in any store."),
    ("faq-track", "Order tracking. Track your order from the Orders page using the tracking number in your confirmation email."),
];

// Labeled eval set: each query maps to the FAQ that answers it, or None
// when there is no confident answer (the bot SHOULD ask, not guess).
fn labeled_queries() -> Vec<(&'static str, Option<&'static str>)> {
    vec![
        ("how do I return something for a refund", Some("faq-refund")),
        ("when are you open on sunday", Some("faq-hours")),
        ("how do I track my package", Some("faq-track")),
        ("do gift cards expire", Some("faq-giftcard")),
        ("can you help me", None),            // too vague — should ask
        ("do you price match competitors", None), // not in the KB — should ask
    ]
}

// Illustrative threshold. DERIVE this on a dev set in production (see guide):
// sweep tau and pick the smallest value that hits your auto-precision target.
const TAU: f32 = 0.2;

fn build() -> anyhow::Result<Document> {
    let chunks: Vec<Chunk> = FAQ
        .iter()
        .map(|(id, text)| {
            Chunk::new(
                ChunkId::new(*id),
                *text,
                "faq",
                TokenCount(text.split_whitespace().count()),
            )
        })
        .collect();
    Ok(chunks_typed(chunks, &LoadOptions::default())?)
}

fn main() -> anyhow::Result<()> {
    let mut doc = build()?;
    println!("Routing each query AUTO vs CLARIFY on redhop's confidence signals.");
    println!("(AUTO only when retrieval is confident: not low_confidence AND grounding >= {TAU})\n");
    println!(
        "  {:<38} {:>9} {:>9} {:>8}  {}",
        "query", "low_conf", "grounding", "route", "outcome"
    );

    let (mut auto_total, mut auto_correct, mut unsafe_auto, mut clarify_total) = (0, 0, 0, 0);

    for (q, gold) in labeled_queries() {
        let ctx = doc.context(q)?;
        // One eval per query: with gold (when we have it) for correctness,
        // and `mean_grounding` is a self-eval populated regardless of gold.
        let gold_slice: Vec<&str> = gold.iter().copied().collect();
        let gold_eval = if gold_slice.is_empty() {
            EvalGold::None
        } else {
            EvalGold::Chunks(&gold_slice)
        };
        let r = evaluate(&Query::new(q), &ctx, None, gold_eval, None, EvalConfig::default());

        let low = ctx.report.low_confidence_retrieval;
        let grounding = r.mean_grounding;
        let auto = !low && grounding >= TAU;
        let gold_present = gold.is_some() && r.context_recall.unwrap_or(0.0) >= 1.0;

        let outcome = if auto {
            auto_total += 1;
            match gold {
                Some(_) if gold_present => {
                    auto_correct += 1;
                    "AUTO ✓ correct"
                }
                Some(_) => "AUTO ✗ WRONG (auto-answered, missed the gold)",
                None => {
                    unsafe_auto += 1;
                    "AUTO ☠ UNSAFE (should have asked)"
                }
            }
        } else {
            clarify_total += 1;
            "clarify (asks the user)"
        };

        println!(
            "  {:<38} {:>9} {:>9.2} {:>8}  {}",
            &q[..q.len().min(38)],
            low,
            grounding,
            if auto { "AUTO" } else { "CLARIFY" },
            outcome,
        );
    }

    let auto_precision = if auto_total > 0 {
        auto_correct as f32 / auto_total as f32
    } else {
        1.0
    };
    println!("\n─── Scorecard ────────────────────────────────────");
    println!("  auto-resolve rate   : {auto_total}/{} answered without asking", labeled_queries().len());
    println!("  auto-precision ⭐    : {auto_precision:.3}  (correct among auto-answered; aim >= 0.99)");
    println!("  unsafe-auto ☠       : {unsafe_auto}      (auto-answered when it should have asked; target 0)");
    println!("  clarify rate        : {clarify_total}/{} routed to a question", labeled_queries().len());
    println!("\nThe gate degrades weak retrievals to clarify, so the bot");
    println!("'gets cautious, not wrong'. DERIVE tau on your own dev set by");
    println!("sweeping it to your precision target — see the safe-auto-answers guide.");
    Ok(())
}
