//! 10 · Assembly strategies — auto / raw_topk / reasoning_preserving.
//!
//! Real-world scenario:
//!     A research team has multi-hop questions over Wikipedia-style
//!     content ("Who invented the safety lamp, and what's their
//!     nationality?"). Retrieval surfaces two relevant chunks: one
//!     naming the inventor ("Davy invented the lamp"), one carrying the
//!     second-hop fact ("Davy was British"). A naive relevance-only
//!     filter would keep the high-scoring inventor chunk and drop the
//!     "Davy was British" chunk as a low-grounding distractor — and the
//!     LLM downstream would never see the bridge fact.
//!
//! What this demonstrates:
//!     - The four strategies via `build_context(&query, &chunks, &cfg)`:
//!         - `auto` (size-gated default)
//!         - `raw_topk` (pass-through)
//!         - `distractor_filtered` (naive baseline)
//!         - `reasoning_preserving` (second-hop rescue)
//!     - `ctx.report.second_hop_rescue_count` on the report confirming
//!       the rescue fired.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 10_strategy_choice --release

use redhop::context::{ContextConfig, ContextStrategy};
use redhop::core::{
    Chunk, ChunkId, Query, RetrievalMethod, RetrievalResult, Score, ScoreBreakdown, TokenCount,
};
use redhop::{build_context, BuiltContext};

const QUERY: &str = "what nationality was the inventor of the miners' safety lamp";

fn make_chunk(id: &str, text: &str) -> Chunk {
    let tok = text.split_whitespace().count();
    Chunk::new(ChunkId::new(id), text, "wiki", TokenCount(tok))
}

fn corpus() -> Vec<RetrievalResult> {
    vec![
        // Hop 1: discriminator + bridge entity.
        make_chunk(
            "hop1",
            "The miners' safety lamp was invented by Humphry Davy in 1815.",
        ),
        // Hop 2: low query-grounding, but linked via "Humphry Davy".
        make_chunk(
            "hop2",
            "Humphry Davy was a British chemist, born in Penzance, Cornwall, England.",
        ),
        // Distractor: no overlap with the query or the bridge.
        make_chunk(
            "d1",
            "Photosynthesis converts sunlight into glucose and oxygen in plants.",
        ),
    ]
    .into_iter()
    .map(|chunk| RetrievalResult {
        chunk,
        score: Score {
            value: 1.0,
            method: RetrievalMethod::Dense,
        },
        breakdown: ScoreBreakdown::default(),
    })
    .collect()
}

fn run(strategy: ContextStrategy, distractor_min_grounding: f32) -> BuiltContext {
    let cfg = ContextConfig {
        strategy,
        distractor_min_grounding,
        ..ContextConfig::default()
    };
    let q = Query::new(QUERY);
    let chunks = corpus();
    build_context(&q, &chunks, &cfg)
}

fn show_arm(label: &str, ctx: &BuiltContext) {
    let bridge_kept = ctx.text().contains("British");
    let discr_kept = ctx.text().contains("safety lamp");
    println!("─── {} ──────────────────────────", label);
    println!("  strategy           : {:?}", ctx.report.strategy);
    println!("  auto decision      : {:?}", ctx.report.auto_decision());
    println!(
        "  selected / input   : {} / {}",
        ctx.report.n_selected, ctx.report.n_input_chunks
    );
    println!(
        "  second-hop rescues : {}",
        ctx.report.second_hop_rescue_count
    );
    println!(
        "  bridge fact kept?  : {}",
        if bridge_kept { "yes ✓" } else { "no ✗" }
    );
    println!(
        "  discriminator kept?: {}",
        if discr_kept { "yes ✓" } else { "no ✗" }
    );
    println!();
}

fn main() {
    println!("Query: {:?}\n", QUERY);
    println!("(The gold answer is 'British' — the bridge fact in hop2,");
    println!("which has low query-grounding.)\n");

    show_arm(
        "Arm A · strategy=Auto (default)",
        &run(ContextStrategy::Auto, 0.10),
    );
    show_arm(
        "Arm B · strategy=RawTopK",
        &run(ContextStrategy::RawTopK, 0.10),
    );
    show_arm(
        "Arm C · strategy=DistractorFiltered (naive — drops bridge)",
        &run(ContextStrategy::DistractorFiltered, 0.30),
    );
    show_arm(
        "Arm D · strategy=ReasoningPreserving (the rescue)",
        &run(ContextStrategy::ReasoningPreserving, 0.30),
    );

    println!("─── How to read this ─────────────────────────────");
    println!("- `Auto` picks `RawTopK` under the size gate, `ReasoningPreserving`");
    println!("  over. Gate threshold: ContextConfig::auto_passthrough_max_tokens");
    println!("  (default 1500).");
    println!("- `RawTopK` for short, high-density chunks (code, schemas).");
    println!("- `ReasoningPreserving` for multi-hop QA where the bridge");
    println!("  between hops can sit below the naive grounding threshold");
    println!("  (docs/findings/SECOND_HOP_TAX.md).");
    println!("- `DistractorFiltered` is the relevance-only baseline that");
    println!("  `ReasoningPreserving` improves on.");
    println!();
    println!("Full strategy decision tree: docs/CHOOSING_A_CONFIG.md.");
}
