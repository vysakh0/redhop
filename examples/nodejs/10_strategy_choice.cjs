// 10 · Assembly strategies — auto / raw_topk / reasoning_preserving.
//
// Real-world scenario:
//   A research team has multi-hop questions over Wikipedia-style
//   content ("Who invented the safety lamp, and what's their
//   nationality?"). Retrieval surfaces two relevant chunks: one
//   naming the inventor ("Davy invented the lamp"), one carrying the
//   second-hop fact ("Davy was British"). A naive relevance-only
//   filter would keep the high-scoring inventor chunk and drop the
//   "Davy was British" chunk as low-grounding — and the LLM never
//   sees the bridge fact.
//
//   RedHop's "reasoning_preserving" strategy keeps both: it rescues
//   low-grounding chunks linked to high-grounding ones via term-set
//   Jaccard overlap (the "bridge" between hops).
//
// Run:
//   node examples/nodejs/10_strategy_choice.cjs

const { buildContext, Chunk } = require("redhop");

const CHUNKS = [
  new Chunk(
    "The miners' safety lamp was invented by Humphry Davy in 1815.",
    { id: "hop1" },
  ),
  new Chunk(
    "Humphry Davy was a British chemist, born in Penzance, Cornwall, England.",
    { id: "hop2" },
  ),
  new Chunk(
    "Photosynthesis converts sunlight into glucose and oxygen in plants.",
    { id: "d1" },
  ),
];

const QUERY = "what nationality was the inventor of the miners' safety lamp";

function showArm(label, strategy, options = {}) {
  const ctx = buildContext(QUERY, CHUNKS, { strategy, ...options });
  console.log(`─── ${label} ──────────────────────────`);
  console.log(`  strategy           : ${ctx.report.strategy}`);
  console.log(`  autoDecision       : ${ctx.report.autoDecision}`);
  console.log(`  selected / input   : ${ctx.report.nSelected} / ${ctx.report.nInputChunks}`);
  console.log(`  second-hop rescues : ${ctx.report.secondHopRescueCount}`);
  const bridgeKept = ctx.text.includes("British");
  const discrKept = ctx.text.includes("safety lamp");
  console.log(`  bridge fact kept?  : ${bridgeKept ? "yes ✓" : "no ✗"}`);
  console.log(`  discriminator kept?: ${discrKept ? "yes ✓" : "no ✗"}`);
  console.log();
}

function main() {
  console.log(`Query: ${JSON.stringify(QUERY)}\n`);
  console.log("(The gold answer is 'British' — the bridge fact in hop2,");
  console.log("which has low query-grounding.)\n");

  showArm("Arm A · strategy='auto' (default)", "auto");
  showArm("Arm B · strategy='raw_topk'", "raw_topk");
  showArm(
    "Arm C · strategy='distractor_filtered' (naive — drops bridge)",
    "distractor_filtered",
    { distractorMinGrounding: 0.30 },
  );
  showArm(
    "Arm D · strategy='reasoning_preserving' (the rescue)",
    "reasoning_preserving",
    { distractorMinGrounding: 0.30 },
  );

  console.log("─── How to read this ─────────────────────────────");
  console.log("- `auto` is the default; it picks `raw_topk` under the size");
  console.log("  gate and `reasoning_preserving` over (where pruning recovers");
  console.log("  accuracy via dilution control). Gate threshold is the");
  console.log("  autoPassthroughMaxTokens knob (default 1500).");
  console.log("- `raw_topk` is what you want when chunks are short and");
  console.log("  high-density — code, schemas, error codes.");
  console.log("- `reasoning_preserving` is what you want for multi-hop QA");
  console.log("  where the bridge between hops can sit below the naive");
  console.log("  grounding threshold (docs/findings/SECOND_HOP_TAX.md).");
  console.log("- `distractor_filtered` is the relevance-only baseline that");
  console.log("  reasoning_preserving improves on.");
  console.log();
  console.log("Full strategy decision tree: docs/CHOOSING_A_CONFIG.md.");
}

main();
