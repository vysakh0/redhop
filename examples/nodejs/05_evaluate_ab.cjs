// 05 · Deterministic A/B with redhop.evaluate(...) — no LLM judge.
//
// Real-world scenario:
//   The legal-ops team from 03_templated_workload.cjs wants to know
//   whether adding clause-name synonyms (the Vocabulary step) actually
//   lifts retrieval on *their* contracts, not on a published benchmark.
//   They have a small gold set: for each query they labeled which chunk
//   id(s) should appear in the assembled context. They compare two arms
//   — baseline vs strip + vocab — and the `evaluate` API returns
//   contextRecall, contextPrecision, and a composite `overall` for
//   each, all from the same primitives the runtime uses for its
//   Decision Report. No LLM judge, no API key, no money spent,
//   deterministic across runs.
//
// Run:
//   node examples/nodejs/05_evaluate_ab.cjs

const {
  Document,
  Chunk,
  Stripper,
  Vocabulary,
  evaluate,
} = require("redhop");

const SECTIONS = [
  {
    id: "sec-7",
    heading: "Change of Control",
    text: "SECTION 7. CHANGE OF CONTROL. In the event of a Change of Control of either party, including any merger, consolidation, or sale of substantially all assets, the non-acquired party shall have the right to terminate this Agreement on thirty days' written notice.",
  },
  {
    id: "sec-8",
    heading: "Non-Compete",
    text: "SECTION 8. NON-COMPETE. During the Term and for two years thereafter, the Distributor shall not, directly or indirectly, engage in any business competitive with the Company within the Territory.",
  },
  {
    id: "sec-9",
    heading: "Indemnification",
    text: "SECTION 9. INDEMNIFICATION. Each party shall indemnify and hold harmless the other from any third-party claims arising from the indemnifying party's gross negligence or willful misconduct.",
  },
  {
    id: "sec-10",
    heading: "Confidentiality",
    text: "SECTION 10. CONFIDENTIALITY. Each party shall keep confidential all non-public information disclosed by the other party in connection with this Agreement.",
  },
  {
    id: "sec-11",
    heading: "Termination",
    text: "SECTION 11. TERMINATION. Either party may terminate this Agreement upon thirty days' written notice in the event of a material breach by the other party.",
  },
  {
    id: "sec-12",
    heading: "Notices",
    text: "SECTION 12. NOTICES. Any notice required under this Agreement shall be in writing and delivered to the address set forth on the signature page.",
  },
];

// Gold set: each templated query maps to the chunk id we expect the
// assembled context to contain.
const GOLD_QUERIES = [
  [
    'Highlight the parts (if any) of this contract related to "Change of Control" that should be reviewed by a lawyer.',
    ["sec-7"],
  ],
  [
    'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer.',
    ["sec-8"],
  ],
  [
    'Highlight the parts (if any) of this contract related to "Indemnification" that should be reviewed by a lawyer.',
    ["sec-9"],
  ],
  [
    'Highlight the parts (if any) of this contract related to "Confidentiality" that should be reviewed by a lawyer.',
    ["sec-10"],
  ],
  [
    'Highlight the parts (if any) of this contract related to "Termination" that should be reviewed by a lawyer.',
    ["sec-11"],
  ],
];

const CLAUSE_SYNONYMS = {
  "change of control": ["merger", "consolidation", "acquisition"],
  "non-compete": ["restraint", "compete", "competitive"],
  "indemnification": ["indemnify", "hold harmless"],
  "confidentiality": ["confidential", "non-disclosure"],
  "termination": ["terminate", "expire", "end"],
};

function buildDoc() {
  // Build the same Document for both arms — only the query differs.
  return Document.fromChunks(
    SECTIONS.map((s) => new Chunk(s.text, {
      source: "msa.txt",
      id: s.id,
      metadata: { heading: s.heading },
    })),
  );
}

function evaluateArm(label, doc, useRewrites) {
  const boilerplate = [
    "highlight", "the", "parts", "if", "any", "of", "this", "contract",
    "related", "to", "that", "should", "be", "reviewed", "by", "a", "lawyer",
  ];
  const stripper = new Stripper(boilerplate);
  const vocab = new Vocabulary(CLAUSE_SYNONYMS);

  console.log(`─── arm ${label} ──────────────────────────────────`);
  const totals = { contextRecall: 0, contextPrecision: 0, overall: 0 };
  let n = 0;
  for (const [query, goldIds] of GOLD_QUERIES) {
    const ctx = useRewrites
      ? doc.contextWithRewrites(query, [stripper, vocab])
      : doc.context(query);
    const r = evaluate(query, ctx, { goldChunks: goldIds });
    n += 1;
    totals.contextRecall += r.contextRecall ?? 0;
    totals.contextPrecision += r.contextPrecision ?? 0;
    totals.overall += r.overall;
    if (n === 1) {
      console.log(`  example query  : ${query.slice(0, 60)}…`);
      console.log(
        `  contextRecall : ${r.contextRecall.toFixed(2)}  contextPrecision : ${r.contextPrecision.toFixed(2)}  overall : ${r.overall.toFixed(2)}`,
      );
    }
  }
  const means = {
    recall: totals.contextRecall / n,
    precision: totals.contextPrecision / n,
    overall: totals.overall / n,
  };
  console.log(
    `  mean over ${n} queries: recall=${means.recall.toFixed(2)}  precision=${means.precision.toFixed(2)}  overall=${means.overall.toFixed(2)}`,
  );
  console.log();
  return means.overall;
}

function main() {
  const doc = buildDoc();
  console.log("Comparing two retrieval arms on the same gold set.\n");
  const a = evaluateArm("A · baseline (no rewrites)", doc, false);
  const b = evaluateArm("B · stripped + clause-name vocabulary", doc, true);
  console.log("─── Verdict ──────────────────────────────────────");
  const delta = b - a;
  console.log(`  ΔB−A on \`overall\`: ${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`);
  if (b > a + 0.02) {
    console.log("  ✓ The rewrite chain lifted retrieval on this gold set.");
  } else if (b < a - 0.02) {
    console.log("  ✗ The rewrite chain regressed retrieval. Inspect the audit");
    console.log("    trail (ctx.report.queryRewrites) — likely the vocab is");
    console.log("    appending workload-pervasive terms (the CUAD_PRF_NULL");
    console.log("    failure mode).");
  } else {
    console.log("  ~ Within sample noise — re-run with a larger gold set.");
  }
}

main();
