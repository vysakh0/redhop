// 03 · Templated workload — detect → strip → vocabulary → audit trail.
//
// Real-world scenario:
//   A legal-ops team uses a fixed query template across hundreds of
//   contracts: each query is shaped like
//       Highlight the parts (if any) of this contract related to "<X>"
//       that should be reviewed by a lawyer. Details: <…>
//   where only <X> varies. The boilerplate words dilute BM25's signal
//   on the discriminating clause name, costing retention on the
//   framework comparison (CUAD: 81% raw → 88% stripped → 90.7%
//   stripped + clause-synonyms). RedHop's 0.3.0 surface ships three
//   things they need:
//     - analyzeQuerySet(queries) to detect the template.
//     - new Stripper(boilerplate) to drop the wrapper at retrieval time.
//     - new Vocabulary({...}) to append clause-name synonyms.
//   Both rewrites run inside doc.contextWithRewrites(query, [stripper, vocab])
//   so the per-stage audit lands on ctx.report.queryRewrites and the
//   chain stays observable.
//
// Run:
//   node examples/nodejs/03_templated_workload.cjs

const {
  Document,
  Stripper,
  Vocabulary,
  analyzeQuerySet,
} = require("redhop");

const CONTRACT = `
SECTION 7. CHANGE OF CONTROL

In the event of a Change of Control of either party, including any
merger, consolidation, or sale of substantially all assets, the
non-acquired party shall have the right to terminate this Agreement on
thirty days' written notice.

SECTION 8. NON-COMPETE

During the Term and for two years thereafter, the Distributor shall
not, directly or indirectly, engage in any business competitive with
the Company within the Territory.

SECTION 9. INDEMNIFICATION

Each party shall indemnify and hold harmless the other from any third-
party claims arising from the indemnifying party's gross negligence or
willful misconduct.

SECTION 10. CONFIDENTIALITY

Each party shall keep confidential all non-public information disclosed
by the other party in connection with this Agreement.
`;

const SAMPLE_QUERIES = [
  'Highlight the parts (if any) of this contract related to "Change of Control" that should be reviewed by a lawyer.',
  'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer.',
  'Highlight the parts (if any) of this contract related to "Indemnification" that should be reviewed by a lawyer.',
  'Highlight the parts (if any) of this contract related to "Confidentiality" that should be reviewed by a lawyer.',
  'Highlight the parts (if any) of this contract related to "Termination" that should be reviewed by a lawyer.',
];

// Workload-specific clause-name synonyms. The library deliberately does
// NOT ship a CUAD dict — Vocabulary is the mechanism; your dict is
// your workload knowledge.
const CLAUSE_SYNONYMS = {
  "change of control": ["merger", "consolidation", "acquisition", "successor"],
  "non-compete": ["restraint", "compete", "competitive"],
  "indemnification": ["indemnify", "hold harmless", "third-party claims"],
  "confidentiality": ["confidential", "non-disclosure", "non-public"],
  "termination": ["terminate", "expire", "end"],
};

function main() {
  // ── Step 1: Detect ────────────────────────────────────────────────
  console.log("─── Step 1 · Detect the template ─────────────────");
  const report = analyzeQuerySet(SAMPLE_QUERIES);
  console.log(`  isTemplated            : ${report.isTemplated}`);
  console.log(`  templateWordShare      : ${report.templateWordShare.toFixed(2)}`);
  console.log(`  estimatedDilutionCost  : ${report.estimatedDilutionCost}`);
  console.log(`  boilerplateTerms       : ${JSON.stringify(report.boilerplateTerms)}`);
  console.log(`  suggestedAction        : ${report.suggestedAction}`);
  console.log();
  if (!report.isTemplated) {
    console.log("(Template not detected — for non-templated workloads skip");
    console.log(" the Stripper and use doc.context(query) directly.)");
    return;
  }

  // ── Step 2: Compile the rewrites ─────────────────────────────────
  // Compile once, reuse for every query. The token-level matcher
  // makes the analyzer pass once at construction time — chatbot
  // hot paths don't pay it per request.
  const stripper = new Stripper(report.boilerplateTerms);
  const vocab = new Vocabulary(CLAUSE_SYNONYMS);
  console.log("─── Step 2 · Compile the rewrites ────────────────");
  console.log(`  Stripper: ${stripper.length} boilerplate forms`);
  console.log(`  Vocabulary: ${vocab.length} clause classes`);
  console.log();

  // ── Step 3: Run a query through the chain ────────────────────────
  console.log("─── Step 3 · Run a query through the chain ───────");
  const doc = Document.fromText(CONTRACT, { source: "msa.txt" });
  const query = SAMPLE_QUERIES[0];
  console.log(`  raw query: ${JSON.stringify(query)}\n`);

  const ctx = doc.contextWithRewrites(query, [stripper, vocab]);

  // The per-stage audit trail. Each RewriteRecord documents what
  // one rewrite stage did: input → output, what was matched, what
  // was added, what was removed.
  console.log("  queryRewrites audit trail:");
  for (const rec of ctx.report.queryRewrites) {
    console.log(`    [${rec.stage}]`);
    console.log(`      from   : ${JSON.stringify(rec.fromQuery)}`);
    console.log(`      to     : ${JSON.stringify(rec.toQuery)}`);
    console.log(`      matched: ${JSON.stringify(rec.matched)}`);
    console.log(`      added  : ${JSON.stringify(rec.added)}`);
    console.log(`      removed: ${JSON.stringify(rec.removed)}`);
  }
  console.log();

  console.log("  Top citation source : ", ctx.citations[0].source);
  console.log(
    "  Top citation text   : ",
    ctx.citations[0].text.slice(0, 80).replace(/\n/g, " "),
    "…",
  );
  console.log();
  console.log("  Decision: ", ctx.report.autoDecision, "/", ctx.report.strategy);
}

main();
