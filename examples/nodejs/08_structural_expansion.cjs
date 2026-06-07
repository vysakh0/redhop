// 08 · Structural expansion — neighbors=N and includeHeading=true.
//
// Real-world scenario:
//   A SaaS company's internal handbook is heavily structured: each
//   policy section has a heading + multiple paragraphs. When an
//   employee asks a question, the BM25 top hit lands on the specific
//   paragraph that answers the question — but for an LLM to write a
//   grounded answer, it usually wants the surrounding context: the
//   section heading (so it knows what topic it's in), plus the
//   paragraphs immediately before and after.
//
// What this demonstrates:
//   - doc.context(query, budget, neighbors, includeHeading) — the
//     Node signature is positional. Pass `undefined` for budget if
//     you only want to tune neighbors / includeHeading.
//   - ctx.report.nExpanded — how many extra chunks the structural
//     expansion added beyond raw retrieval selection.
//
// Run:
//   node examples/nodejs/08_structural_expansion.cjs

const { Document } = require("redhop");

const HANDBOOK = `
# PTO (Paid Time Off)

Full-time employees accrue 1.5 days of PTO per month, totaling 18 days per year.

PTO carries over up to a maximum of 30 days at the end of the calendar year. Beyond that, unused PTO is forfeited.

To request PTO, submit a request through Workday at least two weeks in advance. Manager approval is required.

# Sick Leave

Sick leave is separate from PTO. Employees may take up to 10 paid sick days per year for personal illness or family caregiving.

Sick days do not carry over and do not count against your PTO balance.

# Parental Leave

New parents are eligible for 16 weeks of paid parental leave following the birth or adoption of a child.

Leave can be taken continuously or split into two blocks of at least four weeks each, within the first 12 months.
`;

function showArm(label, query, opts) {
  // candidateK=2 constrains retrieval to the 2 top-scoring chunks so
  // the *expansion* contrast is visible — otherwise on this small
  // corpus the budget swallows everything and there's nothing to
  // expand.
  const doc = Document.fromText(HANDBOOK, {
    source: "handbook.md",
    chunkSize: 20,
    candidateK: 2,
  });
  const ctx = doc.context(query, undefined, opts.neighbors, opts.includeHeading);
  console.log(`─── ${label} ─────────────────────────`);
  console.log(`  nSelected     : ${ctx.report.nSelected}`);
  console.log(`  nExpanded     : ${ctx.report.nExpanded}`);
  console.log(`  totalTokens   : ${ctx.report.totalTokens}`);
  console.log("  assembled context:");
  for (const line of ctx.text.split("\n")) {
    console.log(`    ${line}`);
  }
  console.log();
}

function main() {
  const query = "how many PTO days do I get?";
  console.log(`Query: ${JSON.stringify(query)}\n`);

  showArm("Arm A · plain context (no expansion)", query, {});
  showArm("Arm B · includeHeading=true", query, { includeHeading: true });
  showArm("Arm C · neighbors=1", query, { neighbors: 1 });
  showArm(
    "Arm D · neighbors=1 + includeHeading=true (recommended for handbooks)",
    query,
    { neighbors: 1, includeHeading: true },
  );

  console.log("─── When to use each ─────────────────────────────");
  console.log("- includeHeading=true : structured docs (handbooks,");
  console.log("  contracts, runbooks) where the topic label matters.");
  console.log("- neighbors=1         : when the answer often spans");
  console.log("  adjacent chunks (a fact stated, then qualified).");
  console.log("- both                : the safe default for structured");
  console.log("  document QA.");
  console.log("- neither             : code search, transcripts,");
  console.log("  high-density technical content where the hit IS the answer.");
}

main();
