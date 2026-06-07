// 01 · Quickstart — load a document, ask a question, read the Decision Report.
//
// Real-world scenario:
//   A contract analyst has a Master Services Agreement (MSA) and needs to
//   answer "what's the governing law?" before handing the snippet to an
//   LLM for summarization. They want a citation back to the clause and
//   a reason RedHop chose those chunks.
//
// What this demonstrates:
//   - The 3-call surface: Document.fromText(...), doc.context(query),
//     ctx.text / ctx.citations / ctx.report.
//   - The Decision Report explaining what RedHop did and why.
//   - That for a small document, the runtime *deliberately* leaves the
//     context untouched (the "Auto → passthrough" decision).
//
// Run:
//   npm install redhop
//   node examples/nodejs/01_quickstart.cjs

const { Document } = require("redhop");

// A short Master Services Agreement excerpt. In production this would be
// `Document.fromFile("msa.pdf")`, but for a self-contained demo we paste
// the text inline.
const MSA = `
SECTION 8. CONFIDENTIALITY

Each party shall keep confidential all non-public information disclosed by
the other party in connection with this Agreement. The receiving party
shall not use such information for any purpose other than performance of
this Agreement.

SECTION 9. GOVERNING LAW AND JURISDICTION

This Agreement shall be governed by and construed in accordance with the
laws of the State of New York, without regard to its conflict-of-laws
principles. The parties consent to the exclusive jurisdiction of the
state and federal courts located in New York County, New York.

SECTION 10. ENTIRE AGREEMENT

This Agreement constitutes the entire agreement between the parties and
supersedes all prior negotiations, representations, and agreements,
whether written or oral, with respect to its subject matter.

SECTION 11. NOTICES

Any notice required under this Agreement shall be in writing and
delivered to the address set forth on the signature page.
`;

function main() {
  // 1. Load. `fromText` runs the default sentence chunker and indexes
  //    every chunk with BM25 — no model download, no vector DB.
  const doc = Document.fromText(MSA, { source: "acme_msa.txt" });
  console.log(`Indexed ${doc.chunkCount} chunks from acme_msa.txt\n`);

  // 2. Ask. RedHop retrieves, scores, and budgets the prompt all
  //    in-process. The BuiltContext carries the assembled text,
  //    citations back to source chunks, and the Decision Report.
  const ctx = doc.context("what's the governing law?");

  // 3. Hand the assembled text to whatever LLM you use — RedHop has no
  //    LLM lock-in. For the demo we just print it.
  console.log("─── Prompt (ctx.text) ────────────────────────────");
  console.log(ctx.text);
  console.log();

  // Citations: where did each kept chunk come from?
  console.log("─── Citations ────────────────────────────────────");
  for (const c of ctx.citations) {
    console.log(`  source=${JSON.stringify(c.source)} text=${JSON.stringify(c.text.slice(0, 80))}…`);
  }
  console.log();

  // The Decision Report explains the runtime's choice. For a small
  // document like this, the size gate fires and the context is passed
  // through untouched — pruning small contexts is wash-to-harmful per
  // docs/findings/CONTEXT_DILUTION.md.
  console.log("─── Decision Report ──────────────────────────────");
  console.log(`  strategy          : ${ctx.report.strategy}`);
  console.log(`  autoDecision      : ${ctx.report.autoDecision}`);
  console.log(`  inputChunks       : ${ctx.report.nInputChunks}`);
  console.log(`  selectedChunks    : ${ctx.report.nSelected}`);
  console.log(`  totalTokens       : ${ctx.report.totalTokens}`);
  console.log(`  retainedEvidence  : ${ctx.report.retainedEvidenceRatio.toFixed(2)}`);
  console.log();
  console.log("(For a human-readable version, log ctx.report.rendered)");
}

main();
