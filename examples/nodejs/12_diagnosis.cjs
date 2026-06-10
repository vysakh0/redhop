// 12 · Diagnosis — when retrieval looks weak, the Decision Report tells you why.
//
// Real-world scenario:
//   A support team is wiring up Q&A over a policy doc. A user asks
//   "how long do I have to cancel and get my money back?" and gets
//   an empty answer. The doc uses *refund* and *termination*, not
//   *cancel* and *money back*, so BM25 has nothing to match.
//
// What this demonstrates:
//   - report.diagnosis populated on every context() call.
//   - Layer-2 facts: queryTerms, zeroMatchTerms, termStats computed
//     against the corpus vocabulary.
//   - The closed hints registry: one bounded hint per documented
//     failure shape, each citing the finding that justifies it.
//   - A healthy query produces zero hints.
//
// Run:
//   npm install redhop
//   node examples/nodejs/12_diagnosis.cjs

const { Chunk, Document } = require("redhop");

function main() {
  const doc = Document.fromChunks([
    new Chunk("Refund Policy. Refunds are available within thirty days of purchase.", { id: "a", source: "policy.md" }),
    new Chunk("Termination for convenience. Either party may terminate this agreement.", { id: "b", source: "policy.md" }),
    new Chunk("Governing Law. This agreement is governed by the laws of California.", { id: "c", source: "policy.md" }),
  ]);

  // 1. Healthy query: facts populated, no hints.
  const healthy = doc.context("refund policy thirty days").report;
  console.log("Healthy query:");
  console.log("  queryTerms             =", healthy.diagnosis.queryTerms);
  console.log("  corpusStatsAvailable   =", healthy.diagnosis.corpusStatsAvailable);
  console.log("  zeroMatchTerms         =", healthy.diagnosis.zeroMatchTerms);
  console.log("  hints                  =", healthy.diagnosis.hints);
  console.log();

  // 2. Vocabulary-mismatch query: vocab_mismatch hint fires.
  const paraphrase = doc.context("How long do I have to cancel and get my money back?").report;
  console.log("Vocabulary-mismatch query:");
  console.log("  queryTerms             =", paraphrase.diagnosis.queryTerms);
  console.log("  zeroMatchTerms         =", paraphrase.diagnosis.zeroMatchTerms);
  console.log("  emptyContext           =", paraphrase.diagnosis.emptyContext);
  for (const hint of paraphrase.diagnosis.hints) {
    console.log(`  hint ${JSON.stringify(hint.code)}`);
    console.log(`    evidence : ${hint.evidence}`);
    console.log(`    message  : ${hint.message}`);
  }
  console.log();

  // 3. The same data appears in the rendered Decision Report.
  const rendered = paraphrase.rendered;
  if (rendered.includes("Query diagnosis")) {
    console.log("Rendered report (excerpt):");
    console.log(rendered.slice(rendered.indexOf("Query diagnosis")));
  }
}

main();
