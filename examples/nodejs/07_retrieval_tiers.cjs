// 07 · Retrieval tiers — lexical / hybrid / semantic on the same query.
//
// Real-world scenario:
//   A B2C support team's FAQ uses the company's polite phrasings
//   ("refund", "return") but customers ask in colloquial English
//   ("send back", "money back"). The same five-line FAQ corpus, hit
//   with three different retrieval tiers, shows where each one fails
//   and where each one succeeds — the trade-off documented in
//   docs/findings/SEMANTIC_MISMATCH.md.
//
// What this demonstrates:
//   - The three `retrieval=` tiers: "lexical" (BM25, default, no model),
//     "hybrid" (BM25 candidate pool + dense rerank, ~80MB model),
//     "semantic" (global exact-cosine dense, ~80MB model).
//   - That for a synonym-mismatch query, lexical (and sometimes hybrid)
//     can miss the right chunk because hybrid only reranks within
//     BM25's pool — if BM25 didn't surface it, hybrid can't recover it.
//   - Why "semantic" exists: bounded synonym-heavy corpora where global
//     dense scoring catches what lexical pruning would lose.
//
// First-run note:
//   `retrieval: "hybrid"` and `retrieval: "semantic"` need an embedding
//   model. The first call to either downloads `bge-small` (~80MB) to
//   your local model cache; subsequent runs are fast.
//
// Run:
//   node examples/nodejs/07_retrieval_tiers.cjs

const { Document } = require("redhop");

const SUPPORT_FAQ = `
Q: When will my package arrive?
A: Standard shipping takes 3-5 business days from when your order leaves our warehouse.

Q: How do I get my money back if I'm not satisfied?
A: We offer a full refund within 30 days of delivery. Return the item using the prepaid label.

Q: What's the warranty?
A: Our products have a one-year manufacturer warranty against defects.

Q: Can I cancel a subscription?
A: You can cancel anytime from Settings, no fee.

Q: Do you ship internationally?
A: Yes, we ship to 50 countries. Express international is 5-7 days.
`;

// "send back" / "do not want" — neither phrase appears in the right-
// answer FAQ ("refund", "return"). Pure synonym-mismatch.
const QUERY = "how do I send back something I do not want?";

function tryTier(label, options) {
  const t0 = Date.now();
  const doc = Document.fromText(SUPPORT_FAQ, { chunkSize: 30, ...options });
  const ctx = doc.context(QUERY);
  const elapsed = (Date.now() - t0) / 1000;
  const top = ctx.citations[0] ? ctx.citations[0].text.slice(0, 80) : "(none)";
  console.log(`  ${label.padEnd(10)} build+query: ${elapsed.toFixed(2)}s`);
  console.log(`               top hit  : ${JSON.stringify(top)}`);
  console.log();
}

function main() {
  console.log(`Query: ${JSON.stringify(QUERY)}`);
  console.log(`Gold (the right answer): "How do I get my money back …" /`);
  console.log(`                         "We offer a full refund within 30 days …"\n`);

  console.log("─── Arm A · retrieval='lexical' (BM25, default, no model) ─");
  tryTier("lexical", { retrieval: "lexical" });

  console.log("─── Arm B · retrieval='hybrid' (BM25 pool + dense rerank) ─");
  // First run downloads bge-small (~80MB).
  tryTier("hybrid", { retrieval: "hybrid", model: "bge-small" });

  console.log("─── Arm C · retrieval='semantic' (global exact-cosine dense) ─");
  tryTier("semantic", { retrieval: "semantic", model: "bge-small" });

  console.log("─── How to read this ─────────────────────────────");
  console.log("On this tiny 5-chunk corpus, BM25's candidate pool happens");
  console.log("to fit all 5 chunks, so `hybrid` can find the right answer");
  console.log("too. On a *real* synonym-heavy corpus (HR FAQs, support");
  console.log("tickets translated from internal phrasing, multilingual");
  console.log("content), BM25's top-K will often *exclude* the synonym-");
  console.log("mismatch answer entirely — and then hybrid can't recover");
  console.log("it because it only reranks within BM25's pool. That's the");
  console.log("regime where `semantic` (global, no pruning) earns its keep,");
  console.log("at the cost of embedding every chunk per query (only");
  console.log("practical on small to medium corpora).");
  console.log();
  console.log("Don't read this as 'always use semantic.' For most document");
  console.log("QA — code, runbooks, contracts, financial filings — the");
  console.log("question and answer DO share surface words, and lexical");
  console.log("wins on latency. Climb the ladder only when measured.");
  console.log("Decision tree: docs/CHOOSING_A_CONFIG.md.");
  console.log("Mechanism + measurement: docs/findings/SEMANTIC_MISMATCH.md.");
}

main();
