// 14 · Catalog search — short, noisy queries over a near-duplicate catalog.
//
// Real-world scenario:
//   A corner-store ordering assistant takes short, messy product
//   requests ("liberty root beer", "summit cola", and plenty of typos
//   like "1iberty"). The catalog is a near-duplicate lattice: one brand
//   has the same product at several sizes and prices that differ by a
//   token or two. Three things break here that don't break on prose QA,
//   and each has a lever.
//
// What this demonstrates:
//   - language: "char_ngram" — the subword typo tier. A typo
//     ("1iberty") still matches via shared character n-grams, no model.
//     Word-token BM25 scores it at zero.
//   - bm25FieldWeights: [text, source, heading] — per-field BM25 boosts,
//     a domain lever (default equal weight is bit-for-bit unchanged).
//   - evaluate(..., { goldFamilies }) -> setCoverage — a catalog query
//     maps to a SET (all sizes); recall@k hides a half-retrieved family,
//     setCoverage catches it.
//
//   Honest framing (docs/findings/CATALOG_REGIME.md): char-ngram is a
//   recall booster, not a drop-in. Field weights help only when the
//   boosted field separates the answer from its near-duplicates.
//
// Run:
//   npm install redhop
//   node examples/nodejs/14_catalog_search.cjs

const { Document, Chunk, evaluate } = require("redhop");

// [sku id, brand+product key, full product line] — a small American
// convenience-store catalog with near-duplicate size/price variants.
const CATALOG = [
  ["summit-cola-12", "Summit Cola", "Summit Cola 12 oz 1.49"],
  ["summit-cola-20", "Summit Cola", "Summit Cola 20 oz 1.99"],
  ["summit-cola-2l", "Summit Cola", "Summit Cola 2 liter 2.49"],
  ["summit-diet-12", "Summit Diet Cola", "Summit Diet Cola 12 oz 1.49"],
  ["summit-diet-20", "Summit Diet Cola", "Summit Diet Cola 20 oz 1.99"],
  ["liberty-rb-12", "Liberty Root Beer", "Liberty Root Beer 12 oz 1.49"],
  ["liberty-rb-20", "Liberty Root Beer", "Liberty Root Beer 20 oz 1.99"],
  ["eagle-bbq-2", "Eagle Potato Chips", "Eagle Potato Chips BBQ 2 oz 1.29"],
  ["eagle-bbq-8", "Eagle Potato Chips", "Eagle Potato Chips BBQ 8 oz 3.49"],
  ["eagle-salt-2", "Eagle Potato Chips", "Eagle Potato Chips Salted 2 oz 1.29"],
  ["pioneer-jerky-3", "Pioneer Beef Jerky", "Pioneer Beef Jerky Original 3 oz 5.99"],
  ["coastal-mix-6", "Coastal Trail Mix", "Coastal Trail Mix 6 oz 4.29"],
];

function build(language, bm25FieldWeights) {
  const chunks = CATALOG.map(
    ([id, heading, text]) => new Chunk(text, { id, source: "catalog", metadata: { heading } }),
  );
  return Document.fromChunks(chunks, { language, bm25FieldWeights });
}

// Distinct brand+product labels in the assembled context, in order
// (citations carry the `heading` we set per chunk).
function products(ctx) {
  const seen = [];
  for (const c of ctx.citations) {
    if (c.heading && !seen.includes(c.heading)) seen.push(c.heading);
  }
  return seen;
}

function main() {
  // ── 1. Transcription typo: char-ngram recovers what word-BM25 drops ──
  // A realistic noisy order: the brand is typo'd ("1iberty") AND the
  // product is run together ("rootbeer"), so word-BM25 has no exact token
  // to match. char-ngram bridges both via shared character n-grams.
  console.log("1) Typo recovery — query: '1iberty rootbeer'\n");
  const word = build("raw", undefined); // default word-token analyzer
  const ngram = build("char_ngram", undefined); // subword typo tier
  const q = "1iberty rootbeer";
  console.log(`   word-BM25  found : ${JSON.stringify(products(word.context(q)))}`);
  const ngramFound = products(ngram.context(q));
  console.log(`   char-ngram found : ${JSON.stringify(ngramFound)}`);
  console.log(
    `   -> char-ngram recovered Liberty Root Beer despite the typo: ${ngramFound.includes("Liberty Root Beer")}\n`,
  );

  // ── 2. Per-field weighting is a knob (default = equal weight) ─────────
  console.log("2) Field weights — boost the brand/product 'heading' field 2x\n");
  const boosted = build("char_ngram", [1.0, 1.0, 2.0]);
  console.log(`   'summit cola' -> ${JSON.stringify(products(boosted.context("summit cola")))}`);
  console.log("   (a domain lever: sweep on your own gold set, it is not a");
  console.log("    guaranteed lift; see docs/findings/CATALOG_REGIME.md)\n");

  // ── 3. setCoverage: did we retrieve the WHOLE variant family? ────────
  console.log("3) Set coverage — 'summit cola' should return ALL its sizes\n");
  const ctx = ngram.context("summit cola");
  const family = ["summit-cola-12", "summit-cola-20", "summit-cola-2l"];
  const r = evaluate("summit cola", ctx, { goldFamilies: [family] });
  console.log(`   products offered : ${JSON.stringify(products(ctx))}`);
  console.log(`   setCoverage      : ${r.setCoverage}   (1.0 = whole family offerable)`);
  console.log(`   contextRecall    : ${r.contextRecall}`);
  console.log("   recall@k can read fine while a family is half-retrieved;");
  console.log("   setCoverage is the metric a disambiguation UX should gate on.");
}

main();
