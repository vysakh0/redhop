// Binding-surface tests for the query-set analyzer.
//
// Mirrors the Rust unit tests on `redhop::analyze_query_set` through
// the napi binding so a dropped field on QuerySetReport or a wrong
// Vec<String> ↔ Array<string> mapping at the FFI boundary surfaces
// here, not in user code.
//
// Mechanism + thresholds: docs/findings/QUERY_SET_ANALYZER.md.
//
// Run with: node test/query_set_analyzer.cjs (or `npm test`).

const assert = require("node:assert");
const { analyzeQuerySet, Stripper } = require("../index.js");

// ── analyzeQuerySet ──────────────────────────────────────────────────────

const cuadShape = () => [
  'Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer.',
  'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer.',
  'Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer.',
  'Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer.',
  'Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer.',
  'Highlight the parts (if any) of this contract related to "Renewal Term" that should be reviewed by a lawyer.',
];

const diverse = () => [
  "Who is the current president of France?",
  "When was the Eiffel Tower built?",
  "What language do they speak in Brazil?",
  "How tall is Mount Everest?",
  "Which planet is closest to the sun?",
  "When did World War II end?",
  "Who wrote Pride and Prejudice?",
  "What is the capital of Japan?",
];

// True positive on CUAD shape.
{
  const r = analyzeQuerySet(cuadShape());
  assert.ok(r.isTemplated, `expected CUAD-shape to be flagged; got: ${JSON.stringify(r)}`);
  assert.ok(r.templateWordShare > 0.6, `share ${r.templateWordShare}`);
  assert.strictEqual(r.estimatedDilutionCost, "high");
  for (const word of ["highlight", "contract", "lawyer"]) {
    assert.ok(
      r.boilerplateTerms.includes(word),
      `expected ${word} in boilerplateTerms; got ${JSON.stringify(r.boilerplateTerms)}`,
    );
  }
}

// No false positive on diverse natural-language queries.
{
  const r = analyzeQuerySet(diverse());
  assert.ok(!r.isTemplated, `expected diverse not flagged; got: ${JSON.stringify(r)}`);
  assert.ok(
    ["low", "none"].includes(r.estimatedDilutionCost),
    `unexpected cost: ${r.estimatedDilutionCost}`,
  );
}

// Empty input.
{
  const r = analyzeQuerySet([]);
  assert.strictEqual(r.nQueries, 0);
  assert.ok(!r.isTemplated);
  assert.strictEqual(r.templateWordShare, 0.0);
  assert.deepStrictEqual(r.boilerplateTerms, []);
  assert.strictEqual(r.estimatedDilutionCost, "none");
  assert.ok(
    r.suggestedAction.toLowerCase().includes("empty"),
    `expected mention of 'empty' in suggested_action; got: ${r.suggestedAction}`,
  );
}

// End-to-end: detect → strip pattern via Stripper.
{
  const queries = cuadShape();
  const r = analyzeQuerySet(queries);
  assert.ok(r.isTemplated);
  const stripper = new Stripper(r.boilerplateTerms);
  const stripped = stripper.apply(queries[0]);
  assert.ok(
    stripped.includes("Document Name"),
    `expected discriminator preserved; got: ${stripped}`,
  );
  for (const noise of ["Highlight", "contract", "lawyer"]) {
    assert.ok(
      !stripped.toLowerCase().includes(noise.toLowerCase()),
      `expected ${noise} stripped; got: ${stripped}`,
    );
  }
}

console.log("query_set_analyzer: OK");
