// Binding-surface tests for the query-set analyzer.
//
// Mirrors the Rust unit tests on `redhop::analyze_query_set` and
// `redhop::drop_template_terms` through the napi binding so a dropped
// field on QuerySetReport or a wrong Vec<String> ↔ Array<string> mapping
// at the FFI boundary surfaces here, not in user code.
//
// Mechanism + thresholds: docs/findings/QUERY_SET_ANALYZER.md.
//
// Run with: node test/query_set_analyzer.cjs (or `npm test`).

const assert = require("node:assert");
const { analyzeQuerySet, dropTemplateTerms } = require("../index.js");

// ── dropTemplateTerms ────────────────────────────────────────────────────

{
  const q = "Highlight the parts of this contract related to Change of Control";
  const out = dropTemplateTerms(q, [
    "highlight",
    "the",
    "parts",
    "of",
    "this",
    "contract",
    "related",
    "to",
  ]);
  assert.strictEqual(out, "Change Control", "basic strip");
}

{
  const out = dropTemplateTerms("Find the Document Name", ["the", "find"]);
  assert.strictEqual(out, "Document Name", "case-insensitive match");
}

{
  const q = 'Highlight the parts related to "Change of Control".';
  const out = dropTemplateTerms(q, [
    "highlight",
    "the",
    "parts",
    "of",
    "related",
    "to",
  ]);
  assert.strictEqual(out, '"Change Control".', "punctuation preserved");
}

{
  const q = "Highlight the parts of this contract";
  assert.strictEqual(
    dropTemplateTerms(q, []),
    q,
    "empty boilerplate is identity",
  );
}

{
  const out = dropTemplateTerms("the the the", ["the"]);
  assert.strictEqual(out, "", "all-filtered returns empty string");
}

// CJK queries have no whitespace; analyzer surfaces boilerplate as
// punctuation-bounded phrases. dropTemplateTerms must do substring
// removal (script-aware), not whitespace-token matching.
{
  const q = "请标注本合同中与「文档名称」相关的、应由律师审核的部分（如有）。";
  const boilerplate = ["请标注本合同中与", "应由律师审核的部分", "相关的", "如有"];
  const stripped = dropTemplateTerms(q, boilerplate);
  for (const noise of boilerplate) {
    assert.ok(!stripped.includes(noise), `expected ${noise} stripped; got ${stripped}`);
  }
  assert.ok(stripped.includes("文档名称"), `discriminator should survive; got ${stripped}`);
}

// Latin-script word-boundary safety: "of" should not erase "office".
{
  const out = dropTemplateTerms("the office is open", ["of", "the"]);
  assert.strictEqual(out, "office is open", "word boundary preserved on Latin script");
}

// ─── expandQueryTerms ────────────────────────────────────────────────────

const { expandQueryTerms } = require("../index.js");

{
  // Matched key → synonyms appended; non-matching key's syns absent.
  const expansions = {
    "change of control": ["merger", "successor", "acquisition"],
    "non-compete": ["restraint", "non-competition"],
  };
  const expanded = expandQueryTerms(
    '"Change of Control" the right to terminate',
    expansions,
  );
  assert.ok(
    expanded.startsWith('"Change of Control" the right to terminate'),
    `original query must be preserved verbatim; got: ${expanded}`,
  );
  for (const syn of ["merger", "successor", "acquisition"]) {
    assert.ok(expanded.includes(syn), `expected ${syn} appended; got: ${expanded}`);
  }
  for (const absent of ["restraint", "non-competition"]) {
    assert.ok(
      !expanded.includes(absent),
      `expected ${absent} NOT appended; got: ${expanded}`,
    );
  }
}

{
  assert.strictEqual(expandQueryTerms("anything", {}), "anything", "empty dict is identity");
}

{
  const expanded = expandQueryTerms(
    "What about CHANGE OF CONTROL clauses?",
    { "change of control": ["merger"] },
  );
  assert.ok(expanded.includes("merger"), "case-insensitive key match");
}

{
  // Synonyms must not chain: "merger" is a key, but its synonyms shouldn't
  // be appended just because "merger" was added as a synonym of an earlier
  // key.
  const expanded = expandQueryTerms(
    "change of control clause",
    {
      "change of control": ["merger"],
      "merger": ["consolidation"],
    },
  );
  assert.ok(expanded.includes("merger"));
  assert.ok(
    !expanded.includes("consolidation"),
    `must not recursively expand; got: ${expanded}`,
  );
}

{
  // Two keys share a synonym — dedup: it appears once.
  const expanded = expandQueryTerms(
    "change of control and termination for convenience",
    {
      "change of control": ["merger", "assignment"],
      "termination for convenience": ["assignment", "rescission"],
    },
  );
  const occurrences = expanded.split("assignment").length - 1;
  assert.strictEqual(
    occurrences,
    1,
    `expected dedup of shared synonym; got: ${expanded}`,
  );
}

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

// End-to-end: detect → strip pattern from the docs.
{
  const queries = cuadShape();
  const r = analyzeQuerySet(queries);
  assert.ok(r.isTemplated);
  const stripped = dropTemplateTerms(queries[0], r.boilerplateTerms);
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

// Field types — guard against silent regressions at the FFI boundary.
{
  const r = analyzeQuerySet(cuadShape());
  assert.strictEqual(typeof r.nQueries, "number");
  assert.strictEqual(typeof r.isTemplated, "boolean");
  assert.strictEqual(typeof r.templateWordShare, "number");
  assert.ok(Array.isArray(r.boilerplateTerms));
  assert.ok(r.boilerplateTerms.every((t) => typeof t === "string"));
  assert.strictEqual(typeof r.estimatedDilutionCost, "string");
  assert.strictEqual(typeof r.suggestedAction, "string");
}

console.log("query_set_analyzer.cjs: all assertions passed.");
