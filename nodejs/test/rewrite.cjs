// Binding-surface tests for Stripper, Vocabulary, and the
// `Document.contextWithRewrites(...)` chain.
//
// Mirrors `crates/redhop/src/rewrite.rs` tests through the napi
// boundary, so a dropped field on RewriteRecord or a wrong
// Vec<String> ↔ Array<string> mapping at the FFI edge surfaces here,
// not in user code.
//
// Mechanism: docs/findings/CUAD_CLAUSE_EXPANSION.md and the
// `QueryRewrite` trait in `crates/redhop/src/rewrite.rs`.
//
// Run with: node test/rewrite.cjs (or `npm test`).

const assert = require("node:assert");
const { Document, Stripper, Vocabulary } = require("../index.js");

// ── Stripper ───────────────────────────────────────────────────────────────

{
  const stripper = new Stripper(["highlight", "the", "parts", "of", "related", "to"]);
  const out = stripper.apply('Highlight the parts related to "Change of Control".');
  // "of" is token-level stripped everywhere, including inside the quoted phrase.
  assert.strictEqual(out, '"Change Control".', "drops listed tokens, preserves punctuation");
}

{
  const stripper = new Stripper(["highlight", "the", "parts", "related", "to"]);
  const out = stripper.apply('Highlight the parts related to "Change of Control".');
  assert.strictEqual(out, '"Change of Control".', "preserves internal words not in boilerplate");
}

{
  const stripper = new Stripper(["of", "the"]);
  const out = stripper.apply("the office is open");
  assert.strictEqual(out, "office is open", "word-boundary safe for short tokens");
}

{
  const stripper = new Stripper([]);
  const q = "Highlight the parts of this contract";
  assert.strictEqual(stripper.apply(q), q, "empty boilerplate is identity");
}

{
  const stripper = new Stripper(["a", "b", "c"]);
  assert.strictEqual(stripper.length, 3, "len reflects boilerplate count");
}

// ── Vocabulary ─────────────────────────────────────────────────────────────

{
  const vocab = new Vocabulary({
    "change of control": ["merger", "successor", "acquisition"],
  });
  const out = vocab.apply('"Change of Control" the right to terminate');
  assert.ok(out.startsWith('"Change of Control" the right to terminate'));
  for (const syn of ["merger", "successor", "acquisition"]) {
    assert.ok(out.includes(syn), `expected synonym ${syn} appended; got: ${out}`);
  }
}

{
  // Short-acronym key must NOT substring-fire inside "recipient".
  const vocab = new Vocabulary({ ip: ["intellectual property"] });
  const out = vocab.apply("the recipient agrees to the terms");
  assert.ok(
    !out.includes("intellectual property"),
    `expected no substring fire on 'ip' inside 'recipient'; got: ${out}`,
  );
}

{
  const vocab = new Vocabulary({ ip: ["intellectual property"] });
  const out = vocab.apply("the IP license terms");
  assert.ok(out.includes("intellectual property"), `expected token match on 'IP'; got: ${out}`);
}

{
  const vocab = Vocabulary.bidirectional({ pto: ["paid time off", "vacation"] });
  const out = vocab.apply("how much PTO do I get");
  assert.ok(out.includes("paid time off"));
  assert.ok(out.includes("vacation"));
}

{
  const vocab = Vocabulary.bidirectional({ pto: ["paid time off", "vacation"] });
  const out = vocab.apply("vacation policy details");
  assert.ok(out.toLowerCase().includes("pto"), "bidirectional: synonym side triggers acronym");
  assert.ok(out.includes("paid time off"));
}

{
  const vocab = new Vocabulary({ pto: ["paid time off", "vacation"] });
  const out = vocab.apply("vacation policy details");
  // Asymmetric mode: "vacation" is not the trigger.
  assert.ok(!out.toLowerCase().includes("pto"), `asymmetric should not fire from synonym; got: ${out}`);
}

{
  // No recursive chaining.
  const vocab = new Vocabulary({
    "change of control": ["merger"],
    merger: ["consolidation"],
  });
  const out = vocab.apply("change of control clause");
  assert.ok(out.includes("merger"));
  assert.ok(!out.includes("consolidation"), `expected no recursion; got: ${out}`);
}

{
  // Dedup shared synonyms across matches.
  const vocab = new Vocabulary({
    "change of control": ["merger", "assignment"],
    "termination for convenience": ["assignment", "rescission"],
  });
  const out = vocab.apply("change of control and termination for convenience");
  const occurrences = out.split("assignment").length - 1;
  assert.strictEqual(occurrences, 1, `expected dedup of shared synonym; got: ${out}`);
}

{
  const vocab = new Vocabulary({});
  assert.strictEqual(vocab.apply("anything"), "anything", "empty vocab is identity");
}

{
  const vocab = new Vocabulary({ a: ["b"], c: ["d", "e"] });
  assert.strictEqual(vocab.length, 2, "len reflects class count");
}

// ── Document.contextWithRewrites + audit trail ────────────────────────────

{
  const text = [
    "Change of Control means a merger or sale of substantially all assets.",
    "The parties to this Agreement are Acme Co. and Beta Inc.",
    "Notices shall be sent to the address listed in Schedule A.",
  ].join("\n\n");
  const doc = Document.fromText(text);
  const stripper = new Stripper([
    "highlight", "the", "parts", "of", "this", "contract", "related",
    "to", "reviewed", "by", "a", "lawyer",
  ]);
  const vocab = new Vocabulary({
    "change of control": ["merger", "successor", "acquisition"],
  });
  const ctx = doc.contextWithRewrites(
    'Highlight the parts of this contract related to "Change of Control" reviewed by a lawyer.',
    [stripper, vocab],
  );
  const records = ctx.report.queryRewrites;
  assert.strictEqual(records.length, 2, "two stages → two records");
  assert.strictEqual(records[0].stage, "strip");
  assert.strictEqual(records[1].stage, "vocabulary");
  assert.ok(records[1].added.includes("merger"), `expected merger appended; got: ${JSON.stringify(records[1])}`);
  // Second stage's output is what retrieval ran against; "to" string must
  // start with the input "from" string (vocab appends, doesn't replace).
  assert.ok(records[1].toQuery.startsWith(records[1].fromQuery));
}

// Empty chain.
{
  const doc = Document.fromText("Alpha clause about X.\n\nBeta clause about Y.\n\nGamma clause about Z.");
  const ctx = doc.contextWithRewrites("X clause", []);
  assert.deepStrictEqual(ctx.report.queryRewrites, []);
}

console.log("rewrite: OK");
