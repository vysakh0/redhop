// Binding-surface tests for `redhop.evaluate` + `EvalReport`.
//
// Mirrors the Rust unit tests on `redhop::evaluate` through the napi
// binding so a dropped EvalReport field, a wrong `goldChunks` shape, or
// a misrouted `EvalGold` variant at the FFI boundary surfaces here, not
// in user code.
//
// The mechanism + "refraction not independent measurement" design
// choice are documented in docs/findings/EVALUATE_API.md.
//
// Run with: node test/evaluate.cjs (or `npm test`).

const assert = require("node:assert");
const {
  analyzeQuerySet,
  buildContext,
  Chunk,
  Stripper,
  evaluate,
} = require("../index.js");

const chunksFor = (text, id = "a") => [new Chunk(text, { id })];

// ─── evaluate without any gold ──────────────────────────────────────────────

{
  const ctx = buildContext(
    "refund window",
    chunksFor("the refund window is thirty days"),
    { strategy: "raw_topk" },
  );
  const r = evaluate("refund window", ctx);
  assert.ok(r.contextRecall == null, "no goldChunks → contextRecall is null/undefined");
  assert.ok(r.contextPrecision == null);
  assert.ok(r.answerTokenRecall == null);
  assert.ok(r.meanGrounding > 0, "matching chunk should produce positive grounding");
  assert.ok(r.overall > 0 && r.overall <= 1);
  assert.strictEqual(typeof r.lowConfidence, "boolean");
}

// ─── goldChunks only ────────────────────────────────────────────────────────

{
  const ctx = buildContext(
    "refund window",
    [
      new Chunk("the refund window is thirty days", { id: "hit1" }),
      new Chunk("refund policy details and timing", { id: "hit2" }),
    ],
    { strategy: "raw_topk" },
  );
  const r = evaluate("refund window", ctx, { goldChunks: ["hit1", "hit2"] });
  assert.strictEqual(r.contextRecall, 1);
  assert.strictEqual(r.contextPrecision, 1);
  assert.ok(r.answerTokenRecall == null);
}

// goldChunks: partial recall, distinct precision.
{
  const ctx = buildContext(
    "policy",
    [
      new Chunk("policy section about refunds", { id: "hit" }),
      new Chunk("totally unrelated cooking recipe", { id: "noise_a" }),
      new Chunk("more cooking instructions", { id: "noise_b" }),
    ],
    { strategy: "raw_topk" },
  );
  const r = evaluate("policy", ctx, { goldChunks: ["hit", "missing"] });
  assert.strictEqual(r.contextRecall, 0.5);
  assert.ok(
    Math.abs(r.contextPrecision - 1 / 3) < 1e-5,
    `expected precision ≈ 1/3; got ${r.contextPrecision}`,
  );
}

// goldChunks: empty list → vacuously perfect recall, undefined precision.
{
  const ctx = buildContext("q", chunksFor("some text"), { strategy: "raw_topk" });
  const r = evaluate("q", ctx, { goldChunks: [] });
  assert.strictEqual(r.contextRecall, 1);
  assert.ok(r.contextPrecision == null);
}

// ─── goldAnswer only ────────────────────────────────────────────────────────

{
  const ctx = buildContext(
    "refund window",
    chunksFor("the refund window is thirty days"),
    { strategy: "raw_topk" },
  );
  const r = evaluate("refund window", ctx, { goldAnswer: "thirty days" });
  assert.ok(r.contextRecall == null);
  assert.ok(r.contextPrecision == null);
  assert.ok(r.answerTokenRecall != null && r.answerTokenRecall > 0);
}

// goldAnswer uses Snowball stemming on both sides ("refunds" → "refund").
{
  const ctx = buildContext(
    "refund window",
    chunksFor("the refund window is thirty days from purchase"),
    { strategy: "raw_topk" },
  );
  const r = evaluate("refund window", ctx, {
    goldAnswer: "refunds within thirty days",
  });
  assert.ok(
    r.answerTokenRecall >= 0.5,
    `stemming should match refunds↔refund; got ${r.answerTokenRecall}`,
  );
}

// ─── both gold signals at once ──────────────────────────────────────────────

{
  const ctx = buildContext(
    "refund window",
    [
      new Chunk("the refund window is thirty days", { id: "hit" }),
      new Chunk("shipping policy details", { id: "noise" }),
    ],
    { strategy: "raw_topk" },
  );
  const r = evaluate("refund window", ctx, {
    goldChunks: ["hit"],
    goldAnswer: "thirty days",
  });
  assert.strictEqual(r.contextRecall, 1);
  assert.ok(r.contextPrecision != null);
  assert.ok(r.answerTokenRecall != null && r.answerTokenRecall > 0);
  assert.ok(r.overall > 0 && r.overall <= 1);
}

// ─── low-confidence caps overall ────────────────────────────────────────────

{
  const ctx = buildContext(
    "quantum chromodynamics gluon coupling",
    [
      new Chunk("the refund window is thirty days", { id: "a" }),
      new Chunk("shipping policy and delivery times", { id: "b" }),
    ],
    { strategy: "raw_topk" },
  );
  const r = evaluate("quantum chromodynamics gluon coupling", ctx);
  assert.strictEqual(r.lowConfidence, true, "off-topic query should flag lowConfidence");
  assert.ok(r.overall <= 0.25, `lowConfidence should cap overall ≤ 0.25; got ${r.overall}`);
}

// ─── detect → strip → evaluate workflow (the user-visible promise) ──────────

{
  const cuadShape = [
    'Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer.',
    'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer.',
    'Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer.',
    'Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer.',
    'Highlight the parts (if any) of this contract related to "Renewal Term" that should be reviewed by a lawyer.',
    'Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer.',
  ];
  const report = analyzeQuerySet(cuadShape);
  assert.ok(report.isTemplated);
  const stripper = new Stripper(report.boilerplateTerms);
  const strip = (q) => stripper.apply(q);
  const raw = cuadShape[0];
  const stripped = strip(raw);
  assert.ok(stripped.includes("Document Name"));
  const chunks = chunksFor("Document Name: Acme Co. Master Agreement", "hit");
  const ctxA = buildContext(raw, chunks, { strategy: "raw_topk" });
  const ctxB = buildContext(stripped, chunks, { strategy: "raw_topk" });
  const evalA = evaluate(raw, ctxA, { goldChunks: ["hit"] });
  const evalB = evaluate(stripped, ctxB, { goldChunks: ["hit"] });
  assert.strictEqual(typeof evalA.overall, "number");
  assert.strictEqual(typeof evalB.overall, "number");
}

// ─── field shapes guard ─────────────────────────────────────────────────────

{
  const ctx = buildContext(
    "refund",
    chunksFor("the refund window is thirty days"),
    { strategy: "raw_topk" },
  );
  const r = evaluate("refund", ctx, {
    goldChunks: ["a"],
    goldAnswer: "thirty days",
  });
  assert.strictEqual(typeof r.contextRecall, "number");
  assert.strictEqual(typeof r.contextPrecision, "number");
  assert.strictEqual(typeof r.answerTokenRecall, "number");
  assert.strictEqual(typeof r.meanGrounding, "number");
  assert.strictEqual(typeof r.evidenceDensity, "number");
  assert.strictEqual(typeof r.retainedEvidenceRatio, "number");
  assert.strictEqual(typeof r.secondHopRescues, "number");
  assert.strictEqual(typeof r.lowConfidence, "boolean");
  assert.strictEqual(typeof r.estimatedWasteTokens, "number");
  assert.strictEqual(typeof r.overall, "number");
}

console.log("evaluate.cjs: all assertions passed.");
