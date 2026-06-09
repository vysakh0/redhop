// LLM-judged metrics via the async `evaluateWithJudge` path.
// JS is single-threaded so the judge callback can't fire during a sync
// napi call — the binding moves the eval onto a tokio spawn_blocking
// worker, calls back into JS via a ThreadsafeFunction, and blocks the
// worker on a future until the callback returns. From the JS user's
// perspective it's just `await evaluateWithJudge(...)`.
//
// The Rust unit tests in crates/redhop/src/context/eval.rs and the
// Python tests in python/tests/test_evaluate.py are authoritative on
// the metric semantics; this file guards the napi callback surface.

const assert = require("node:assert");
const { Document, Judge, evaluateWithJudge } = require("../index.js");

const chunkText = "the refund window is thirty days from purchase. customers may return items in original packaging.";

(async function main() {
  // 1. End-to-end: judge populates all three _judged metrics.
  {
    const doc = Document.fromText(chunkText);
    const ctx = doc.context("refund window");
    let calls = 0;
    const judge = Judge.fromCallable((err, prompt, system) => {
      calls++;
      return 0.85;
    }, "stub");
    const report = await evaluateWithJudge("refund window", ctx, judge, {
      answer: "Thirty days from purchase.",
      goldAnswer: "thirty days",
    });
    assert.ok(report.faithfulnessJudged != null, "faithfulnessJudged populated");
    assert.ok(report.relevancyJudged != null, "relevancyJudged populated");
    assert.ok(report.correctnessJudged != null, "correctnessJudged populated");
    assert.strictEqual(calls, 3, `expected 3 judge calls, got ${calls}`);
  }

  // 2. correctness_judged requires goldAnswer (otherwise stays null).
  {
    const doc = Document.fromText(chunkText);
    const ctx = doc.context("refund window");
    let calls = 0;
    const judge = Judge.fromCallable((err, prompt, system) => { calls++; return 0.7; }, "stub");
    const report = await evaluateWithJudge("refund window", ctx, judge, {
      answer: "Thirty days from purchase.",
    });
    assert.ok(report.faithfulnessJudged != null);
    assert.ok(report.relevancyJudged != null);
    assert.ok(report.correctnessJudged == null, "no goldAnswer → no correctness");
    assert.strictEqual(calls, 2, `expected 2 judge calls (no correctness), got ${calls}`);
  }

  // 3. `.cached()` suppresses repeat calls for identical prompts.
  {
    const doc = Document.fromText(chunkText);
    const ctx = doc.context("refund window");
    let calls = 0;
    const judge = Judge.fromCallable((err, prompt, system) => { calls++; return 0.9; }, "stub").cached();
    await evaluateWithJudge("refund window", ctx, judge, {
      answer: "Thirty days.",
      goldAnswer: "thirty days",
    });
    assert.strictEqual(calls, 3, "first run hits all 3 prompts");
    await evaluateWithJudge("refund window", ctx, judge, {
      answer: "Thirty days.",
      goldAnswer: "thirty days",
    });
    assert.strictEqual(calls, 3, "second run with identical inputs serves cache");
  }

  // 4. JS-side exception in the callable leaves _judged metrics null,
  //    but lexical fields stay populated — failure is isolated.
  {
    const doc = Document.fromText(chunkText);
    const ctx = doc.context("refund window");
    const judge = Judge.fromCallable((err, prompt, system) => { throw new Error("transport"); }, "failing");
    const report = await evaluateWithJudge("refund window", ctx, judge, {
      answer: "Thirty days.",
      goldAnswer: "thirty days",
    });
    assert.ok(report.faithfulnessJudged == null, "JS error → judged stays null");
    assert.ok(report.relevancyJudged == null);
    assert.ok(report.correctnessJudged == null);
    assert.ok(report.faithfulnessLexical != null, "lexical fields unaffected");
    assert.ok(report.relevancyLexical != null);
  }

  // 5. Judge.name + Judge.cached().name match.
  {
    const judge = Judge.fromCallable((err, prompt, system) => 0.5, "myname");
    assert.strictEqual(judge.name, "myname");
    assert.strictEqual(judge.cached().name, "myname");
  }

  // 6. claim-decomposition path. With decomposeFaithfulness:true the
  //    judge is called once for extraction (system mentions "Decompose
  //    answers") to return raw claim text, once for batched verification
  //    (returns one `N: SCORE` line per claim), and once for relevancy.
  //    Three total LLM calls regardless of how many claims were extracted.
  {
    const ctx = Document.fromText(chunkText).context("refund window");
    let extractionCalls = 0;
    let batchedVerificationCalls = 0;
    const judge = Judge.fromCallable((err, prompt, system) => {
      if (system && system.includes("Decompose answers")) {
        extractionCalls++;
        return "claim a\nclaim b\nclaim c";
      }
      // Batched verification: return one `N: score` per claim.
      if (prompt.includes("Claims to verify")) {
        batchedVerificationCalls++;
        return "1: 0.8\n2: 0.8\n3: 0.8";
      }
      // Other arms (relevancy, etc.)
      return 0.8;
    }, "decomposer");
    const r = await evaluateWithJudge("refund window", ctx, judge, {
      answer: "Three claims.",
      decomposeFaithfulness: true,
    });
    assert.ok(r.faithfulnessJudged != null);
    assert.ok(Math.abs(r.faithfulnessJudged - 0.8) < 0.01,
      `faithfulnessJudged ≈ 0.8, got ${r.faithfulnessJudged}`);
    assert.strictEqual(r.nFaithfulnessClaimsExtracted, 3);
    assert.strictEqual(r.nFaithfulnessClaimsSupported, 3, "0.8 >= 0.5 → all supported");
    assert.strictEqual(extractionCalls, 1, "exactly 1 extraction call");
    assert.strictEqual(batchedVerificationCalls, 1, "exactly 1 batched verification call");
  }

  // 7. Zero claims extracted → metric stays null.
  {
    const ctx = Document.fromText(chunkText).context("refund window");
    const judge = Judge.fromCallable((err, prompt, system) => {
      if (system && system.includes("Decompose answers")) {
        return "";  // empty string = no claims
      }
      return 0.5;
    }, "decomposer-empty");
    const r = await evaluateWithJudge("refund window", ctx, judge, {
      answer: "I cannot answer that.",
      decomposeFaithfulness: true,
    });
    assert.ok(r.faithfulnessJudged == null,
      "no claims extracted → faithfulnessJudged null");
    assert.ok(r.nFaithfulnessClaimsExtracted == null);
    assert.ok(r.nFaithfulnessClaimsSupported == null);
  }

  console.log("judge.cjs: all assertions passed.");
})().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
