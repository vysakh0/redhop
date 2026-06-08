// Tier-2 LLM-judged metrics via the async `evaluateWithJudge` path.
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
    const judge = Judge.fromCallable((prompt, system) => {
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
    const judge = Judge.fromCallable(() => { calls++; return 0.7; }, "stub");
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
    const judge = Judge.fromCallable(() => { calls++; return 0.9; }, "stub").cached();
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
    const judge = Judge.fromCallable(() => { throw new Error("transport"); }, "failing");
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
    const judge = Judge.fromCallable(() => 0.5, "myname");
    assert.strictEqual(judge.name, "myname");
    assert.strictEqual(judge.cached().name, "myname");
  }

  console.log("judge.cjs: all assertions passed.");
})().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
