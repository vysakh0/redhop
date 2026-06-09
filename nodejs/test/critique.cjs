// Phase 7: aspect-critique smoke / behavior tests via the async
// `critique` napi entry-point. Same Judge surface as
// `evaluateWithJudge`; one judge call per aspect; polarity-corrected
// scores so high = good across the report regardless of an aspect's
// `highIsGood` flag.

const assert = require("node:assert");
const { Judge, critique } = require("../index.js");

(async function main() {
  // 1. Each aspect produces one judge call; scores preserved in order.
  {
    let calls = 0;
    const judge = Judge.fromCallable((err, prompt, system) => {
      if (err) throw err;
      calls++;
      return 0.7;
    }, "stub");
    const aspects = [
      { name: "a", definition: "First aspect" },
      { name: "b", definition: "Second aspect" },
      { name: "c", definition: "Third aspect" },
    ];
    const report = await critique("Some answer.", aspects, judge);
    assert.strictEqual(report.n, 3);
    assert.strictEqual(report.scores.length, 3);
    assert.strictEqual(calls, 3);
    for (const s of report.scores) {
      assert.ok(s.score != null && Math.abs(s.score - 0.7) < 0.01,
        `${s.name} score ≈ 0.7, got ${s.score}`);
    }
    // Order preserved.
    assert.deepStrictEqual(
      report.scores.map((s) => s.name),
      ["a", "b", "c"],
    );
  }

  // 2. highIsGood:false inverts the raw score.
  {
    const judge = Judge.fromCallable((err) => 0.9, "harmful-stub");
    const report = await critique("anything", [
      { name: "harmfulness", definition: "Is it harmful?", highIsGood: false },
    ], judge);
    const s = report.scores[0].score;
    assert.ok(Math.abs(s - 0.1) < 0.01,
      `harmfulness raw 0.9 → inverted 0.1, got ${s}`);
  }

  // 3. Empty aspects makes zero judge calls.
  {
    let calls = 0;
    const judge = Judge.fromCallable(() => { calls++; return 0.5; }, "stub");
    const report = await critique("x", [], judge);
    assert.strictEqual(report.n, 0);
    assert.strictEqual(report.scores.length, 0);
    assert.strictEqual(calls, 0);
  }

  // 4. Judge error on one aspect leaves only that one's score null.
  {
    let n = 0;
    const judge = Judge.fromCallable((err, prompt, system) => {
      if (err) throw err;
      n++;
      if (n === 2) throw new Error("transient");
      return 0.6;
    }, "flaky");
    const aspects = [
      { name: "a", definition: "first" },
      { name: "b", definition: "second" },
      { name: "c", definition: "third" },
    ];
    const report = await critique("x", aspects, judge);
    assert.ok(report.scores[0].score != null, "a should have a score");
    assert.ok(report.scores[1].score == null, "b should be null on transient error");
    assert.ok(report.scores[2].score != null, "c should have a score");
  }

  // 5. context + query are optional.
  {
    const judge = Judge.fromCallable(() => 0.8, "stub");
    const aspects = [{ name: "x", definition: "test" }];
    const r1 = await critique("answer", aspects, judge);
    const r2 = await critique("answer", aspects, judge, { context: "ctx" });
    const r3 = await critique("answer", aspects, judge, { context: "ctx", query: "q" });
    for (const r of [r1, r2, r3]) {
      assert.ok(Math.abs(r.scores[0].score - 0.8) < 0.01);
    }
  }

  // 6. highIsGood defaults to true (omitting it gives raw score back).
  {
    const judge = Judge.fromCallable(() => 0.6, "stub");
    const report = await critique("x", [
      { name: "x", definition: "Some property" },  // no highIsGood
    ], judge);
    assert.ok(Math.abs(report.scores[0].score - 0.6) < 0.01,
      "default polarity should keep raw score 0.6");
  }

  console.log("critique.cjs: all assertions passed.");
})().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
