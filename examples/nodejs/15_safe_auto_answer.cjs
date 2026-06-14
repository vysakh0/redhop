// 15 · Safe auto-answers — when should a chatbot answer vs ask?
//
// Real-world scenario:
//   A US store's help bot answers FAQs. The expensive failure is a
//   *confident wrong answer*, so the bot should auto-answer only when
//   retrieval clearly matched, and otherwise ask a clarifying question
//   (or hand off). RedHop does not ship a router or a threshold — it
//   gives you the *signals* and a deterministic eval to measure the gate.
//   You own the "if confident then answer, else ask" logic. This is the
//   pattern from the safe-auto-answers guide.
//
// What this demonstrates:
//   - ctx.report.lowConfidenceRetrieval — the primary gate ("nothing
//     relevant matched").
//   - evaluate(query, ctx).meanGrounding — a no-gold confidence
//     *strength* in [0,1] (how query-relevant the assembled context is).
//     Confidence is a measured signal, not the model's self-report.
//   - evaluate(query, ctx, { goldChunks }) to MEASURE the gate on a
//     labeled set: auto-precision (correct among auto-answered) and
//     unsafe-auto (auto-answered when it should have asked, target 0).
//   - The headline: a good gate "gets cautious, not wrong" — it routes
//     weak retrievals to clarify, keeping auto-precision high and
//     unsafe-auto at 0.
//
//   tau here is illustrative. In production you DERIVE it: sweep on a
//   labeled dev set and pick the smallest tau hitting your precision
//   target (e.g. 99%). See the guide.
//
// Run:
//   npm install redhop
//   node examples/nodejs/15_safe_auto_answer.cjs

const { Document, Chunk, evaluate } = require("redhop");

const FAQ = [
  ["faq-refund", "Refunds. Return any item within 30 days for a full refund, no questions asked."],
  ["faq-shipping", "Shipping. Standard shipping is free on orders over 35 dollars and arrives in 5 to 7 business days."],
  ["faq-hours", "Store hours. Our stores are open 9am to 9pm Monday through Saturday, and 10am to 6pm on Sunday."],
  ["faq-giftcard", "Gift cards. Gift cards never expire and can be used online or in any store."],
  ["faq-track", "Order tracking. Track your order from the Orders page using the tracking number in your confirmation email."],
];

// Labeled eval set: each query maps to the FAQ id that answers it, or null
// when there is no confident answer (the bot SHOULD ask, not guess).
const LABELED = [
  ["how do I return something for a refund", "faq-refund"],
  ["when are you open on sunday", "faq-hours"],
  ["how do I track my package", "faq-track"],
  ["do gift cards expire", "faq-giftcard"],
  ["can you help me", null],             // too vague — should ask
  ["do you price match competitors", null],  // not in the KB — should ask
];

// Illustrative threshold. DERIVE this on a dev set in production (see guide):
// sweep tau and pick the smallest value that hits your auto-precision target.
const TAU = 0.2;

const pad = (s, n) => String(s).slice(0, n).padEnd(n);
const padL = (s, n) => String(s).padStart(n);

function main() {
  const doc = Document.fromChunks(FAQ.map(([id, text]) => new Chunk(text, { id, source: "faq" })));
  console.log("Routing each query AUTO vs CLARIFY on redhop's confidence signals.");
  console.log(`(AUTO only when retrieval is confident: not low_confidence AND grounding >= ${TAU})\n`);
  console.log(`  ${pad("query", 38)} ${padL("low_conf", 9)} ${padL("grounding", 9)} ${padL("route", 8)}  outcome`);

  let autoTotal = 0, autoCorrect = 0, unsafeAuto = 0, clarifyTotal = 0;
  for (const [query, gold] of LABELED) {
    const ctx = doc.context(query);
    // One eval per query; meanGrounding is a self-eval populated with or
    // without gold. Pass gold (when we have it) to also check correctness.
    const r = gold ? evaluate(query, ctx, { goldChunks: [gold] }) : evaluate(query, ctx);
    const low = ctx.report.lowConfidenceRetrieval;
    const grounding = r.meanGrounding;
    const auto = !low && grounding >= TAU;
    const goldPresent = !!gold && (r.contextRecall ?? 0) >= 1.0;

    let outcome;
    if (auto) {
      autoTotal += 1;
      if (gold && goldPresent) { autoCorrect += 1; outcome = "AUTO ✓ correct"; }
      else if (gold) { outcome = "AUTO ✗ WRONG (auto-answered, missed the gold)"; }
      else { unsafeAuto += 1; outcome = "AUTO ☠ UNSAFE (should have asked)"; }
    } else {
      clarifyTotal += 1;
      outcome = "clarify (asks the user)";
    }

    console.log(
      `  ${pad(query, 38)} ${padL(low, 9)} ${padL(grounding.toFixed(2), 9)} ${padL(auto ? "AUTO" : "CLARIFY", 8)}  ${outcome}`,
    );
  }

  const autoPrecision = autoTotal ? autoCorrect / autoTotal : 1.0;
  const n = LABELED.length;
  console.log("\n─── Scorecard ────────────────────────────────────");
  console.log(`  auto-resolve rate   : ${autoTotal}/${n} answered without asking`);
  console.log(`  auto-precision ⭐    : ${autoPrecision.toFixed(3)}  (correct among auto-answered; aim >= 0.99)`);
  console.log(`  unsafe-auto ☠       : ${unsafeAuto}      (auto-answered when it should have asked; target 0)`);
  console.log(`  clarify rate        : ${clarifyTotal}/${n} routed to a question`);
  console.log("\nThe gate degrades weak retrievals to clarify, so the bot");
  console.log("'gets cautious, not wrong'. DERIVE tau on your own dev set by");
  console.log("sweeping it to your precision target — see the safe-auto-answers guide.");
}

main();
