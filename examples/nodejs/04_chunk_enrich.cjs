// 04 · Chunk-side enrich — vocab.enrich(chunkText) at ingest.
//
// Real-world scenario:
//   A platform engineering team maintains a runbook keyed by short
//   error codes (ERR_4012, EVT_CHRGBCK, DB_5001). When alerts fire,
//   on-call engineers search the runbook in natural language ("payment
//   declined", "checkout broken", "database timeout") — almost never by
//   the code itself. The runbook entries are short and coded; the
//   natural-language queries share no surface words with them. That's
//   exactly the regime where chunk-side enrich is *predicted* to help.
//
// ⚠ Honest framing (read before applying to your corpus):
//   Enrich is shipped as a primitive on mechanism reasoning with
//   asymmetric measured evidence:
//     - Measured negative: CUAD prose chunks regressed −2.0pt
//       (docs/findings/CUAD_ENRICH_DEFINITIONS_NULL.md).
//     - Measured positive: none on RedHop's eval rigs yet.
//     - This example shows the mechanism on short opaque coded units
//       (the regime where it's predicted to help) — but it is a
//       synthetic demo with a hand-crafted dictionary, not a
//       benchmark. Whether it lifts retention on your runbook depends
//       on your specific corpus and your dictionary.
//   Always A/B with redhop.evaluate(...) against your gold set before
//   adopting in production. See 05_evaluate_ab.cjs for how.
//
// Run:
//   node examples/nodejs/04_chunk_enrich.cjs

const { Document, Chunk, Vocabulary } = require("redhop");

const RUNBOOK_ENTRIES = [
  {
    code: "ERR_4012",
    title: "ERR_4012: PAYMENT_GATEWAY_DECLINED",
    body: "Stripe returned a 4012. Check the customer's card. Common causes: insufficient funds, expired card, blocked transaction. Retry strategy: exponential backoff with a max of 3 attempts.",
  },
  {
    code: "ERR_5001",
    title: "ERR_5001: DB_CONNECTION_TIMEOUT",
    body: "The Postgres pool exhausted. Check `pg_stat_activity` for long-running queries. Restart the worker if connections aren't returning to the pool.",
  },
  {
    code: "EVT_CHRGBCK",
    title: "EVT_CHRGBCK: chargeback notification",
    body: "Stripe sent a chargeback webhook. Flag the order, freeze the customer's account pending review. Respond to Stripe within 7 days with evidence.",
  },
  {
    code: "ERR_6201",
    title: "ERR_6201: SHIPPING_LABEL_INVALID",
    body: "ShipStation rejected the label. Check the customer's address validity. Re-print the label after the address is corrected.",
  },
  {
    code: "ERR_7301",
    title: "ERR_7301: EMAIL_DELIVERY_FAILED",
    body: "SendGrid bounced. Check the recipient's domain status. Most common cause: customer mistyped their email at signup.",
  },
];

// Workload-specific decoder dictionary — *the user supplies this*.
// Each key gets a small set of TERM-SPECIFIC synonyms. Do not add
// generic words like "error", "system", "alert" — those are workload-
// pervasive and re-create the CUAD_PRF_NULL low-IDF dilution failure
// mode (which CUAD_ENRICH_DEFINITIONS_NULL just measured on the chunk
// side).
const ERROR_CODE_VOCAB = {
  ERR_4012: ["payment", "card", "charge", "stripe declined"],
  ERR_5001: ["database", "postgres", "timeout", "connection pool"],
  EVT_CHRGBCK: ["chargeback", "dispute", "refund request"],
  ERR_6201: ["shipping", "label", "address", "delivery"],
  ERR_7301: ["email", "bounce", "deliverability"],
};

function main() {
  const vocab = new Vocabulary(ERROR_CODE_VOCAB);
  console.log(`Compiled vocabulary with ${vocab.length} classes\n`);

  console.log("─── Step 1 · Enrich chunks at ingest ─────────────");
  const chunks = [];
  for (const entry of RUNBOOK_ENTRIES) {
    const chunkText = `${entry.title}\n${entry.body}`;
    // vocab.enrich(text) returns { text, record }. The record is the
    // audit trail — what was matched, what was added — so you can log
    // it at ingest time.
    const { text, record } = vocab.enrich(chunkText);
    if (record.matched.length) {
      console.log(
        `  ${entry.code.padStart(14)} ← matched=${JSON.stringify(record.matched)} added=${JSON.stringify(record.added)}`,
      );
    }
    chunks.push(new Chunk(text, {
      source: `runbook/${entry.code}.md`,
      id: entry.code,
      metadata: { heading: entry.title },
    }));
  }
  console.log();

  const doc = Document.fromChunks(chunks);
  const query = "customer's card got declined at checkout, what do we do?";
  console.log("─── Step 2 · Query (natural language) ────────────");
  console.log(`  ${JSON.stringify(query)}\n`);

  const ctx = doc.context(query);
  console.log("─── Top hit ──────────────────────────────────────");
  const top = ctx.citations[0];
  console.log(`  source : ${top.source}`);
  console.log(`  heading: ${top.heading}`);
  console.log(`  excerpt: ${top.text.slice(0, 100).replace(/\n/g, " ")}…`);
  console.log();

  console.log("Mechanism: the query has no overlap with the bare error code");
  console.log("`ERR_4012` — the match landed via the appended `payment`/`card`/");
  console.log("`charge` tokens that enrich attached at ingest. On your real");
  console.log("runbook, A/B with redhop.evaluate(...) against a gold set (see");
  console.log("05_evaluate_ab.cjs) before committing to this in production.");
}

main();
