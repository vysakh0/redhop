// 02 · Structured corpus — `new Chunk(...)` for content you already
//      chunked elsewhere, with metadata that flows through to citations.
//
// Real-world scenario:
//   A SaaS company has a customer-support knowledge base: each FAQ pair
//   is one row in a database (question, answer, category, lastUpdated,
//   articleUrl). Support agents query it in natural language while
//   chatting with a customer. They need:
//     - Citations that point back to a specific article (`articleUrl`)
//     - Metadata visible on the citation (e.g. `category`, `lastUpdated`)
//   The 0.3.0 typed `Chunk` constructor + open `metadata` object is what
//   makes this clean — pre-0.3.0 the dict path couldn't carry arbitrary
//   metadata onto citations.
//
// What this demonstrates:
//   - `new Chunk(text, { source, id, metadata })` — typed constructor
//     for hand-built chunks.
//   - source vs id: `source` is *provenance* (what citations display);
//     `id` is *identity* (stable handle for dedup / gold).
//   - Citations pick up known metadata keys: `page`, `heading`, `line`
//     are surfaced on `ctx.citations[i]`. Arbitrary metadata (your own
//     keys like `category`) is preserved on the chunk but not yet
//     surfaced through citations — keep a parallel object keyed by
//     chunk id if you need them at display time.
//   - `Document.fromChunks(chunks)` — no chunker re-split; what you
//     pass in is what gets indexed, 1-to-1.
//
// Run:
//   node examples/nodejs/02_structured_corpus.cjs

const { Document, Chunk } = require("redhop");

// Toy FAQ corpus — eight Q&A pairs across four categories. In production
// you'd pull these from your DB / CMS / CSV; the shape doesn't change.
const FAQ_ROWS = [
  {
    id: "faq-001",
    category: "billing",
    question: "When is my credit card charged?",
    answer: "Your card is charged on the first day of each billing cycle. You can view upcoming charges under Settings → Billing.",
    url: "https://help.acme.com/billing/charge-date",
    lastUpdated: "2026-04-12",
  },
  {
    id: "faq-002",
    category: "billing",
    question: "How do I request a refund?",
    answer: "Refunds are available within 30 days of charge. Email finance@acme.com with your invoice number and reason. We process refunds within 5 business days.",
    url: "https://help.acme.com/billing/refunds",
    lastUpdated: "2026-05-03",
  },
  {
    id: "faq-003",
    category: "account",
    question: "How do I change my email address?",
    answer: "Settings → Account → Email. We send a confirmation link to the new address; click it within 24 hours to complete the change.",
    url: "https://help.acme.com/account/email",
    lastUpdated: "2026-03-21",
  },
  {
    id: "faq-004",
    category: "account",
    question: "How do I delete my account?",
    answer: "Settings → Account → Delete Account. We retain billing records for 7 years for tax compliance but anonymize all profile data immediately.",
    url: "https://help.acme.com/account/delete",
    lastUpdated: "2026-02-18",
  },
  {
    id: "faq-005",
    category: "shipping",
    question: "When will my order arrive?",
    answer: "Standard shipping is 3-5 business days. Express is 1-2 days. You'll get a tracking link by email once the package leaves our warehouse.",
    url: "https://help.acme.com/shipping/delivery-time",
    lastUpdated: "2026-05-30",
  },
  {
    id: "faq-006",
    category: "shipping",
    question: "Can I change my shipping address after ordering?",
    answer: "Yes, if the order hasn't shipped yet. Go to Orders → Edit. After shipment we cannot reroute — you'll need to contact the carrier directly.",
    url: "https://help.acme.com/shipping/change-address",
    lastUpdated: "2026-04-05",
  },
  {
    id: "faq-007",
    category: "returns",
    question: "What is your return policy?",
    answer: "Unworn items in original packaging may be returned within 30 days of delivery for a full refund. Print a prepaid label from Orders → Return.",
    url: "https://help.acme.com/returns/policy",
    lastUpdated: "2026-05-15",
  },
  {
    id: "faq-008",
    category: "returns",
    question: "Do you cover return shipping?",
    answer: "Yes — return shipping is free in the US for unworn items. International returns are paid by the customer.",
    url: "https://help.acme.com/returns/shipping-costs",
    lastUpdated: "2026-04-22",
  },
];

function buildChunks(rows) {
  // Decisions worth noting:
  //   - `text` combines question + answer so retrieval sees both.
  //   - `source` is the article URL — that's what `ctx.citations[*].source`
  //     will display.
  //   - `id` is the FAQ row id — stable across runs.
  //   - `metadata` carries the rest. `heading` is a known citation
  //     key; others ride along.
  return rows.map((r) => new Chunk(
    `Q: ${r.question}\nA: ${r.answer}`,
    {
      source: r.url,
      id: r.id,
      metadata: {
        category: r.category,
        lastUpdated: r.lastUpdated,
        heading: r.question,
      },
    },
  ));
}

function main() {
  const chunks = buildChunks(FAQ_ROWS);
  const doc = Document.fromChunks(chunks);
  console.log(`Indexed ${doc.chunkCount} FAQ entries.\n`);

  // A real customer query. BM25 matches "refund" + "deadline" against
  // the refunds FAQ; the result is one citation pointing at the
  // billing/refunds article URL.
  const query = "what's the deadline for getting a refund?";
  console.log(`Query: ${JSON.stringify(query)}\n`);

  const ctx = doc.context(query);
  console.log("─── Top hit ──────────────────────────────────────");
  const cite = ctx.citations[0];
  console.log(`  source        : ${cite.source}`);
  console.log(`  heading       : ${cite.heading}`);

  // `category` and `lastUpdated` aren't first-class citation fields,
  // but we attached them to the chunk's metadata. Look them up by
  // source URL from the original rows.
  const row = FAQ_ROWS.find((r) => r.url === cite.source);
  if (row) {
    console.log(`  category      : ${row.category}`);
    console.log(`  lastUpdated   : ${row.lastUpdated}`);
  }
  console.log(`  text (excerpt): ${cite.text.slice(0, 80)}…`);
  console.log();

  console.log("─── Decision Report ──────────────────────────────");
  console.log(`  Final context tokens : ${ctx.report.totalTokens}`);
  console.log(`  Decision             : ${ctx.report.autoDecision} (strategy=${ctx.report.strategy})`);
  console.log(`  Chunks selected      : ${ctx.report.nSelected} of ${ctx.report.nInputChunks}`);
}

main();
