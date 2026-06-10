// 13 · Workload audit — point RedHop's diagnostics at your existing pipeline.
//
// Real-world scenario:
//   A team already has a retrieval pipeline (LangChain BM25 over their
//   contracts, in this sketch). They are not ready to migrate. They
//   want to know, across their last few hundred production queries,
//   *why* retrieval sometimes fails, and which single knob the data
//   says to reach for first.
//
// What this demonstrates:
//   - The bring-your-own-retrieval (BYO) loop: caller-supplied chunks
//     via `redhop.analyzeContext(query, chunks)`. RedHop never owns
//     the retriever; it observes what the retriever returned.
//   - Workload-level aggregation via `redhop.summarizeDiagnoses`.
//     One focus recommendation per workload, with a finding citation.
//   - Layer 1 (BYO, no corpus access) vs Layer 2 (full corpus
//     diagnosis via `Document.fromChunks`). Two lines to upgrade.
//
// Run:
//   npm install redhop
//   node examples/nodejs/13_workload_audit.cjs

const { Chunk, Document, analyzeContext, summarizeDiagnoses } = require("redhop");

// Stand-in for "your existing retriever".
const CORPUS = [
  "Refund Policy. Refunds are available within thirty days of purchase.",
  "Termination for convenience. Either party may terminate this agreement.",
  "Governing Law. This agreement is governed by the laws of California.",
  "Limitation of Liability. The cap is twelve months of fees.",
  "Confidentiality. Each party shall keep the other party's information confidential.",
];

function externalSearch(query, k = 3) {
  const qTerms = new Set(query.toLowerCase().split(/\s+/));
  return CORPUS.map((text) => {
    const score = text.toLowerCase().split(/\s+/).filter((w) => qTerms.has(w)).length;
    return { score, text };
  })
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((r) => r.text);
}

const QUERIES = [];
for (let i = 0; i < 6; i++) {
  QUERIES.push(
    "how do I cancel and get my money back",
    "when can I quit this contract",
    "what is the cap on damages",
    "who keeps secrets",
  );
}
for (let i = 0; i < 4; i++) {
  QUERIES.push(
    "refund policy",
    "termination for convenience",
    "governing law",
    "limitation of liability cap",
  );
}

function main() {
  // ── Layer 1: BYO retrieval ──────────────────────────────────────────
  const layer1Reports = QUERIES.map((q) => {
    const texts = externalSearch(q);
    const chunks = texts.map((t, i) => new Chunk(t, { id: String(i), source: "external" }));
    return analyzeContext(q, chunks);
  });

  console.log("── Layer 1: observe what your retriever returned ──");
  console.log(summarizeDiagnoses(layer1Reports).rendered);

  // ── Layer 2: also point RedHop at the same corpus, once ────────────
  const doc = Document.fromChunks(
    CORPUS.map((t, i) => new Chunk(t, { id: String(i), source: "corpus" })),
  );
  const layer2Reports = QUERIES.map((q) => doc.context(q).report);
  console.log("\n── Layer 2: same queries against an in-memory corpus index ──");
  console.log(summarizeDiagnoses(layer2Reports).rendered);

  // ── Ship attributes to telemetry (OTel/Langfuse-compatible) ────────
  // Same conventions as the Python helper; see docs/DIAGNOSE_YOUR_PIPELINE.md.
  const r = layer2Reports[0];
  const attrs = {
    "redhop.strategy": r.strategy,
    "redhop.auto_decision": r.autoDecision,
    "redhop.input_tokens": Number(r.inputTokens),
    "redhop.total_tokens": Number(r.totalTokens),
    "redhop.retained_evidence_ratio": Number(r.retainedEvidenceRatio),
    "redhop.low_confidence": Boolean(r.lowConfidenceRetrieval),
    "redhop.diagnosis.hints": r.diagnosis.hints.map((h) => h.code),
    "redhop.diagnosis.zero_match_terms": r.diagnosis.zeroMatchTerms.slice(0, 16),
  };
  console.log("\n── Sample telemetry attributes for the first report ──");
  for (const [k, v] of Object.entries(attrs)) {
    console.log(`  ${k} = ${JSON.stringify(v)}`);
  }
}

main();
