// 06 · Chat RAG with chronology preserved — `preserveOrder: true`.
//
// Real-world scenario:
//   A customer-support agent's chat session has been going for an hour
//   and has 30+ turns. Rather than summarizing or compacting the
//   history (lossy), the team retrieves the few past turns relevant to
//   the user's *current* question and pulls those into the LLM prompt.
//   But causality breaks if the retrieved turns are presented in
//   relevance order — "after the refund came in" reads strangely if
//   it's shown before "ordered the laptop." They want the same
//   relevance-driven selection but with chronological emission.
//
// What this demonstrates:
//   - Document.fromChunks(chunks, { preserveOrder: true }) — selection
//     stays relevance-driven, emission becomes chronological.
//   - The contrast between the two modes on the same chat history —
//     the same chunks come back, in different order.
//
// Run:
//   node examples/nodejs/06_chat_rag.cjs

const { Document, Chunk } = require("redhop");

// A 12-turn synthetic chat history. Each turn is one chunk.
const CHAT_HISTORY = [
  ["turn-00", "Hi, I have a question about my order."],
  ["turn-01", "I ordered a laptop last Tuesday."],
  ["turn-02", "It was the new MacBook Air, 15-inch."],
  ["turn-03", "Shipping confirmation came in yesterday — said tomorrow."],
  ["turn-04", "Actually I'd like to cancel and get my money back."],
  ["turn-05", "Sure — what is your refund policy on a shipped order?"],
  ["turn-06", "We offer a thirty-day refund window from the delivery date."],
  ["turn-07", "So I just send it back after it arrives?"],
  ["turn-08", "Yes — print the return label from your Orders page and drop it off."],
  ["turn-09", "Does the refund come right away?"],
  ["turn-10", "We refund within five business days of receiving the return."],
  ["turn-11", "Got it, thanks for your help!"],
];

function buildDoc(preserveOrder) {
  const chunks = CHAT_HISTORY.map(([tid, text]) =>
    new Chunk(text, { source: "chat", id: tid, metadata: { heading: tid } }),
  );
  return Document.fromChunks(chunks, { preserveOrder });
}

function showArm(label, preserveOrder, query) {
  const doc = buildDoc(preserveOrder);
  const ctx = doc.context(query);
  console.log(`─── ${label} ──────────────────────────────────────`);
  console.log("  selected turns, in emission order:");
  for (const c of ctx.citations) {
    console.log(`    ${c.heading}: ${c.text}`);
  }
  console.log();
}

function main() {
  const query = "remind me — when does the refund actually come back?";
  console.log(`Current user question: ${JSON.stringify(query)}\n`);

  // Arm A: default (preserveOrder=false). RedHop selects by relevance
  // and emits in the strategy's order. Wrong for chat.
  showArm("Arm A · default (relevance-emitted)", false, query);

  // Arm B: preserveOrder=true. Same selection, sorted back into
  // source-document order so the LLM reads the turns chronologically.
  showArm("Arm B · preserveOrder=true (chronological)", true, query);

  console.log("Both arms select the same turns by relevance; only the");
  console.log("emission order differs. For the LLM, that ordering controls");
  console.log("whether causality reads correctly — `refund` after `return`");
  console.log("after `ordered`.");
}

main();
