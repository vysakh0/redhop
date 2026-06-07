// 09 · Multilingual analyzer — `language: "german"` / "french" / ...
//
// Real-world scenario:
//   A pharmaceutical company has internal policy documents in
//   English, German, and French. They want BM25-quality retrieval on
//   each — which means the tokenizer needs to understand each
//   language's morphology. German `Bücher` should find a chunk that
//   only contains `Buch`; French `manger` should find a chunk with
//   `mange`. The default English analyzer would miss both.
//
//   RedHop ships an 18-language analyzer matrix (the Snowball Porter2
//   family). Pass `language: "german"` to `Document.from*` and the
//   whole pipeline (chunking-side tokenization, BM25 stemming,
//   grounding scorer) uses the German analyzer.
//
// What this demonstrates:
//   - `Document.fromChunks(chunks, { language: "german" })` and the
//     same on `fromText`, `fromFile`, `fromFolder`.
//   - The 18 supported languages: arabic, danish, dutch, english,
//     finnish, french, german, greek, hungarian, italian, norwegian,
//     portuguese, romanian, russian, spanish, swedish, tamil, turkish.
//   - Why morphology unification matters: a German query for `Buch`
//     lands on a chunk containing only the plural `Bücher`.
//   - That unknown language strings ERROR (no silent fallback to
//     English) — caught by docs/findings/MULTILINGUAL_ANALYZER.md.
//
// Run:
//   node examples/nodejs/09_multilingual.cjs

const { Document, Chunk } = require("redhop");

const GERMAN_CORPUS = [
  new Chunk("Ich habe viele Bücher im Regal stehen.", { id: "de-1" }),
  new Chunk("Ein Kind spielt fröhlich im Garten.", { id: "de-2" }),
  new Chunk("Der Hund läuft schnell durch den Park.", { id: "de-3" }),
];

const FRENCH_CORPUS = [
  new Chunk("Il aime manger des pommes chaque matin.", { id: "fr-1" }),
  new Chunk("Le chien court dans la rue très vite.", { id: "fr-2" }),
  new Chunk("Les enfants jouent au parc le weekend.", { id: "fr-3" }),
];

function demo(label, corpus, query, language) {
  const doc = Document.fromChunks(corpus, { language });
  const ctx = doc.context(query);
  console.log(`─── ${label} ────────────────────────────────`);
  console.log(`  language=${JSON.stringify(language)}, query=${JSON.stringify(query)}`);
  if (ctx.citations.length) {
    console.log(`  top hit: ${ctx.citations[0].text}`);
  } else {
    console.log("  (no hits)");
  }
  console.log();
}

function main() {
  // German Snowball: "Buch" (singular) should reach "Bücher" (plural).
  demo("Arm A · German morphology", GERMAN_CORPUS, "Buch", "german");

  // French Snowball: "manger" (infinitive) should reach "mange".
  demo("Arm B · French morphology", FRENCH_CORPUS, "manger", "french");

  // Counter-example: same German corpus, default English analyzer.
  demo(
    "Arm C · German corpus + English analyzer (the bug it prevents)",
    GERMAN_CORPUS,
    "Buch",
    "english",
  );

  // Unknown language: deliberate error rather than silent English
  // fallback.
  console.log("─── Arm D · Unknown language string ──────────────");
  try {
    Document.fromChunks(GERMAN_CORPUS, { language: "germann" });
    console.log("  (oops — should have raised)");
  } catch (e) {
    console.log(`  Error: ${e.message.slice(0, 140)}…`);
  }
  console.log();

  console.log("─── How to read this ─────────────────────────────");
  console.log("Arm A and B: language=… routes the whole pipeline through");
  console.log("  the right Snowball stemmer.");
  console.log("Arm C: same German corpus + default English analyzer → miss.");
  console.log("  Picking the wrong language SILENTLY (e.g. forgetting to");
  console.log("  pass language=) is the real failure mode.");
  console.log("Arm D: unknown language strings ERROR with the supported");
  console.log("  list so a typo'd 'germann' is caught at construction.");
  console.log();
  console.log("Validated cross-language by docs/findings/MULTILINGUAL_ANALYZER.md.");
}

main();
