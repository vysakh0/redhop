// Binding-surface tests for the pluggable lexical analyzer.
//
// Mirrors the Rust `quality_suite::t41`-`t44` matrix through the napi
// binding so a dropped `language` field or a wrong string-to-analyzer
// mapping at the FFI boundary surfaces here, not in user code.
//
// Run with: node test/analyzer.cjs   (or `npm test` — wired into the
// default test script).

const assert = require("node:assert");
const { Chunk, Document } = require("../index.js");

const germanCorpus = () => [
  new Chunk("ich habe viele Bücher im Regal stehen"),
  new Chunk("ein Kind spielt fröhlich im Garten"),
];

const frenchCorpus = () => [
  new Chunk("il aime manger des pommes chaque matin"),
  new Chunk("le chien court dans la rue très vite"),
];

// ── Per-language behavior (mirrors Rust T41/T42) ─────────────────────────

// T41: German Snowball — `Buch` query should reach a chunk containing
// only the plural `Bücher`.
{
  const ctx = Document.fromChunks(germanCorpus(), { language: "german" }).context("Buch");
  assert.ok(
    ctx.text.includes("Bücher"),
    `German analyzer should unify Bücher↔Buch; got: ${JSON.stringify(ctx.text)}`,
  );
}

// T42: French Snowball — `manger` query should reach a chunk with the
// conjugated form `mange`.
{
  const ctx = Document.fromChunks(frenchCorpus(), { language: "french" }).context("manger");
  assert.ok(
    ctx.text.includes("mange"),
    `French analyzer should unify manger↔mange; got: ${JSON.stringify(ctx.text)}`,
  );
}

// from_text is a different code path from from_chunks — both must route.
{
  const text = "ich habe viele Bücher im Regal stehen.\n\nein Kind spielt fröhlich im Garten.";
  const ctx = Document.fromText(text, { language: "german" }).context("Buch");
  assert.ok(
    ctx.text.includes("Bücher"),
    `fromText + language='german' should find Bücher; got: ${JSON.stringify(ctx.text)}`,
  );
}

// ── Default preserved when omitted (negative case) ───────────────────────
//
// English stemming doesn't unify Bücher↔Buch (different languages), so
// the German query MISSES under default English. Proves the kwarg
// actually does something rather than always returning the chunk.
{
  const ctx = Document.fromChunks(germanCorpus()).context("Buch");
  assert.ok(
    !ctx.text.includes("Bücher"),
    `Default English analyzer should NOT find Bücher from "Buch"; got: ${JSON.stringify(ctx.text)}`,
  );
}

// ── Validation: unknown language must throw (mirrors Rust T44) ───────────
//
// Silent fallback to English on a typo would let a ranking regression
// hide in production.
for (const ctor of ["fromChunks", "fromText"]) {
  let threw = false;
  let msg = "";
  try {
    if (ctor === "fromChunks") {
      Document.fromChunks(germanCorpus(), { language: "germann" });
    } else {
      Document.fromText("ich habe Bücher", { language: "germann" });
    }
  } catch (e) {
    threw = true;
    msg = String(e.message || e);
  }
  assert.ok(threw, `${ctor}({language:"germann"}) should throw, not fall back to English`);
  assert.ok(
    /unknown language/i.test(msg),
    `${ctor} error should mention 'unknown language'; got: ${JSON.stringify(msg)}`,
  );
  assert.ok(
    msg.includes("germann"),
    `${ctor} error should echo the bad name; got: ${JSON.stringify(msg)}`,
  );
}

// ── All 18 advertised Snowball builtins are reachable ────────────────────
//
// Catches drift between the unknown-language error message's list and
// the actual by_name() mapping — a name listed in one but not the other
// would leave a builtin unreachable from Node while looking supported.
const ALL_18 = [
  "english", "german", "french", "spanish", "italian", "portuguese",
  "dutch", "russian", "swedish", "norwegian", "danish", "finnish",
  "romanian", "hungarian", "turkish", "arabic", "greek", "tamil",
];
for (const language of ALL_18) {
  const ctx = Document.fromChunks([new Chunk("the quick brown fox jumps over the lazy dog")], { language }).context("fox");
  assert.ok(typeof ctx.text === "string", `language="${language}" should round-trip`);
}

// ── Char-ngram subword tier + per-field BM25 weights (catalog regime) ─────
//
// Pins the string -> CharNgramAnalyzer mapping and the bm25FieldWeights
// option at the napi boundary.
const catalogCorpus = () => [
  new Chunk("lays classic salted potato chips"),
  new Chunk("kurkure masala munch namkeen"),
  new Chunk("bingo mad angles tomato"),
];

// language="char_ngram" recovers a brand typo ("lays" -> "1ays") that the
// word-token analyzer scores at zero.
{
  const raw = Document.fromChunks(catalogCorpus(), { language: "raw" }).context("1ays");
  const ng = Document.fromChunks(catalogCorpus(), { language: "char_ngram" }).context("1ays");
  assert.ok(!raw.text.includes("lays"), "word-token analyzer should miss the typo'd brand");
  assert.ok(ng.text.includes("lays"), "char_ngram should recover the typo'd brand");
}

// The "char_ngram:MIN-MAX" tuning form round-trips.
{
  const ctx = Document.fromChunks(catalogCorpus(), { language: "char_ngram:2-4" }).context("1ays");
  assert.ok(ctx.text.includes("lays"), "char_ngram:2-4 should recover the typo'd brand");
}

// Per-field BM25 weights: a 3-vector is accepted; wrong arity throws.
{
  const ctx = Document.fromChunks(catalogCorpus(), { bm25FieldWeights: [1.0, 1.0, 2.0] }).context("chips");
  assert.ok(ctx.text.length > 0, "field-weighted retrieval should build + retrieve");
  assert.throws(
    () => Document.fromChunks(catalogCorpus(), { bm25FieldWeights: [1.0, 1.0] }),
    "wrong-arity bm25FieldWeights should throw",
  );
}

console.log(`✓ node analyzer tests passed (${4 + 2 + ALL_18.length + 5} assertions)`);
