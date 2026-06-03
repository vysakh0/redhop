// Binding-surface tests for the pluggable lexical analyzer.
//
// Mirrors the Rust `quality_suite::t41`-`t44` matrix through the napi
// binding so a dropped `language` field or a wrong string-to-analyzer
// mapping at the FFI boundary surfaces here, not in user code.
//
// Run with: node test/analyzer.cjs   (or `npm test` — wired into the
// default test script).

const assert = require("node:assert");
const { Document } = require("../index.js");

const germanCorpus = () => [
  "ich habe viele Bücher im Regal stehen",
  "ein Kind spielt fröhlich im Garten",
];

const frenchCorpus = () => [
  "il aime manger des pommes chaque matin",
  "le chien court dans la rue très vite",
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
  const ctx = Document.fromChunks(["the quick brown fox jumps over the lazy dog"], { language }).context("fox");
  assert.ok(typeof ctx.text === "string", `language="${language}" should round-trip`);
}

console.log(`✓ node analyzer tests passed (${4 + 2 + ALL_18.length} assertions)`);
