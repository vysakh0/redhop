// Lexical, offline, self-contained smoke tests for the Node binding.
const assert = require("node:assert");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { Chunk, Document, groundingScore, linkStrength } = require("../index.js");

// from_text → context → citations + report
let ctx = Document.fromText("the refund window is thirty days from purchase").context("refund window");
assert.ok(ctx.text.toLowerCase().includes("refund"));
assert.strictEqual(ctx.citations[0].source, "document");
assert.ok(ctx.citations[0].page == null); // None -> undefined in JS
assert.strictEqual(ctx.report.autoDecision, "passthrough");
assert.ok(ctx.report.rendered.includes("RedHop Decision Report"));

// from_chunks (typed Chunk only)
assert.ok(
  Document.fromChunks([new Chunk("alpha one"), new Chunk("beta two refund")])
    .context("refund").chunks.length >= 1,
);

// analyze: returns a Report without assembling the prompt; same shape as ctx.report
{
  const doc = Document.fromText("the refund window is thirty days from purchase");
  const report = doc.analyze("refund");
  assert.ok(typeof report.autoDecision === "string", "analyze().autoDecision should be a string");
  assert.ok(typeof report.totalTokens === "number", "analyze().totalTokens should be a number");
  assert.ok(typeof report.rendered === "string" && report.rendered.length > 0, "analyze().rendered should be non-empty");
  assert.ok(!("chunks" in report), "analyze() returns a Report, NOT a BuiltContext (no .chunks/.citations)");
}

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rh-node-"));
fs.writeFileSync(path.join(dir, "notes.txt"), "the warranty lasts one year");
fs.writeFileSync(path.join(dir, "m.py"), "def login(u):\n    return make_token(u)\n");
fs.mkdirSync(path.join(dir, "sub"));
fs.writeFileSync(path.join(dir, "sub", "ship.md"), "# Shipping\norders ship in two days");

// from_file: text source tracked
ctx = Document.fromFile(path.join(dir, "notes.txt")).context("warranty year");
assert.ok(ctx.citations[0].source.endsWith("notes.txt"));

// from_file: code → symbol citation, kept verbatim
ctx = Document.fromFile(path.join(dir, "m.py")).context("make token login");
assert.strictEqual(ctx.citations[0].heading, "def login(u)");
assert.ok(ctx.citations[0].text.includes("\n"));

// from_bytes: markdown heading
ctx = Document.fromBytes(Buffer.from("# Policy\n\nrefund within thirty days"), "p.md").context("refund thirty");
assert.strictEqual(ctx.citations[0].heading, "Policy");

// from_folder + ignore globs
{
  const folderDoc = Document.fromFolder(dir);
  assert.ok(folderDoc.chunkCount >= 3);
  assert.ok(folderDoc.nFiles >= 3, `expected ≥3 indexed files; got ${folderDoc.nFiles}`);
  assert.ok(Array.isArray(folderDoc.skippedFiles), "skippedFiles should be an array");
}
ctx = Document.fromFolder(dir, { ignore: ["**/*.md"] }).context("shipping orders ship");
assert.ok(ctx.citations.every((c) => !c.source.endsWith(".md")));

// single-source ctors default to nFiles=1, skippedFiles=[]
{
  const sd = Document.fromText("hello world");
  assert.strictEqual(sd.nFiles, 1, "single-source ctor should have nFiles=1");
  assert.deepStrictEqual(sd.skippedFiles, [], "single-source ctor should have empty skippedFiles");
}

// from_folder with a bad file: the bad one is in skippedFiles with a reason
{
  const baddir = fs.mkdtempSync(path.join(os.tmpdir(), "rh-skip-"));
  fs.writeFileSync(path.join(baddir, "good.md"), "# Title\n\nrefund within 30 days");
  fs.writeFileSync(path.join(baddir, "bad.txt"), "");  // empty — NoText
  const fd = Document.fromFolder(baddir);
  assert.strictEqual(fd.nFiles, 1, `only the good file should index; got nFiles=${fd.nFiles}`);
  assert.strictEqual(fd.skippedFiles.length, 1, `the empty file should be skipped; got ${JSON.stringify(fd.skippedFiles)}`);
  assert.ok(fd.skippedFiles[0].source.endsWith("bad.txt"));
  assert.ok(typeof fd.skippedFiles[0].reason === "string" && fd.skippedFiles[0].reason.length > 0);
  fs.rmSync(baddir, { recursive: true, force: true });
}

// structural expansion
ctx = Document.fromText("Alpha clause here. The xyzzy term governs payment. Gamma clause here.", { chunkSize: 6 })
  .context("xyzzy term", undefined, 1);
assert.ok(ctx.report.nExpanded >= 1);

fs.rmSync(dir, { recursive: true, force: true });
// from_folder persist: writes an incremental on-disk index, reload reuses it
const pdir = fs.mkdtempSync(path.join(os.tmpdir(), "rh-persist-"));
fs.writeFileSync(path.join(pdir, "x.txt"), "the refund window is thirty days");
Document.fromFolder(pdir, { persist: true });
assert.ok(fs.existsSync(path.join(pdir, ".redhop", "index.json")), "persist index written");
let pctx = Document.fromFolder(pdir, { persist: true }).context("refund window");
assert.ok(pctx.citations.some((c) => c.source.endsWith("x.txt")));
fs.rmSync(pdir, { recursive: true, force: true });

// observability primitives — match Python's redhop.grounding_score / link_strength
{
  const g = groundingScore("refund window", "the refund window is thirty days");
  assert.ok(g > 0 && g <= 1, `groundingScore should be in (0,1] when query terms appear in text; got ${g}`);
  const g0 = groundingScore("xyzzy frobnicate", "the refund window is thirty days");
  assert.strictEqual(g0, 0, `groundingScore should be 0 when no terms overlap; got ${g0}`);
  const l = linkStrength("refund within thirty days", "thirty-day refund policy");
  assert.ok(l > 0 && l <= 1, `linkStrength should be in (0,1] when chunks share terms; got ${l}`);
}

// low-level context functions — caller brings their own chunks (mirrors
// Python's redhop.build_context / filter_context / analyze_context /
// context_economics).
{
  const { buildContext, filterContext, analyzeContext, contextEconomics } = require("../index.js");
  const query = "what is the refund window?";
  const chunks = [
    new Chunk("The refund window is thirty days from the purchase date.", { id: "g1" }),
    new Chunk("Customers may return items within 30 days for a full refund.", { id: "g2" }),
    new Chunk("Photosynthesis converts sunlight into glucose in plants.", { id: "d1" }),
  ];

  // buildContext: assembles + filters + reports
  const ctx = buildContext(query, chunks);
  assert.ok(ctx.text.includes("refund"), "buildContext should keep the on-topic chunks");
  assert.ok(!ctx.text.includes("Photosynthesis"), "buildContext should drop the off-topic chunk");
  assert.ok(ctx.report.totalTokens > 0, "buildContext should report token totals");

  // filterContext: same as buildContext but no budget cap
  const filtered = filterContext(query, chunks);
  assert.ok(filtered.text.includes("refund"), "filterContext should keep on-topic");

  // analyzeContext: report only, no assembly
  const report = analyzeContext(query, chunks);
  assert.ok(typeof report.totalTokens === "number", "analyzeContext should return a Report");
  assert.ok(!("chunks" in report), "analyzeContext result is a Report, not a BuiltContext");

  // contextEconomics: returns a JSON string with economics fields
  const econJson = contextEconomics(query, chunks);
  const econ = JSON.parse(econJson);
  assert.ok(typeof econ === "object" && econ !== null, "contextEconomics should JSON-parse to an object");
  assert.ok("evidence_density" in econ || "evidenceDensity" in econ, `contextEconomics object should include evidence_density: ${JSON.stringify(econ)}`);
}

// Document-config kwargs surfaced in 0.3.2 (codeNeighborsDefault, proseHeadingDefault).
// See docs/findings/CODE_NEIGHBORS_DEFAULT.md and PROSE_HEADING_DEFAULT.md.
{
  // Re-create a code file (the earlier tmpdir was cleaned up around line 86).
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "rh-kwargs-"));
  const codeFile = path.join(tmp, "m.py");
  fs.writeFileSync(
    codeFile,
    "def login(u):\n    return make_token(u)\n\ndef logout(u):\n    return revoke(u)\n",
  );
  const q = "make token login";

  // Default: code chunks auto-pull neighbors=1 (matches Python parity).
  const rDefault = Document.fromFile(codeFile, { tokenBudget: 400 })
    .context(q).report;
  // Explicit 0: no auto-expansion.
  const rOff = Document.fromFile(codeFile, { tokenBudget: 400, codeNeighborsDefault: 0 })
    .context(q).report;
  assert.ok(
    rOff.nExpanded === 0,
    `codeNeighborsDefault=0 must suppress auto-expansion, got nExpanded=${rOff.nExpanded}`,
  );
  // On this tiny one-function file the default may or may not have neighbors to pull;
  // what matters is that =0 produces nExpanded=0. The Python suite has the larger
  // body-marker probe with multi-chunk files.

  // proseHeadingDefault flag is accepted on every constructor and doesn't crash.
  const dt = Document.fromText("the refund window is thirty days", { proseHeadingDefault: false });
  assert.ok(dt.context("refund").chunks.length > 0, "proseHeadingDefault=false must still produce chunks");

  fs.rmSync(tmp, { recursive: true, force: true });
}

// report.diagnosis surface and the vocab-mismatch hint case.
{
  const refundDoc = Document.fromChunks([
    new Chunk("Refund Policy. Refunds are available within thirty days of purchase.", { id: "a", source: "policy.md" }),
    new Chunk("Termination for convenience. Either party may terminate this agreement.", { id: "b", source: "policy.md" }),
    new Chunk("Governing Law. This agreement is governed by the laws of California.", { id: "c", source: "policy.md" }),
  ]);

  const healthy = refundDoc.context("refund policy thirty days").report;
  assert.ok(healthy.diagnosis, "report.diagnosis should be present on every report");
  assert.strictEqual(healthy.diagnosis.corpusStatsAvailable, true, "Document path should populate Layer 2");
  assert.ok(Array.isArray(healthy.diagnosis.queryTerms) && healthy.diagnosis.queryTerms.length > 0);
  assert.deepStrictEqual(healthy.diagnosis.hints, [], "healthy query should fire no hints");

  const mismatch = refundDoc.context("How long do I have to cancel and get my money back?").report;
  assert.ok(mismatch.diagnosis.zeroMatchTerms.includes("cancel"), "expected 'cancel' in zeroMatchTerms");
  const codes = mismatch.diagnosis.hints.map((h) => h.code);
  assert.ok(codes.includes("vocab_mismatch"), `expected vocab_mismatch hint, got ${codes}`);
  const h2 = mismatch.diagnosis.hints.find((h) => h.code === "vocab_mismatch");
  assert.ok(h2.evidence.endsWith("MULTIHOP_HYBRID.md"), `wrong evidence: ${h2.evidence}`);
  assert.ok(!h2.message.includes("—"), "hint message must not contain an em dash");
}

console.log("✓ node smoke tests passed");
