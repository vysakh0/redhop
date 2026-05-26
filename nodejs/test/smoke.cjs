// Lexical, offline, self-contained smoke tests for the Node binding.
const assert = require("node:assert");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { Document } = require("../index.js");

// from_text → context → citations + report
let ctx = Document.fromText("the refund window is thirty days from purchase").context("refund window");
assert.ok(ctx.text.toLowerCase().includes("refund"));
assert.strictEqual(ctx.citations[0].source, "document");
assert.ok(ctx.citations[0].page == null); // None -> undefined in JS
assert.strictEqual(ctx.report.autoDecision, "passthrough");
assert.ok(ctx.report.rendered.includes("RedHop Decision Report"));

// from_chunks
assert.ok(Document.fromChunks(["alpha one", "beta two refund"]).context("refund").chunks.length >= 1);

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
assert.ok(Document.fromFolder(dir).chunkCount >= 3);
ctx = Document.fromFolder(dir, { ignore: ["**/*.md"] }).context("shipping orders ship");
assert.ok(ctx.citations.every((c) => !c.source.endsWith(".md")));

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

console.log("✓ node smoke tests passed");
