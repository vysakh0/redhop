// 11 · Folder indexing — Document.fromFolder(path, { ... }) with
//      .gitignore, ignore globs, and incremental persistent on-disk index.
//
// Real-world scenario:
//   An engineering team has a `docs/` directory with mixed Markdown,
//   code samples, and the occasional vendored upstream file they don't
//   want indexed. They want:
//     - One combined index over all readable files.
//     - .gitignore honored automatically.
//     - Custom ignore globs for the vendored-but-not-gitignored files.
//     - persist:true so the second invocation skips re-indexing
//       unchanged files (incremental on-disk cache, default location
//       `<folder>/.redhop/`).
//
// What this demonstrates:
//   - Document.fromFolder(path) — recursive indexing over a directory.
//   - recursive:false to flat-index just one level.
//   - gitignore:true (default).
//   - ignore: ["glob1", "glob2", ...] — extra gitignore-style globs.
//   - persist:true — incremental on-disk index.
//   - doc.nFiles / doc.skippedFiles — observability.
//   - Document.fromBytes(data, source, ...) — for S3/GCS/DB blobs.
//
// Run:
//   node examples/nodejs/11_folder_indexing.cjs

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { Document } = require("redhop");

function setupDemoDocs(root) {
  fs.writeFileSync(
    path.join(root, "README.md"),
    "# Acme Inc Engineering Handbook\n\nWelcome. Start with onboarding.md for new hires.\n",
  );
  fs.writeFileSync(
    path.join(root, "onboarding.md"),
    "# Onboarding\n\nNew hires get a laptop on day 1 and access provisioned in 24 hours.\nTalk to it@acme.com if something is missing.\n",
  );

  fs.mkdirSync(path.join(root, "policies"));
  fs.writeFileSync(
    path.join(root, "policies", "refunds.md"),
    "# Refund Policy\n\nCustomers get a full refund within 30 days of delivery.\n",
  );
  fs.writeFileSync(
    path.join(root, "policies", "shipping.md"),
    "# Shipping Policy\n\nStandard ships in 3-5 business days. Express in 1-2.\n",
  );

  // Vendored upstream file we want to ignore.
  fs.mkdirSync(path.join(root, "vendored"));
  fs.writeFileSync(
    path.join(root, "vendored", "third_party_license.md"),
    "# Apache 2.0 license text\n\nIRRELEVANT BOILERPLATE\n".repeat(30),
  );

  // .gitignore that excludes build/.
  fs.mkdirSync(path.join(root, "build"));
  fs.writeFileSync(path.join(root, "build", "generated.md"), "# generated, ignore me\n");
  fs.writeFileSync(path.join(root, ".gitignore"), "build/\n");
}

function main() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "redhop-folder-demo-"));
  try {
    setupDemoDocs(root);
    console.log(`Demo directory: ${root}\n`);

    // ── Arm A: vanilla recursive index ─────────────────────────
    console.log("─── Arm A · Document.fromFolder(path) ─────────");
    const docA = Document.fromFolder(root);
    console.log(`  files indexed   : ${docA.nFiles}`);
    console.log(`  total chunks    : ${docA.chunkCount}`);
    console.log(`  files skipped   : ${docA.skippedFiles.length}`);
    for (const sk of docA.skippedFiles.slice(0, 3)) {
      console.log(`    - ${sk.source}: ${sk.reason}`);
    }
    console.log();

    const ctx = docA.context("how long do I have to get a refund?");
    if (ctx.citations.length) {
      const top = ctx.citations[0];
      console.log(`  top hit source : ${top.source}`);
      console.log(`  top hit heading: ${top.heading}`);
    }
    console.log();

    // ── Arm B: custom ignore globs ────────────────────────────
    console.log("─── Arm B · ignore: ['vendored/**'] ───────────");
    const docB = Document.fromFolder(root, { ignore: ["vendored/**"] });
    console.log(`  files indexed   : ${docB.nFiles}  (vs Arm A: ${docA.nFiles})`);
    console.log(`  total chunks    : ${docB.chunkCount}`);
    console.log();

    // ── Arm C: persist: true (incremental on-disk index) ──────
    console.log("─── Arm C · persist: true ─────────────────────");
    let t0 = process.hrtime.bigint();
    Document.fromFolder(root, { persist: true });
    const firstRunMs = Number(process.hrtime.bigint() - t0) / 1e6;

    t0 = process.hrtime.bigint();
    const docC2 = Document.fromFolder(root, { persist: true });
    const secondRunMs = Number(process.hrtime.bigint() - t0) / 1e6;

    const cachePath = path.join(root, ".redhop", "index.json");
    console.log(`  cache written   : ${fs.existsSync(cachePath)}`);
    console.log(`  first  run      : ${firstRunMs.toFixed(1).padStart(5)} ms (cold)`);
    console.log(`  second run      : ${secondRunMs.toFixed(1).padStart(5)} ms (warm — re-read cache)`);
    console.log(`  same nFiles     : ${docC2.nFiles}`);
    console.log();

    // ── Arm D: fromBytes (bytes you fetched yourself) ─────────
    console.log("─── Arm D · fromBytes (for S3 / GCS / blobs) ──");
    const data = fs.readFileSync(path.join(root, "policies", "refunds.md"));
    const docD = Document.fromBytes(data, "refunds.md");
    console.log(`  indexed         : ${docD.nFiles} file, ${docD.chunkCount} chunks`);
    const ctxD = docD.context("refund window");
    if (ctxD.citations.length) {
      console.log(`  citation source : ${ctxD.citations[0].source}`);
    }
    console.log();
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }

  console.log("─── When to use what ─────────────────────────────");
  console.log("- fromFolder(path)                 : one combined index");
  console.log("  over a directory. Default `recursive:true`,");
  console.log("  `gitignore:true`.");
  console.log("- ignore: [...]                    : add gitignore-style");
  console.log("  globs (vendored code, generated docs).");
  console.log("- persist: true                    : incremental cache.");
  console.log("- fromBytes(buffer, 'source.pdf')  : bytes from S3/GCS/DB.");
}

main();
