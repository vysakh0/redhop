// Cross-binding parity runner. Reads a JSON request from stdin, runs the
// requested redhop call, writes a JSON response to stdout. The Python
// parity tests use this so they can hand the exact same inputs to both
// bindings and diff the outputs.
//
// Request schema:
//   { "fn": "buildContext" | "filterContext" | "analyzeContext"
//          | "contextEconomics" | "groundingScore" | "linkStrength",
//     "args": [ ... ] }
//
// Response schema:
//   { "ok": true,  "result": <the function's return value> }
//   { "ok": false, "error":  "<error message>" }
//
// Run by hand:
//   echo '{"fn":"groundingScore","args":["refund","the refund window"]}' \
//     | node test/parity_runner.cjs

const {
  buildContext,
  filterContext,
  analyzeContext,
  contextEconomics,
  groundingScore,
  linkStrength,
  Chunk,
} = require("../index.js");

// The Python side ships JSON-wire chunks: `[{ id, text }, ...]`. As of
// 0.3.0 the typed `Chunk` class is required on both bindings, so the
// runner wraps each wire-format dict back into a `new Chunk(...)` before
// handing it to the binding function. Other arg positions pass through
// unchanged.
function rewrapArgs(fn, args) {
  const chunkAware = new Set([
    "buildContext",
    "filterContext",
    "analyzeContext",
    "contextEconomics",
  ]);
  if (!chunkAware.has(fn)) return args;
  // Convention from python/tests/test_parity_node.py: args[1] is the chunk list.
  return args.map((a, i) => {
    if (i !== 1 || !Array.isArray(a)) return a;
    return a.map((c) => {
      if (c && typeof c === "object" && typeof c.text === "string") {
        const { text, id, source, metadata, tokenCount, embedding } = c;
        return new Chunk(text, { id, source, metadata, tokenCount, embedding });
      }
      return c;
    });
  });
}

let buf = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { buf += chunk; });
process.stdin.on("end", () => {
  let req;
  try {
    req = JSON.parse(buf);
  } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, error: `bad request JSON: ${e.message}` }));
    return;
  }

  // BuiltContext became a `#[napi]` class (not a plain object) when
  // `redhop.evaluate(...)` was added — it now carries the underlying Rust
  // struct in addition to the four exposed properties. Class getters
  // aren't enumerable, so JSON.stringify(ctx) yields `{}`. Project to a
  // plain object here so the Python parity diff sees the same shape it
  // always did.
  function projectBuiltContext(ctx) {
    return {
      text: ctx.text,
      chunks: ctx.chunks,
      citations: ctx.citations,
      report: ctx.report,
    };
  }

  try {
    let result;
    const args = rewrapArgs(req.fn, req.args);
    switch (req.fn) {
      case "buildContext":     result = projectBuiltContext(buildContext(...args)); break;
      case "filterContext":    result = projectBuiltContext(filterContext(...args)); break;
      case "analyzeContext":   result = analyzeContext(...args); break;
      case "contextEconomics":
        // Returns a JSON string; reparse so the python diff compares the same
        // typed shape on both sides.
        result = JSON.parse(contextEconomics(...args));
        break;
      case "groundingScore":   result = groundingScore(...args); break;
      case "linkStrength":     result = linkStrength(...args); break;
      default:
        process.stdout.write(JSON.stringify({ ok: false, error: `unknown fn '${req.fn}'` }));
        return;
    }
    process.stdout.write(JSON.stringify({ ok: true, result }));
  } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, error: e.message || String(e) }));
  }
});
