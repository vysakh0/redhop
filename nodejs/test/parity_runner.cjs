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
} = require("../index.js");

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

  try {
    let result;
    switch (req.fn) {
      case "buildContext":     result = buildContext(...req.args); break;
      case "filterContext":    result = filterContext(...req.args); break;
      case "analyzeContext":   result = analyzeContext(...req.args); break;
      case "contextEconomics":
        // Returns a JSON string; reparse so the python diff compares the same
        // typed shape on both sides.
        result = JSON.parse(contextEconomics(...req.args));
        break;
      case "groundingScore":   result = groundingScore(...req.args); break;
      case "linkStrength":     result = linkStrength(...req.args); break;
      default:
        process.stdout.write(JSON.stringify({ ok: false, error: `unknown fn '${req.fn}'` }));
        return;
    }
    process.stdout.write(JSON.stringify({ ok: true, result }));
  } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, error: e.message || String(e) }));
  }
});
