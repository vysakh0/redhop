# redhop (Node.js)

[![npm](https://img.shields.io/npm/v/redhop?label=npm&color=e11d48)](https://www.npmjs.com/package/redhop)
[![Node](https://img.shields.io/node/v/redhop?color=e11d48)](https://www.npmjs.com/package/redhop)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/vysakh0/redhop/blob/main/LICENSE)
[![Evidence layer](https://img.shields.io/badge/evidence-layer-blue)](https://github.com/vysakh0/redhop/tree/main/docs/findings)

Reasoning-aware context runtime for RAG — hand it a document and a question, get
back the context the model should actually see, with citations and a Decision
Report. No vector database, no LLM, in-process. A native addon (napi-rs) over the
RedHop Rust core; the embedding engine and document parsers are bundled — no extra
deps.

```js
const { Document } = require("redhop");

// Point it at a file — PDF, DOCX, PPTX, XLSX, or text/code — and ask.
const doc = Document.fromFile("contract.pdf");
const ctx = doc.context("What is the governing law?");

llm.generate(ctx.text);          // feed any provider — no lock-in
for (const c of ctx.citations) { // where the answer's context came from
  console.log(c.source, c.page, c.heading); // e.g. contract.pdf 3 null
}
console.log(ctx.report.rendered); // the Decision Report — what it kept, and why
```

## How it compares

Measured on identical documents + budgets + BM25 retrieval, RedHop **beats both
frameworks on multi-hop evidence retention** (80% vs LangChain 71%, LlamaIndex 72%)
and **beats LangChain on contracts** (82% vs 73%). It trails LlamaIndex by 4 points
on CUAD's raw-template query — that gap is mechanism-known and closeable with a
6-line query preprocessor (RedHop reaches 88%, +2 over LlamaIndex); see
[CUAD_RECALL_GAP.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_RECALL_GAP.md).
All without a vector database, an agent framework, or model finetuning.

<p align="center">
  <img src="https://raw.githubusercontent.com/vysakh0/redhop/main/.github/retention_vs_frameworks.svg" alt="Evidence retention vs LangChain vs LlamaIndex" width="100%">
</p>

Methodology + raw runs: [FRAMEWORK_COMPARISON.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/FRAMEWORK_COMPARISON.md)
· [framework_comparison_2026-06-06.txt](https://github.com/vysakh0/redhop/blob/main/reports/framework_comparison_2026-06-06.txt).

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/vysakh0/redhop/main/.github/architecture.svg" alt="RedHop pipeline" width="100%">
</p>

Five stages: you bring documents and a query, RedHop owns parsing, chunking,
retrieval, and context allocation, and you get a `BuiltContext` with the
assembled prompt, citations, and a Decision Report. Each stage has an
evidence-backed default that traces to a finding in
[`docs/findings/`](https://github.com/vysakh0/redhop/tree/main/docs/findings).

## The Decision Report

<p align="center">
  <img src="https://raw.githubusercontent.com/vysakh0/redhop/main/.github/decision_report.svg" alt="Sample Decision Report" width="100%">
</p>

`ctx.report.rendered` carries the human-readable text above; individual fields
(`autoDecision`, `totalTokens`, `retainedEvidenceRatio`, etc.) are on
`ctx.report` directly. `Document.analyze(query)` returns the same `Report`
shape without paying assembly cost.

## Loaders

```js
Document.fromText(text, options?)
Document.fromChunks(["…", "…"], options?)
Document.fromFile(path, options?)                 // PDF/DOCX/PPTX/XLSX + text/code
Document.fromBytes(buffer, "key.pdf", options?)   // S3 / GCS / Azure / HTTP / DB blobs
Document.fromFolder(path, folderOptions?)         // one combined index over a dir
```

`fromFolder` honors `.gitignore` and accepts extra `ignore` globs:

```js
Document.fromFolder("./repo", { recursive: true, gitignore: true,
  ignore: ["*.lock", "tests/**"], options: { retrieval: "hybrid", model: "bge-small" } });
```

## Retrieval — start with the default

Start at the lexical default — it handles most document QA because the words
in the question are usually the words in the answer — and climb only when the
failure shape calls for it.

```js
// Default — most docs (code, API refs, runbooks, financial reports, handbooks)
Document.fromFile("contract.pdf").context("What is the governing law?");

// Structured docs with parallel clauses (regional overrides, per-region sub-sections):
Document.fromFile("msa.pdf", { retrieval: "hybrid", model: "bge-small" })
  .context("What law applies in the UK?", undefined, 1, true);  // neighbors=1, includeHeading=true

// Synonym-mismatch corpora (HR FAQs, support tickets where users phrase
// things very differently from the docs). Cross-encoder adds 5–10× latency
// — verify it helps on your corpus before enabling.
Document.fromFile("support.md",
  { retrieval: "hybrid", model: "bge-small", rerank: "cross-encoder" });
```

`options.retrieval` is `"lexical"` (default), `"hybrid"` (BM25 → dense rerank),
or `"semantic"` (dense over every chunk). Dense tiers download a small model
named by `options.model` (`"bge-small"` / `"bge-base"`). The 60-second
decision guide:
[CHOOSING_A_CONFIG](https://github.com/vysakh0/redhop/blob/main/docs/CHOOSING_A_CONFIG.md).

## Non-English content

Default is English Snowball. Swap with `options.language` — any of the
18 Snowball Porter2 languages (`arabic, danish, dutch, english,
finnish, french, german, greek, hungarian, italian, norwegian,
portuguese, romanian, russian, spanish, swedish, tamil, turkish`):

```javascript
const doc = Document.fromText(germanText, { language: "german" });
// Now `Buch` finds chunks containing `Bücher` (and vice versa)
```

One analyzer drives both BM25 retrieval AND the grounding scorer, so
they can't drift on what "the same term" means. Unknown names throw
(we don't silently fall back to English). See the
[language guide](https://github.com/vysakh0/redhop/blob/main/docs/LANGUAGE.md)
for the full breakdown and a calibration disclaimer.

## The result

`context(query, budget?, neighbors?, includeHeading?)` returns:

- `text` — the assembled prompt string
- `chunks` — the selected chunk texts, in order
- `citations` — `{ source, page, heading, line, text }` per chunk (`null`/absent
  fields where a format doesn't provide them)
- `report` — the Decision Report, with the same field surface as Python's
  `ctx.report`. Read `strategy` / `requestedStrategy` for the resolved
  vs requested allocation; `autoDecision` for the Auto gate's verdict
  (`"passthrough"` | `"prune"` | `"not_auto"`); `inputTokens` /
  `tokenBudget` / `totalTokens` / `tokenUtilization` for budget
  accounting; `nInputChunks` / `nSelected` / `nExpanded` for chunk
  counts; `inputDistractorRatio` / `retainedEvidenceRatio` /
  `evidenceDensity` / `distractorRatio` / `estimatedWasteTokens` for
  context economics; `secondHopRescues` (or its longer alias
  `secondHopRescueCount`) and `reasoningPreservationDelta` for the
  reasoning-preserving accounting; `lowConfidenceRetrieval` /
  `lowConfidenceThreshold` for the "did anything actually match"
  signal; and `rendered` for the human-readable Decision Report
  string. The full shape is in `index.d.ts`.

`neighbors` / `includeHeading` turn on structural context expansion (adjacent
chunks / section headings, in document order).

`analyze(query)` is the same retrieve + score pass without assembling the
prompt — useful for auditing what RedHop would do before paying assembly
cost. Returns just the `report`.

Standalone observability primitives (the same scoring the strategies use
internally, exposed so external code never has to reimplement and drift):

```javascript
const { groundingScore, linkStrength } = require("redhop");

groundingScore("refund window", chunkText);   // → number in [0, 1]
linkStrength(chunkA, chunkB);                  // → number in [0, 1]
```

Both use the default English analyzer; non-English content reaches the
configured analyzer through `Document.context(...).report` instead.

`fromFolder` exposes two more getters: `doc.nFiles` (count of indexed
files) and `doc.skippedFiles` (`{ source, reason }[]` — files that
couldn't be parsed: unsupported formats, unreadable bytes, scanned PDFs
without OCR, etc.). Single-source constructors default to `nFiles=1`
and `skippedFiles=[]`.

## Templated workloads — the +6 retention lift

If every query in your workload follows a fixed template — legal QA
(`"Highlight the parts (if any) of this contract related to X. Details: …"`),
support-ticket triage (`"Help me with X, my account is Y, the error is Z"`),
form-filled queries from a structured UI — **BM25 weights every query term
by corpus IDF, not by how often the term repeats across your query set**.
The boilerplate words dilute the real signal words, and retention suffers.
This is the mechanism behind the 4-point CUAD gap on the head-to-head;
stripping the template at *your* boundary closes it.

**Measured** on the CUAD framework comparison (n=300, BM25, budget 2,000 tok):
≥0.8 evidence retention goes **82% → 88%** with a six-line stripper,
overtaking LlamaIndex's 86% by 2 points. Full mechanism + numbers:
[CUAD_RECALL_GAP.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_RECALL_GAP.md).

Recommended workflow: **detect → strip → A/B**, with two helpers in the
public API:

```javascript
const redhop = require("redhop");

// 1 — Detect. Hand a representative sample of your queries to the analyzer.
const report = redhop.analyzeQuerySet(myQueries.slice(0, 300));
// report.isTemplated            → true / false
// report.templateWordShare      → e.g. 0.66 on CUAD
// report.boilerplateTerms       → ["highlight", "contract", "lawyer", …]
// report.estimatedDilutionCost  → "high" | "medium" | "low" | "none"
// report.suggestedAction        → human-readable recommendation

if (report.isTemplated) {
  // 2 — Strip. Use the boilerplate the analyzer found.
  const strip = (q) => redhop.dropTemplateTerms(q, report.boilerplateTerms);

  // 3 — A/B. redhop.evaluate scores both arms deterministically,
  //     no LLM judge — see EVALUATE_API.md for the design.
  const doc = await redhop.Document.fromFile("contract.pdf");
  const evalA = redhop.evaluate(
    userQuery,
    doc.context(userQuery, { strategy: "raw_topk" }),
    { goldChunks: yourGoldChunkIds },
  );
  const evalB = redhop.evaluate(
    strip(userQuery),
    doc.context(strip(userQuery), { strategy: "raw_topk" }),
    { goldChunks: yourGoldChunkIds },
  );
  console.log(evalB.overall - evalA.overall);  // the lift, deterministically
}
```

- **Only matters if your queries are templated.** `analyzeQuerySet` is
  conservative by design — HotpotQA and MuSiQue both register quiet
  (`isTemplated: false`) in the cross-workload probe; CUAD fires. If
  yours doesn't fire, skip this section.
- **The analyzer measures the *shape* of your query set, not your
  retention.** It says "this *looks* like a templated workload" with
  the boilerplate terms it found; it does **not** promise a specific
  lift. Always A/B on your gold-evidence sample before committing.
- **For single-doc extraction workloads also set `strategy: "raw_topk"`.**
  `auto` routes large contexts to `reasoning_preserving`, which solves a
  multi-hop problem contract extraction doesn't have. RawTopK beats it
  by ~4 points at every chunk size on CUAD.
- **We deliberately don't ship a CUAD-specific `stripTemplate()`
  helper.** Templates are workload-specific; baking one in would make
  the wrong call for the next workload. `dropTemplateTerms` takes
  *your* boilerplate so the call stays on your side.

Decision rule + the recipe on the docs site:
[Choosing a configuration → "Templated queries with heavy boilerplate"](https://www.redhopai.com/docs/choosing-a-config/#3-templated-queries-with-heavy-boilerplate).
Cross-workload probe that validated the analyzer:
[QUERY_SET_ANALYZER.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/QUERY_SET_ANALYZER.md).
Design rationale + tradeoffs for `evaluate`:
[EVALUATE_API.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/EVALUATE_API.md).

## Build from source

```sh
npm install        # gets @napi-rs/cli
npm run build      # builds the native .node (release)
npm test
```
