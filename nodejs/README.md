# redhop (Node.js)

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

We measured 121 labeled queries across 6 real document shapes (legal MSA, API
ref, financial report, incident runbook, 101-page handbook, multi-file folder)
and **lexical won or tied on 5 of 6.** Don't reach for a model unless you have
a measured reason. Full data:
[CORPUS_CONFIG_MATRIX](https://github.com/vysakh0/redhop/blob/main/docs/findings/CORPUS_CONFIG_MATRIX.md).

```js
// Default — works for most docs (code, API refs, runbooks, financials, handbooks)
Document.fromFile("contract.pdf").context("What is the governing law?");

// Structured docs with parallel clauses (regional overrides, sub-policies):
Document.fromFile("msa.pdf", { retrieval: "hybrid", model: "bge-small" })
  .context("What law applies in the UK?", undefined, 1, true);  // neighbors=1, includeHeading=true

// Synonym-heavy corpora — verify on your corpus first; rerank adds 5–10× latency
// and added 0 measured accuracy on the 6 we tested.
Document.fromFile("support.md",
  { retrieval: "hybrid", model: "bge-small", rerank: "cross-encoder" });
```

`options.retrieval` is `"lexical"` (default), `"hybrid"` (BM25 → dense rerank),
or `"semantic"` (dense over every chunk). Dense tiers download a small model
named by `options.model` (`"bge-small"` / `"bge-base"`). The 60-second
decision guide:
[CHOOSING_A_CONFIG](https://github.com/vysakh0/redhop/blob/main/docs/CHOOSING_A_CONFIG.md).

## The result

`context(query, budget?, neighbors?, includeHeading?)` returns:

- `text` — the assembled prompt string
- `chunks` — the selected chunk texts, in order
- `citations` — `{ source, page, heading, line, text }` per chunk (`null`/absent
  fields where a format doesn't provide them)
- `report` — `{ autoDecision, totalTokens, retainedEvidenceRatio, nExpanded, rendered }`

`neighbors` / `includeHeading` turn on structural context expansion (adjacent
chunks / section headings, in document order).

## Build from source

```sh
npm install        # gets @napi-rs/cli
npm run build      # builds the native .node (release)
npm test
```
