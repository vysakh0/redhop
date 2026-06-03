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
- `report` — `{ autoDecision, totalTokens, retainedEvidenceRatio, nExpanded, rendered }`

`neighbors` / `includeHeading` turn on structural context expansion (adjacent
chunks / section headings, in document order).

`analyze(query)` is the same retrieve + score pass without assembling the
prompt — useful for auditing what RedHop would do before paying assembly
cost. Returns just the `report`.

`fromFolder` exposes two more getters: `doc.nFiles` (count of indexed
files) and `doc.skippedFiles` (`{ source, reason }[]` — files that
couldn't be parsed: unsupported formats, unreadable bytes, scanned PDFs
without OCR, etc.). Single-source constructors default to `nFiles=1`
and `skippedFiles=[]`.

## Build from source

```sh
npm install        # gets @napi-rs/cli
npm run build      # builds the native .node (release)
npm test
```
